"""Walking task for ZBot."""

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Self, TypedDict

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from ksim.actuators import NoiseType, StatefulActuators
from ksim.types import Metadata, PhysicsData
from ksim.utils.mujoco import get_ctrl_data_idx_by_name

logger = logging.getLogger(__name__)

NUM_JOINTS = 20
NUM_COMMANDS = 7
NUM_ACTOR_INPUTS = 43 + NUM_COMMANDS      # 50
NUM_CRITIC_INPUTS = 476 + NUM_COMMANDS     # 483


def get_servo_deadband() -> tuple[float, float]:
    """Get deadband values based on current servo configuration."""
    encoder_resolution = 0.087 * jnp.pi / 180  # radians

    pos_deadband = 2 * encoder_resolution
    neg_deadband = 2 * encoder_resolution

    return pos_deadband, neg_deadband


# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("right_hip_yaw", 0.0),
    ("right_hip_roll", 0.0),
    ("right_hip_pitch", 0.0),
    ("right_knee_pitch", 0.0),
    ("right_ankle_pitch", 0.0),
    ("right_ankle_roll", 0.0),
    ("left_hip_yaw", 0.0),
    ("left_hip_roll", 0.0),
    ("left_hip_pitch", 0.0),
    ("left_knee_pitch", 0.0),
    ("left_ankle_pitch", 0.0),
    ("left_ankle_roll", 0.0),
    ("left_shoulder_pitch", 0.0),
    ("left_shoulder_roll", 0.0),
    ("left_elbow_roll", 0.0),
    ("left_gripper_roll", 0.0),
    ("right_shoulder_pitch", 0.0),
    ("right_shoulder_roll", 0.0),
    ("right_elbow_roll", 0.0),
    ("right_gripper_roll", 0.0),
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlannerState:
    position: Array
    velocity: Array
    last_computed_torque: Array


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in ZEROS}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Comment out because we now sample deltas not absolute positions
        # mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_CRITIC_INPUTS
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        self.actor = Actor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=1.0,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class ZbotWalkingTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )
    render_distance: float = xax.field(
        value=0.8,
        help="The distance to the render camera.",
    )


class FeetechParams(TypedDict):
    sysid: str
    max_torque: float
    armature: float
    frictionloss: float
    damping: float
    vin: float
    kt: float
    R: float
    vmax: float
    amax: float
    max_velocity: float
    max_pwm: float
    error_gain: float


def trapezoidal_step(
    state: PlannerState,
    target_position: Array,
    dt: float,
    v_max: Array,
    a_max: Array,
    positive_deadband: float,
    negative_deadband: float,
) -> tuple[PlannerState, tuple[Array, Array]]:
    position_error = target_position - state.position

    # Determine which deadband to use based on error direction
    deadband_threshold = jnp.where(position_error >= 0, positive_deadband, negative_deadband)

    in_deadband = jnp.abs(position_error) <= deadband_threshold

    # Deadband behavior: gradually decay velocity
    decay_factor = 0.8  # Tunable parameter - could be measured from real servo
    deadband_velocity = state.velocity * decay_factor
    deadband_position = state.position + deadband_velocity * dt

    # Planning behavior: normal trapezoidal planning
    target_direction = jnp.sign(position_error)

    # Calculate stopping distance for current velocity
    stopping_distance = jnp.abs(state.velocity**2) / (2 * a_max)

    # Check if velocity is aligned with target direction
    velocity_direction = jnp.sign(state.velocity)
    moving_towards_target = velocity_direction * target_direction >= 0

    should_accelerate = jnp.logical_and(moving_towards_target, jnp.abs(position_error) > stopping_distance)

    # Choose acceleration
    acceleration = jnp.where(
        should_accelerate,
        target_direction * a_max,  # Accelerate towards target
        -velocity_direction * a_max,  # Decelerate (oppose current velocity)
    )

    # Handle zero velocity case
    acceleration = jnp.where(
        jnp.abs(state.velocity) < 1e-6,
        target_direction * a_max,
        acceleration,  # If stopped, accelerate towards target
    )

    planning_velocity = state.velocity + acceleration * dt
    planning_velocity = jnp.clip(planning_velocity, -v_max, v_max)
    planning_position = state.position + planning_velocity * dt

    # Use element-wise where to select behavior for each joint
    new_velocity = jnp.where(in_deadband, deadband_velocity, planning_velocity)
    new_position = jnp.where(in_deadband, deadband_position, planning_position)

    new_state = PlannerState(
        position=new_position, velocity=new_velocity, last_computed_torque=state.last_computed_torque
    )

    return new_state, (new_position, new_velocity)


class FeetechActuators(StatefulActuators):
    """Feetech actuator controller."""

    def __init__(
        self,
        max_torque_j: Array,
        kp_j: Array,
        kd_j: Array,
        max_velocity_j: Array,
        max_pwm_j: Array,
        vin_j: Array,
        kt_j: Array,
        r_j: Array,
        vmax_j: Array,
        amax_j: Array,
        error_gain_j: Array,
        dt: float,
        action_noise: float = 0.005,
        action_noise_type: NoiseType = "gaussian",
        torque_noise: float = 0.02,
        torque_noise_type: NoiseType = "gaussian",
    ) -> None:
        self.max_torque_j = max_torque_j
        self.kp_j = kp_j
        self.kd_j = kd_j
        self.max_velocity_j = max_velocity_j
        self.max_pwm_j = max_pwm_j
        self.vin_j = vin_j
        self.kt_j = kt_j
        self.r_j = r_j
        self.vmax_j = vmax_j
        self.amax_j = amax_j
        self.error_gain_j = error_gain_j
        self.dt = dt
        # self.prev_qtarget_j = jnp.zeros_like(self.kp_j)
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type
        self.debug_counter = 0
        self.positive_deadband = get_servo_deadband()[0]
        self.negative_deadband = get_servo_deadband()[1]

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PlannerState,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PlannerState]:
        """Compute torque control with velocity smoothing and duty cycle clipping (JAX friendly)."""
        pos_rng, tor_rng = jax.random.split(rng)

        current_pos_j = physics_data.qpos[7:]
        current_vel_j = physics_data.qvel[6:]

        planner_state = actuator_state
        planner_state, (desired_position, desired_velocity) = trapezoidal_step(
            planner_state, action, self.dt, self.vmax_j, self.amax_j, self.positive_deadband, self.negative_deadband
        )

        pos_error_j = desired_position - current_pos_j
        vel_error_j = desired_velocity - current_vel_j

        # Compute raw duty cycle and clip by max_pwm
        raw_duty_j = self.kp_j * self.error_gain_j * pos_error_j + self.kd_j * vel_error_j
        duty_j = jnp.clip(raw_duty_j, -self.max_pwm_j, self.max_pwm_j)

        # Compute torque
        volts_j = duty_j * self.vin_j
        torque_j = volts_j * self.kt_j / self.r_j

        new_planner_state = PlannerState(
            position=planner_state.position,  # Updated by trapezoidal_step
            velocity=planner_state.velocity,  # Updated by trapezoidal_step
            last_computed_torque=torque_j,  # New computed torque
        )

        # Add noise to torque
        torque_j_noisy = self.add_noise(self.torque_noise, self.torque_noise_type, torque_j, tor_rng)

        return torque_j_noisy, new_planner_state

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        return physics_data.qpos[7:]

    def get_default_state(
        self, initial_position: Array, initial_velocity: Array, initial_last_computed_torque: Array
    ) -> PlannerState:
        """Initialize the planner state with the provided position and velocity."""
        return PlannerState(
            position=initial_position, velocity=initial_velocity, last_computed_torque=initial_last_computed_torque
        )

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PlannerState:
        """Implement abstract method to initialize planner state from physics data."""
        initial_position = physics_data.qpos[7:]
        initial_velocity = physics_data.qvel[6:]
        initial_last_computed_torque = jnp.zeros_like(initial_position)
        return self.get_default_state(initial_position, initial_velocity, initial_last_computed_torque)


@attrs.define(frozen=True, kw_only=True)
class FeetechTorqueObservation(ksim.Observation):
    """Observation that returns the actual computed Feetech torque."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # Simply return the torque that was computed by the actuator
        actuator_state = state.physics_state.actuator_state
        return actuator_state.last_computed_torque


class ZbotWalkingTask(ksim.PPOTask[ZbotWalkingTaskConfig]):
    delta_max_j: jnp.ndarray | None = None  # set later in get_actuators

    def get_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("zbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("zbot"))
        # Ensure we're returning a proper RobotURDFMetadataOutput
        if not isinstance(metadata, Metadata):
            raise ValueError("Metadata is not a Metadata")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: Metadata | None = None,
    ) -> FeetechActuators:
        vmax_default = 5.0  # rad · s⁻¹ fallback
        amax_default = 17.45

        if metadata is None:
            raise ValueError("metadata must be provided")
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata must be provided")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata must be provided")

        joint_meta = metadata.joint_name_to_metadata
        actuator_meta = metadata.actuator_type_to_metadata

        ctrl_indices = get_ctrl_data_idx_by_name(physics_model)  # {actuator_name: idx}
        joint_order = sorted(ctrl_indices.keys(), key=lambda x: ctrl_indices[x])

        for actuator_name in joint_order:
            joint_name = actuator_name.split("_ctrl")[0]
            if joint_name not in joint_meta:
                raise ValueError(f"Joint '{joint_name}' not found in metadata")

        def param(joint: str, field: str, *, default: float | None = None) -> float:
            if joint_meta[joint].actuator_type is None:
                raise ValueError(f"Joint '{joint}' has no actuator type specified in metadata")

            actuator_type = joint_meta[joint].actuator_type
            if actuator_type not in actuator_meta:
                raise ValueError(f"Actuator type '{actuator_type}' for joint '{joint}' not found in metadata")

            if actuator_meta[actuator_type] is None:
                raise ValueError(f"Actuator metadata for type '{actuator_type}' is None")

            value = getattr(actuator_meta[actuator_type], field)
            if value is None:
                if default is None:
                    raise ValueError(f"Parameter '{field}' missing for joint '{joint}'")
                return default
            return float(value)

        max_torque = []
        max_vel = []
        max_pwm = []
        vin = []
        kt = []
        r = []
        vmax = []
        amax = []
        err_gain = []
        kp = []
        kd = []

        for joint in joint_order:
            name = joint.split("_ctrl")[0]
            max_torque.append(param(name, "max_torque"))
            max_vel.append(param(name, "max_velocity"))
            max_pwm.append(param(name, "max_pwm"))
            vin.append(param(name, "vin"))
            kt.append(param(name, "kt"))
            r.append(param(name, "R"))
            vmax.append(param(name, "vmax", default=vmax_default))
            amax.append(param(name, "amax", default=amax_default))
            err_gain.append(param(name, "error_gain"))

            if (kp_str := joint_meta[name].kp) is None:
                raise ValueError(f"Joint '{name}' has no kp specified in metadata")
            kp.append(float(kp_str))

            if (kd_str := joint_meta[name].kd) is None:
                raise ValueError(f"Joint '{name}' has no kd specified in metadata")
            kd.append(float(kd_str))

        # Reachability vector -- Δmax = v max · Δt  +  ½ a max · Δt²
        delta_max = jnp.array(vmax) * self.config.ctrl_dt + 0.5 * jnp.array(amax) * self.config.ctrl_dt**2
        self.delta_max_j = delta_max

        # ------------------------------------------------------------------
        # 4. Instantiate controller (lists → jnp.array on‑the‑fly)
        # ------------------------------------------------------------------
        return FeetechActuators(
            max_torque_j=jnp.array(max_torque),
            kp_j=jnp.array(kp),
            kd_j=jnp.array(kd),
            max_velocity_j=jnp.array(max_vel),
            max_pwm_j=jnp.array(max_pwm),
            vin_j=jnp.array(vin),
            kt_j=jnp.array(kt),
            r_j=jnp.array(r),
            vmax_j=jnp.array(vmax),
            amax_j=jnp.array(amax),
            error_gain_j=jnp.array(err_gain),
            dt=self.config.dt,
            action_noise=0.01,
            action_noise_type="none",
            torque_noise=0.01,
            torque_noise_type="none",
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            # ksim.StaticFrictionRandomizer(),
            # ksim.ArmatureRandomizer(),
            # ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.15),
            # ksim.JointDampingRandomizer(),
            # ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
            # ksim.FloorFrictionRandomizer.from_geom_name(
            #     model=physics_model, floor_geom_name="floor", scale_lower=0.3, scale_upper=1.5
            # ),
            # # 1σ ≈ 1.5°, gives ~99.7% within 4.5°
            # # enable yaw randomization with 1σ ≈ 1°
            # # 5mm standard deviation
            # ksim.IMUAlignmentRandomizer(
            #     site_name="imu_site",
            #     tilt_std_rad=math.radians(5),
            #     yaw_std_rad=math.radians(1.0),
            #     translate_std_m=0.005
            # )
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            #    ksim.PushEvent(
            #        x_linvel=0.1,
            #        y_linvel=0.1,
            #        z_linvel=0.05,
            #        x_angvel=0.0,  # angular velocity in rad/s
            #        y_angvel=0.0,
            #        z_angvel=0.0,
            #        vel_range=(0.05, 0.15),
            #        interval_range=(2.0, 4.0)
            #    ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            # ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            # ksim.RandomJointVelocityReset(),
            # ksim.RandomHeadingReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        obs_list = [
            ksim.JointPositionObservation(noise=math.radians(0.00)),
            ksim.JointVelocityObservation(noise=math.radians(0.0)),
            ksim.ActuatorForceObservation(),
            FeetechTorqueObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.05),
                noise=math.radians(1),
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=0.5,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(0),
            ),
        ]

        # Add action-position observation for each joint
        obs_list.extend(
            [
                ksim.ActPosObservation.create(
                    physics_model=physics_model,
                    joint_name=joint_name,
                )
                for joint_name, _ in ZEROS
            ]
        )

        return obs_list

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.JoystickCommand(
                sample_probs=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Always "stand still" during training
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.StayAliveReward(scale=1.0),
            ksim.UprightReward(scale=1.0),
            # ksim.AngularVelocityPenalty(index=("x", "y", "z"), scale=-0.005,scale_by_curriculum=True),
            # ksim.LinearVelocityPenalty(index=("x", "y", "z"), scale=-0.005,scale_by_curriculum=True),
            # ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            # ksim.ReachabilityPenalty(
            #     delta_max_j=tuple(float(x) for x in self.delta_max_j),
            #     scale=-1.0,
            #     squared=True,
            #     scale_by_curriculum=True,
            # ),
            JointPositionPenalty.create_from_names(
                physics_model=physics_model,
                names=[name for name, _ in ZEROS],
                scale=-0.1,
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.05, unhealthy_z_upper=0.5),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=0.0,
            step_every_n_epochs=10,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_inputs=NUM_ACTOR_INPUTS,
            num_outputs=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
        rng: PRNGKeyArray,
    ) -> tuple[distrax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        cmd_7       = commands["joystick_command"]          # shape (...,7)

        def maybe_drop(x: Array, p: float, key: PRNGKeyArray) -> Array:
            mask = jax.random.bernoulli(key, p, x.shape)
            return jnp.where(mask, 0.0, x)

        # Occasionally zeroing the gravity vector forces the policy to back‑up
        # on joint encoders / base‑quat and stops it from keying on one cue
        rng, sub = jax.random.split(rng)
        proj_grav_3 = maybe_drop(proj_grav_3, p=0.0, key=sub)  # don't drop for now

        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                proj_grav_3,  # 3
                cmd_7,        # 7
            ],
            axis=-1,
        )

        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        cmd_7       = commands["joystick_command"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                cmd_7,  # 7
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        step_keys = jax.random.split(rng, trajectory.action.shape[0])

        def scan_fn(
            actor_critic_carry: tuple[Array, Array],
            scan_inputs: tuple[ksim.Trajectory, PRNGKeyArray],
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            transition, step_key = scan_inputs
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
                rng=step_key,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y),
                self.get_initial_model_carry(rng),
                (next_actor_carry, next_critic_carry),
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, (trajectory, step_keys))

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        rng, actor_rng = jax.random.split(rng)
        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
            rng=actor_rng,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
        )


if __name__ == "__main__":
    ZbotWalkingTask.launch(
        ZbotWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            # Simulation parameters.
            dt=0.001,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            valid_every_n_steps=5,
            render_full_every_n_seconds=10,
            # valid_first_n_steps=1,
        ),
    )
