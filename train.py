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

from jaxtyping import Array, PRNGKeyArray, PyTree

logger = logging.getLogger(__name__)

NUM_JOINTS = 20
NUM_COMMANDS = 6
NUM_ACTOR_INPUTS = 20 + 20 + 4 + NUM_COMMANDS           # 50
# Critic inputs: joint_pos + joint_vel + com_inertia + com_vel + imu_acc + imu_gyro + imu_quat + full_cmd + act_force + base_pos + base_quat
NUM_CRITIC_INPUTS = 484   # 340


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
    ("right_hip_pitch", -0.4),
    ("right_knee_pitch", -0.8),
    ("right_ankle_pitch", -0.4),
    ("right_ankle_roll", 0.0),
    ("left_hip_yaw", 0.0),
    ("left_hip_roll", 0.0),
    ("left_hip_pitch", -0.4),
    ("left_knee_pitch", -0.8),
    ("left_ankle_pitch", -0.4),
    ("left_ankle_roll", 0.0),
    ("left_shoulder_pitch", 0.0),
    ("left_shoulder_roll", 0.2),
    ("left_elbow_roll", 0.2),
    ("left_gripper_roll", 0.0),
    ("right_shoulder_pitch", 0.0),
    ("right_shoulder_roll", -0.2),
    ("right_elbow_roll", -0.2),
    ("right_gripper_roll", 0.0),
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlannerState:
    position: Array
    velocity: Array
    last_computed_torque: Array

@attrs.define(kw_only=True)
class UnifiedLinearVelocityCommandMarker(ksim.vis.Marker):
    """Visualise the planar (x,y) velocity command from unified command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.03)
    arrow_scale: float = attrs.field(default=0.1)
    height: float = attrs.field(default=0.5)
    base_length: float = attrs.field(default=0.25)

    def update(self, trajectory: ksim.Trajectory) -> None:
        cmd = trajectory.command[self.command_name]
        vx, vy = float(cmd[0]), float(cmd[1])
        speed = (vx * vx + vy * vy) ** 0.5
        self.pos = (0.0, 0.0, self.height)

        # Always show arrow with base length plus scaling
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        arrow_length = self.base_length + self.arrow_scale * speed
        self.scale = (self.size, self.size, arrow_length)

        if speed < 1e-4:  # zero command → point forward, grey color
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:  # non-zero command → point in command direction, green color
            self.orientation = self.quat_from_direction((vx, vy, 0.0))
            self.rgba = (0.2, 0.8, 0.2, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.1,
        height: float = 0.5,
        base_length: float = 0.25,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(0.03, 0.03, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=True,
        )


@attrs.define(kw_only=True)
class UnifiedAbsoluteYawCommandMarker(ksim.vis.Marker):
    """Visualise the absolute yaw command from unified command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.02)
    height: float = attrs.field(default=0.7)
    arrow_scale: float = attrs.field(default=0.1)
    base_length: float = attrs.field(default=0.25)

    def update(self, trajectory: ksim.Trajectory) -> None:
        cmd = trajectory.command[self.command_name]
        yaw = float(cmd[3])  # yaw command is in position 3
        self.pos = (0.0, 0.0, self.height)

        # Always show arrow with base length plus scaling
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        arrow_length = self.base_length + self.arrow_scale * abs(yaw)
        self.scale = (self.size, self.size, arrow_length)

        if abs(yaw) < 1e-4:  # zero command → point forward, grey color
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:  # non-zero command → point in yaw direction, blue color
            # Convert yaw to direction vector (rotate around z-axis)
            direction_x = jnp.cos(yaw)
            direction_y = jnp.sin(yaw)
            self.orientation = self.quat_from_direction((float(direction_x), float(direction_y), 0.0))
            self.rgba = (0.2, 0.2, 0.8, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.1,
        height: float = 0.7,
        base_length: float = 0.25,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(0.02, 0.02, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=False,
        )


@attrs.define(frozen=True)
class UnifiedCommand(ksim.Command):
    """Unifiying all commands into one to allow for covariance control."""

    vx_range: tuple[float, float] = attrs.field()
    vy_range: tuple[float, float] = attrs.field()
    wz_range: tuple[float, float] = attrs.field()
    bh_range: tuple[float, float] = attrs.field()
    bh_standing_range: tuple[float, float] = attrs.field()
    rx_range: tuple[float, float] = attrs.field()
    ry_range: tuple[float, float] = attrs.field()
    ctrl_dt: float = attrs.field()
    switch_prob: float = attrs.field()

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h = jax.random.split(rng, 8)

        # cmd  = [vx, vy, wz, bh, rx, ry]
        vx = jax.random.uniform(rng_b, (1,), minval=self.vx_range[0], maxval=self.vx_range[1])
        vy = jax.random.uniform(rng_c, (1,), minval=self.vy_range[0], maxval=self.vy_range[1])
        wz = jax.random.uniform(rng_d, (1,), minval=self.wz_range[0], maxval=self.wz_range[1])
        bh = jax.random.uniform(rng_e, (1,), minval=self.bh_range[0], maxval=self.bh_range[1])
        bhs = jax.random.uniform(rng_f, (1,), minval=self.bh_standing_range[0], maxval=self.bh_standing_range[1])
        rx = jax.random.uniform(rng_g, (1,), minval=self.rx_range[0], maxval=self.rx_range[1])
        ry = jax.random.uniform(rng_h, (1,), minval=self.ry_range[0], maxval=self.ry_range[1])

        # don't like super small velocity commands
        vx = jnp.where(jnp.abs(vx) < 0.09, 0.0, vx)
        vy = jnp.where(jnp.abs(vy) < 0.09, 0.0, vy)
        wz = jnp.where(jnp.abs(wz) < 0.09, 0.0, wz)

        _ = jnp.zeros_like(vx)

        # Create each mode's command vector
        forward_cmd = jnp.concatenate([vx, _, _, bh, _, _])
        sideways_cmd = jnp.concatenate([_, vy, _, bh, _, _])
        rotate_cmd = jnp.concatenate([_, _, wz, bh, _, _])
        omni_cmd = jnp.concatenate([vx, vy, wz, bh, _, _])
        stand_cmd = jnp.concatenate([_, _, _, bhs, rx, ry])

        # randomly select a mode
        mode = jax.random.randint(rng_a, (), minval=0, maxval=6) # 0 1 2 3 4s 5s -- 2/6 standing
        cmd = jnp.where(mode == 0, forward_cmd,
              jnp.where(mode == 1, sideways_cmd,
              jnp.where(mode == 2, rotate_cmd,
              jnp.where(mode == 3, omni_cmd,
              stand_cmd))))

        # get initial heading
        init_euler = xax.quat_to_euler(physics_data.xquat[1])
        init_heading = init_euler[2] + self.ctrl_dt * cmd[2]  # add 1 step of yaw vel cmd to initial heading.
        cmd = jnp.concatenate([cmd[:3], jnp.array([init_heading]), cmd[3:]])
        assert cmd.shape == (7,)

        return cmd

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        def update_heading(prev_command: Array) -> Array:
            """Update the heading by integrating the angular velocity."""
            wz_cmd, heading = prev_command[2], prev_command[3]
            heading = heading + wz_cmd * self.ctrl_dt
            prev_command = prev_command.at[3].set(heading)
            return prev_command

        continued_command = update_heading(prev_command)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_command, continued_command)

    def get_markers(self) -> list[ksim.vis.Marker]:
        """Return markers for visualizing the unified command components."""
        return [
            UnifiedAbsoluteYawCommandMarker.get(
                command_name=self.command_name,
                height=0.7,
            ),
            UnifiedLinearVelocityCommandMarker.get(
                command_name=self.command_name,
                height=0.5,
            ),
        ]



@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="base_linear_velocity_observation")
    command_name: str = attrs.field(default="unified_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=1e-2)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # need to get lin vel obs from sensor, because xvel is not available in Trajectory.
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        # Get global frame velocities
        global_vel = trajectory.obs[self.linvel_obs_name]

        # get base quat, only yaw.
        # careful to only rotate in z, disregard rx and ry, bad conflict with roll and pitch.
        base_euler = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        base_euler = base_euler.at[:, :2].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # rotate local frame commands to global frame
        robot_vel_cmd = jnp.zeros_like(global_vel).at[:, :2].set(trajectory.command[self.command_name][:, :2])
        global_vel_cmd = xax.rotate_vector_by_quat(robot_vel_cmd, base_z_quat, inverse=False)

        # drop vz. vz conflicts with base height reward.
        global_vel_xy_cmd = global_vel_cmd[:, :2]
        global_vel_xy = global_vel[:, :2]

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < self.stand_still_threshold
        vel_error = jnp.linalg.norm(global_vel_xy - global_vel_xy_cmd, axis=-1)
        error = jnp.where(zero_cmd_mask, vel_error, 2 * jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the heading using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        base_yaw = xax.quat_to_euler(trajectory.xquat[:, 1, :])[:, 2]
        base_yaw_cmd = trajectory.command[self.command_name][:, 3]

        base_yaw_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw], axis=-1)
        )
        base_yaw_target_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw_cmd], axis=-1)
        )

        # Compute quaternion error
        quat_error = 1 - jnp.sum(base_yaw_target_quat * base_yaw_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        euler_orientation = euler_orientation.at[:, 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack(
            [
                trajectory.command[self.command_name][:, 5],
                trajectory.command[self.command_name][:, 6],
                jnp.zeros_like(trajectory.command[self.command_name][:, 6]),
            ],
            axis=-1,
        )
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)

        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int
    foot_right_idx: int
    floor_threshold: float = 0.0
    in_robot_frame: bool = True

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
        in_robot_frame: bool = True,
    ) -> Self:
        fl = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        fr = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(foot_left_idx=fl, foot_right_idx=fr, floor_threshold=floor_threshold, in_robot_frame=in_robot_frame)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        fl_ndarray = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        fr_ndarray = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )

        if self.in_robot_frame:
            # Transform foot positions to robot frame
            base_quat = state.physics_state.data.qpos[3:7]  # Base quaternion
            fl = xax.rotate_vector_by_quat(jnp.array(fl_ndarray), base_quat, inverse=True)
            fr = xax.rotate_vector_by_quat(jnp.array(fr_ndarray), base_quat, inverse=True)

        return jnp.concatenate([fl, fr], axis=-1)

@attrs.define(frozen=True)
class BaseHeightReward(ksim.Reward):
    """Reward for keeping the base height at the commanded height."""

    error_scale: float = attrs.field(default=0.25)
    standard_height: float = attrs.field(default=0.28)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_height = trajectory.xpos[:, 1, 2]  # 1st body, because world is 0. 2nd element is z.
        commanded_height = trajectory.command["unified_command"][:, 4] + self.standard_height

        height_error = jnp.abs(current_height - commanded_height)
        # is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        # height_error = jnp.where(is_zero_cmd, height_error, height_error**2)  # smooth kernel for walking.
        return jnp.exp(-height_error / self.error_scale)
    

@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases and penalizing quick stepping."""

    scale: float = 1.0
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    scale_by_curriculum: bool = False

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        # initial left and right airtime
        return jnp.array([0.0, 0.0])

    def _airtime_sequence(self, initial_airtime: Array, contact_bool: Array, done: Array) -> tuple[Array, Array]:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        # or with done to reset the airtime counter when the episode is done
        contact_or_done = jnp.logical_or(contact_bool, done)
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False)[:, 0]

        # airtime counters
        left_carry, left_air = self._airtime_sequence(reward_carry[0], left_contact, traj.done)
        right_carry, right_air = self._airtime_sequence(reward_carry[1], right_contact, traj.done)

        reward_carry = jnp.array([left_carry, right_carry])

        # touchdown boolean (0→1 transition)
        def touchdown(c: Array) -> Array:
            prev = jnp.concatenate([jnp.array([False]), c[:-1]])
            return jnp.logical_and(c, jnp.logical_not(prev))

        td_l = touchdown(left_contact)
        td_r = touchdown(right_contact)

        left_air_shifted = jnp.roll(left_air, 1)
        right_air_shifted = jnp.roll(right_air, 1)

        left_feet_airtime_reward = (left_air_shifted - self.touchdown_penalty) * td_l.astype(jnp.float32)
        right_feet_airtime_reward = (right_air_shifted - self.touchdown_penalty) * td_r.astype(jnp.float32)

        reward = left_feet_airtime_reward + right_feet_airtime_reward

        # standing mask
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 0.0, reward)

        return reward, reward_carry

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

@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "left_hip_roll",
                "left_hip_yaw",
                "right_hip_roll",
                "right_hip_yaw",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


class AnkleKneePenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=["left_knee_pitch", "left_ankle_pitch", "right_knee_pitch", "right_ankle_pitch"],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True)
class FeetOrientationReward(ksim.Reward):
    """Reward for keeping feet pitch and roll oriented parallel to the ground."""

    scale: float = attrs.field(default=1.0)
    error_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        left_foot_euler = xax.quat_to_euler(trajectory.xquat[:, 23, :])
        right_foot_euler = xax.quat_to_euler(trajectory.xquat[:, 18, :])

        straight_foot_euler = jnp.stack([-jnp.pi / 2, 0], axis=-1)  # ignore yaw

        left_error = jnp.abs(left_foot_euler[:, :2] - straight_foot_euler).sum(axis=-1)
        right_error = jnp.abs(right_foot_euler[:, :2] - straight_foot_euler).sum(axis=-1)

        total_error = left_error + right_error
        return jnp.exp(-total_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class SimpleSingleFootContactReward(ksim.Reward):
    """Reward having one and only one foot in contact with the ground, while walking."""

    scale: float = 1.0

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False).squeeze()
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False).squeeze()
        single = jnp.logical_xor(left_contact, right_contact).squeeze()

        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 1.0, single)
        return reward


def rotate_quat_by_quat(quat_to_rotate: Array, rotating_quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates one quaternion by another quaternion through quaternion multiplication.

    This performs the operation: rotating_quat * quat_to_rotate * rotating_quat^(-1) if inverse=False
    or rotating_quat^(-1) * quat_to_rotate * rotating_quat if inverse=True

    Args:
        quat_to_rotate: The quaternion being rotated (w,x,y,z), shape (*, 4)
        rotating_quat: The quaternion performing the rotation (w,x,y,z), shape (*, 4)
        inverse: If True, rotate by the inverse of rotating_quat
        eps: Small epsilon value to avoid division by zero in normalization

    Returns:
        The rotated quaternion (w,x,y,z), shape (*, 4)
    """
    # Normalize both quaternions
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)

    # If inverse requested, conjugate the rotating quaternion (negate x,y,z components)
    if inverse:
        rotating_quat = rotating_quat.at[..., 1:].multiply(-1)

    # Extract components of both quaternions
    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)  # rotating quaternion
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)  # quaternion being rotated

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize result
    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)

@attrs.define(frozen=True, kw_only=True)
class ImuOrientationObservation(ksim.StatefulObservation):
    """Observes the IMU orientation, back spun in yaw heading, as commanded.

    This provides an approximation of reading the IMU orientation from
    the IMU on the physical robot, backspun by commanded heading. The `framequat_name` should be the name of
    the framequat sensor attached to the IMU.

    Example: if yaw cmd = 3.14, and IMU reading is [0, 0, 0, 1], then back spun IMU heading obs is [1, 0, 0, 0]

    The policy learns to keep the IMU heading obs around [1, 0, 0, 0].
    """

    framequat_idx_range: tuple[int, int | None] = attrs.field()
    lag_range: tuple[float, float] = attrs.field(
        default=(0.01, 0.1),
        validator=attrs.validators.deep_iterable(
            attrs.validators.and_(
                attrs.validators.ge(0.0),
                attrs.validators.lt(1.0),
            ),
        ),
    )

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        framequat_name: str,
        lag_range: tuple[float, float] = (0.01, 0.1),
        noise: float = 0.0,
    ) -> Self:
        """Create a IMU orientation observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            framequat_name: The name of the framequat sensor
            lag_range: The range of EMA factors to use, to approximate the
                variation in the amount of smoothing of the Kalman filter.
            noise: The observation noise
        """
        sensor_name_to_idx_range = ksim.get_sensor_data_idxs_by_name(physics_model)
        if framequat_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{framequat_name} not found in model. Available:\n{options}")

        return cls(
            framequat_idx_range=sensor_name_to_idx_range[framequat_name],
            lag_range=lag_range,
            noise=noise,
        )

    def initial_carry(self, physics_state: ksim.PhysicsState, rng: PRNGKeyArray) -> tuple[Array, Array]:
        minval, maxval = self.lag_range
        return jnp.zeros((4,)), jax.random.uniform(rng, (1,), minval=minval, maxval=maxval)

    def observe_stateful(
        self,
        state: ksim.ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array]]:
        framequat_start, framequat_end = self.framequat_idx_range
        framequat_data = state.physics_state.data.sensordata[framequat_start:framequat_end].ravel()

        # Add noise
        # # BUG? noise is added twice? also in ksim rl.py
        # framequat_data = add_noise(framequat_data, rng, "gaussian", self.noise, curriculum_level)

        # get heading cmd
        heading_yaw_cmd = state.commands["unified_command"][3]

        # spin back
        heading_yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))
        backspun_framequat = rotate_quat_by_quat(framequat_data, heading_yaw_cmd_quat, inverse=True)
        # ensure positive quat hemisphere
        backspun_framequat = jnp.where(backspun_framequat[..., 0] < 0, -backspun_framequat, backspun_framequat)

        # Get current Kalman filter state
        x, lag = state.obs_carry
        x = x * lag + backspun_framequat * (1 - lag)

        return x, (x, lag)

@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):
    """Single-scalar z of body-1 (the robot base)."""
    def observe(self, state: ksim.ObservationInput, curriculum_level, rng):
        # body 0 is world; body 1 is the floating base
        return state.physics_state.data.xpos[1, 2:]

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

        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

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
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.0),
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
            BaseHeightObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ImuOrientationObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.ActuatorAccelerationObservation(),
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
        # get_observations
        obs_list += [
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_touch",  noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_touch", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force",  noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.0,
                in_robot_frame=True,
            )
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
            UnifiedCommand(
                vx_range=(-0.1, 0.3),  # m/s
                vy_range=(-0.075, 0.075),  # m/s
                wz_range=(-0.5, 0.5),  # rad/s
                # bh_range=(-0.05, 0.05), # m
                bh_range=(0.0, 0.0),  # m # disabled for now, does not work on this robot. reward conflicts
                bh_standing_range=(0.0, 0.0), # m
                # bh_standing_range=(0.0, 0.0),  # m
                rx_range=(-0.0, 0.0),  # rad
                ry_range=(-0.0, 0.0),  # rad
                ctrl_dt=self.config.ctrl_dt,
                switch_prob=self.config.ctrl_dt / 4,  # once per x seconds
            ),
        ]

    def get_rewards(self, physics_model):
        return [
            ksim.StayAliveReward(scale=1.0),
            # ksim.UprightReward(scale=1.0),

            # --- command-tracking ---
            LinearVelocityTrackingReward(scale=0.8,  error_scale=0.2),
            AngularVelocityTrackingReward(scale=0.1, error_scale=0.005),
            XYOrientationReward(scale=0.2,          error_scale=0.03),
            BaseHeightReward(scale=0.1,             error_scale=0.05,
                            standard_height=0.28),          # adjust if COM is lower

            SimpleSingleFootContactReward(scale=0.3),

            # keep the old posture prior (optional)
            # JointPositionPenalty.create_from_names(
            #     physics_model=physics_model,
            #     names=[name for name, _ in ZEROS],
            #     scale=-0.1,
            # ),
            FeetAirtimeReward(
                scale=2.5,
                ctrl_dt=self.config.ctrl_dt,
                touchdown_penalty=0.6,
            ),
            FeetOrientationReward(scale=0.1, error_scale=0.25),
            StraightLegPenalty.create_penalty(physics_model, scale=-0.05, scale_by_curriculum=True),
            AnkleKneePenalty.create_penalty(physics_model, scale=-0.05, scale_by_curriculum=True),
            ksim.ActionVelocityPenalty(scale=-0.01,  scale_by_curriculum=True),
            ksim.JointVelocityPenalty (scale=-0.01,  scale_by_curriculum=True),
            ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.05, unhealthy_z_upper=0.5),
        ]

    def get_curriculum(self, physics_model):
        return ksim.EpisodeLengthCurriculum(
            num_levels=30,
            increase_threshold=30.0,
            decrease_threshold=10.0,
            min_level_steps=10,
            min_level=0.5,
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
        imu_quat_4 = observations["imu_orientation_observation"]
        cmd = commands["unified_command"]          # shape (...,7)

        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                imu_quat_4,   # 4
                cmd[..., :2],      # vx, vy
                cmd[..., 3:4],     # heading   (index 3)
                cmd[..., 4:],      # bh, rx, ry
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
        imu_quat_4 = observations["imu_orientation_observation"]
        cmd_7       = commands["unified_command"]  # still 7 elements for critic
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
                imu_quat_4,  # 4
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
