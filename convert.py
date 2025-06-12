"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import Model, ZbotWalkingTask, rotate_quat_by_quat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: ZbotWalkingTask = ZbotWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    # Loads the Mujoco model and gets the joint names.
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    NUM_COMMANDS = 6              # exactly what run_actor concatenates

    # Constant values.
    carry_shape = (task.config.depth, task.config.hidden_size)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,            # 20
        joint_angular_velocities: Array,  # 20
        imu_quaternion: Array,          # 4 (w,x,y,z) – raw IMU reading
        initial_heading: Array,         # scalar – heading at power-on
        command: Array,                 # 6-D vector (vx,vy,heading,bh,rx,ry)
        carry: Array,
    ) -> tuple[Array, Array]:

        # 1.   spin the IMU quaternion into the same frame used for training
        heading_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, command[..., 2]]))   # heading term
        init_quat    = xax.euler_to_quat(jnp.array([0.0, 0.0, initial_heading.squeeze()]))

        rel_quat   = rotate_quat_by_quat(imu_quaternion, init_quat, inverse=True)
        spun_quat  = rotate_quat_by_quat(rel_quat, heading_quat, inverse=True)
        spun_quat  = jnp.where(spun_quat[..., 0] < 0, -spun_quat, spun_quat)        # hemisphere trick

        # 2.   build exactly the 50-D actor input
        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                spun_quat,
                command[..., :2],      # vx, vy
                command[..., 2:3],     # heading (already at index 2 after drop)
                command[..., 3:],      # bh, rx, ry
            ],
            axis=-1,
        )

        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS,
        carry_size=carry_shape,   # tuple is fine
    )

    init_onnx = export_fn(init_fn, metadata)
    step_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
