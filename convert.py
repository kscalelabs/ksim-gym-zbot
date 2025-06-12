"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import Model, ZbotWalkingTask


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

    num_commands: int = 7

    # Constant values.
    carry_shape = (task.config.depth, task.config.hidden_size)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        accelerometer: Array,
        gyroscope: Array,
        command: Array,
        time: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        obs_components = [
            joint_angles,
            joint_angular_velocities,
            projected_gravity,
            command[..., :num_commands],   # 7-D slice
        ]

        # if task.config.use_acc_gyro:
        #    obs_components.extend([accelerometer, gyroscope])

        obs = jnp.concatenate(obs_components, axis=-1)

        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=num_commands,
        carry_size=list(carry_shape),
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
