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

from train import Model, ZbotWalkingTask


def rotate_quat_by_quat(
    quat_to_rotate: Array,
    rotating_quat: Array,
    *,
    inverse: bool = False,
    eps: float = 1e-6,
) -> Array:
    """Return rotating_quat * quat_to_rotate * rotating_quat⁻¹ (optionally inverse)."""
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)

    if inverse:
        rotating_quat = jnp.concatenate([rotating_quat[..., :1], -rotating_quat[..., 1:]], axis=-1)

    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = jnp.concatenate([w, x, y, z], axis=-1)
    return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + eps)


NUM_COMMANDS = 6  # vx, vy, heading, bh, rx, ry


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

    carry_shape = (task.config.depth, task.config.hidden_size)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        initial_heading: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        heading_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, command[..., 2]]))
        init_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, initial_heading.squeeze()]))

        rel_quat = rotate_quat_by_quat(quaternion, init_quat, inverse=True)
        spun_quat = rotate_quat_by_quat(rel_quat, heading_quat, inverse=True)
        spun_quat = jnp.where(spun_quat[..., 0] < 0, -spun_quat, spun_quat)

        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                spun_quat,
                command[..., :2],  # vx, vy
                command[..., 2:3],  # heading
                command[..., 3:],  # bh, rx, ry
            ],
            axis=-1,
        )

        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS,
        carry_size=carry_shape,
    )

    init_onnx = export_fn(init_fn, metadata)
    step_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kinfer_model)
    print(f"Kinfer model written to {out_path}")


if __name__ == "__main__":
    main()
