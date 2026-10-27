"""Reference Roban2 motions, with optional compact GPU storage."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    mat_to_6d,
    quat_slerp,
    quat_to_mat,
    so3_log,
    yaw_mat,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import Roban2Kinematics

CONTROL_DT = 0.02
BIN_FRAMES = 50
COMPACT_DIM = 59
COMPACT_VERSION = 1


def resample_clip(file: str | Path, src_fps: float = 120.0) -> torch.Tensor:
    """Return a 50 Hz float32 clip as p, quat, q, v, w, qd."""
    raw = _load_npz(str(file), src_fps)
    position, quaternion, joints = raw[:, :3], raw[:, 3:7], raw[:, 7:30]
    source_frames = raw.shape[0]
    duration = (source_frames - 1) / src_fps
    num_frames = math.floor(duration / CONTROL_DT) + 1
    sample = torch.arange(num_frames, dtype=torch.float32) * CONTROL_DT * src_fps
    lower = torch.clamp(sample.floor().long(), max=source_frames - 2)
    alpha = (sample - lower.float()).clamp(0, 1)
    position = position[lower] * (1 - alpha[:, None]) + position[lower + 1] * alpha[:, None]
    joints = joints[lower] * (1 - alpha[:, None]) + joints[lower + 1] * alpha[:, None]
    quaternion = quat_slerp(quaternion[lower], quaternion[lower + 1], alpha)
    rotation = quat_to_mat(quaternion)
    linear_velocity = _gradient(position, CONTROL_DT)
    joint_velocity = _gradient(joints, CONTROL_DT)
    angular_velocity = torch.zeros_like(position)
    angular_velocity[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (
        2 * CONTROL_DT
    )
    angular_velocity[0] = so3_log(rotation[1] @ rotation[0].transpose(-1, -2)) / CONTROL_DT
    angular_velocity[-1] = so3_log(rotation[-1] @ rotation[-2].transpose(-1, -2)) / CONTROL_DT
    return torch.cat(
        (position, quaternion, joints, linear_velocity, angular_velocity, joint_velocity), -1
    )


class MotionLib:
    def __init__(
        self,
        files: Sequence[str],
        kin: Roban2Kinematics,
        device: str | torch.device,
        src_fps: float = 120.0,
    ):
        if not files:
            raise ValueError("MotionLib requires at least one NPZ file")
        self.device = torch.device(device)
        self.compact = len(files) == 1 and Path(files[0]).suffix == ".pt"
        if self.compact:
            packed = torch.load(files[0], map_location="cpu", weights_only=True)
            if packed.get("version") != COMPACT_VERSION or packed.get("fps") != 50:
                raise ValueError("Unsupported Roban2 compact library format")
            names, lengths, reference = packed["names"], packed["lengths"], packed["reference"]
            if (
                not isinstance(names, list)
                or not isinstance(lengths, list)
                or not isinstance(reference, torch.Tensor)
                or reference.ndim != 2
                or reference.shape[1] != COMPACT_DIM
                or sum(lengths) != len(reference)
                or len(names) != len(lengths)
                or len(set(names)) != len(names)
                or any(length < 2 for length in lengths)
            ):
                raise ValueError("Invalid Roban2 compact library contents")
            self.names = names
            self.reference = reference.to(self.device, dtype=torch.float16)
        else:
            self.names = [os.path.splitext(os.path.basename(file))[0] for file in files]
            clips = [resample_clip(file, src_fps) for file in files]
            lengths = [len(clip) for clip in clips]

        lengths_cpu = torch.tensor(lengths, dtype=torch.long)
        starts_cpu = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.long), lengths_cpu[:-1])), 0
        )
        self.clip_len = lengths_cpu.to(self.device)
        self.clip_start = starts_cpu.to(self.device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = sum(lengths)
        index_dtype = torch.int32 if self.compact else torch.long
        self.clip_id = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=index_dtype),
            lengths_cpu,
        ).to(self.device)

        if not self.compact:
            reference = torch.cat(clips).to(self.device)
            self.root_p = reference[:, :3]
            self.root_R = quat_to_mat(reference[:, 3:7])
            self.q = reference[:, 7:30]
            self.root_v = reference[:, 30:33]
            self.root_w = reference[:, 33:36]
            self.qd = reference[:, 36:59]
            with torch.no_grad():
                body_p, body_R, body_v, body_w = [], [], [], []
                for start in range(0, self.num_frames, 65536):
                    end = min(start + 65536, self.num_frames)
                    output = kin.forward(*self.ref_state(slice(start, end)))
                    body_p.append(output[0])
                    body_R.append(output[1])
                    body_v.append(output[2])
                    body_w.append(output[3])
                self.body_p, self.body_R, self.body_v, self.body_w = (
                    torch.cat(values) for values in (body_p, body_R, body_v, body_w)
                )
            root_rotation_t = self.root_R.transpose(-1, -2)
            self.root_v_local = (root_rotation_t @ self.root_v.unsqueeze(-1)).squeeze(-1)
            self.root_w_local = (root_rotation_t @ self.root_w.unsqueeze(-1)).squeeze(-1)
            self.root_gz = self.root_R[:, 2, 2]
        self.kin = kin

        bins_per_clip = (lengths_cpu + BIN_FRAMES - 1) // BIN_FRAMES
        bin_clip_cpu = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=index_dtype), bins_per_clip
        )
        self.num_bins = len(bin_clip_cpu)
        bin_offsets = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        local_bin = torch.arange(self.num_bins) - bin_offsets[bin_clip_cpu.long()]
        bin_start_cpu = starts_cpu[bin_clip_cpu.long()] + local_bin * BIN_FRAMES
        self.bin_clip = bin_clip_cpu.to(self.device)
        self.bin_start = bin_start_cpu.to(self.device)
        self.frame_bin = torch.repeat_interleave(
            torch.arange(self.num_bins, dtype=index_dtype),
            torch.minimum(
                torch.full((self.num_bins,), BIN_FRAMES),
                starts_cpu[bin_clip_cpu.long()] + lengths_cpu[bin_clip_cpu.long()] - bin_start_cpu,
            ),
        ).to(self.device)

    def to(self, device: str | torch.device) -> MotionLib:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        new.kin = self.kin.to(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        return new

    def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Return frames+1 through frames+K, clamped at each clip's last frame."""
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Build future reference features in the current reference heading frame."""
        future = self.future_frames(frames, num_frames)
        position, rotation, velocity, angular, joints, joint_velocity = self.ref_state(future)
        current_p, current_R, _, _, _, _ = self.ref_state(frames)
        heading = yaw_mat(current_R)
        heading_t = heading.transpose(-1, -2).unsqueeze(1)
        position_delta = (heading_t @ (position - current_p.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d(heading_t @ rotation)
        rotation_t = rotation.transpose(-1, -2)
        features = torch.cat(
            (
                position_delta,
                relative_rotation,
                (rotation_t @ velocity.unsqueeze(-1)).squeeze(-1),
                (rotation_t @ angular.unsqueeze(-1)).squeeze(-1),
                joints,
                joint_velocity * 0.1,
            ),
            -1,
        )
        return features.reshape(frames.shape[0], -1)

    def ref_state(self, frames: torch.Tensor | slice) -> State:
        if self.compact:
            reference = self.reference[frames].float()
            quaternion = torch.nn.functional.normalize(reference[..., 3:7], dim=-1)
            return (
                reference[..., :3],
                quat_to_mat(quaternion),
                reference[..., 30:33],
                reference[..., 33:36],
                reference[..., 7:30],
                reference[..., 36:59],
            )
        return (
            self.root_p[frames],
            self.root_R[frames],
            self.root_v[frames],
            self.root_w[frames],
            self.q[frames],
            self.qd[frames],
        )

    def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.compact:
            return self.kin.forward(*self.ref_state(frames))
        return (
            self.body_p[frames],
            self.body_R[frames],
            self.body_v[frames],
            self.body_w[frames],
        )


def _load_npz(path: str, src_fps: float) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        key = "qpos" if "qpos" in archive else "data" if "data" in archive else None
        if key is None:
            raise ValueError(f"{path} must contain qpos or data")
        values = archive[key]
        fps = float(archive["fps"]) if "fps" in archive else float("nan")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] not in (28, 30):
        raise ValueError(f"{path} must have at least two rows and 28 or 30 columns")
    if not math.isclose(fps, src_fps, abs_tol=1e-6):
        raise ValueError(f"{path} has fps={fps}, expected {src_fps}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite qpos values")
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-2):
        raise ValueError(f"{path} contains invalid root quaternions")
    if values.shape[1] == 28:
        values = np.pad(values, ((0, 0), (0, 2)))
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
    gradient = torch.zeros_like(value)
    gradient[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
    gradient[0] = (value[1] - value[0]) / dt
    gradient[-1] = (value[-1] - value[-2]) / dt
    return gradient
