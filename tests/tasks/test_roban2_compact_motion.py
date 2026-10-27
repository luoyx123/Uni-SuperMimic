"""Compact Roban2 references must preserve full-library tracking calculations."""

import json
from pathlib import Path

import numpy as np
import torch
from tools.build_roban2_compact_library import pack, selected_paths

from unisupermimic.tasks.motion_tracking.roban2.supermimic.kinematics import Roban2Kinematics
from unisupermimic.tasks.motion_tracking.roban2.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.roban2.supermimic.motion import MotionLib

SCENE = Path("src/unisupermimic/assets/robots/roban2/roban2.xml")


def make_clip(path: Path, frames: int, phase: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames, dtype=np.float32) / 120
    qpos = np.zeros((frames, 30), dtype=np.float32)
    qpos[:, 0] = 0.2 * time
    qpos[:, 2] = 0.78 + 0.01 * np.sin(3 * time + phase)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = 0.05 * np.sin(2 * time[:, None] + phase + np.arange(23))
    np.savez(path, qpos=qpos, fps=120.0)


def test_compact_library_matches_full(tmp_path):
    root = tmp_path / "dataset"
    names = [
        "Non-Interactive_Actions/walk/clip_a.npz",
        "Non-Interactive_Actions/jump/clip_b.npz",
    ]
    make_clip(root / names[0], 121, 0.0)
    make_clip(root / names[1], 185, 0.7)
    manifest = root / "passed_motions.json"
    manifest.write_text(json.dumps({"passed": names, "num_passed": 2}))
    paths = selected_paths(root, manifest, ["Non-Interactive_Actions"])
    packed = tmp_path / "packed.pt"
    pack(root, paths, packed)

    kin = Roban2Kinematics(str(SCENE), "cpu")
    full = MotionLib([str(root / path) for path in paths], kin, "cpu")
    compact = MotionLib([str(packed)], kin, "cpu")
    assert compact.compact
    assert compact.names == [Path(path).with_suffix("").as_posix() for path in names]
    assert torch.equal(full.clip_len, compact.clip_len)
    assert torch.equal(full.clip_id, compact.clip_id)
    assert torch.equal(full.frame_bin, compact.frame_bin)

    frames = torch.tensor([0, 10, 35, full.clip_start[1], full.num_frames - 2])
    for actual, expected in zip(compact.ref_state(frames), full.ref_state(frames), strict=True):
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    for actual, expected in zip(compact.body_state(frames), full.body_state(frames), strict=True):
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(
        compact.goal(frames, 10), full.goal(frames, 10), atol=0.03, rtol=0.03
    )

    default_q = torch.zeros(23)
    full_mdp, compact_mdp = MDP(kin, full, default_q), MDP(kin, compact, default_q)
    state = full.ref_state(frames)
    action = torch.zeros(len(frames), 23)
    for method in ("reward", "termination", "critic_features", "obs"):
        if method == "reward":
            old = full_mdp.reward(state, frames, action, action)
            new = compact_mdp.reward(state, frames, action, action)
        elif method == "termination":
            old = full_mdp.termination(state, frames)
            new = compact_mdp.termination(state, frames)
        elif method == "critic_features":
            old = full_mdp.critic_features(state, frames, torch.zeros(len(frames), 27))
            new = compact_mdp.critic_features(state, frames, torch.zeros(len(frames), 27))
        else:
            old = full_mdp.obs(state, action, torch.zeros(len(frames), 27), frames)
            new = compact_mdp.obs(state, action, torch.zeros(len(frames), 27), frames)
        torch.testing.assert_close(new, old, atol=0.03, rtol=0.03)


def test_manifest_directory_with_trailing_dot(tmp_path):
    root = tmp_path / "dataset"
    actual = "Dance/Dance/D0109_Calc/clip.npz"
    make_clip(root / actual, 3, 0.0)
    manifest = root / "passed_motions.json"
    manifest.write_text(json.dumps({"passed": ["Dance/Dance/D0109_Calc./clip.npz"]}))
    assert selected_paths(root, manifest, [], limit=1) == [actual]
