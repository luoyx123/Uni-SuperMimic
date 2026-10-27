#!/usr/bin/env python3
"""Pack OpenLET Roban2 motions into a compact 50 Hz FP16 library."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from unisupermimic.tasks.motion_tracking.roban2.supermimic.motion import (
    COMPACT_DIM,
    COMPACT_VERSION,
    CONTROL_DT,
    _load_npz,
    resample_clip,
)


def selected_paths(
    dataset_root: Path, manifest: Path, prefixes: list[str], limit: int | None = None
) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    paths = data["passed"]
    if prefixes:
        paths = [path for path in paths if any(path.startswith(p + "/") for p in prefixes)]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No motions selected from the passed-motions manifest")
    resolved = []
    for path in paths:
        if (dataset_root / path).is_file():
            resolved.append(path)
            continue
        parent = Path(path).parent
        corrected = parent.with_name(parent.name.rstrip(".")).joinpath(Path(path).name)
        if corrected != Path(path) and (dataset_root / corrected).is_file():
            print(f"Corrected manifest path: {path} -> {corrected}", flush=True)
            resolved.append(corrected.as_posix())
        else:
            raise FileNotFoundError(f"Missing motion from manifest: {path}")
    paths = resolved
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate paths in the passed-motions manifest")
    names = [Path(path).with_suffix("").as_posix() for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Motion names must be unique after removing .npz")
    return paths


def pack(dataset_root: Path, paths: list[str], output: Path) -> None:
    lengths = []
    for index, relative in enumerate(paths, 1):
        raw = _load_npz(str(dataset_root / relative), 120.0)
        length = math.floor(((len(raw) - 1) / 120.0) / CONTROL_DT) + 1
        lengths.append(length)
        if index % 5000 == 0:
            print(f"Validated {index}/{len(paths)} motions", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
    reference = None
    try:
        reference = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(sum(lengths), COMPACT_DIM)
        )
        offset = 0
        for index, (relative, length) in enumerate(zip(paths, lengths, strict=True), 1):
            clip = resample_clip(dataset_root / relative)
            if len(clip) != length:
                raise RuntimeError(f"Resampled length changed for {relative}")
            reference[offset : offset + length] = clip.numpy().astype(np.float16)
            offset += length
            if index % 5000 == 0:
                reference.flush()
                print(f"Packed {index}/{len(paths)} motions", flush=True)
        reference.flush()
        payload = {
            "version": COMPACT_VERSION,
            "fps": 50,
            "names": [Path(path).with_suffix("").as_posix() for path in paths],
            "lengths": lengths,
            "reference": torch.from_numpy(reference),
        }
        staged = output.with_name(output.name + ".tmp")
        try:
            torch.save(payload, staged)
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)
        print(
            f"Saved {len(paths)} motions, {sum(lengths) / 50 / 3600:.2f} h "
            f"({sum(lengths) * COMPACT_DIM * 2 / 2**30:.2f} GiB reference): {output}",
            flush=True,
        )
    finally:
        del reference
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Defaults to passed_motions.json")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Dataset category prefix; repeat to select several categories",
    )
    parser.add_argument("--limit", type=int, help="Pack only the first N clips for testing")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.dataset_root.resolve()
    manifest = args.manifest or root / "passed_motions.json"
    paths = selected_paths(root, manifest, args.include_prefix, args.limit)
    pack(root, paths, args.output.resolve())


if __name__ == "__main__":
    main()
