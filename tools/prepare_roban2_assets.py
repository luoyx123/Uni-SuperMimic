#!/usr/bin/env python3
"""Prepare the OpenLET Roban2 MJCF for the SuperMimic position-control task."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from unisupermimic.tasks.motion_tracking.roban2.supermimic.robot import (
    COLLISION_GEOM_NAMES,
    EFFORT,
    KD,
    KP,
    ROBAN2_JOINT_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/lyx/train/dataset/LET-HumanMotion-Roban2/scripts/robot")
DEST = ROOT / "src/unisupermimic/assets/robots/roban2"


def prepare(source: Path) -> None:
    source_model = source / "biped_s17.xml"
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    reference = mujoco.MjModel.from_xml_path(str(source_model))
    joints = [reference.joint(i).name for i in range(1, reference.njnt)]
    if joints != ROBAN2_JOINT_NAMES or reference.nq != 30 or reference.nu != 23:
        raise ValueError("OpenLET model joint order or dimensions differ from Roban2 task")

    root = ET.parse(source_model).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", "meshes")
    for geom in root.findall("./default//geom"):
        geom.attrib.pop("material", None)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for geom in list(worldbody):
        if geom.get("name") == "floor":
            worldbody.remove(geom)
    collisions = [geom for geom in worldbody.iter("geom") if geom.get("contype", "1") != "0"]
    if len(collisions) != len(COLLISION_GEOM_NAMES):
        raise ValueError(
            f"Expected {len(COLLISION_GEOM_NAMES)} robot collision geoms, found {len(collisions)}"
        )
    for geom, name in zip(collisions, COLLISION_GEOM_NAMES, strict=True):
        geom.set("name", name)

    actuators = root.find("actuator")
    assert actuators is not None
    actuators.clear()
    for name, kp, kd, effort in zip(ROBAN2_JOINT_NAMES, KP, KD, EFFORT, strict=True):
        ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=str(kp),
            kv=str(kd),
            forcerange=f"{-effort} {effort}",
        )
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "meshes", DEST / "meshes", dirs_exist_ok=True)
    ET.indent(root, space="  ")
    robot_path = DEST / "roban2.xml"
    ET.ElementTree(root).write(robot_path, encoding="unicode", xml_declaration=True)

    scene = ET.Element("mujoco", model="roban2 supermimic scene")
    ET.SubElement(scene, "include", file="roban2.xml")
    ET.SubElement(scene, "option", integrator="implicitfast", timestep="0.005", cone="pyramidal")
    scene_world = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        scene_world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        friction="0.6 0.005 0.0001",
    )
    keyframes = ET.SubElement(scene, "keyframe")
    ET.SubElement(
        keyframes,
        "key",
        name="stand",
        qpos="0 0 0.77 1 0 0 0 " + " ".join("0" for _ in ROBAN2_JOINT_NAMES),
        ctrl=" ".join("0" for _ in ROBAN2_JOINT_NAMES),
    )
    ET.indent(scene, space="  ")
    scene_path = DEST / "scene_supermimic.xml"
    ET.ElementTree(scene).write(scene_path, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("Prepared Roban2 scene has unexpected dimensions")
    print(f"Prepared and validated {scene_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    prepare(args.source.resolve())


if __name__ == "__main__":
    main()
