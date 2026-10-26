"""Roban2 23-joint order and simulation control constants."""

ROBAN2_JOINT_NAMES = [
    "waist_yaw_joint",
    *(f"leg_l{i}_joint" for i in range(1, 7)),
    *(f"leg_r{i}_joint" for i in range(1, 7)),
    *(f"zarm_l{i}_joint" for i in range(1, 5)),
    *(f"zarm_r{i}_joint" for i in range(1, 5)),
    "zhead_1_joint",
    "zhead_2_joint",
]

TRACKED_BODIES = [
    "base_link",
    "waist_yaw_link",
    "leg_l2_link",
    "leg_l4_link",
    "leg_l6_link",
    "leg_r2_link",
    "leg_r4_link",
    "leg_r6_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_r2_link",
    "zarm_r4_link",
    "zhead_1_link",
    "zhead_2_link",
]

# Matched by prepare_roban2_assets.py when it names the source collision geoms.
COLLISION_GEOM_NAMES = tuple(f"roban2_collision_{i:03d}" for i in range(55))

# The 21 body gains follow LejuLab-Deploy's Roban2 mimic configuration.
# Head gains are simulation-only defaults and must be tuned before hardware use.
KP = (
    40.1792,
    90.1792,
    150.0984,
    40.1792,
    150.0984,
    34.2506,
    34.2506,
    90.1792,
    150.0984,
    40.1792,
    150.0984,
    34.2506,
    34.2506,
    14.25,
    6.25,
    14.25,
    6.25,
    14.25,
    6.25,
    14.25,
    6.25,
    5.0,
    5.0,
)
KD = (
    3.557,
    3.557,
    8.308,
    3.557,
    8.308,
    2.907,
    2.907,
    3.557,
    8.308,
    3.557,
    8.308,
    2.907,
    2.907,
    1.907,
    4.907,
    1.907,
    4.907,
    1.907,
    4.907,
    1.907,
    4.907,
    0.2,
    0.2,
)
EFFORT = (
    50.0,
    150.0,
    150.0,
    70.0,
    150.0,
    74.0,
    74.0,
    150.0,
    150.0,
    70.0,
    150.0,
    74.0,
    74.0,
    14.1,
    37.0,
    37.0,
    37.0,
    14.1,
    37.0,
    37.0,
    37.0,
    1.5,
    12.0,
)


def action_scale(joint_name: str) -> float:
    index = ROBAN2_JOINT_NAMES.index(joint_name)
    return 0.25 * EFFORT[index] / KP[index]


def default_joint_pos_list() -> list[float]:
    return [0.0] * len(ROBAN2_JOINT_NAMES)
