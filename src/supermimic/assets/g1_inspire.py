"""Unitree G1 (29 DoF) with an Inspire dexterous hand on the right wrist.

The body is mjlab's tuned ``g1.xml`` unchanged except at the right wrist, where the
rubber hand is replaced by the Inspire hand from ``inspire_hand/right_hand.xml``
(extracted once from the G1+Inspire URDF in the WEAVE repository). The left
hand stays the rubber hand.

The Inspire hand has 12 joints but only 6 are actuated (thumb yaw, thumb pitch and
the four finger proximals); the other 6 follow through joint equality constraints
that mirror the URDF ``<mimic>`` tags. The action space is 29 + 6 = 35.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

HAND_XML = Path(__file__).parent / "inspire_hand" / "right_hand.xml"
assert HAND_XML.exists()

HAND_ROOT = "R_hand_base_link"
WRIST = "right_wrist_yaw_link"

HAND_MOUNT_POS = (0.0415, 0.0, 0.0)
HAND_MOUNT_QUAT = (0.0, 0.70710678, -0.70710678, 0.0)

WRIST_MASS = 0.08457647
WRIST_IPOS = (0.02200381568, -0.00049485096, 0.00053861123)
WRIST_FULLINERTIA = (
  4.929128828e-05,
  5.973338134e-05,
  3.928083826e-05,
  4.5735494e-07,
  4.45867591e-06,
  -4.3217198e-07,
)

HAND_ACTUATED_JOINTS = (
  "R_thumb_proximal_yaw_joint",
  "R_thumb_proximal_pitch_joint",
  "R_index_proximal_joint",
  "R_middle_proximal_joint",
  "R_ring_proximal_joint",
  "R_pinky_proximal_joint",
)
FINGERTIP_SITES = (
  "R_thumb_tip",
  "R_index_tip",
  "R_middle_tip",
  "R_ring_tip",
  "R_pinky_tip",
)
PALM_SITE = "right_palm"


def _principal_inertia(full: tuple[float, ...]) -> tuple[list[float], list[float]]:
  """(ixx, iyy, izz, ixy, ixz, iyz) -> (diagonal inertia, principal-frame quat)."""
  ixx, iyy, izz, ixy, ixz, iyz = full
  tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
  diag, axes = np.linalg.eigh(tensor)
  if np.linalg.det(axes) < 0:
    axes[:, 0] *= -1
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, axes.flatten())
  return diag.tolist(), quat.tolist()


def copy_body(src: mujoco.MjsBody, parent: mujoco.MjsBody) -> mujoco.MjsBody:
  """Deep-copy ``src`` under ``parent`` keeping names (MjSpec.attach with an empty
  prefix works too but logs a warning on every compile)."""
  body = parent.add_body(
    name=src.name,
    pos=src.pos,
    quat=src.quat,
    mass=src.mass,
    ipos=src.ipos,
    iquat=src.iquat,
    inertia=src.inertia,
    explicitinertial=True,
  )
  for j in src.joints:
    body.add_joint(
      name=j.name,
      type=j.type,
      axis=j.axis,
      pos=j.pos,
      range=j.range,
      armature=j.armature,
      damping=j.damping,
      actfrcrange=j.actfrcrange,
    )
  for g in src.geoms:
    body.add_geom(
      name=g.name,
      type=g.type,
      meshname=g.meshname,
      pos=g.pos,
      quat=g.quat,
      size=g.size,
      contype=g.contype,
      conaffinity=g.conaffinity,
      group=g.group,
      density=g.density,
      rgba=g.rgba,
    )
  for s in src.sites:
    body.add_site(name=s.name, pos=s.pos, quat=s.quat, size=s.size, group=s.group)
  for child in src.bodies:
    copy_body(child, body)
  return body


def get_spec() -> mujoco.MjSpec:
  spec = g1.get_spec()
  wrist = spec.body(WRIST)

  for geom in list(wrist.geoms):
    if geom.meshname == "right_rubber_hand" or geom.name == "right_hand_collision":
      spec.delete(geom)
  spec.delete(spec.mesh("right_rubber_hand"))
  for site in list(wrist.sites):
    if site.name == PALM_SITE:
      spec.delete(site)

  wrist.mass = WRIST_MASS
  wrist.ipos = list(WRIST_IPOS)
  wrist.inertia, wrist.iquat = _principal_inertia(WRIST_FULLINERTIA)

  hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
  mesh_dir = HAND_XML.parent / hand_spec.compiler.meshdir
  for mesh in hand_spec.meshes:
    spec.add_mesh(name=mesh.name, file=str(mesh_dir / mesh.file))
  hand = copy_body(hand_spec.body(HAND_ROOT), wrist)
  hand.pos = list(HAND_MOUNT_POS)
  hand.quat = list(HAND_MOUNT_QUAT)
  for eq in hand_spec.equalities:
    spec.add_equality(
      type=eq.type,
      name=eq.name,
      name1=eq.name1,
      name2=eq.name2,
      objtype=eq.objtype,
      data=eq.data,
    )
  # In R_hand_base_link the fingers extend along -y and the palm faces -x. The
  # site sits on the palm surface, midway between the wrist and the finger roots.
  hand.add_site(name=PALM_SITE, pos=[-0.02, -0.09, 0.0], size=[0.01, 0, 0], group=5)
  return spec


##
# Actuators.
##

# Finger PD after WEAVE (G1 + Inspire): kp 5, kd 0.5, 1 Nm, armature 0.01.
INSPIRE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=HAND_ACTUATED_JOINTS,
  stiffness=5.0,
  damping=0.5,
  effort_limit=1.0,
  armature=0.01,
)

G1_INSPIRE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*g1.G1_ARTICULATION.actuators, INSPIRE_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

##
# Collisions.
##

_FOOT = r"^(left|right)_foot[1-7]_collision$"
_HAND = r"^R_.*_collision$"

# Same as mjlab's G1 FULL_COLLISION, plus the Inspire hand. Hand geoms use
# contype 2 / conaffinity 1: they touch the body (contype 1), objects and the
# world, but not each other (2 & 1 == 0), avoiding finger-finger mesh contacts.
# They get condim 3 so grasps have friction (other body geoms use condim 1).
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={_HAND: 2, ".*": 1},
  conaffinity=1,
  condim={_FOOT: 3, _HAND: 3, ".*": 1},
  priority={_FOOT: 1, _HAND: 1, ".*": 0},
  friction={_FOOT: (0.6,), _HAND: (1.0,)},
)

##
# Keyframe: mjlab's knees-bent stance with the hand open.
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=g1.KNEES_BENT_KEYFRAME.pos,
  joint_pos={**g1.KNEES_BENT_KEYFRAME.joint_pos, "R_.*": 0.0},
  joint_vel={".*": 0.0},
)


def get_g1_inspire_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_INSPIRE_ARTICULATION,
  )


# Body joints keep mjlab's G1 rule, 0.25 * effort / stiffness. For the weak, soft
# Inspire finger actuators (1 N m, kp 5) that rule gives 0.05 rad per unit action,
# so closing a finger to its ~0.7 rad grasp needs an action of ~14 and the policy
# never closed the hand (training run 3: zero hand contacts at every lift-phase
# failure). Fingers get a range-based scale instead: ~1.2 units closes a finger.
HAND_ACTION_SCALE = 0.6

G1_INSPIRE_ACTION_SCALE: dict[str, float] = {}
for _a in G1_INSPIRE_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    G1_INSPIRE_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness
for _n in HAND_ACTUATED_JOINTS:
  G1_INSPIRE_ACTION_SCALE[_n] = HAND_ACTION_SCALE
