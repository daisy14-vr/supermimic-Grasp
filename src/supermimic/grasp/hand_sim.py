"""Physics check of a grasp clip with a floating Inspire hand.

The hand base is welded to the clip's reference hand-base trajectory (ideal arm
tracking), the six active finger joints are driven by the real PD actuators, and
the object is a free cylinder resting on the table. This answers "can this hand
motion grasp and lift this object?" independently of whole-body tracking, and
gives a physically consistent object trajectory and contact record to use as the
clip's object reference and contact labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from supermimic.assets import g1_inspire as gi
from supermimic.grasp import scene

SIM_DT = 0.002
FINGER_KP = gi.INSPIRE_ACTUATOR.stiffness
FINGER_KD = gi.INSPIRE_ACTUATOR.damping
FINGER_EFFORT = gi.INSPIRE_ACTUATOR.effort_limit
CONTACT_FORCE = 0.5  # N; a body "touches" the object above this normal force
HAND_FRICTION = 1.0


@dataclass
class HandSimResult:
  object_pos_w: np.ndarray  # (T, 3)
  object_quat_w: np.ndarray  # (T, 4) wxyz
  object_lin_vel_w: np.ndarray  # (T, 3)
  object_ang_vel_w: np.ndarray  # (T, 3)
  contact: np.ndarray  # (T, len(body_names)) bool
  body_names: tuple[str, ...]
  lift: float  # object rise over the lift + hold phases
  tilt_deg: float  # object tilt at the end


def hand_base_trajectory(
  robot: mujoco.MjModel, clip: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
  data = mujoco.MjData(robot)
  body = robot.body(gi.HAND_ROOT).id
  num = len(clip["joint_pos"])
  pos, quat = np.zeros((num, 3)), np.zeros((num, 4))
  for t in range(num):
    data.qpos[:3] = clip["root_pos_w"][t]
    data.qpos[3:7] = clip["root_quat_w"][t]
    data.qpos[7:] = clip["joint_pos"][t]
    mujoco.mj_kinematics(robot, data)
    pos[t], quat[t] = data.xpos[body], data.xquat[body]
  return pos, quat


def _build(clip: dict[str, np.ndarray], pos0, quat0) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.compiler.degree = False
  spec.option.timestep = SIM_DT
  hand_spec = mujoco.MjSpec.from_file(str(gi.HAND_XML))
  mesh_dir = gi.HAND_XML.parent / hand_spec.compiler.meshdir
  for mesh in hand_spec.meshes:
    spec.add_mesh(name=mesh.name, file=str(mesh_dir / mesh.file))
  hand = gi.copy_body(hand_spec.body(gi.HAND_ROOT), spec.worldbody)
  hand.pos, hand.quat = pos0.tolist(), quat0.tolist()
  hand.add_freejoint()
  for eq in hand_spec.equalities:
    spec.add_equality(
      type=eq.type,
      name=eq.name,
      name1=eq.name1,
      name2=eq.name2,
      objtype=eq.objtype,
      data=eq.data,
    )
  for geom in spec.geoms:
    if geom.name.endswith("_collision"):
      geom.contype, geom.conaffinity, geom.condim = 2, 1, 3
      geom.friction = [HAND_FRICTION, 0.005, 0.0001]
  spec.worldbody.add_body(name="target", mocap=True, pos=pos0.tolist(), quat=quat0)
  spec.add_equality(
    type=mujoco.mjtEq.mjEQ_WELD,
    name1="target",
    name2=gi.HAND_ROOT,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    data=[0.0] * 6 + [1.0, 0.0, 0.0, 0.0] + [1.0, 0.0],
  )

  table_h = float(clip["table_height"])
  obj_x, obj_y = float(clip["object_x"]), float(clip["object_y"])
  table = spec.worldbody.add_body(
    name="table", pos=list(scene.table_center(obj_x, obj_y, table_h))
  )
  scene.add_table(table)
  obj = spec.worldbody.add_body(name="object", pos=clip["object_pos_w"][0].tolist())
  obj.add_freejoint()
  scene.add_cylinder(
    obj,
    float(clip["object_radius"]),
    float(clip["object_height"]),
    float(clip["object_mass"]),
  )
  for joint in gi.HAND_ACTUATED_JOINTS:
    act = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gainprm = [FINGER_KP] + [0.0] * 9
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm = [0.0, -FINGER_KP, -FINGER_KD] + [0.0] * 7
    act.forcelimited = True
    act.forcerange = [-FINGER_EFFORT, FINGER_EFFORT]
  return spec


def simulate(
  robot: mujoco.MjModel, clip: dict[str, np.ndarray], fps: int
) -> HandSimResult:
  hand_pos, hand_quat = hand_base_trajectory(robot, clip)
  model = _build(clip, hand_pos[0], hand_quat[0]).compile()
  data = mujoco.MjData(model)
  joint_names = clip["joint_names"].tolist()
  act_cols = [joint_names.index(j) for j in gi.HAND_ACTUATED_JOINTS]
  obj_qadr = model.jnt_qposadr[model.body("object").jntadr[0]]
  obj_dadr = model.jnt_dofadr[model.body("object").jntadr[0]]
  obj_geom = model.geom("object").id

  hand_bodies = tuple(
    model.body(i).name
    for i in range(model.nbody)
    if any(
      model.geom(g).name.endswith("_collision")
      for g in range(
        model.body_geomadr[i], model.body_geomadr[i] + model.body_geomnum[i]
      )
    )
  )
  body_col = {model.body(n).id: k for k, n in enumerate(hand_bodies)}

  num = len(clip["joint_pos"])
  substeps = round(1.0 / (fps * SIM_DT))
  out_pos, out_quat = np.zeros((num, 3)), np.zeros((num, 4))
  out_lin, out_ang = np.zeros((num, 3)), np.zeros((num, 3))
  contact = np.zeros((num, len(hand_bodies)), dtype=bool)
  force = np.zeros(6)
  ff = FINGER_KD / FINGER_KP  # velocity feed-forward cancels PD damping lag
  for t in range(num):
    data.mocap_pos[0], data.mocap_quat[0] = hand_pos[t], hand_quat[t]
    data.ctrl[:] = clip["joint_pos"][t][act_cols] + ff * clip["joint_vel"][t][act_cols]
    for _ in range(substeps):
      mujoco.mj_step(model, data)
    out_pos[t] = data.qpos[obj_qadr : obj_qadr + 3]
    out_quat[t] = data.qpos[obj_qadr + 3 : obj_qadr + 7]
    out_lin[t] = data.qvel[obj_dadr : obj_dadr + 3]
    # Free-joint angular velocity is in the body frame; rotate to world.
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, out_quat[t])
    out_ang[t] = rot.reshape(3, 3) @ data.qvel[obj_dadr + 3 : obj_dadr + 6]
    for i in range(data.ncon):
      con = data.contact[i]
      if obj_geom not in (con.geom1, con.geom2):
        continue
      other = con.geom2 if con.geom1 == obj_geom else con.geom1
      col = body_col.get(int(model.geom_bodyid[other]))
      if col is None:
        continue
      mujoco.mj_contactForce(model, data, i, force)
      if force[0] > CONTACT_FORCE:
        contact[t, col] = True

  phase = dict(zip(clip["phase_names"].tolist(), clip["phase_starts"], strict=True))
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, out_quat[-1])
  return HandSimResult(
    object_pos_w=out_pos,
    object_quat_w=out_quat,
    object_lin_vel_w=out_lin,
    object_ang_vel_w=out_ang,
    contact=contact,
    body_names=hand_bodies,
    lift=float(out_pos[-1, 2] - out_pos[phase["lift"] - 1, 2]),
    tilt_deg=float(np.degrees(np.arccos(np.clip(rot[8], -1.0, 1.0)))),
  )
