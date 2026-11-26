"""Procedural grasp-and-lift reference clips for the G1 + right Inspire hand.

Each clip: the robot stands at a table, squats/leans as the table height requires,
reaches its right hand to the side of an upright cylinder, wraps the fingers
around it, lifts it and holds it. A whole-body IK (feet planted, center of mass
over the feet, pelvis-height schedule, palm pose target, posture prior) produces
the robot reference; the object rests on the table until the lift and then
moves rigidly with the palm.

Frame conventions: world z up, the robot faces +x and its feet stay put. In
``R_hand_base_link`` the fingers extend along -y and the palm faces -x; the side
grasp places the palm normal along +y (the palm faces the robot's midline) with the
fingers pointing +x.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from mjlab.entity.entity import Entity

from supermimic.assets import g1_inspire as gi

FPS = 50

# Phases and their keyframe segments (s). The palm first rises above the table,
# moves over to hover above the pre-grasp, descends while the thumb rotates into
# opposition (clear of the object), slides in sideways, and closes the fingers.
PHASES = (
  ("settle", (0.4,)),
  ("reach", (0.9, 1.3)),  # rise; move over to hover
  ("approach", (0.5, 0.6)),  # descend to pre-grasp, opposing the thumb; slide in
  ("close", (0.5, 0.3)),  # flex fingers; settle the squeeze
  ("lift", (1.0,)),
  ("hold", (0.8,)),
)

LIFT_HEIGHT = 0.15
PREGRASP_OFFSET = 0.08  # palm starts this far outside (−y) the grasp pose
HOVER_CLEARANCE = 0.10  # palm height above the table while moving over it
GRASP_HEIGHT_FRAC = 0.45  # palm height on the cylinder
PALM_CLEARANCE = 0.006  # palm surface to cylinder surface at the grasp
PALM_BEHIND_AXIS = 0.065  # palm sits behind the axis so the fingers wrap it
THUMB_YAW = 1.1  # thumb opposition during the grasp
SQUEEZE = 0.3  # rad of closure past first contact, so PD presses the object
TIP_PAD = 0.006  # fingertip site to finger surface

# IK task weights (squared-error cost): feet >> pelvis, CoM > palm >> posture.
W_FEET = 100.0
W_PELVIS = 20.0
W_COM = 10.0
W_PALM_POS = 10.0
W_PALM_ROT = 1.0
W_POSTURE = 1e-3
# Per-joint posture weights overriding W_POSTURE. Waist roll is held near zero:
# a sideways torso sway is not needed to reach a table in front of the robot,
# and without this the solver uses it freely (it saturated its limit in every
# clip). Waist yaw gets a milder pull.
POSTURE_WEIGHTS = {
  "waist_roll_joint": 1.0,
  "waist_yaw_joint": 0.05,
  "waist_pitch_joint": 0.01,
}
# Hard IK bounds (rad). No sideways sway; no backward lean (which, with the CoM
# held over the feet, tips the robot over backwards under physics).
JOINT_BOUNDS = {
  "waist_roll_joint": (-0.15, 0.15),
  "waist_pitch_joint": (-0.05, 0.45),
}
# Pelvis may rise at most this far (m) above the knees-bent home height, and rises
# by up to LIFT_RISE while lifting.
MAX_RISE = 0.03
LIFT_RISE = 0.05
# The CoM is kept within this distance (m) of the feet's support center.
COM_TOLERANCE = 0.02

# Bodies that get a contact label, in this order.
CONTACT_BODIES = (
  "R_hand_base_link",
  "R_thumb_distal",
  "R_index_intermediate",
  "R_middle_intermediate",
  "R_ring_intermediate",
  "R_pinky_intermediate",
)
# Leader joint per finger tip site, for the closure search.
_FINGERS = (
  ("R_thumb_tip", "R_thumb_proximal_pitch_joint", "R_thumb_distal"),
  ("R_index_tip", "R_index_proximal_joint", "R_index_intermediate"),
  ("R_middle_tip", "R_middle_proximal_joint", "R_middle_intermediate"),
  ("R_ring_tip", "R_ring_proximal_joint", "R_ring_intermediate"),
  ("R_pinky_tip", "R_pinky_proximal_joint", "R_pinky_intermediate"),
)
_MIMIC = tuple(
  (f, leader, mult)
  for f, leader, mult in (
    ("R_thumb_intermediate_joint", "R_thumb_proximal_pitch_joint", 1.6),
    ("R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4),
    ("R_index_intermediate_joint", "R_index_proximal_joint", 1.0),
    ("R_middle_intermediate_joint", "R_middle_proximal_joint", 1.0),
    ("R_ring_intermediate_joint", "R_ring_proximal_joint", 1.0),
    ("R_pinky_intermediate_joint", "R_pinky_proximal_joint", 1.0),
  )
)

# Palm orientation of the side grasp: hand axes expressed in world.
_PALM_ROT = np.array(
  [
    [0.0, -1.0, 0.0],  # columns: x_h = -y_w, y_h = -x_w, z_h = -z_w
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
  ]
)


@dataclass
class ClipParams:
  table_height: float
  object_radius: float
  object_height: float
  object_mass: float
  object_x: float
  object_y: float


# v1 trains on one object: an upright cylinder of fixed size.
V1_OBJECT = dict(object_radius=0.03, object_height=0.15, object_mass=0.3)


def sample_params(rng: np.random.Generator, fixed_object: bool = True) -> ClipParams:
  """Table height and object placement vary; with ``fixed_object`` the object is
  ``V1_OBJECT``, otherwise its size and mass vary too."""
  if fixed_object:
    obj = dict(V1_OBJECT)
  else:
    obj = dict(
      # The Inspire hand fails to wrap cylinders much wider than r = 3.5 cm.
      object_radius=float(rng.uniform(0.025, 0.035)),
      object_height=float(rng.uniform(0.12, 0.18)),
      object_mass=float(rng.uniform(0.2, 0.5)),
    )
  return ClipParams(
    table_height=float(rng.uniform(0.55, 0.85)),
    object_x=float(rng.uniform(0.34, 0.42)),
    object_y=float(rng.uniform(-0.24, -0.16)),
    **obj,
  )


def _smoothstep(x: np.ndarray) -> np.ndarray:
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3.0 - 2.0 * x)


class WholeBodyIK:
  """Damped least-squares IK on the G1 + Inspire model with a fixed-orientation
  floating base. Arms other than the right one and the fingers are set directly."""

  def __init__(self) -> None:
    self.model = Entity(gi.get_g1_inspire_robot_cfg()).spec.compile()
    m = self.model
    self.data = mujoco.MjData(m)
    self.joint_names = [m.joint(i).name for i in range(1, m.njnt)]

    init = gi.KNEES_BENT_KEYFRAME
    self.q_home = np.zeros(m.nq)
    self.q_home[:3] = init.pos
    self.q_home[3] = 1.0
    for name in self.joint_names:
      self.q_home[self._qadr(name)] = _match(init.joint_pos, name)
    self._place_feet_on_ground(self.q_home)

    # IK variables: root translation, legs, waist, right arm.
    leg = re.compile(r"^(left|right)_(hip|knee|ankle)_.*")
    arm = re.compile(r"^(waist|right_(shoulder|elbow|wrist))_.*")
    ik_joints = [n for n in self.joint_names if leg.match(n) or arm.match(n)]
    self.dofs = np.array([0, 1, 2] + [self._dadr(n) for n in ik_joints])
    self.ik_qadr = np.array([self._qadr(n) for n in ik_joints])
    lo, hi = m.jnt_range[[m.joint(n).id for n in ik_joints]].T
    margin = 0.05 * (hi - lo)
    self.lo, self.hi = lo + margin, hi - margin
    for name, (b_lo, b_hi) in JOINT_BOUNDS.items():
      i = ik_joints.index(name)
      self.lo[i], self.hi[i] = max(self.lo[i], b_lo), min(self.hi[i], b_hi)
    self.posture_w = np.array([POSTURE_WEIGHTS.get(n, W_POSTURE) for n in ik_joints])

    self.palm = m.site(gi.PALM_SITE).id
    self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    self.data.qpos[:] = self.q_home
    mujoco.mj_forward(m, self.data)
    self.foot_pose = [
      (self.data.xpos[b].copy(), self.data.xquat[b].copy()) for b in self.feet
    ]
    # Support center: mean of the foot contact capsules (ahead of the ankles).
    foot = re.compile(r"^(left|right)_foot[1-7]_collision$")
    foot_geoms = [g for g in range(m.ngeom) if foot.match(m.geom(g).name)]
    self.com_xy = self.data.geom_xpos[foot_geoms, :2].mean(axis=0)
    self.pelvis_home = float(self.q_home[2])

  def _qadr(self, name: str) -> int:
    return int(self.model.jnt_qposadr[self.model.joint(name).id])

  def _dadr(self, name: str) -> int:
    return int(self.model.jnt_dofadr[self.model.joint(name).id])

  def _place_feet_on_ground(self, q: np.ndarray) -> None:
    d = self.data
    d.qpos[:] = q
    mujoco.mj_forward(self.model, d)
    foot = re.compile(r"^(left|right)_foot[1-7]_collision$")
    lowest = min(
      d.geom_xpos[g][2] - self.model.geom_size[g][0]  # capsule radius
      for g in range(self.model.ngeom)
      if foot.match(self.model.geom(g).name)
    )
    q[2] -= lowest

  def set_hand(self, q: np.ndarray, closure: dict[str, float], thumb_yaw: float):
    q[self._qadr("R_thumb_proximal_yaw_joint")] = thumb_yaw
    for leader, value in closure.items():
      q[self._qadr(leader)] = value
    for follower, leader, mult in _MIMIC:
      q[self._qadr(follower)] = mult * q[self._qadr(leader)]

  def solve(
    self,
    q: np.ndarray,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    pelvis_height: float,
    palm_weight: float = 1.0,
    iters: int = 60,
  ) -> np.ndarray:
    m, d = self.model, self.data
    q = q.copy()
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    for _ in range(iters):
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      # (jacobian, error, weight) per task.
      tasks: list[tuple[np.ndarray, np.ndarray, float | np.ndarray]] = []
      for body, (pos, quat) in zip(self.feet, self.foot_pose, strict=True):
        mujoco.mj_jacBody(m, d, jp, jr, body)
        tasks.append((jp.copy(), pos - d.xpos[body], W_FEET))
        tasks.append((jr.copy(), _rot_err(quat, d.xquat[body]), W_FEET))
      mujoco.mj_jacSite(m, d, jp, jr, self.palm)
      palm_err = palm_pos - d.site_xpos[self.palm]
      tasks.append((jp.copy(), palm_err, W_PALM_POS * palm_weight))
      cur = d.site_xmat[self.palm].reshape(3, 3)
      rot_err = _so3_log(palm_rot @ cur.T)
      tasks.append((jr.copy(), rot_err, W_PALM_ROT * palm_weight))
      jc = np.zeros((3, m.nv))
      mujoco.mj_jacSubtreeCom(m, d, jc, 1)
      com_err = self.com_xy - d.subtree_com[1][:2]
      com_err = np.sign(com_err) * np.maximum(np.abs(com_err) - COM_TOLERANCE, 0.0)
      tasks.append((jc[:2], com_err, W_COM))
      jz = np.zeros((1, m.nv))
      jz[0, 2] = 1.0
      tasks.append((jz, np.array([pelvis_height - q[2]]), W_PELVIS))
      # Posture prior toward home (root translation excluded).
      jpost = np.zeros((len(self.dofs) - 3, m.nv))
      jpost[np.arange(len(self.dofs) - 3), self.dofs[3:]] = 1.0
      posture_err = self.q_home[self.ik_qadr] - q[self.ik_qadr]
      tasks.append((jpost, posture_err, self.posture_w))

      rows = [j[:, self.dofs] for j, _, _ in tasks]
      errs = [e for _, e, _ in tasks]
      weights = [np.broadcast_to(w, e.shape) for _, e, w in tasks]
      J = np.vstack(rows)
      e = np.concatenate(errs)
      w = np.concatenate(weights)
      H = J.T @ (w[:, None] * J) + 1e-4 * np.eye(J.shape[1])
      dx = np.linalg.solve(H, J.T @ (w * e))
      dq = np.zeros(m.nv)
      dq[self.dofs] = dx
      mujoco.mj_integratePos(m, q, dq, 1.0)
      q[self.ik_qadr] = np.clip(q[self.ik_qadr], self.lo, self.hi)
      if np.linalg.norm(dx) < 1e-5:
        break
    return q

  def closure_at_contact(
    self, q: np.ndarray, axis_xy: np.ndarray, radius: float, z_range
  ) -> tuple[dict[str, float], dict[str, bool]]:
    """Per finger, the smallest leader angle whose tip reaches the cylinder
    surface (+SQUEEZE), and whether it reaches the cylinder at all."""
    m, d = self.model, self.data
    closure, touches = {}, {}
    for tip, leader, _ in _FINGERS:
      lo, hi = m.jnt_range[m.joint(leader).id]
      angle, hit = 0.8 * hi, False
      for a in np.linspace(lo, hi, 80):
        qq = q.copy()
        self.set_hand(qq, {leader: a}, THUMB_YAW)
        d.qpos[:] = qq
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[m.site(tip).id]
        inside_z = z_range[0] <= p[2] <= z_range[1]
        if inside_z and np.linalg.norm(p[:2] - axis_xy) <= radius + TIP_PAD:
          angle, hit = min(a + SQUEEZE, hi), True
          break
      closure[leader], touches[leader] = float(angle), hit
    return closure, touches


def _grasp_geometry(p: ClipParams):
  axis_xy = np.array([p.object_x, p.object_y])
  grasp_z = p.table_height + GRASP_HEIGHT_FRAC * p.object_height
  palm_grasp = np.array(
    [
      p.object_x - PALM_BEHIND_AXIS,
      p.object_y - (p.object_radius + PALM_CLEARANCE),
      grasp_z,
    ]
  )
  return axis_xy, palm_grasp


def generate_clip(ik: WholeBodyIK, p: ClipParams) -> dict[str, np.ndarray]:
  m = ik.model
  axis_xy, palm_grasp = _grasp_geometry(p)
  palm_pre = palm_grasp - np.array([0.0, PREGRASP_OFFSET, 0.0])
  clear_z = max(palm_pre[2], p.table_height + HOVER_CLEARANCE)
  # Squat for low grasps; for high grasps and the lift, stand taller (straighten
  # the knees) rather than lean back. Positive drop lowers the pelvis.
  drop = float(np.clip(0.82 - palm_grasp[2], -MAX_RISE, 0.2)) * 0.7
  pelvis_home, pelvis_grasp = ik.pelvis_home, ik.pelvis_home - drop
  pelvis_lift = min(pelvis_grasp + LIFT_RISE, pelvis_home + MAX_RISE)

  ik.data.qpos[:] = ik.q_home
  mujoco.mj_kinematics(m, ik.data)
  palm_home = ik.data.site_xpos[ik.palm].copy()
  rot_home = ik.data.site_xmat[ik.palm].reshape(3, 3).copy()

  # Finger closure that just reaches the cylinder at the grasp pose.
  q_grasp = ik.solve(ik.q_home, palm_grasp, _PALM_ROT, pelvis_grasp, iters=200)
  z_range = (p.table_height + 0.005, p.table_height + p.object_height)
  closure, touches = ik.closure_at_contact(q_grasp, axis_xy, p.object_radius, z_range)

  # Keyframes: (palm, palm rotation, pelvis, thumb yaw, finger flexion). The
  # palm keeps its home orientation while rising and turns into the grasp
  # orientation while moving over to the hover point.
  lift = palm_grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
  rise = np.array([*palm_home[:2], clear_z])
  hover = np.array([*palm_pre[:2], clear_z])
  grasp_rot, yaw = _PALM_ROT, THUMB_YAW
  keys = [
    (palm_home, rot_home, pelvis_home, 0.0, 0.0),  # start
    (palm_home, rot_home, pelvis_home, 0.0, 0.0),  # settle
    (rise, rot_home, pelvis_home, 0.0, 0.0),  # rise
    (hover, grasp_rot, pelvis_grasp, 0.0, 0.0),  # hover
    (palm_pre, grasp_rot, pelvis_grasp, yaw, 0.0),  # pre-grasp, thumb opposed
    (palm_grasp, grasp_rot, pelvis_grasp, yaw, 0.0),  # slide in
    (palm_grasp, grasp_rot, pelvis_grasp, yaw, 1.0),  # flex
    (palm_grasp, grasp_rot, pelvis_grasp, yaw, 1.0),  # settle the squeeze
    (lift, grasp_rot, pelvis_lift, yaw, 1.0),  # lift
    (lift, grasp_rot, pelvis_lift, yaw, 1.0),  # hold
  ]
  durations = [d for _, segs in PHASES for d in segs]
  seg_frames = [round(d * FPS) for d in durations]
  num_frames = sum(seg_frames) + 1
  phase_starts, i = [], 0
  for _, segs in PHASES:
    phase_starts.append(i)
    i += sum(round(d * FPS) for d in segs)

  qs = np.zeros((num_frames, m.nq))
  palm_traj = np.zeros((num_frames, 3))
  q = ik.q_home.copy()
  t = 0
  for k, n in enumerate([1, *seg_frames]):
    a, b = keys[max(k - 1, 0)], keys[k]
    for j in range(n):
      s = float(_smoothstep(np.array((j + 1) / n)))
      palm = (1 - s) * a[0] + s * b[0]
      rot = _so3_exp(s * _so3_log(b[1] @ a[1].T)) @ a[1]
      pelvis = (1 - s) * a[2] + s * b[2]
      thumb_yaw = (1 - s) * a[3] + s * b[3]
      flex = (1 - s) * a[4] + s * b[4]
      q = ik.solve(q, palm, rot, pelvis)
      ik.set_hand(q, {key: flex * v for key, v in closure.items()}, thumb_yaw)
      qs[t] = q
      ik.data.qpos[:] = q
      mujoco.mj_kinematics(m, ik.data)
      palm_traj[t] = ik.data.site_xpos[ik.palm]
      t += 1
  assert t == num_frames

  phase = dict(zip([name for name, _ in PHASES], phase_starts, strict=True))

  # Object: at rest until the lift, then rigid with the palm translation.
  object_z = p.table_height + 0.5 * p.object_height
  obj_pos = np.tile([p.object_x, p.object_y, object_z], (num_frames, 1))
  lift0 = phase["lift"]
  obj_pos[lift0:] += palm_traj[lift0:] - palm_traj[lift0]
  obj_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1))

  # Contact labels: forbidden while reaching, free while approaching, required
  # for fingers that reach the object once they flex.
  labels = np.zeros((num_frames, len(CONTACT_BODIES)), dtype=np.int8)
  labels[: phase["approach"]] = -1
  flex_start = phase["close"] + round(PHASES[3][1][0] * FPS) // 2
  for i, (_, leader, body) in enumerate(_FINGERS, start=1):
    assert CONTACT_BODIES[i] == body
    labels[flex_start:, i] = 1 if touches[leader] else 0

  dt = 1.0 / FPS
  joint_pos = qs[:, 7:]
  root_pos = qs[:, :3]
  root_quat = qs[:, 3:7]
  return {
    "fps": np.array(FPS),
    "joint_names": np.array(ik.joint_names),
    "root_pos_w": root_pos.astype(np.float32),
    "root_quat_w": root_quat.astype(np.float32),
    "root_lin_vel_w": np.gradient(root_pos, dt, axis=0).astype(np.float32),
    # The base orientation is held fixed by the IK.
    "root_ang_vel_w": np.zeros_like(root_pos, dtype=np.float32),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": np.gradient(joint_pos, dt, axis=0).astype(np.float32),
    "object_pos_w": obj_pos.astype(np.float32),
    "object_quat_w": obj_quat.astype(np.float32),
    "object_lin_vel_w": np.gradient(obj_pos, dt, axis=0).astype(np.float32),
    "object_ang_vel_w": np.zeros_like(obj_pos, dtype=np.float32),
    "contact_label": labels,
    "contact_body_names": np.array(CONTACT_BODIES),
    "phase_starts": np.array(phase_starts, dtype=np.int32),
    "phase_names": np.array([name for name, _ in PHASES]),
    **{k: np.array(v) for k, v in asdict(p).items()},
  }


def apply_hand_sim(clip: dict[str, np.ndarray], sim) -> dict[str, np.ndarray]:
  """Replace the scripted object trajectory and contact labels of ``clip`` with
  the physically simulated ones from ``hand_sim.simulate``.

  Labels per hand body: -1 (must not touch) while reaching, 0 (free) while
  approaching, then +1 where the simulated grasp touches the object and 0
  elsewhere.
  """
  phase = dict(zip(clip["phase_names"].tolist(), clip["phase_starts"], strict=True))
  labels = np.where(sim.contact, 1, 0).astype(np.int8)
  labels[: phase["close"]] = 0
  labels[: phase["approach"]] = -1
  return {
    **clip,
    "object_pos_w": sim.object_pos_w.astype(np.float32),
    "object_quat_w": sim.object_quat_w.astype(np.float32),
    "object_lin_vel_w": sim.object_lin_vel_w.astype(np.float32),
    "object_ang_vel_w": sim.object_ang_vel_w.astype(np.float32),
    "contact_label": labels,
    "contact_body_names": np.array(sim.body_names),
    "sim_lift": np.array(sim.lift),
    "sim_tilt_deg": np.array(sim.tilt_deg),
  }


def _match(patterns: dict[str, float], name: str) -> float:
  for pat, value in patterns.items():
    if re.fullmatch(pat, name):
      return value
  return 0.0


def _rot_err(target_quat: np.ndarray, cur_quat: np.ndarray) -> np.ndarray:
  """World-frame rotation vector taking ``cur_quat`` to ``target_quat``."""
  res = np.zeros(3)
  mujoco.mju_subQuat(res, target_quat, cur_quat)
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, cur_quat)
  return mat.reshape(3, 3) @ res


def _so3_exp(vec: np.ndarray) -> np.ndarray:
  angle = float(np.linalg.norm(vec))
  if angle < 1e-12:
    return np.eye(3)
  quat, mat = np.zeros(4), np.zeros(9)
  mujoco.mju_axisAngle2Quat(quat, vec / angle, angle)
  mujoco.mju_quat2Mat(mat, quat)
  return mat.reshape(3, 3)


def _so3_log(rot: np.ndarray) -> np.ndarray:
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rot.flatten())
  vel = np.zeros(3)
  mujoco.mju_quat2Vel(vel, quat, 1.0)
  return vel
