"""World-model state with a grasped object: GenCoord robot + rigid object block.

Full state: ``[robot GenCoord full state (13 + 2N), obj_pos(3), obj_quat(4),
obj_lin_vel_w(3), obj_ang_vel_w(3)]``; the object position has env origins
removed like the robot root.

Rep state: ``[robot GenCoord rep state (10 + 2N), obj_pos_b(3), obj_rot6d_b(6),
obj_v_b(3), obj_w_b(3)]``; the object pose and twist are expressed in the robot
root frame, so the state stays translation/heading invariant.

Prediction: ``[robot GenCoord prediction (6 + N), Δv_obj(3), Δw_obj(3)]``,
acceleration-level deltas of the object's world twist expressed in the current
root frame, integrated semi-implicitly like the root (contact jumps become velocity
changes; pose stays kinematically consistent).

Besides the state, the dynamics predicts one contact logit per hand body
(``contact_dim``), supervised by the env's ``contact`` obs group. Contacts are not
fed back into the state (as in Dex-SuperMimic): they feed the imagined
contact-match reward only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_matrix,
)

from supermimic.assets.g1_inspire import HAND_ACTUATED_JOINTS
from supermimic.representations import GenCoordRepresentation, register_representation
from supermimic.utils.gencoord_helpers import (
  compute_gencoord_state,
  infer_num_joints_from_state_dim,
)
from supermimic.utils.helpers import rot6d_to_matrix

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.grasp.command import GraspMotionCommand
  from supermimic.sim_context import SimContext

OBJ_FULL_DIM = 13
OBJ_REP_DIM = 15
OBJ_PRED_DIM = 6


def _mat_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
  return mat[..., :2].flatten(-2, -1)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
  """Rotation matrix of the rotation vector ``omega`` (Rodrigues), ``(..., 3, 3)``."""
  angle = omega.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  axis = omega / angle
  x, y, z = axis.unbind(-1)
  zero = torch.zeros_like(x)
  k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).unflatten(-1, (3, 3))
  s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
  eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(k)
  return eye + s * k + (1 - c) * (k @ k)


def rotation_angle(r_a: torch.Tensor, r_b: torch.Tensor) -> torch.Tensor:
  """Geodesic angle between rotation matrices, via atan2 (finite gradient at 0)."""
  rel = r_a.transpose(-1, -2) @ r_b
  trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
  skew = rel - rel.transpose(-1, -2)
  sin = 0.5 * torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], -1)
  return torch.atan2(sin.norm(dim=-1), 0.5 * (trace - 1.0))


class GenCoordObjectRepresentation(GenCoordRepresentation):
  LOSS_WEIGHT_CONFIG = {
    **GenCoordRepresentation.LOSS_WEIGHT_CONFIG,
    "object_pos": "nominal_error_object_pos",
    "object_rot": "nominal_error_object_rot",
    "object_vel": "nominal_error_object_vel",
    "object_avel": "nominal_error_object_avel",
  }

  def __init__(
    self, num_joints: int, default_joint_pos: torch.Tensor, contact_dim: int
  ):
    super().__init__(num_joints, default_joint_pos)
    self.robot_full_dim = self.full_state_dim
    self.robot_rep_dim = self.rep_state_dim
    self.robot_pred_dim = self.prediction_dim
    self.full_state_dim += OBJ_FULL_DIM
    self.rep_state_dim += OBJ_REP_DIM
    self.prediction_dim += OBJ_PRED_DIM
    self.contact_dim = contact_dim

  # -- splitting ------------------------------------------------------------

  def _robot_rep(self, rep_state: torch.Tensor) -> torch.Tensor:
    return rep_state[..., : self.robot_rep_dim]

  def object_rep(self, rep_state: torch.Tensor):
    o = rep_state[..., self.robot_rep_dim :]
    return o[..., 0:3], o[..., 3:9], o[..., 9:12], o[..., 12:15]

  def object_world(self, aux_state: torch.Tensor, rep_state: torch.Tensor):
    """Object ``(pos, rot_mat, lin_vel, ang_vel)`` in the world (env-local) frame."""
    pos_b, rot6d_b, v_b, w_b = self.object_rep(rep_state)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    root_rot = matrix_from_quat(root_quat)
    return (
      root_pos + quat_apply(root_quat, pos_b),
      root_rot @ rot6d_to_matrix(rot6d_b),
      quat_apply(root_quat, v_b),
      quat_apply(root_quat, w_b),
    )

  def _object_relative(self, aux_state, pos_w, rot_w, v_w, w_w) -> torch.Tensor:
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    root_rot = matrix_from_quat(root_quat)
    return torch.cat(
      [
        quat_apply_inverse(root_quat, pos_w - root_pos),
        _mat_to_rot6d(root_rot.transpose(-1, -2) @ rot_w),
        quat_apply_inverse(root_quat, v_w),
        quat_apply_inverse(root_quat, w_w),
      ],
      dim=-1,
    )

  # -- StateRepresentation ------------------------------------------------

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    robot_full = full_state[..., : self.robot_full_dim]
    flat = robot_full.reshape(-1, self.robot_full_dim)
    robot = compute_gencoord_state(flat).reshape(*robot_full.shape[:-1], -1)
    obj = full_state[..., self.robot_full_dim :]
    aux = full_state[..., :7]
    rel = self._object_relative(
      aux,
      obj[..., 0:3],
      matrix_from_quat(obj[..., 3:7]),
      obj[..., 7:10],
      obj[..., 10:13],
    )
    return torch.cat([robot, rel], dim=-1)

  def apply_prediction(self, aux_state, curr_rep_state, prediction, dt):
    next_aux, next_robot = super().apply_prediction(
      aux_state,
      self._robot_rep(curr_rep_state),
      prediction[..., : self.robot_pred_dim],
      dt,
    )
    c = self.substep_coeff
    pos_w, rot_w, v_w, w_w = self.object_world(aux_state, curr_rep_state)
    root_quat = aux_state[..., 3:7]
    dv = quat_apply(
      root_quat, prediction[..., self.robot_pred_dim : self.robot_pred_dim + 3]
    )
    dw = quat_apply(root_quat, prediction[..., self.robot_pred_dim + 3 :])
    next_v = v_w + dv
    next_w = w_w + dw
    next_pos = pos_w + (v_w + c * dv) * dt
    next_rot = _so3_exp((w_w + c * dw) * dt) @ rot_w
    rel = self._object_relative(next_aux, next_pos, next_rot, next_v, next_w)
    return next_aux, torch.cat([next_robot, rel], dim=-1)

  def reconstruct_full_state(self, aux_state, rep_state):
    robot = super().reconstruct_full_state(aux_state, self._robot_rep(rep_state))
    pos_w, rot_w, v_w, w_w = self.object_world(aux_state, rep_state)
    return torch.cat([robot, pos_w, quat_from_matrix(rot_w), v_w, w_w], dim=-1)

  def loss_blocks(self, rep_state, aux_state):
    blocks = super().loss_blocks(self._robot_rep(rep_state), aux_state)
    pos_b, rot6d_b, v_b, w_b = self.object_rep(rep_state)
    return {
      **blocks,
      "object_pos": pos_b,
      "object_rot": rot6d_b,
      "object_vel": v_b,
      "object_avel": w_b,
    }

  def world_frame_bodies(self, aux_state, rep_state):
    return super().world_frame_bodies(aux_state, self._robot_rep(rep_state))

  def joint_state(self, rep_state):
    return super().joint_state(self._robot_rep(rep_state))

  def compute_wm_obs(self, rep_state, aux_state, joint_state, sim_ctx):
    return super().compute_wm_obs(
      self._robot_rep(rep_state), aux_state, joint_state, sim_ctx
    )

  # -- actor view ------------------------------------------------------------

  # Linear velocities are dropped from the actor's input: a real robot has no
  # reliable root linear-velocity sensor, and the object's twist would come from
  # perception. Robot rep: [z, g_b, v_b(4:7), w_b, qj, vj]; object rep:
  # [pos_b, rot6d_b, v_b(9:12), w_b].
  @property
  def actor_state_dim(self) -> int:
    return self.rep_state_dim - 6

  def actor_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    robot = self._robot_rep(rep_state)
    obj = rep_state[..., self.robot_rep_dim :]
    return torch.cat([robot[..., :4], robot[..., 7:], obj[..., :9], obj[..., 12:]], -1)

  # -- grasp terms for the world-model env ---------------------------------

  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    super().bind(env, sim_context)
    cmd = cast("GraspMotionCommand", env.command_manager.get_term("motion"))
    self.ref_object_pos = cmd.ref_object_pos
    self.ref_object_rot = matrix_from_quat(cmd.ref_object_quat)
    self.ref_contact_label = cmd.ref_contact_label
    rewards, dt = env.cfg.rewards, env.step_dt
    # Same weights/stds as the simulator's reward terms, dt-scaled like mjlab's
    # reward manager, so imagined and real returns share one scale.
    self._w_obj_pos = rewards["object_pos"].weight * dt
    self._std_obj_pos = rewards["object_pos"].params["std"]
    self._w_obj_rot = rewards["object_ori"].weight * dt
    self._std_obj_rot = rewards["object_ori"].params["std"]
    self._w_contact = rewards["contact_match"].weight * dt
    self._obj_pos_threshold = env.cfg.terminations["object_pos"].params["threshold"]
    hand = rewards["hand_joint_pos"]
    self._w_hand, self._std_hand = hand.weight * dt, hand.params["std"]
    robot = env.scene["robot"]
    ids = robot.find_joints(HAND_ACTUATED_JOINTS, preserve_order=True)[0]
    self.hand_joint_ids = torch.tensor(ids, device=cmd.device)
    self.ref_joint_pos = cmd.motion.joint_pos

  def object_reward(
    self,
    next_aux: torch.Tensor,
    next_rep: torch.Tensor,
    contact_logits: torch.Tensor,
    motion_times: torch.Tensor,
  ) -> torch.Tensor:
    """Object tracking + contact match + hand-joint tracking on the predicted
    state; the contact strength is the predicted contact probability."""
    pos_w, rot_w, _, _ = self.object_world(next_aux, next_rep)
    pos_err = (pos_w - self.ref_object_pos[motion_times]).square().sum(-1)
    rot_err = rotation_angle(rot_w, self.ref_object_rot[motion_times]).square()
    label = self.ref_contact_label[motion_times]
    strength = torch.sigmoid(contact_logits)
    active = (label != 0).float()
    score = 1.0 - torch.abs((label > 0).float() - strength)
    count = active.sum(-1)
    contact = torch.where(
      count > 0, (score * active).sum(-1) / count.clamp(min=1), torch.ones_like(count)
    )
    qj = self.joint_state(next_rep)[..., : self.num_joints]
    ids = self.hand_joint_ids
    hand_err = (qj[..., ids] - self.ref_joint_pos[motion_times][..., ids]).square()
    return (
      self._w_obj_pos * torch.exp(-pos_err / self._std_obj_pos**2)
      + self._w_obj_rot * torch.exp(-rot_err / self._std_obj_rot**2)
      + self._w_contact * contact
      + self._w_hand * torch.exp(-hand_err.mean(-1) / self._std_hand**2)
    )

  @property
  def max_object_step_reward(self) -> float:
    return self._w_obj_pos + self._w_obj_rot + self._w_contact + self._w_hand

  def object_term_prob(
    self,
    next_aux: torch.Tensor,
    next_rep: torch.Tensor,
    motion_times: torch.Tensor,
    softness: float,
  ) -> torch.Tensor:
    """Soft version of the env's ``object_pos`` termination."""
    pos_w, _, _, _ = self.object_world(next_aux, next_rep)
    err = (pos_w - self.ref_object_pos[motion_times]).norm(dim=-1)
    return torch.sigmoid((err / self._obj_pos_threshold - 1.0) / softness)


def _create(full_state_dim, num_actions, default_joint_pos, contact_dim):
  del num_actions
  num_joints = infer_num_joints_from_state_dim(full_state_dim - OBJ_FULL_DIM)
  return GenCoordObjectRepresentation(num_joints, default_joint_pos, contact_dim)


register_representation("gencoord_object", _create)
