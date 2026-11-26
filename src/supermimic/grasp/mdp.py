"""Observation, reward and termination terms for the grasp task.

Object quantities are tracked in the world frame (the object does not move with
the robot), following WEAVE; robot-relative observations use the robot anchor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
)

from supermimic.assets.g1_inspire import HAND_ACTUATED_JOINTS
from supermimic.representations import goal_window_indices
from supermimic.utils.helpers import quat_to_rot6d

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

  from supermimic.grasp.command import GraspMotionCommand


def _command(env: ManagerBasedRlEnv, name: str) -> GraspMotionCommand:
  return cast("GraspMotionCommand", env.command_manager.get_term(name))


def _contact_forces(
  env: ManagerBasedRlEnv, sensor_name: str, command: GraspMotionCommand
) -> torch.Tensor:
  """Net contact force magnitude per hand body, ``(E, B)``, in the command's
  ``contact_body_names`` order."""
  sensor = cast("ContactSensor", env.scene.sensors[sensor_name])
  order = getattr(sensor, "_grasp_order", None)
  if order is None:
    short = [n.split("/")[-1] for n in sensor.primary_names]
    order = torch.tensor(
      [short.index(n) for n in command.contact_body_names], device=env.device
    )
    sensor._grasp_order = order  # type: ignore[attr-defined]
  force = sensor.data.force
  assert force is not None
  return torch.norm(force[:, order], dim=-1)


# Observations.


def object_pos_w(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Simulated object position, env origins removed (like the robot root)."""
  cmd = _command(env, command_name)
  return cmd.robot_object_pos_w - env.scene.env_origins


def object_quat_w(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  return _command(env, command_name).robot_object_quat_w


def object_lin_vel_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  return _command(env, command_name).object.data.root_link_lin_vel_w


def object_ang_vel_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  return _command(env, command_name).object.data.root_link_ang_vel_w


def object_pos_b(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Simulated object position in the robot anchor frame."""
  cmd = _command(env, command_name)
  return quat_apply_inverse(
    cmd.robot_anchor_quat_w, cmd.robot_object_pos_w - cmd.robot_anchor_pos_w
  )


def object_ori_b(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Simulated object orientation (rot6d) in the robot anchor frame."""
  cmd = _command(env, command_name)
  return quat_to_rot6d(
    quat_mul(quat_inv(cmd.robot_anchor_quat_w), cmd.robot_object_quat_w)
  )


def object_lin_vel_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  cmd = _command(env, command_name)
  return quat_apply_inverse(
    cmd.robot_anchor_quat_w, cmd.object.data.root_link_lin_vel_w
  )


def object_ref_window_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Reference object pose over the goal window, relative to the robot anchor:
  per frame ``[pos(3), rot6d(6)]``."""
  cmd = _command(env, command_name)
  idx = goal_window_indices(cmd.motion, cmd.time_steps)  # (E, N)
  num = idx.shape[1]
  pos = cmd.ref_object_pos[idx] + env.scene.env_origins[:, None]
  quat = cmd.ref_object_quat[idx]
  anchor_pos = cmd.robot_anchor_pos_w[:, None].expand(-1, num, -1)
  anchor_quat = cmd.robot_anchor_quat_w[:, None].expand(-1, num, -1)
  rel_pos = quat_apply_inverse(anchor_quat, pos - anchor_pos)
  rel_ori = quat_to_rot6d(quat_mul(quat_inv(anchor_quat), quat))
  return torch.cat([rel_pos, rel_ori], dim=-1).reshape(env.num_envs, -1)


def contact_label_window(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Reference contact labels (-1/0/+1 per hand body) over the goal window."""
  cmd = _command(env, command_name)
  idx = goal_window_indices(cmd.motion, cmd.time_steps)
  return cmd.ref_contact_label[idx].reshape(env.num_envs, -1)


def hand_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_object",
  command_name: str = "motion",
  threshold: float = 1.0,
) -> torch.Tensor:
  """Binary contact per hand body with the object (privileged)."""
  cmd = _command(env, command_name)
  return (_contact_forces(env, sensor_name, cmd) > threshold).float()


# Rewards.


def object_pos_error_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str = "motion"
) -> torch.Tensor:
  cmd = _command(env, command_name)
  err = torch.sum(torch.square(cmd.object_pos_w - cmd.robot_object_pos_w), dim=-1)
  return torch.exp(-err / std**2)


def object_ori_error_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str = "motion"
) -> torch.Tensor:
  cmd = _command(env, command_name)
  err = quat_error_magnitude(cmd.object_quat_w, cmd.robot_object_quat_w) ** 2
  return torch.exp(-err / std**2)


def contact_match(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_object",
  command_name: str = "motion",
  force_scale: float = 5.0,
) -> torch.Tensor:
  """WEAVE's tri-state contact reward: over hand bodies whose reference label is
  non-zero, mean of ``1 - |1[label > 0] - min(|F| / force_scale, 1)|``. Rewards
  required contacts and penalizes forbidden ones; 1 when no body is labeled."""
  cmd = _command(env, command_name)
  label = cmd.contact_label
  strength = torch.clamp(_contact_forces(env, sensor_name, cmd) / force_scale, max=1.0)
  active = (label != 0).float()
  score = 1.0 - torch.abs((label > 0).float() - strength)
  count = active.sum(dim=-1)
  return torch.where(
    count > 0, (score * active).sum(dim=-1) / count.clamp(min=1), torch.ones_like(count)
  )


def hand_joint_pos_error_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str = "motion"
) -> torch.Tensor:
  """Dense hand-shape tracking on the 6 actuated hand joints:
  ``exp(-mean_j (q_j - q_ref_j)^2 / std^2)``. Body-pose tracking alone barely
  shapes the hand (5 of 20 bodies), so the policy half-closed it (run 4)."""
  cmd = _command(env, command_name)
  ids = _hand_joint_ids(env, cmd)
  err = (cmd.robot_joint_pos[:, ids] - cmd.joint_pos[:, ids]).square().mean(-1)
  return torch.exp(-err / std**2)


def _hand_joint_ids(env: ManagerBasedRlEnv, cmd: GraspMotionCommand) -> torch.Tensor:
  ids = getattr(cmd, "_hand_joint_ids", None)
  if ids is None:
    found = cmd.robot.find_joints(HAND_ACTUATED_JOINTS, preserve_order=True)[0]
    ids = torch.tensor(found, device=env.device)
    cmd._hand_joint_ids = ids  # type: ignore[attr-defined]
  return ids


# Terminations.


def bad_object_pos(
  env: ManagerBasedRlEnv, threshold: float, command_name: str = "motion"
) -> torch.Tensor:
  cmd = _command(env, command_name)
  return torch.norm(cmd.object_pos_w - cmd.robot_object_pos_w, dim=-1) > threshold
