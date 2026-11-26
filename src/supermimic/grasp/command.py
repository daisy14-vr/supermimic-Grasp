"""Motion command for grasp clips: robot reference plus object reference.

Extends mjlab's ``MotionCommand`` with the per-frame object pose/velocity and
per-hand-body contact labels stored in the grasp clips (see
``supermimic.grasp.clip_gen``), loaded in the same file order as the robot motion.
Every reset (RSI) also writes the object's reference state and moves the env's
table to the clip's table pose, so robot, object and table start consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  resolve_motion_file,
)
from mjlab.utils.lab_api.math import quat_error_magnitude

from supermimic.grasp import clip_gen, scene

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Squeezed finger leaders (thumb yaw is a placement, not a squeeze) and the
# (follower, leader, multiplier) mimic couplings of the Inspire hand.
_SQUEEZED_LEADERS = (
  "R_thumb_proximal_pitch_joint",
  "R_index_proximal_joint",
  "R_middle_proximal_joint",
  "R_ring_proximal_joint",
  "R_pinky_proximal_joint",
)
_FOLLOWERS = (
  ("R_thumb_intermediate_joint", "R_thumb_proximal_pitch_joint", 1.6),
  ("R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4),
  ("R_index_intermediate_joint", "R_index_proximal_joint", 1.0),
  ("R_middle_intermediate_joint", "R_middle_proximal_joint", 1.0),
  ("R_ring_intermediate_joint", "R_ring_proximal_joint", 1.0),
  ("R_pinky_intermediate_joint", "R_pinky_proximal_joint", 1.0),
)

_OBJECT_FIELDS = (
  "object_pos_w",
  "object_quat_w",
  "object_lin_vel_w",
  "object_ang_vel_w",
  "contact_label",
)


class GraspMotionCommand(MotionCommand):
  cfg: GraspMotionCommandCfg

  def __init__(self, cfg: GraspMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.object_name]
    self.table: Entity = env.scene[cfg.table_name]

    files = resolve_motion_file(cfg.motion_file)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in _OBJECT_FIELDS}
    table_pos: list[np.ndarray] = []
    body_names = None
    for path in files:
      with np.load(path) as clip:
        frames = clip["joint_pos"].shape[0]
        for key in _OBJECT_FIELDS:
          arrays[key].append(clip[key])
        names = clip["contact_body_names"].tolist()
        if body_names is None:
          body_names = names
        elif names != body_names:
          raise ValueError(f"{path}: contact bodies {names} differ from {body_names}")
        center = scene.table_center(
          float(clip["object_x"]), float(clip["object_y"]), float(clip["table_height"])
        )
        table_pos.append(np.tile(center, (frames, 1)))
    assert body_names is not None

    def stack(values: list[np.ndarray], dtype=torch.float32) -> torch.Tensor:
      return torch.tensor(np.concatenate(values), dtype=dtype, device=self.device)

    self.ref_object_pos = stack(arrays["object_pos_w"])
    self.ref_object_quat = stack(arrays["object_quat_w"])
    self.ref_object_lin_vel = stack(arrays["object_lin_vel_w"])
    self.ref_object_ang_vel = stack(arrays["object_ang_vel_w"])
    self.ref_contact_label = stack(arrays["contact_label"])
    self.ref_table_pos = stack(table_pos)
    self.contact_body_names: tuple[str, ...] = tuple(body_names)
    if self.ref_object_pos.shape[0] != self.motion.time_step_total:
      raise ValueError(
        f"object frames {self.ref_object_pos.shape[0]} != motion frames "
        f"{self.motion.time_step_total}"
      )

    # Finger leaders and their mimic followers, for resets at first contact.
    find = self.robot.find_joints
    self._leader_ids = torch.tensor(
      find(_SQUEEZED_LEADERS, preserve_order=True)[0], device=self.device
    )
    self._follower_ids = torch.tensor(
      find(tuple(f for f, _, _ in _FOLLOWERS), preserve_order=True)[0],
      device=self.device,
    )
    leader_col = {n: i for i, n in enumerate(_SQUEEZED_LEADERS)}
    self._follower_leader = torch.tensor(
      [leader_col[lead] for _, lead, _ in _FOLLOWERS], device=self.device
    )
    self._follower_mult = torch.tensor(
      [m for _, _, m in _FOLLOWERS], device=self.device
    )

    self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

  # Reference (current frame, world frame with env origins).

  @property
  def object_pos_w(self) -> torch.Tensor:
    return self.ref_object_pos[self.time_steps] + self._env.scene.env_origins

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.ref_object_quat[self.time_steps]

  @property
  def contact_label(self) -> torch.Tensor:
    return self.ref_contact_label[self.time_steps]

  # Simulated object.

  @property
  def robot_object_pos_w(self) -> torch.Tensor:
    return self.object.data.root_link_pos_w

  @property
  def robot_object_quat_w(self) -> torch.Tensor:
    return self.object.data.root_link_quat_w

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["error_object_pos"] = torch.norm(
      self.object_pos_w - self.robot_object_pos_w, dim=-1
    )
    self.metrics["error_object_rot"] = quat_error_magnitude(
      self.object_quat_w, self.robot_object_quat_w
    )

  def _write_scene_state(self, env_ids: torch.Tensor) -> None:
    frames = self.time_steps[env_ids]
    origins = self._env.scene.env_origins[env_ids]
    object_state = torch.cat(
      [
        self.ref_object_pos[frames] + origins,
        self.ref_object_quat[frames],
        self.ref_object_lin_vel[frames],
        self.ref_object_ang_vel[frames],
      ],
      dim=-1,
    )
    self.object.write_root_state_to_sim(object_state, env_ids=env_ids)
    self.object.reset(env_ids=env_ids)
    identity = torch.zeros(len(env_ids), 4, device=self.device)
    identity[:, 0] = 1.0
    table_pose = torch.cat([self.ref_table_pos[frames] + origins, identity], dim=-1)
    self.table.write_mocap_pose_to_sim(table_pose, env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self._write_scene_state(env_ids)
    self._open_fingers_to_contact(env_ids)

  def _open_fingers_to_contact(self, env_ids: torch.Tensor) -> None:
    """Reset fingers at first contact, not at the squeezed reference.

    The reference closes each finger ``SQUEEZE`` past first contact so PD presses
    the object. Writing that pose at a reset puts the fingers inside the object
    and pushes it out of the hand (61-79% of close/lift-phase resets lost it).
    Opening the leaders by ``SQUEEZE`` (clamped at open) starts them at contact;
    the policy has to produce the squeeze. Mimic followers are kept consistent.
    """
    data = self.robot.data
    joint_pos = data.joint_pos[env_ids].clone()
    joint_vel = data.joint_vel[env_ids].clone()
    leaders = (joint_pos[:, self._leader_ids] - clip_gen.SQUEEZE).clamp(min=0.0)
    joint_pos[:, self._leader_ids] = leaders
    joint_pos[:, self._follower_ids] = (
      leaders[:, self._follower_leader] * self._follower_mult
    )
    finger_ids = torch.cat([self._leader_ids, self._follower_ids])
    joint_vel[:, finger_ids] = 0.0
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    super().reset_to_frame(env_ids, frame)
    self._write_scene_state(env_ids)


@dataclass(kw_only=True)
class GraspMotionCommandCfg(MotionCommandCfg):
  object_name: str = "object"
  table_name: str = "table"

  def build(self, env: ManagerBasedRlEnv) -> GraspMotionCommand:
    return GraspMotionCommand(self, env)
