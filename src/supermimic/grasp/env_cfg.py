"""Env config for v1 grasp-and-lift: G1 + right Inspire hand, one cylinder.

Built on SuperMimic's generalized-coordinate tracking cfg (same ``actor``,
``full_state`` and ``goal`` groups the runner expects), with:

- the G1 + Inspire robot, a free cylinder and a per-env table;
- ``GraspMotionCommand`` reading robot, object and contact references from the
  procedural clips (``scripts/gen_grasp_clips.py``) and resetting all three;
- hand bodies added to the tracked bodies so finger shape is rewarded;
- object tracking and WEAVE's contact-match rewards, an object-error
  termination, and an ``object`` observation group (object state, its reference
  window and contact labels) for the world model and critic.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from supermimic.assets import g1_inspire as gi
from supermimic.grasp import clip_gen, mdp, scene
from supermimic.grasp.command import GraspMotionCommandCfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg

DEFAULT_CLIPS = str(
  Path(__file__).resolve().parents[3] / "data" / "grasp_clips" / "clip_*.npz"
)

HAND_TRACKED_BODIES = (
  "R_hand_base_link",
  "R_thumb_distal",
  "R_index_intermediate",
  "R_middle_intermediate",
  "R_ring_intermediate",
  "R_pinky_intermediate",
)
# All hand bodies with collision geometry; must match the clips' contact bodies.
HAND_CONTACT_BODIES = (
  "R_hand_base_link",
  "R_thumb_proximal_base",
  "R_thumb_proximal",
  "R_thumb_intermediate",
  "R_thumb_distal",
  "R_index_proximal",
  "R_index_intermediate",
  "R_middle_proximal",
  "R_middle_intermediate",
  "R_ring_proximal",
  "R_ring_intermediate",
  "R_pinky_proximal",
  "R_pinky_intermediate",
)


def grasp_env_cfg(play: bool = False, motion_file: str = DEFAULT_CLIPS):
  cfg: ManagerBasedRlEnvCfg = gencoord_env_cfg(play=play)

  obj = clip_gen.V1_OBJECT
  cfg.scene.entities = {
    "robot": gi.get_g1_inspire_robot_cfg(),
    "object": EntityCfg(
      spec_fn=lambda: scene.cylinder_spec(
        obj["object_radius"], obj["object_height"], obj["object_mass"]
      )
    ),
    "table": EntityCfg(spec_fn=scene.table_spec),
  }
  cfg.actions["joint_pos"].scale = gi.G1_INSPIRE_ACTION_SCALE  # type: ignore[attr-defined]

  hand_object = ContactSensorCfg(
    name="hand_object",
    primary=ContactMatch(mode="body", pattern=HAND_CONTACT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("found", "force"),
    reduce="netforce",
  )
  cfg.scene.sensors = (*cfg.scene.sensors, hand_object)

  # Motion command -> grasp command, keeping the tracking cfg's settings.
  base = cfg.commands["motion"]
  assert isinstance(base, MotionCommandCfg)
  kwargs = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
  kwargs.update(
    motion_file=motion_file,
    body_names=(*base.body_names, *HAND_TRACKED_BODIES),
    # The object is world-fixed: keep the robot's reset pose close to the clip.
    pose_range={
      "x": (-0.02, 0.02),
      "y": (-0.02, 0.02),
      "z": (-0.01, 0.01),
      "roll": (-0.05, 0.05),
      "pitch": (-0.05, 0.05),
      "yaw": (-0.05, 0.05),
    },
    joint_position_range=(-0.05, 0.05),
  )
  cfg.commands["motion"] = GraspMotionCommandCfg(**kwargs)

  # World-model state: robot GenCoord state + the object block
  # (see supermimic.grasp.representation for the layout).
  full_state = cfg.observations["full_state"]
  full_state.terms.update(
    {
      "object_pos": ObservationTermCfg(func=mdp.object_pos_w),
      "object_quat": ObservationTermCfg(func=mdp.object_quat_w),
      "object_lin_vel": ObservationTermCfg(func=mdp.object_lin_vel_w),
      "object_ang_vel": ObservationTermCfg(func=mdp.object_ang_vel_w),
    }
  )
  # Contact targets for the world model's contact head (privileged).
  cfg.observations["contact"] = ObservationGroupCfg(
    terms={"hand_contact": ObservationTermCfg(func=mdp.hand_contact)},
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["object"] = ObservationGroupCfg(
    terms={
      "object_pos_b": ObservationTermCfg(func=mdp.object_pos_b),
      "object_ori_b": ObservationTermCfg(func=mdp.object_ori_b),
      "object_lin_vel_b": ObservationTermCfg(func=mdp.object_lin_vel_b),
      "object_ref": ObservationTermCfg(func=mdp.object_ref_window_b),
      "contact_label": ObservationTermCfg(func=mdp.contact_label_window),
      "hand_contact": ObservationTermCfg(func=mdp.hand_contact),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Narrow enough that an object left on the table (~12 cm below a lifted
  # reference) earns little: exp(-(0.12/0.06)^2) ~ 0.02.
  cfg.rewards["object_pos"] = RewardTermCfg(
    func=mdp.object_pos_error_exp, weight=2.0, params={"std": 0.06}
  )
  cfg.rewards["object_ori"] = RewardTermCfg(
    func=mdp.object_ori_error_exp, weight=1.0, params={"std": 0.4}
  )
  cfg.rewards["contact_match"] = RewardTermCfg(func=mdp.contact_match, weight=2.0)
  # Dense hand-shape signal; the WM mirrors it (GenCoordObjectRepresentation).
  cfg.rewards["hand_joint_pos"] = RewardTermCfg(
    func=mdp.hand_joint_pos_error_exp, weight=1.0, params={"std": 0.3}
  )
  # Contact budget sized from measurements (random actions, 512 worlds): about 22
  # contacts per world on average and at most 210 constraint rows in any world.
  # nconmax is a per-world average of a shared pool; njmax is a hard per-world
  # cap. 120/800 ran out of GPU memory at 4096 envs. The solver needs more
  # than the tracking cfg's 10 iterations for the grasp contacts.
  cfg.sim.nconmax = 48
  cfg.sim.njmax = 350
  cfg.sim.mujoco.iterations = 30
  cfg.sim.mujoco.ls_iterations = 50  # 30 hit its limit occasionally once fingers grasp

  # The world model re-evaluates terminations and rewards from the post-step
  # observation; without this, mjlab scores them on kinematics one physics
  # substep stale and the WM terminates one control step early (99.8% of its
  # false positives in a cross-check).
  cfg.forward_before_termination = True

  # Below the ~12 cm lift: not lifting the object, or dropping it, ends the
  # episode (0.25 let a policy that never grasps survive to the clip end).
  cfg.terminations["object_pos"] = TerminationTermCfg(
    func=mdp.bad_object_pos, params={"threshold": 0.10}
  )
  return cfg
