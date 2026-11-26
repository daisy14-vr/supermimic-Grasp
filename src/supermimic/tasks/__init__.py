"""Register SuperMimic tasks."""

from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

import supermimic.grasp.representation  # noqa: F401  (registers "gencoord_object")
from supermimic.grasp.env_cfg import grasp_env_cfg

from .env_cfgs import gencoord_env_cfg, maxcoord_env_cfg

# We reuse the RL config from the base tracking task
BASE_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"

try:
  register_mjlab_task(
    task_id="SuperMimic-GenCoord-G1",
    env_cfg=gencoord_env_cfg(),
    play_env_cfg=gencoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass

try:
  register_mjlab_task(
    task_id="SuperMimic-MaxCoord-G1",
    env_cfg=maxcoord_env_cfg(),
    play_env_cfg=maxcoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass

try:
  # v1 grasp-and-lift (G1 + right Inspire hand); clips from gen_grasp_clips.py.
  register_mjlab_task(
    task_id="SuperMimic-Grasp-G1Inspire",
    env_cfg=grasp_env_cfg(),
    play_env_cfg=grasp_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass
