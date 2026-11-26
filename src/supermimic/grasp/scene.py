"""Grasp scene: the table and the object, shared by clip generation, the hand
physics check and the training env so all three place them identically."""

from __future__ import annotations

import mujoco
import numpy as np

# Table: a box of fixed size; its height per clip comes from moving it vertically
# (the part below the floor is hidden and, being static, collides with nothing).
TABLE_HALF_SIZE = (0.25, 0.4, 0.45)
TABLE_EDGE_TO_OBJECT = 0.05  # table near edge sits this far behind the object axis
TABLE_FRICTION = 0.6

OBJECT_FRICTION = 1.0


def table_center(object_x, object_y, table_height):
  """Table box center for an object at (x, y) on a table top at ``table_height``.
  Works on floats and on numpy/torch arrays."""
  x = object_x - TABLE_EDGE_TO_OBJECT + TABLE_HALF_SIZE[0]
  z = table_height - TABLE_HALF_SIZE[2]
  return x, object_y, z


def add_table(parent: mujoco.MjsBody, name: str = "table") -> mujoco.MjsGeom:
  return parent.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(TABLE_HALF_SIZE),
    condim=3,
    friction=[TABLE_FRICTION, 0.005, 0.0001],
    rgba=[0.55, 0.4, 0.25, 1.0],
  )


def add_cylinder(
  parent: mujoco.MjsBody, radius: float, height: float, mass: float, name="object"
) -> mujoco.MjsGeom:
  return parent.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[radius, height / 2, 0.0],
    mass=mass,
    condim=3,
    friction=[OBJECT_FRICTION, 0.005, 0.0001],
    rgba=[0.85, 0.2, 0.2, 1.0],
  )


def table_spec() -> mujoco.MjSpec:
  """Fixed-base table entity (mjlab wraps it in a mocap body for per-env poses)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  add_table(body, name="table_collision")
  return spec


def cylinder_spec(radius: float, height: float, mass: float) -> mujoco.MjSpec:
  """Free-floating cylinder entity."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_joint")
  add_cylinder(body, radius, height, mass, name="object_collision")
  return spec


def object_rest_pos(object_x, object_y, table_height, object_height) -> np.ndarray:
  return np.array([object_x, object_y, table_height + 0.5 * object_height])
