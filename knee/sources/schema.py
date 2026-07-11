"""
schema.py
PASS data-source schema - the shared packet/capture contract.

Every PASS source (synthetic, HuGaDB offline, live serial) emits these SAME
types, so the biomechanics engine and the runners never depend on which source
produced the data. Kept in its own module so both sources import the ONE
definition from here rather than from each other.

  Packet   one sample, matching the firmware packet:
           seq, t_ms, knee_angle_deg, quat_thigh[4] (w,x,y,z), quat_shank[4].
  Capture  a finite span of samples as arrays (the get_data return contract).

knee_angle_deg is the source's REFERENCE angle when it has one (synthetic: the
independent forward-model ground truth; firmware: on-device value). A source
with no reference (HuGaDB, which has no labelled knee angle) sets it to NaN - the
engine still computes the angle from the quaternions downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Packet:
    """One sample, matching the firmware packet schema."""
    seq: int
    t_ms: int
    knee_angle_deg: float          # source reference angle, or NaN if none
    quat_thigh: np.ndarray         # (4,) w,x,y,z
    quat_shank: np.ndarray         # (4,) w,x,y,z


@dataclass
class Capture:
    """A finite span of samples as arrays (the get_data return contract).

    activity is an OPTIONAL, source-neutral per-sample label column. It stays
    None for sources that have no labels (synthetic); HuGaDB fills it with its
    activity annotations. It is deliberately optional so one source's extra
    metadata does not bend the shared schema.
    """
    seq: np.ndarray                # (N,)
    t_ms: np.ndarray               # (N,)
    knee_angle_deg: np.ndarray     # (N,) reference angle, or NaN
    quat_thigh: np.ndarray         # (N,4)
    quat_shank: np.ndarray         # (N,4)
    activity: np.ndarray | None = None   # (N,) labels, or None
