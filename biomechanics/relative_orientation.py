"""
relative_orientation.py
PASS Biomechanics — the knee joint relative-orientation layer.

quaternion_math.py is pure algebra: it does not know what a thigh, a shank, or
a knee is. This module is the first place that assigns anatomical meaning. It
turns the two IMU quaternions into ONE quaternion that represents the knee
joint rotation itself, cleaned up so the downstream angle extractor gets a
well-behaved signal. It deliberately stops there: the flexion AXIS and the
scalar flexion ANGLE live in joint_angles.py, not here.

Three responsibilities, each a small pure function:

  knee_relative  - fix the convention "thigh is the reference frame, shank is
                   the moving frame" so the sign of flexion is decided in
                   exactly one place.
  remove_offset  - express a reading relative to a captured straight-leg
                   neutral (calibration). Identity neutral -> passthrough, so
                   the engine is correct even before Step 4 wires calibration.
  canonicalize   - q and -q are the same rotation, but the BNO085 may hand us
                   either; force w >= 0 so a raw sign flip cannot make the
                   downstream angle jump.

Convention (inherited from quaternion_math): (w, x, y, z), unit norm, Hamilton
product, right-handed. All functions accept (4,) or (N,4).
"""

from __future__ import annotations

import numpy as np

from .quaternion_math import normalize, conjugate, multiply, relative


def knee_relative(q_thigh: np.ndarray, q_shank: np.ndarray) -> np.ndarray:
    """
    Knee joint rotation: the rotation that carries the thigh frame onto the
    shank frame, i.e. q_thigh^-1 (x) q_shank.

    Thigh is the reference, shank is the moving segment. This ordering fixes
    the sign convention for flexion once and for all; every other module reads
    the sign from here rather than re-deciding it. The result is independent of
    how the whole leg is oriented in the world (a common world rotation applied
    to both segments cancels exactly).
    """
    return relative(q_thigh, q_shank)


def remove_offset(q_rel: np.ndarray, q_neutral: np.ndarray) -> np.ndarray:
    """
    Express q_rel relative to a captured straight-leg neutral: q_neutral^-1 (x)
    q_rel. This is the calibration application (the capture of q_neutral by
    averaging a quiet-standing window is Step 4).

    When the leg returns to the neutral pose, q_rel == q_neutral and the result
    is the identity quaternion -> 0 deg. When q_neutral is the identity
    quaternion (no calibration yet), identity^-1 (x) q_rel == q_rel, so the
    reading passes straight through unchanged. No special-casing needed.

    q_neutral may be a single (4,) applied across an (N,4) batch of readings.
    """
    return multiply(conjugate(normalize(q_neutral)), normalize(q_rel))


def canonicalize(q: np.ndarray) -> np.ndarray:
    """
    Return the same rotation in a canonical form with w >= 0.

    A unit quaternion and its negation encode the identical rotation (the
    "double cover"). The sensor can legitimately report either, and an
    unflagged flip would make a continuous angle signal jump. Forcing w >= 0
    gives one deterministic representative per rotation.
    """
    q = normalize(np.asarray(q, dtype=float))
    sign = np.where(q[..., 0] < 0.0, -1.0, 1.0)
    return q * sign[..., None]


if __name__ == "__main__":
    print("PASS relative_orientation ready")
