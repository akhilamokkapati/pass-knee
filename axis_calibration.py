"""
axis_calibration.py
PASS knee module — measure the live knee flexion AXIS on the mounted hardware.

WHY THIS EXISTS (the one thing gated by sensor day)
---------------------------------------------------
The knee flexion axis is not (1,0,0) in general — it depends on how the two
BNO085s physically sit on the limb. On the HuGaDB dataset it turned out to be
~ -Y; on our own hardware it is unknown until the sensors are strapped on. Rather
than GUESS the axis, we MEASURE it: capture a straight-leg pose and a clearly
bent pose, and the axis of the rotation between them IS the flexion axis. The
result is a defensible, measured number and it plugs straight into the engine as
the `axis=` argument to joint_angles.knee_flexion_angle.

Everything here reuses validated code:
  * calibrate.calibrate_from_quaternions  — Markley average of each pose (and a
    "held still" residual for free);
  * relative_orientation.remove_offset     — the neutral->bent relative rotation;
  * (the only new bit) the standard axis-angle read of a quaternion.

USAGE ON HARDWARE (the axis calibration step)
---------------------------------------------
    from sources.serial_source import SerialSource
    from axis_calibration import calibrate_flexion_axis

    src = SerialSource(port="COM5")                 # firmware streaming
    input("Hold the leg STRAIGHT and still, then press Enter...")
    n = src.get_data(2.0)                           # ~2 s straight-leg window
    input("Hold the knee CLEARLY BENT (~60 deg) and still, then press Enter...")
    b = src.get_data(2.0)                           # ~2 s bent window

    cal = calibrate_flexion_axis(n.quat_thigh, n.quat_shank,
                                 b.quat_thigh, b.quat_shank)
    print(cal)                                      # axis, bend angle, confidence
    if not cal.reliable:
        # redo: bend further, hold stiller, keep the motion about one axis
        ...
    FLEXION_AXIS = cal.flexion_axis                 # feed to knee_flexion_angle(..., axis=FLEXION_AXIS)

INTERPRETING THE RESULT
-----------------------
  flexion_axis        unit vector; use it as knee_flexion_angle's axis (bending
                      reads POSITIVE with it).
  bend_angle_deg      how far the calibration bend was; small bends make the axis
                      noisy, so aim for >= ~30 deg.
  axis_confidence     0..1, how cleanly the bend was about a SINGLE axis (1.0 =
                      pure single-axis rotation). Low means wobble / mixed motion.
  neutral/bent_residual_deg  how still each pose was held (smaller is better).
  reliable            convenience gate on all of the above; if False, recapture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biomechanics.quaternion_math import normalize
from biomechanics.relative_orientation import knee_relative, remove_offset, canonicalize
from calibrate import calibrate_from_quaternions


@dataclass
class AxisCalibration:
    flexion_axis: np.ndarray        # (3,) unit vector, usable as knee_flexion_angle axis
    bend_angle_deg: float           # magnitude of the neutral->bent rotation
    axis_confidence: float          # 0..1, single-axis cleanliness of the bend
    neutral_residual_deg: float     # how still the straight-leg pose was held
    bent_residual_deg: float        # how still the bent pose was held
    n_neutral: int
    n_bent: int
    reliable: bool


def _axis_angle(q: np.ndarray):
    """Standard axis-angle read of a unit quaternion (w,x,y,z), canonicalized so
    the angle is non-negative. Returns (axis (3,), angle_deg)."""
    q = canonicalize(normalize(np.asarray(q, dtype=float)))
    w = float(np.clip(q[0], -1.0, 1.0))
    vec = q[1:]
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0]), 0.0
    return vec / n, float(np.degrees(2.0 * np.arctan2(n, w)))


def calibrate_flexion_axis(neutral_thigh: np.ndarray, neutral_shank: np.ndarray,
                           bent_thigh: np.ndarray, bent_shank: np.ndarray, *,
                           min_bend_deg: float = 20.0,
                           min_confidence: float = 0.9,
                           max_residual_deg: float = 5.0) -> AxisCalibration:
    """
    Measure the knee flexion axis from a straight-leg window and a bent window.

    Each window is a batch of thigh/shank quaternions (as from source.get_data).
    The straight-leg pose is averaged into a neutral (Markley), likewise the bent
    pose; the rotation from neutral to bent has the flexion axis as its axis.
    Confidence is how tightly every bent sample's rotation axis agrees with that
    consensus, weighted toward larger rotations (whose axis is better determined).
    """
    neu = calibrate_from_quaternions(neutral_thigh, neutral_shank)
    bnt = calibrate_from_quaternions(bent_thigh, bent_shank)

    # neutral -> bent relative rotation; its axis is the flexion axis
    flex_rot = remove_offset(bnt.q_neutral, neu.q_neutral)
    flexion_axis, bend_angle = _axis_angle(flex_rot)

    # per-sample bent rotations (relative to the same neutral) -> axis agreement
    rels = knee_relative(np.atleast_2d(bent_thigh), np.atleast_2d(bent_shank))
    flex_samples = canonicalize(remove_offset(rels, neu.q_neutral))
    vecs = flex_samples[:, 1:]
    norms = np.linalg.norm(vecs, axis=1)
    good = norms > 1e-6
    axes = np.zeros_like(vecs)
    axes[good] = vecs[good] / norms[good, None]
    collinearity = np.abs(axes @ flexion_axis)          # 1 = parallel to consensus axis
    weight = norms                                       # larger rotations weigh more
    confidence = float(np.clip(np.sum(weight * collinearity) / np.sum(weight), 0.0, 1.0)) \
        if np.sum(weight) > 0 else 0.0

    reliable = bool(
        bend_angle >= min_bend_deg
        and confidence >= min_confidence
        and neu.residual_rms_deg <= max_residual_deg
        and bnt.residual_rms_deg <= max_residual_deg
    )

    return AxisCalibration(
        flexion_axis=flexion_axis,
        bend_angle_deg=bend_angle,
        axis_confidence=confidence,
        neutral_residual_deg=neu.residual_rms_deg,
        bent_residual_deg=bnt.residual_rms_deg,
        n_neutral=neu.n_samples,
        n_bent=bnt.n_samples,
        reliable=reliable,
    )


if __name__ == "__main__":
    # No hardware yet: demonstrate on HuGaDB, using standing as the straight-leg
    # neutral and sitting as the bent pose. The helper should rediscover the
    # dataset's ~ -Y flexion axis that we found by hand.
    from pathlib import Path
    from sources.hugadb import HuGaDBSource

    f = Path(__file__).parent / "hugadb" / "HuGaDB_v2_various_01_00.csv"
    src = HuGaDBSource(f)
    cap = src.get_data()
    stand = cap.activity == "standing"
    sit = cap.activity == "sitting"
    cal = calibrate_flexion_axis(cap.quat_thigh[stand], cap.quat_shank[stand],
                                 cap.quat_thigh[sit], cap.quat_shank[sit])
    print("PASS axis_calibration (HuGaDB standing->sitting):")
    print(f"  flexion_axis = {np.round(cal.flexion_axis, 3)}  (expect ~ [0,-1,0])")
    print(f"  bend_angle   = {cal.bend_angle_deg:.1f} deg   confidence = {cal.axis_confidence:.3f}")
    print(f"  reliable     = {cal.reliable}")
