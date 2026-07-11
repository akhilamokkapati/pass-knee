"""
calibrate.py
PASS knee module — straight-leg (quiet-standing) calibration.

Calibration = straight-leg zero: capture a quiet-standing window and treat the
average relative thigh->shank orientation as 0 deg. That average becomes the
q_neutral that relative_orientation.remove_offset already consumes, so a correct
calibration makes the straight leg read 0 deg and removes the sensor mounting
offset properly.

WHY MARKLEY (the quaternion-averaging choice)
---------------------------------------------
Unit quaternions do not average by a simple component mean:
  * q and -q are the SAME rotation, so a naive mean can cancel them toward zero;
  * a naive mean's error grows with the spread of the cluster.
Markley et al. (2007) give the correct average: form M = sum_i q_i q_i^T (4x4)
and take the eigenvector of its largest eigenvalue. This minimizes the L2
chordal distance on the rotation manifold and is INHERENTLY sign-invariant
(q q^T == (-q)(-q)^T), so a BNO085 sign flip mid-window cannot corrupt it. For a
tight quiet-standing cluster it agrees with a sign-aligned normalized mean to
well within our tolerance, but it needs no ad-hoc hemisphere-flip heuristic and
is trivial to defend at review. The 4x4 eigendecomposition is free.

Source-agnostic: calibrate() pulls the window through the same get_data()
interface every PASS source exposes, so it works on live sensor data unchanged.
Library functions here do not print; callers inspect the returned result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biomechanics.quaternion_math import normalize
from biomechanics.relative_orientation import knee_relative, canonicalize


@dataclass
class CalibrationResult:
    """Outcome of a straight-leg calibration.

    q_neutral        : averaged relative orientation to feed remove_offset.
    n_samples        : number of samples averaged.
    residual_rms_deg : RMS angular spread of the window about q_neutral.
    residual_max_deg : worst-case angular deviation in the window.

    The residuals are a quality gate: a large spread means the patient was not
    actually still, so the calibration should be repeated.
    """
    q_neutral: np.ndarray
    n_samples: int
    residual_rms_deg: float
    residual_max_deg: float


def average_quaternion(quats: np.ndarray) -> np.ndarray:
    """
    Markley eigenvector average of unit quaternions. (N,4) -> (4,), canonical.

    M = sum_i q_i q_i^T; the average attitude is the eigenvector of M's largest
    eigenvalue. Sign-invariant by construction, so mixed q/-q samples average
    correctly.
    """
    q = normalize(np.atleast_2d(np.asarray(quats, dtype=float)))
    m = q.T @ q                       # (4,4), symmetric PSD
    _, vecs = np.linalg.eigh(m)       # ascending eigenvalues
    return canonicalize(vecs[:, -1])  # largest-eigenvalue eigenvector, w >= 0


def _angular_deviation_deg(quats: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Per-sample rotation angle (deg) between each quat and q_ref, sign-free."""
    q = normalize(np.atleast_2d(np.asarray(quats, dtype=float)))
    q_ref = normalize(np.asarray(q_ref, dtype=float))
    dots = np.clip(np.abs(q @ q_ref), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def calibrate_from_quaternions(q_thigh: np.ndarray,
                               q_shank: np.ndarray) -> CalibrationResult:
    """
    Compute the straight-leg neutral from a window of thigh/shank quaternions.

    We average the RELATIVE orientation (knee_relative per sample) — that is
    exactly "treat this relative orientation as 0 deg". Returns the neutral plus
    the window's angular spread as a quality metric.
    """
    rel = np.atleast_2d(knee_relative(q_thigh, q_shank))
    q_neutral = average_quaternion(rel)
    dev = _angular_deviation_deg(rel, q_neutral)
    return CalibrationResult(
        q_neutral=q_neutral,
        n_samples=int(rel.shape[0]),
        residual_rms_deg=float(np.sqrt(np.mean(dev ** 2))),
        residual_max_deg=float(np.max(dev)),
    )


def calibrate(source, duration_s: float = 2.0) -> CalibrationResult:
    """
    Straight-leg calibration from any PASS source. Ask the patient to stand
    still, capture `duration_s` of data via source.get_data, average it.

    Source-agnostic: swap SyntheticSource for a live SerialSource and this is
    unchanged.
    """
    cap = source.get_data(duration_s)
    return calibrate_from_quaternions(cap.quat_thigh, cap.quat_shank)


if __name__ == "__main__":
    from sources.synthetic import SyntheticSource
    res = calibrate(SyntheticSource(min_angle_deg=0.0, max_angle_deg=0.0), 1.0)
    print(f"PASS calibrate ready - neutral={np.round(res.q_neutral, 4)} "
          f"n={res.n_samples} residual_max={res.residual_max_deg:.3g} deg")
