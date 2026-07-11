"""
metrics.py
PASS knee module - direct angle-signal metrics (the parameter-free reductions).

These are the metrics that fall straight out of the knee-angle signal with no
tuning: range of motion, max flexion, max extension, and angular velocity. They
are unambiguous single definitions (one per function), so every downstream
consumer reads the same number. Detection-based metrics that DO need tuning
(repetition count, rep consistency) live in repetitions.py, deliberately kept
separate because they have real failure modes and their own tests.

All functions take the knee flexion angle in degrees (an array), computed by the
engine on any source's Capture. Sign convention (from joint_angles): positive is
flexion, negative is hyperextension.

NOTE ON ANGULAR VELOCITY: differentiation amplifies noise, so velocity should be
taken on the FILTERED angle (filters.lowpass_offline / StreamingLowpass), not the
raw per-sample signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def range_of_motion(angle: np.ndarray) -> float:
    """Peak-to-peak knee excursion (deg): max flexion - max extension. Independent
    of the calibration zero, since it is a difference."""
    a = np.asarray(angle, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.max(a) - np.min(a))


def max_flexion(angle: np.ndarray) -> float:
    """Greatest flexion angle reached (deg)."""
    a = np.asarray(angle, dtype=float)
    return float(np.max(a)) if a.size else float("nan")


def max_extension(angle: np.ndarray) -> float:
    """Most-extended angle reached (deg) = minimum flexion. Negative means the
    knee went past neutral into hyperextension."""
    a = np.asarray(angle, dtype=float)
    return float(np.min(a)) if a.size else float("nan")


def angular_velocity(angle: np.ndarray, fs_hz: float) -> np.ndarray:
    """Knee angular velocity (deg/s) as the time derivative of the angle, via
    central differences (np.gradient), same length as the input. Best computed on
    the filtered angle."""
    a = np.asarray(angle, dtype=float)
    if a.size < 2:
        return np.zeros_like(a)
    return np.gradient(a, 1.0 / float(fs_hz))


def peak_angular_velocity(angle: np.ndarray, fs_hz: float) -> float:
    """Largest absolute angular velocity in the signal (deg/s)."""
    v = angular_velocity(angle, fs_hz)
    return float(np.max(np.abs(v))) if v.size else float("nan")


@dataclass
class AngleMetrics:
    range_of_motion_deg: float
    max_flexion_deg: float
    max_extension_deg: float
    peak_angular_velocity_dps: float
    n_samples: int


def summarize(angle: np.ndarray, fs_hz: float) -> AngleMetrics:
    """Bundle the direct metrics for a session (composes the definitions above;
    does not redefine any of them)."""
    a = np.asarray(angle, dtype=float)
    return AngleMetrics(
        range_of_motion_deg=range_of_motion(a),
        max_flexion_deg=max_flexion(a),
        max_extension_deg=max_extension(a),
        peak_angular_velocity_dps=peak_angular_velocity(a, fs_hz),
        n_samples=int(a.size),
    )


if __name__ == "__main__":
    t = np.arange(0, 4, 0.01)
    demo = 30 * (1 - np.cos(2 * np.pi * 0.5 * t))       # 0..60 deg, 0.5 Hz
    print("PASS metrics ready -", summarize(demo, fs_hz=100.0))
