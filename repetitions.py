"""
repetitions.py
PASS knee module — repetition detection and rep consistency.

Kept separate from metrics.py because, unlike the direct reductions, rep counting
is DETECTION-based: it has tuning parameters and real failure modes (missed reps,
partial reps at the window edges, mid-session pauses). Those deserve their own
focused tests, and a caller deserves confidence indicators, not just a bare count.

ADAPTIVE PROMINENCE (why the threshold is not a fixed number)
-------------------------------------------------------------
A fixed prominence floor (say 15 deg) looks perfect on healthy ~60 deg
sit-to-stand but would start MISSING reps for our actual users — stroke rehab
patients whose early-recovery ROM may be only 20-30 deg — i.e. it fails exactly
the population that most needs accurate counting. So the peak prominence adapts
to each session:

    effective_prominence = max(prominence_floor_deg,
                               prominence_fraction * session_ROM)

The fraction scales the threshold to the patient's own movement; the small floor
still rejects pure noise when the patient is nearly still. The computed value is
reported (effective_prominence_deg) so a clinician can see what was applied.

CONFIDENCE INDICATORS returned alongside the count:
  * peak_prominences_deg  — per-rep prominence (how clearly each rep stands out);
  * partial_at_start / partial_at_end — the window boundary cut through a flexed
    state, so an edge rep may be incomplete;
  * amplitude_cv, period_cv — rep consistency in amplitude and in timing (a
    mid-session pause shows up as an elevated period_cv, not a wrong count).

Operates on the (filtered) knee flexion angle in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from metrics import range_of_motion


@dataclass
class RepResult:
    count: int
    peak_indices: np.ndarray
    peak_times_s: np.ndarray
    peak_values_deg: np.ndarray
    peak_prominences_deg: np.ndarray
    effective_prominence_deg: float      # the adaptive threshold actually used
    partial_at_start: bool               # window began mid-flexion (edge rep may be cut)
    partial_at_end: bool                 # window ended mid-flexion
    amplitude_cv: float                  # consistency of rep amplitude (std/mean of prominences)
    period_cv: float                     # consistency of rep timing (std/mean of intervals)


def _cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    m = np.mean(x)
    return float(np.std(x) / m) if m != 0 else float("nan")


def detect_reps(angle: np.ndarray, fs_hz: float, *,
                prominence_floor_deg: float = 6.0,
                prominence_fraction: float = 0.3,
                min_separation_s: float = 0.8,
                edge_fraction: float = 0.25) -> RepResult:
    """
    Count knee-flexion repetitions in a session and report confidence indicators.

    prominence_floor_deg : absolute noise floor for a peak's prominence.
    prominence_fraction  : fraction of the session ROM used for the adaptive
                           prominence; effective = max(floor, fraction * ROM).
    min_separation_s     : refractory distance between rep peaks (anti double-count).
    edge_fraction        : a window edge above baseline + edge_fraction*ROM is
                           flagged as a possibly-cut (partial) rep.
    """
    a = np.asarray(angle, dtype=float)
    rom = range_of_motion(a) if a.size else 0.0
    effective = max(float(prominence_floor_deg), float(prominence_fraction) * (rom or 0.0))

    empty = np.array([], dtype=float)
    if a.size < 3:
        return RepResult(0, np.array([], int), empty, empty, empty,
                         effective, False, False, float("nan"), float("nan"))

    distance = max(1, int(round(min_separation_s * fs_hz)))
    peaks, props = find_peaks(a, prominence=effective, distance=distance)
    prominences = props["prominences"]

    baseline = float(np.min(a))
    partial_start = bool(a[0] - baseline > edge_fraction * rom)
    partial_end = bool(a[-1] - baseline > edge_fraction * rom)

    intervals = np.diff(peaks) / fs_hz
    return RepResult(
        count=int(peaks.size),
        peak_indices=peaks,
        peak_times_s=peaks / fs_hz,
        peak_values_deg=a[peaks],
        peak_prominences_deg=prominences,
        effective_prominence_deg=effective,
        partial_at_start=partial_start,
        partial_at_end=partial_end,
        amplitude_cv=_cv(prominences),
        period_cv=_cv(intervals),
    )


if __name__ == "__main__":
    fs = 100.0
    t = np.arange(0, 8, 1 / fs)
    demo = 30 * (1 - np.cos(2 * np.pi * t / 2.0))       # 4 reps, 0..60 deg
    r = detect_reps(demo, fs)
    print(f"PASS repetitions ready - count={r.count} "
          f"prominence={r.effective_prominence_deg:.1f} deg amp_cv={r.amplitude_cv:.3f}")
