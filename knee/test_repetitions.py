"""
test_repetitions.py
Known-answer tests for repetition detection, with focused cases for the real
failure modes: partial reps at edges, small-wiggle rejection, mid-session pause,
and - the clinically important one - SHALLOW low-ROM rehab reps that a fixed
prominence floor would miss.

Run:  python -m pytest test_repetitions.py -v
"""

import numpy as np
from scipy.signal import find_peaks

from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle
from calibrate import calibrate_from_quaternions
from sources.hugadb import HUGADB_FLEXION_AXIS
from repetitions import detect_reps

FS = 100.0


def rep_signal(n_reps, amp, period_s=2.0, fs=FS, baseline=0.0):
    """n_reps clean flexion cycles, each 0->amp->0 (baseline offset optional)."""
    t = np.arange(0, n_reps * period_s, 1 / fs)
    return baseline + amp * (1 - np.cos(2 * np.pi * t / period_s)) / 2.0


def reps_varied(amps, period_s=2.0, fs=FS):
    """One full rep per amplitude in `amps`, concatenated."""
    t = np.arange(0, period_s, 1 / fs)
    return np.concatenate([a * (1 - np.cos(2 * np.pi * t / period_s)) / 2.0 for a in amps])


def test_counts_known_reps():
    r = detect_reps(rep_signal(4, 60.0), FS)
    assert r.count == 4
    assert not r.partial_at_start and not r.partial_at_end


def test_per_peak_prominence_reported():
    r = detect_reps(rep_signal(4, 60.0), FS)
    assert r.peak_prominences_deg.size == 4
    assert np.allclose(r.peak_prominences_deg, 60.0, atol=1.0)


def test_partial_rep_at_end_flagged():
    """A signal ending mid-flexion (at a peak) is flagged partial_at_end."""
    sig = rep_signal(4, 60.0)
    sig = np.concatenate([sig, rep_signal(1, 60.0)[: len(sig) // 8]])  # append a rising edge
    r = detect_reps(sig, FS)
    assert r.partial_at_end


def test_partial_rep_at_start_flagged():
    """A signal that begins already flexed is flagged partial_at_start."""
    full = rep_signal(4, 60.0)
    started_mid = full[len(full) // 16:]        # drop the first bit -> starts elevated
    # ensure it truly starts elevated
    assert started_mid[0] > 10.0
    r = detect_reps(started_mid, FS)
    assert r.partial_at_start


def test_small_wiggle_is_not_counted():
    """A tiny bump between two real reps must not add a rep."""
    two = rep_signal(2, 50.0)
    wiggle = 3.0 * (1 - np.cos(2 * np.pi * np.arange(0, 0.5, 1 / FS) / 0.5)) / 2.0
    sig = np.concatenate([two, wiggle, two])
    r = detect_reps(sig, FS)
    assert r.count == 4                          # the 3 deg wiggle is ignored


def test_low_rom_rehab_reps_are_counted():
    """~25 deg early-recovery reps are counted (adaptive prominence scales down)."""
    r = detect_reps(rep_signal(4, 25.0), FS)
    assert r.count == 4
    assert r.effective_prominence_deg < 25.0     # threshold below the rep amplitude


def test_adaptive_prominence_beats_a_fixed_floor():
    """
    THE CLINICAL POINT. Shallow ~12 deg reps: the adaptive threshold counts all
    four, whereas a fixed 15 deg prominence floor (what we almost shipped) would
    miss them entirely - failing exactly the low-ROM patients who need counting.
    """
    sig = rep_signal(4, 12.0)
    adaptive = detect_reps(sig, FS)
    fixed15, _ = find_peaks(sig, prominence=15.0, distance=int(0.8 * FS))
    assert adaptive.count == 4
    assert fixed15.size < 4                       # fixed floor undercounts the shallow reps


def test_rep_consistency_uniform_vs_varied():
    """Uniform reps have low amplitude_cv; varied amplitudes raise it."""
    uniform = detect_reps(rep_signal(4, 50.0), FS)
    varied = detect_reps(reps_varied([30.0, 60.0, 40.0, 55.0]), FS)
    assert uniform.amplitude_cv < 0.05
    assert varied.amplitude_cv > 0.15


def test_mid_session_pause_keeps_count_but_raises_period_cv():
    """A long pause mid-session must not change the count, but it shows up as
    irregular timing (elevated period_cv)."""
    two = rep_signal(2, 50.0)
    pause = np.zeros(int(6.0 * FS))              # 6 s still
    sig = np.concatenate([two, pause, two])
    paused = detect_reps(sig, FS)
    even = detect_reps(rep_signal(4, 50.0), FS)
    assert paused.count == 4
    assert paused.period_cv > even.period_cv + 0.2


def test_hugadb_sit_to_stand_rep_count(hugadb_source):
    """
    Real validation: this file's sit-to-stand cycles are ~64 deg, and the
    recording begins AND ends mid-sit. The detector reports the unambiguous
    interior reps (3) and - the point of the confidence indicators - flags BOTH
    edges as partial, so a clinician knows movement was cut at both boundaries
    (true count is 3-5) rather than trusting a bare number.
    """
    cap = hugadb_source.get_data()
    stand = np.where(cap.activity == "standing")[0]
    neutral = calibrate_from_quaternions(
        cap.quat_thigh[stand], cap.quat_shank[stand]).q_neutral
    ang = knee_flexion_angle(remove_offset(knee_relative(cap.quat_thigh, cap.quat_shank),
                                           neutral), axis=HUGADB_FLEXION_AXIS)
    r = detect_reps(ang, fs_hz=hugadb_source.fs_hz)
    assert 3 <= r.count <= 5, r.count
    assert np.median(r.peak_values_deg) > 45.0
    assert r.partial_at_start and r.partial_at_end        # begins and ends mid-sit


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"run via: pytest {__file__}")
