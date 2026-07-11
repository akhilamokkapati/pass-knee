"""
test_run_capture.py
The accuracy number we cite has to be defensible too, so the stat computation
gets a known-answer test and the runner is checked to recover the synthetic
ground truth. Also covers the offline filter improving accuracy, and a headless
smoke test of the HuGaDB real-data figure (raw+filtered overlay, activity bands).

Run:  python -m pytest test_run_capture.py -v
"""

import matplotlib
matplotlib.use("Agg")            # headless: no window for the plot smoke test

import numpy as np

from filters import lowpass_offline
from sources.synthetic import SyntheticSource
from sources.hugadb import HUGADB_FLEXION_AXIS
from run_capture import (
    recovered_angle, accuracy_stats, straight_leg_neutral, plot_capture,
    activity_flexion_lines, metric_lines, _reps_line,
)
from repetitions import detect_reps


def test_accuracy_stats_known_answer():
    """Errors of [0, +3, -4] deg -> max 4, RMS sqrt(25/3), bias -1/3."""
    gt = np.array([10.0, 10.0, 10.0])
    rec = np.array([10.0, 13.0, 6.0])
    s = accuracy_stats(gt, rec)
    assert s.n_samples == 3
    assert np.isclose(s.max_abs_error_deg, 4.0)
    assert np.isclose(s.rms_error_deg, np.sqrt(25.0 / 3.0))
    assert np.isclose(s.mean_error_deg, -1.0 / 3.0)


def test_runner_recovers_clean_synthetic_to_precision():
    """Noiseless synthetic -> engine matches ground truth ~exactly."""
    cap = SyntheticSource(seed=0).get_data(5.0)
    s = accuracy_stats(cap.knee_angle_deg, recovered_angle(cap))
    assert s.max_abs_error_deg < 1e-6 and s.rms_error_deg < 1e-6, s


def test_offline_filter_improves_noisy_accuracy():
    """Zero-phase filtering the recovered angle lowers both max and RMS error
    versus the unfiltered angle on noisy synthetic data."""
    cap = SyntheticSource(noise_deg=2.0, seed=3).get_data(5.0)
    raw = recovered_angle(cap)
    filt = lowpass_offline(raw, cutoff_hz=6.0, fs_hz=100.0)
    interior = slice(50, -50)
    raw_s = accuracy_stats(cap.knee_angle_deg[interior], raw[interior])
    filt_s = accuracy_stats(cap.knee_angle_deg[interior], filt[interior])
    assert filt_s.rms_error_deg < raw_s.rms_error_deg
    assert filt_s.max_abs_error_deg < raw_s.max_abs_error_deg


def test_metric_lines_report_full_set():
    """The figure report carries the full metric set, not just headlines."""
    t = np.arange(0, 8, 0.01)
    ang = 30 * (1 - np.cos(2 * np.pi * t / 2.0))         # 4 reps, 0..60 deg
    lines = metric_lines(ang, fs_hz=100.0)
    joined = " ".join(lines)
    for token in ("ROM", "max flexion", "max extension", "peak vel", "reps:"):
        assert token in joined, token


def test_reps_line_is_never_a_bare_number():
    """Rep reporting always includes the adaptive prominence, and edge/partial
    context when present — so a count is never shown without its confidence."""
    fs = 100.0
    t = np.arange(0, 8, 1 / fs)
    full = 30 * (1 - np.cos(2 * np.pi * t / 2.0))
    line = _reps_line(detect_reps(full, fs))
    assert "reps:" in line and "adaptive prominence" in line

    # a signal ending mid-flexion must surface the partial-edge caveat
    trimmed = np.concatenate([full, full[: len(full) // 8]])
    edge_line = _reps_line(detect_reps(trimmed, fs))
    assert "partial" in edge_line and "recording trimmed" in edge_line


def test_hugadb_real_data_figure_smoke(hugadb_source, tmp_path):
    """The SDR real-data figure builds headlessly: raw+filtered overlay with
    activity bands, standing calibrated to ~0, per-activity ROM lines produced."""
    cap = hugadb_source.get_data()
    neutral = straight_leg_neutral(cap, hugadb_source.fs_hz)
    raw = recovered_angle(cap, axis=HUGADB_FLEXION_AXIS, q_neutral=neutral)
    filt = lowpass_offline(raw, cutoff_hz=6.0, fs_hz=hugadb_source.fs_hz)

    assert np.all(np.isfinite(filt))
    stand = cap.activity == "standing"
    assert abs(np.median(filt[stand])) < 10.0

    lines = activity_flexion_lines(cap.activity, filt)
    assert any("sitting" in ln for ln in lines)

    out = tmp_path / "hugadb.png"
    plot_capture(cap, raw, filt, ground_truth=None, cutoff_hz=6.0,
                 title="smoke", caption="smoke", stats_lines=lines,
                 save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"run via: pytest {__file__}")
