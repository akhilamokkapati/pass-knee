"""
test_filters.py
Known-answer / property tests for the knee-angle low-pass filters.

Two modes, one shared Butterworth design:
  * lowpass_offline (filtfilt) - zero phase, for batch/HuGaDB/captured sessions;
  * StreamingLowpass (lfilter + state) - causal, for the live real-time path.

The properties checked are the ones we rely on: unity DC gain, passband
preserved, stopband attenuated, offline has NO lag while streaming does, the
streaming filter warm-starts without a transient, and - the headline claim from
CLAUDE.md - the streaming filter pulls per-sample noise back under +/-2.5 deg.

Run:  python -m pytest test_filters.py -v
"""

import numpy as np
import pytest

from filters import design_lowpass, lowpass_offline, StreamingLowpass


FS = 100.0   # BNO085 sample rate


def _lag_samples(x, y):
    """Lag of y relative to x by cross-correlation (positive = y delayed)."""
    x0 = np.asarray(x) - np.mean(x)
    y0 = np.asarray(y) - np.mean(y)
    corr = np.correlate(y0, x0, mode="full")
    return int(np.argmax(corr) - (len(x) - 1))


# --- design ---------------------------------------------------------------

def test_design_rejects_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError):
        design_lowpass(cutoff_hz=50.0, fs_hz=100.0)     # == Nyquist
    with pytest.raises(ValueError):
        design_lowpass(cutoff_hz=60.0, fs_hz=100.0)     # > Nyquist


# --- offline (zero-phase) -------------------------------------------------

def test_offline_preserves_constant():
    """Unity DC gain: a constant angle passes through unchanged."""
    x = np.full(300, 45.0)
    y = lowpass_offline(x, cutoff_hz=6.0, fs_hz=FS)
    assert np.allclose(y, 45.0, atol=1e-6), y[:5]


def test_offline_preserves_slow_passband_signal():
    """A slow knee-like sine (0.5 Hz) inside the passband is preserved in both
    amplitude AND phase (zero-phase filtering)."""
    t = np.arange(1000) / FS
    x = 30.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)
    y = lowpass_offline(x, cutoff_hz=6.0, fs_hz=FS)
    interior = slice(100, -100)
    assert np.sqrt(np.mean((y[interior] - x[interior]) ** 2)) < 0.2


def test_offline_attenuates_high_frequency():
    """A 30 Hz component (well above 6 Hz cutoff) is strongly suppressed."""
    t = np.arange(1000) / FS
    x = np.sin(2 * np.pi * 30.0 * t)
    y = lowpass_offline(x, cutoff_hz=6.0, fs_hz=FS)
    assert np.max(np.abs(y[100:-100])) < 0.1


def test_offline_is_zero_phase():
    """filtfilt introduces no lag: a passband sine stays time-aligned."""
    t = np.arange(1000) / FS
    x = np.sin(2 * np.pi * 1.0 * t)
    y = lowpass_offline(x, cutoff_hz=6.0, fs_hz=FS)
    assert _lag_samples(x, y) == 0


def test_offline_handles_short_signal():
    """Short arrays must not crash (filtfilt padding guard)."""
    x = np.array([10.0, 10.0, 10.0])
    y = lowpass_offline(x, cutoff_hz=6.0, fs_hz=FS)
    assert np.allclose(y, 10.0, atol=1e-6)


# --- streaming (causal) ---------------------------------------------------

def test_streaming_warm_starts_without_transient():
    """First output sits at the signal level, not ramping up from zero."""
    filt = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS)
    first = filt.process(50.0)
    assert abs(first - 50.0) < 1e-6, first


def test_streaming_preserves_constant():
    filt = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS)
    y = filt.process_array(np.full(200, 50.0))
    assert np.allclose(y, 50.0, atol=1e-9)


def test_streaming_is_causal_and_lags():
    """A causal filter must delay the signal (the price of real-time)."""
    t = np.arange(1000) / FS
    x = np.sin(2 * np.pi * 1.0 * t)
    y = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS).process_array(x)
    assert _lag_samples(x, y) > 0


def test_streaming_process_array_matches_scipy_reference():
    """process_array == sample-by-sample == warm-started scipy lfilter."""
    from scipy.signal import lfilter, lfilter_zi
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=250) + 20.0
    b, a = design_lowpass(cutoff_hz=6.0, fs_hz=FS)

    filt = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS)
    got = filt.process_array(x)

    zi = lfilter_zi(b, a) * x[0]
    ref, _ = lfilter(b, a, x, zi=zi)
    assert np.allclose(got, ref, atol=1e-12)


def test_streaming_reset_clears_state():
    filt = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS)
    filt.process_array(np.full(50, 99.0))
    filt.reset()
    assert abs(filt.process(5.0) - 5.0) < 1e-6      # warm-starts fresh at 5.0


def test_streaming_pulls_per_sample_noise_under_target():
    """
    HEADLINE (validates the CLAUDE.md note). Raw per-sample noise on a held
    straight leg can spike past +/-2.5 deg; the causal filter brings the max
    error back under target for the live path.
    """
    rng = np.random.default_rng(1)
    true = np.full(500, 40.0)                        # held pose, no motion lag
    noise = rng.normal(0.0, 2.0, size=500)
    raw = true + noise
    filt = StreamingLowpass(cutoff_hz=6.0, fs_hz=FS).process_array(raw)

    interior = slice(50, None)                       # skip warm-up
    raw_max = np.max(np.abs(raw[interior] - true[interior]))
    filt_max = np.max(np.abs(filt[interior] - true[interior]))
    assert raw_max > 2.5, raw_max                    # raw breaches target
    assert filt_max < 2.5, filt_max                  # filtered stays within it


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{passed}/{len(tests)} tests passed")
