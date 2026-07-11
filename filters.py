"""
filters.py
PASS knee module — low-pass filtering for the knee-angle signal.

The BNO085 fused quaternions carry per-sample noise that turns into per-sample
angle noise; raw, it can spike past the +/-2.5 deg clinical target even when its
RMS is well under it. Filtering the angle signal is what makes the +/-2.5 deg
claim hold, and it is also what gives HuGaDB and captured sessions clean angles
to derive ROM / velocity / reps from. So this is shared real-data infrastructure,
not a one-off.

TWO MODES, ONE SHARED DESIGN
----------------------------
Both modes use the SAME Butterworth low-pass design (design_lowpass), so their
magnitude responses match; only the way they are RUN differs.

  lowpass_offline (filtfilt)  - forward+backward pass => ZERO phase lag, so peak
      timing is preserved. For batch/offline paths (HuGaDB, captured sessions).
      Needs the whole signal, so it CANNOT run in real time.

  StreamingLowpass (lfilter + state) - causal, one sample at a time, carrying
      filter state. For the LIVE path (drops into a source's stream()). It lags
      by construction - the unavoidable price of real-time - but warm-starts so
      there is no startup transient.

Butterworth is the biomechanics standard (maximally flat passband, no ripple, so
it does not distort ROM or peak amplitudes). Defaults: 2nd order, 6 Hz cutoff at
100 Hz - knee motion in rehab is < ~3 Hz, so 6 Hz keeps the movement and removes
the noise. fs must be supplied by the caller because sources differ (the BNO085
runs ~100 Hz; HuGaDB does not).

Library functions do not print; they return filtered signals.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

DEFAULT_CUTOFF_HZ = 6.0
DEFAULT_ORDER = 2


def design_lowpass(cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                   fs_hz: float = 100.0,
                   order: int = DEFAULT_ORDER):
    """
    Butterworth low-pass coefficients (b, a). The single design shared by the
    offline and streaming filters, so both have identical magnitude response.

    cutoff_hz must be below Nyquist (fs_hz / 2).
    """
    nyquist = fs_hz / 2.0
    if not (0.0 < cutoff_hz < nyquist):
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be in (0, Nyquist={nyquist}); "
            f"fs_hz={fs_hz}"
        )
    return butter(order, cutoff_hz / nyquist, btype="low")


def lowpass_offline(signal: np.ndarray,
                    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                    fs_hz: float = 100.0,
                    order: int = DEFAULT_ORDER) -> np.ndarray:
    """
    Zero-phase low-pass (filtfilt) for offline / batch signals.

    Forward-backward filtering cancels phase, so peaks are not shifted in time -
    essential for ROM, max-flexion timing and rep detection on captured/HuGaDB
    data. Short signals are handled by shrinking the pad length; a signal too
    short to filter (<= 1 sample) is returned unchanged.
    """
    x = np.asarray(signal, dtype=float)
    if x.size <= 1:
        return x.copy()
    b, a = design_lowpass(cutoff_hz, fs_hz, order)
    default_pad = 3 * max(len(a), len(b))
    padlen = min(default_pad, x.size - 1)
    return filtfilt(b, a, x, padlen=padlen)


class StreamingLowpass:
    """
    Causal, stateful low-pass for the live real-time path.

    Same Butterworth design as lowpass_offline, run with lfilter and carried
    filter state (zi) so it can process one sample at a time inside a source's
    stream() loop. On the first sample it warm-starts the state to that sample's
    steady value, so the output starts at the signal level instead of ramping up
    from zero.

    Usage (live):   filt = StreamingLowpass(cutoff_hz, fs_hz)
                    for packet in source.stream():
                        angle = filt.process(raw_angle_from(packet))
    """

    def __init__(self,
                 cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                 fs_hz: float = 100.0,
                 order: int = DEFAULT_ORDER):
        self.b, self.a = design_lowpass(cutoff_hz, fs_hz, order)
        self._zi_unit = lfilter_zi(self.b, self.a)   # steady state for unit DC
        self._zi = None                              # set on first sample

    def reset(self):
        """Forget filter state; the next sample warm-starts afresh."""
        self._zi = None

    def process(self, x: float) -> float:
        """Filter one sample, updating internal state. Returns the filtered value."""
        x = float(x)
        if self._zi is None:
            self._zi = self._zi_unit * x             # warm start at x -> no transient
        y, self._zi = lfilter(self.b, self.a, [x], zi=self._zi)
        return float(y[0])

    def process_array(self, xs: np.ndarray) -> np.ndarray:
        """Filter a sequence sample-by-sample, preserving state across it.
        Equivalent to looping process(); provided for batch use and testing."""
        xs = np.asarray(xs, dtype=float)
        out = np.empty(xs.shape, dtype=float)
        for i, x in enumerate(xs):
            out[i] = self.process(x)
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = np.full(500, 40.0)
    raw = true + rng.normal(0.0, 2.0, size=500)
    off = lowpass_offline(raw)
    on = StreamingLowpass().process_array(raw)
    print("PASS filters ready - "
          f"raw max err {np.max(np.abs(raw - true)):.2f}, "
          f"offline {np.max(np.abs(off - true)):.2f}, "
          f"streaming {np.max(np.abs(on[50:] - true[50:])):.2f} deg")
