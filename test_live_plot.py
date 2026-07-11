"""
test_live_plot.py
Headless tests for the live plot's DATA PATH (the matplotlib animation itself is
interactive and not unit-tested). We verify the hot loop and cadence logic:
buffers stay bounded and scroll, the filtered stream is the CAUSAL
StreamingLowpass over the raw stream (not filtfilt), filtering actually smooths,
and the wall-clock pump decouples sample cadence from redraw cadence.

Run:  python -m pytest test_live_plot.py -v
"""

import itertools

import matplotlib
matplotlib.use("Agg")            # headless: exercise the drawing path with no window

import numpy as np

from sources.synthetic import SyntheticSource
from filters import StreamingLowpass
from live_plot import LiveKneePlot


def test_ingest_filters_and_buffers():
    """Ingesting packets fills the buffers and the filter smooths the noise."""
    src = SyntheticSource(noise_deg=3.0, seed=0)
    plot = LiveKneePlot(src, window_s=100.0, cutoff_hz=6.0)   # big window: keep all
    for pkt in itertools.islice(src.stream(), 400):
        plot.ingest(pkt)
    raw = np.fromiter(plot.buf_raw, float)
    filt = np.fromiter(plot.buf_filt, float)
    assert raw.size == filt.size == 400
    # filtered signal is smoother: smaller sample-to-sample jumps
    assert np.std(np.diff(filt)) < np.std(np.diff(raw))


def test_filtered_stream_is_causal_reference():
    """The live filtered buffer equals a standalone causal StreamingLowpass run
    over the same raw sequence — i.e. it is the streaming filter with state, not
    a zero-phase filtfilt."""
    src = SyntheticSource(noise_deg=2.0, seed=1)
    plot = LiveKneePlot(src, window_s=100.0, cutoff_hz=6.0)
    for pkt in itertools.islice(src.stream(), 300):
        plot.ingest(pkt)
    raw = np.fromiter(plot.buf_raw, float)
    ref = StreamingLowpass(cutoff_hz=6.0, fs_hz=src.rate_hz).process_array(raw)
    assert np.allclose(np.fromiter(plot.buf_filt, float), ref, atol=1e-12)


def test_buffers_are_bounded_and_scroll():
    """Buffers cap at the window length, so old samples scroll off."""
    src = SyntheticSource(rate_hz=100.0)
    plot = LiveKneePlot(src, window_s=2.0)                    # 2 s -> ~201 samples
    cap = plot.buf_t.maxlen
    for pkt in itertools.islice(src.stream(), cap + 500):
        plot.ingest(pkt)
    assert len(plot.buf_t) == cap
    assert len(plot.buf_raw) == cap == len(plot.buf_filt)


def test_pump_decouples_sample_cadence_from_wall_clock():
    """The pump ingests ~ elapsed*rate samples, capped per call — this is what
    lets the ~30 fps redraw stay decoupled from the ~100 Hz sample rate."""
    src = SyntheticSource(rate_hz=100.0)
    plot = LiveKneePlot(src, target_fps=30.0)
    assert plot.pump(now=0.0) == 0                            # sets t0, nothing due yet
    assert plot.pump(now=0.1) == 10                           # 0.1 s * 100 Hz = 10
    # a big jump is capped (4 * rate / fps) so a stall cannot spiral
    capped = plot.pump(now=5.0)
    assert capped == plot._max_pull < 500


def test_drawing_path_renders_headless(tmp_path):
    """build_figure + draw_frame run without a window and produce a PNG."""
    src = SyntheticSource(noise_deg=2.0, seed=2)
    plot = LiveKneePlot(src, window_s=5.0)
    plot.build_figure()
    for pkt in itertools.islice(src.stream(), 300):
        plot.ingest(pkt)
    plot.draw_frame()                        # pump adds ~0 more; renders the buffer
    out = tmp_path / "live.png"
    plot.fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    # scrolling x-window is at most window_s wide
    lo, hi = plot.ax.get_xlim()
    assert hi - lo <= plot.window_s + 1e-6


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
    print(f"\n{passed}/{len(tests)} tests passed")
