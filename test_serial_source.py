"""
test_serial_source.py
Tests for the live serial source, driven entirely by SIMULATED firmware lines —
no hardware, no pyserial. The point is that when the BNO085s arrive, the parser
and the whole path are already proven.

The centerpiece asserts the design contract: the engine recomputes the knee
angle from the RAW QUATERNIONS (the validated path), while the firmware's
on-device angle is only carried as a cross-check — so a wrong firmware angle
does NOT corrupt the measured angle.

Run:  python -m pytest test_serial_source.py -v
"""

import itertools

import numpy as np

from biomechanics.relative_orientation import knee_relative
from biomechanics.joint_angles import knee_flexion_angle
from sources.synthetic import SyntheticSource
from sources.serial_source import (
    SerialSource, parse_packet_line, format_packet_line, FIRMWARE_FIELDS,
)


def _firmware_line(seq, t_ms, angle, qt, qs):
    q = lambda a: ",".join(f"{v:.6f}" for v in a)
    return f"{seq},{t_ms},{angle},{q(qt)},{q(qs)}"


# --- parsing --------------------------------------------------------------

def test_parse_valid_line():
    line = _firmware_line(0, 10, 12.5, [1, 0, 0, 0], [0.966, 0.259, 0, 0])
    p = parse_packet_line(line)
    assert p.seq == 0 and p.t_ms == 10
    assert p.knee_angle_deg == 12.5
    assert p.quat_thigh.shape == (4,) and p.quat_shank.shape == (4,)
    assert np.allclose(p.quat_shank, [0.966, 0.259, 0, 0])


def test_parse_rejects_malformed_lines():
    assert parse_packet_line("") is None
    assert parse_packet_line("   ") is None
    assert parse_packet_line("# a comment") is None
    assert parse_packet_line("1,2,3") is None                 # too few fields
    assert parse_packet_line("a,b,c,d,e,f,g,h,i,j,k") is None  # non-numeric
    assert parse_packet_line("0,1,2,3,4,5,6,7,8,9,10,11") is None  # too many


def test_format_parse_roundtrip():
    pkt = next(SyntheticSource(seed=0).stream())
    reparsed = parse_packet_line(format_packet_line(pkt))
    assert reparsed.seq == pkt.seq and reparsed.t_ms == pkt.t_ms
    assert np.allclose(reparsed.quat_thigh, pkt.quat_thigh, atol=1e-6)
    assert np.allclose(reparsed.quat_shank, pkt.quat_shank, atol=1e-6)


# --- source behaviour -----------------------------------------------------

def test_stream_skips_and_counts_malformed():
    lines = [
        "# header",
        _firmware_line(0, 0, 0.0, [1, 0, 0, 0], [1, 0, 0, 0]),
        "garbage,line",
        _firmware_line(1, 10, 5.0, [1, 0, 0, 0], [1, 0, 0, 0]),
        "",
    ]
    src = SerialSource(line_source=lines)
    got = list(src.stream())
    assert [p.seq for p in got] == [0, 1]
    assert src.n_malformed == 1                    # only "garbage,line" counts


def test_get_data_builds_capture_with_raw_quaternions():
    src = SyntheticSource(seed=1)
    lines = [format_packet_line(p) for p in itertools.islice(src.stream(), 50)]
    cap = SerialSource(line_source=lines).get_data()
    assert cap.seq.size == 50
    assert cap.quat_thigh.shape == (50, 4) and cap.quat_shank.shape == (50, 4)
    assert cap.activity is None                    # serial has no activity labels


def test_engine_is_truth_firmware_angle_is_crosscheck():
    """
    THE CONTRACT. Simulate a firmware that reports a BIASED on-device angle
    (+7 deg). The engine must still recover the TRUE angle from the quaternions,
    and the biased firmware value must survive only in knee_angle_deg as a
    cross-check — never affecting the measured angle.
    """
    src = SyntheticSource(seed=2)
    truth, lines = [], []
    for p in itertools.islice(src.stream(), 120):
        truth.append(p.knee_angle_deg)                         # synthetic ground truth
        lines.append(format_packet_line(
            type(p)(p.seq, p.t_ms, p.knee_angle_deg + 7.0,     # firmware bias
                    p.quat_thigh, p.quat_shank)))
    truth = np.array(truth)

    cap = SerialSource(line_source=lines).get_data()
    engine_angle = knee_flexion_angle(knee_relative(cap.quat_thigh, cap.quat_shank))

    # engine (from quaternions) recovers the true angle
    assert np.max(np.abs(engine_angle - truth)) < 1e-4
    # firmware angle is carried as-is (biased) for cross-check only
    assert np.allclose(cap.knee_angle_deg, truth + 7.0, atol=1e-3)
    # and the two disagree by ~the bias — that's the cross-check signal
    assert abs(np.mean(cap.knee_angle_deg - engine_angle) - 7.0) < 1e-2


def test_get_data_duration_limit():
    """A timed capture stops after duration_s of device time."""
    src = SyntheticSource(rate_hz=100.0, seed=3)          # 10 ms per sample
    lines = [format_packet_line(p) for p in itertools.islice(src.stream(), 500)]
    cap = SerialSource(line_source=lines).get_data(duration_s=0.05)   # 0.05 s -> ~6 samples
    assert cap.seq.size == 6                              # t_ms 0..50 inclusive
    assert cap.t_ms[-1] - cap.t_ms[0] >= 50


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
