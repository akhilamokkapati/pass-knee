"""
test_synthetic_source.py
Tests for the synthetic data source.

The centerpiece is the ROUND-TRIP against an INDEPENDENT ground truth: the
source builds the two quaternions from a chosen angle via a forward model and
reports that chosen angle as knee_angle_deg (never calling the engine). Here we
run the engine (knee_relative -> knee_flexion_angle) on those quaternions and
require it to recover knee_angle_deg. Two independent implementations agreeing
is real evidence the whole chain is correct.

Run:  python -m pytest test_synthetic_source.py -v
  or: python test_synthetic_source.py
"""

import itertools

import numpy as np

from biomechanics.relative_orientation import knee_relative
from biomechanics.joint_angles import knee_flexion_angle
from sources.synthetic import SyntheticSource, Packet, axis_angle_quat


def _engine_angle(p):
    """Independently reduce a packet's quaternions to a flexion angle."""
    return knee_flexion_angle(knee_relative(p.quat_thigh, p.quat_shank))


def test_packet_matches_firmware_schema():
    """A packet carries exactly the firmware fields, quats length 4."""
    p = next(SyntheticSource().stream())
    assert isinstance(p, Packet)
    assert p.seq == 0 and p.t_ms == 0
    assert np.shape(p.quat_thigh) == (4,) and np.shape(p.quat_shank) == (4,)
    assert np.isfinite(p.knee_angle_deg)


def test_stream_increments_seq_and_time():
    """seq counts up; t_ms advances by the sample period."""
    src = SyntheticSource(rate_hz=100.0)
    ps = list(itertools.islice(src.stream(), 4))
    assert [p.seq for p in ps] == [0, 1, 2, 3]
    assert [p.t_ms for p in ps] == [0, 10, 20, 30]     # 100 Hz -> 10 ms


def test_round_trip_recovers_independent_ground_truth():
    """
    THE KEY TEST. knee_angle_deg is set by the forward model; the engine
    reconstructs the angle from the emitted quaternions; the two must match to
    numerical precision. No noise, so recovery is exact.
    """
    src = SyntheticSource(seed=0)                        # noise defaults to 0
    for p in itertools.islice(src.stream(), 250):       # ~2.5 s, > one full rep
        assert abs(_engine_angle(p) - p.knee_angle_deg) < 1e-6, (
            p.seq, _engine_angle(p), p.knee_angle_deg)


def test_round_trip_holds_for_a_nontrivial_leg_pose():
    """World-invariance in the round trip: an explicit tilted/turned leg pose
    must not change the recovered knee angle."""
    pose = axis_angle_quat([1.0, -0.4, 0.6], 120.0)
    src = SyntheticSource(leg_pose=pose)
    cap = src.get_data(1.0)
    for i in range(len(cap.seq)):
        rel = knee_relative(cap.quat_thigh[i], cap.quat_shank[i])
        assert abs(knee_flexion_angle(rel) - cap.knee_angle_deg[i]) < 1e-6


def test_get_data_shapes_and_duration():
    """get_data returns arrays of the right length and shape."""
    src = SyntheticSource(rate_hz=100.0)
    cap = src.get_data(1.0)
    assert len(cap.seq) == 100
    assert cap.quat_thigh.shape == (100, 4)
    assert cap.quat_shank.shape == (100, 4)
    assert cap.knee_angle_deg.shape == (100,)


def test_ground_truth_stays_within_profile_bounds():
    """The chosen angle never leaves [min, max]."""
    src = SyntheticSource(min_angle_deg=5.0, max_angle_deg=70.0)
    cap = src.get_data(3.0)
    assert cap.knee_angle_deg.min() >= 5.0 - 1e-9
    assert cap.knee_angle_deg.max() <= 70.0 + 1e-9


def test_profile_reaches_both_extremes():
    """Over a full rep the profile actually sweeps to min and to max."""
    src = SyntheticSource(min_angle_deg=0.0, max_angle_deg=60.0, rep_period_s=2.0)
    cap = src.get_data(2.0)                              # exactly one period
    assert cap.knee_angle_deg.min() < 0.5
    assert cap.knee_angle_deg.max() > 59.5


def test_noise_perturbs_quaternions_but_not_ground_truth():
    """
    Reinforces the independence: with noise on, knee_angle_deg stays the clean
    forward-model value (identical across seeds), while the engine reading now
    differs from it by roughly the injected noise - small but nonzero.
    """
    clean = SyntheticSource(noise_deg=0.0).get_data(1.0)
    noisy = SyntheticSource(noise_deg=3.0, seed=1).get_data(1.0)
    # ground truth is unaffected by noise
    assert np.allclose(clean.knee_angle_deg, noisy.knee_angle_deg, atol=1e-12)
    # but the engine now disagrees with ground truth by a small, real amount
    err = np.array([
        knee_flexion_angle(knee_relative(noisy.quat_thigh[i], noisy.quat_shank[i]))
        - noisy.knee_angle_deg[i]
        for i in range(len(noisy.seq))
    ])
    assert np.any(np.abs(err) > 1e-6), "noise should move the engine reading"
    assert np.sqrt(np.mean(err**2)) < 10.0, "3 deg noise should stay bounded"


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
