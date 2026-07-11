"""
test_axis_calibration.py
Known-answer tests for the flexion-axis calibration: build poses whose flexion
axis we CHOOSE, confirm the helper recovers it (and that the recovered axis is
directly usable, sign-correct, in the engine). Plus a real check that on HuGaDB
the helper rediscovers the dataset's ~ -Y axis we found by hand, and that a
wobbly bend lowers the confidence.

Run:  python -m pytest test_axis_calibration.py -v
"""

import numpy as np

from biomechanics.quaternion_math import multiply, normalize
from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle
from calibrate import calibrate_from_quaternions
from sources.synthetic import axis_angle_quat
from sources.hugadb import HUGADB_FLEXION_AXIS
from axis_calibration import calibrate_flexion_axis, _axis_angle


def _angle_between(a, b):
    a = normalize(np.asarray(a, float)); b = normalize(np.asarray(b, float))
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def _poses(true_axis, theta_deg, n=150, jitter_deg=0.3, axis_wobble=0.0, seed=0):
    """Build straight-leg and bent windows whose flexion axis is `true_axis`.
    knee_relative(neutral) = offset; remove_offset(knee_relative(bent), offset)
    = rotation of theta about true_axis."""
    rng = np.random.default_rng(seed)
    leg = axis_angle_quat([0.3, 1.0, 0.2], 40.0)      # arbitrary whole-leg pose
    offset = axis_angle_quat([1.0, 0.2, -0.3], 12.0)  # mounting misalignment

    def jit(q):
        return multiply(q, axis_angle_quat(rng.normal(size=3), rng.normal(0, jitter_deg)))

    n_thigh = np.array([jit(leg) for _ in range(n)])
    n_shank = np.array([jit(multiply(leg, offset)) for _ in range(n)])

    b_thigh = np.array([jit(leg) for _ in range(n)])
    b_shank = np.empty((n, 4))
    for i in range(n):
        ax = true_axis if axis_wobble == 0 else np.asarray(true_axis, float) + rng.normal(0, axis_wobble, 3)
        flex = axis_angle_quat(ax, theta_deg + rng.normal(0, 0.5))
        b_shank[i] = jit(multiply(multiply(leg, offset), flex))
    return n_thigh, n_shank, b_thigh, b_shank


def test_axis_angle_reads_known_rotation():
    axis, ang = _axis_angle(axis_angle_quat([0, -1, 0], 55.0))
    assert _angle_between(axis, [0, -1, 0]) < 1e-6
    assert abs(ang - 55.0) < 1e-6


def test_recovers_known_flexion_axis():
    true_axis = normalize([0.1, -1.0, 0.25])
    cal = calibrate_flexion_axis(*_poses(true_axis, 55.0))
    assert _angle_between(cal.flexion_axis, true_axis) < 2.0
    assert abs(cal.bend_angle_deg - 55.0) < 2.0
    assert cal.axis_confidence > 0.98
    assert cal.reliable


def test_recovered_axis_gives_positive_flexion_in_engine():
    """The measured axis, used in knee_flexion_angle, reads the bend as positive
    and ~= the true bend magnitude — i.e. it drops straight into the engine."""
    true_axis = normalize([0.0, -1.0, 0.0])
    nt, ns, bt, bs = _poses(true_axis, 50.0)
    cal = calibrate_flexion_axis(nt, ns, bt, bs)
    neutral = calibrate_from_quaternions(nt, ns).q_neutral
    rel = remove_offset(knee_relative(bt, bs), neutral)
    ang = knee_flexion_angle(rel, axis=cal.flexion_axis)
    assert np.median(ang) > 0                      # bending reads positive
    assert abs(np.median(ang) - 50.0) < 2.0


def test_small_bend_is_flagged_unreliable():
    """A tiny bend leaves the axis under-determined -> reliable is False."""
    cal = calibrate_flexion_axis(*_poses(normalize([0, -1, 0]), 5.0))
    assert cal.bend_angle_deg < 20.0
    assert not cal.reliable


def test_wobbly_bend_lowers_confidence():
    """A bend whose axis wanders sample-to-sample scores lower confidence than a
    clean single-axis bend."""
    clean = calibrate_flexion_axis(*_poses(normalize([0, -1, 0]), 50.0, axis_wobble=0.0))
    wobbly = calibrate_flexion_axis(*_poses(normalize([0, -1, 0]), 50.0, axis_wobble=0.6))
    assert wobbly.axis_confidence < clean.axis_confidence
    assert wobbly.axis_confidence < 0.9


def test_hugadb_rediscovers_minus_y_axis(hugadb_source):
    """Real validation: standing->sitting on HuGaDB recovers an axis dominated by
    -Y (the mounting axis we found by hand), with a substantial, confident bend."""
    cap = hugadb_source.get_data()
    stand = cap.activity == "standing"
    sit = cap.activity == "sitting"
    cal = calibrate_flexion_axis(cap.quat_thigh[stand], cap.quat_shank[stand],
                                 cap.quat_thigh[sit], cap.quat_shank[sit])
    ax = cal.flexion_axis
    assert abs(ax[1]) > abs(ax[0]) and abs(ax[1]) > abs(ax[2])    # Y dominant
    assert ax[1] < 0                                             # -Y
    assert _angle_between(ax, HUGADB_FLEXION_AXIS) < 15.0
    assert 55.0 < cal.bend_angle_deg < 75.0
    assert cal.axis_confidence > 0.85


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"run via: pytest {__file__}")
