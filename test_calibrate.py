"""
test_calibrate.py
Known-answer tests for straight-leg (quiet-standing) calibration.

The math to defend is the quaternion average (Markley eigenvector method) and
the end-to-end promise: a captured neutral must zero out the mounting offset so
a straight leg reads 0 deg. The centerpiece test builds a quiet-standing window
as a KNOWN offset plus small jitter and checks the recovered neutral removes it.

Run:  python -m pytest test_calibrate.py -v
"""

import numpy as np

from biomechanics.quaternion_math import multiply, normalize
from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle
from sources.synthetic import SyntheticSource, axis_angle_quat
from calibrate import (
    average_quaternion, calibrate_from_quaternions, calibrate, CalibrationResult,
)


def quat_angle_between_deg(a, b):
    """Rotation angle (deg) between two unit quaternions, sign-independent."""
    a = normalize(np.asarray(a, float))
    b = normalize(np.asarray(b, float))
    d = np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


X = np.array([1.0, 0.0, 0.0])


def test_average_of_identical_quaternions_is_that_quaternion():
    q = axis_angle_quat([0.2, 1.0, 0.3], 40.0)
    avg = average_quaternion(np.tile(q, (20, 1)))
    assert quat_angle_between_deg(avg, q) < 1e-6, avg


def test_average_is_sign_invariant_double_cover():
    """
    Mixing q and -q (the same rotation) must not cancel — the whole reason we
    use Markley (q q^T is sign-invariant) instead of a naive component mean.
    """
    q = axis_angle_quat([0.5, -0.2, 0.8], 33.0)
    stack = np.array([q, -q, q, -q, q])
    avg = average_quaternion(stack)
    assert quat_angle_between_deg(avg, q) < 1e-6, avg


def test_average_recovers_center_of_symmetric_jitter():
    """q0 jittered by +delta and -delta about an axis averages back to q0."""
    q0 = axis_angle_quat([0.1, 0.2, 1.0], 25.0)
    plus = multiply(q0, axis_angle_quat(X, 1.5))
    minus = multiply(q0, axis_angle_quat(X, -1.5))
    avg = average_quaternion(np.array([plus, minus]))
    assert quat_angle_between_deg(avg, q0) < 1e-6, avg


def test_average_output_is_unit_and_canonical():
    q = average_quaternion(np.array([axis_angle_quat([1, 1, 0], 200.0)]))
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert q[0] >= 0.0            # canonical w >= 0


def test_neutral_recovers_known_offset_and_zeros_straight_leg():
    """
    THE CENTERPIECE. Quiet standing has a fixed mounting offset (the relative
    thigh->shank rotation at straight leg) plus small orientation jitter. The
    recovered neutral must (a) match the offset and (b) drive the straight-leg
    flexion angle to ~0 after remove_offset.
    """
    rng = np.random.default_rng(0)
    offset = axis_angle_quat([0.2, 1.0, 0.4], 12.0)      # true mounting offset
    leg_pose = axis_angle_quat([0.3, 0.5, 1.0], 30.0)    # whole-leg standing pose

    n = 200
    q_thigh = np.zeros((n, 4))
    q_shank = np.zeros((n, 4))
    for i in range(n):
        jitter = axis_angle_quat(rng.normal(size=3), rng.normal(0.0, 0.4))  # ~0.4 deg
        q_thigh[i] = leg_pose
        q_shank[i] = multiply(multiply(leg_pose, offset), jitter)

    result = calibrate_from_quaternions(q_thigh, q_shank)
    assert isinstance(result, CalibrationResult)
    assert result.n_samples == n

    # (a) recovered neutral matches the true offset
    assert quat_angle_between_deg(result.q_neutral, offset) < 0.3, result.q_neutral

    # (b) straight-leg angle after calibration is ~0 across the window
    rel = knee_relative(q_thigh, q_shank)
    ang = knee_flexion_angle(remove_offset(rel, result.q_neutral))
    assert np.max(np.abs(ang)) < 1.5, np.max(np.abs(ang))


def test_residual_reports_window_spread():
    """A jittered window has small-but-nonzero residual; a still window ~0."""
    q0 = axis_angle_quat([0.2, 1.0, 0.3], 20.0)
    still = calibrate_from_quaternions(np.tile(q0, (10, 1)), np.tile(q0, (10, 1)))
    assert still.residual_rms_deg < 1e-6 and still.residual_max_deg < 1e-6

    # a window with a 2 deg swing shows it in the residual
    shank = np.array([multiply(q0, axis_angle_quat(X, a)) for a in (-2, -1, 0, 1, 2)])
    thigh = np.tile(q0, (5, 1))
    swayed = calibrate_from_quaternions(thigh, shank)
    assert swayed.residual_max_deg > 1.0


def test_calibrate_reads_through_source_interface():
    """calibrate() pulls a quiet-standing window from any source via get_data.
    Synthetic source at zero flexion -> neutral is (near) identity, tiny spread."""
    src = SyntheticSource(min_angle_deg=0.0, max_angle_deg=0.0)   # perfectly still
    result = calibrate(src, duration_s=1.0)
    assert result.n_samples == 100
    assert result.residual_max_deg < 1e-6
    # zero flexion + no mounting offset in the synthetic model -> identity neutral
    assert quat_angle_between_deg(result.q_neutral, [1.0, 0.0, 0.0, 0.0]) < 1e-6


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
