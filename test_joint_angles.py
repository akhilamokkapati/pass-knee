"""
test_joint_angles.py
Known-answer tests for the knee flexion-angle metric - the single scalar the
whole knee module exists to produce. Every case is hand-checkable, so a pass is
real evidence the metric is correct.

Two conventions are pinned here on purpose, because they are decided in exactly
one place (joint_angles.py) and everything downstream trusts them:
  * SIGN: positive degrees == flexion (knee bending), negative == hyperextension.
  * AXIS: the flexion axis is a named parameter (a mounting/calibration choice),
    defaulting to +x; reversing it negates the reading.

Run:  python -m pytest test_joint_angles.py -v
  or: python test_joint_angles.py
"""

import numpy as np

from biomechanics.quaternion_math import multiply
from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle, DEFAULT_FLEXION_AXIS


def quat_about_axis(axis, deg):
    """Build a quaternion for a rotation of `deg` degrees about `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
X = np.array([1.0, 0.0, 0.0])   # default flexion axis


def test_identity_relative_is_zero_degrees():
    """No knee rotation -> 0 deg."""
    assert abs(knee_flexion_angle(IDENTITY)) < 1e-9


def test_flexion_is_positive():
    """
    SIGN CONVENTION. A pure +30 deg bend about the flexion axis must read
    +30, not -30. This single assertion fixes the sign for the whole module.
    """
    rel = knee_relative(quat_about_axis(X, 0.0), quat_about_axis(X, 30.0))
    ang = knee_flexion_angle(rel)
    assert ang > 0, f"flexion must be positive, got {ang}"
    assert abs(ang - 30.0) < 1e-6, f"expected +30, got {ang}"


def test_hyperextension_is_negative():
    """The other side of the sign convention: a -10 deg rotation reads negative."""
    rel = knee_relative(quat_about_axis(X, 0.0), quat_about_axis(X, -10.0))
    ang = knee_flexion_angle(rel)
    assert ang < 0, f"hyperextension must be negative, got {ang}"
    assert abs(ang + 10.0) < 1e-6, f"expected -10, got {ang}"


def test_axis_default_is_plus_x():
    """The documented default equals passing the x-axis explicitly."""
    rel = knee_relative(quat_about_axis(X, 0.0), quat_about_axis(X, 42.0))
    assert np.isclose(knee_flexion_angle(rel),
                      knee_flexion_angle(rel, axis=(1.0, 0.0, 0.0)))
    assert tuple(DEFAULT_FLEXION_AXIS) == (1.0, 0.0, 0.0)


def test_axis_parameter_governs_sign():
    """
    Reversing the flexion axis negates the reading. This is WHY the axis is a
    parameter and not a constant: the real mounting direction sets the sign, so
    it must be a pinnable calibration input.
    """
    rel = knee_relative(quat_about_axis(X, 0.0), quat_about_axis(X, 30.0))
    ang_plus = knee_flexion_angle(rel, axis=(1.0, 0.0, 0.0))
    ang_minus = knee_flexion_angle(rel, axis=(-1.0, 0.0, 0.0))
    assert np.isclose(ang_plus, -ang_minus), f"{ang_plus} vs {ang_minus}"


def test_isolates_flexion_from_off_axis_rotation():
    """Swing-twist: an added twist about another axis must not inflate flexion."""
    q_shank = multiply(quat_about_axis(X, 30.0), quat_about_axis([0, 1, 0], 15.0))
    rel = knee_relative(quat_about_axis(X, 0.0), q_shank)
    ang = knee_flexion_angle(rel)
    assert abs(ang - 30.0) < 2.0, f"flexion should stay ~30, got {ang}"


def test_invariant_to_quaternion_double_cover():
    """q and -q are the same rotation, so the angle must be identical."""
    rel = knee_relative(quat_about_axis(X, 0.0), quat_about_axis(X, 55.0))
    assert np.isclose(knee_flexion_angle(rel), knee_flexion_angle(-rel))


def test_vectorized_matches_scalar():
    """Whole-session array path equals looping the scalar path."""
    angs = np.array([0, 10, -5, 45, 90, 120], dtype=float)
    q_thigh = np.tile(quat_about_axis(X, 0.0), (len(angs), 1))
    q_shank = np.array([quat_about_axis(X, a) for a in angs])
    rel = knee_relative(q_thigh, q_shank)
    got = knee_flexion_angle(rel)
    assert np.allclose(got, angs, atol=1e-6), f"{got} vs {angs}"


def test_full_chain_with_calibration_offset():
    """
    Integration: raw segments carry a mounting offset captured as the neutral;
    after remove_offset the flexion reads the true bend. Proves joint_angles
    plugs into the relative_orientation chain end to end.
    """
    mount = quat_about_axis([0.2, 1.0, 0.3], 18.0)          # sensor mounting offset
    q_thigh = quat_about_axis(X, 0.0)
    q_shank_neutral = multiply(mount, quat_about_axis(X, 0.0))
    q_shank_flexed = multiply(mount, quat_about_axis(X, 40.0))

    q_neutral = knee_relative(q_thigh, q_shank_neutral)     # captured straight-leg
    rel = remove_offset(knee_relative(q_thigh, q_shank_flexed), q_neutral)
    assert abs(knee_flexion_angle(rel) - 40.0) < 1e-6, knee_flexion_angle(rel)


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
