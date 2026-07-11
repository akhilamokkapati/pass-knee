"""
test_quaternion_math.py
Verify the quaternion primitives against rotations whose answers we KNOW
by hand. If these pass, the knee-angle math is trustworthy - this is the
evidence you can point to if anyone questions the software's correctness.

Run:  python -m pytest test_quaternion_math.py -v
  or: python test_quaternion_math.py
"""

import numpy as np

from biomechanics.quaternion_math import (
    normalize, conjugate, multiply, relative, angle_about_axis,
)


def quat_about_axis(axis, deg):
    """Build a quaternion for a rotation of `deg` degrees about `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


X = np.array([1.0, 0.0, 0.0])   # we treat x as the knee flexion axis


def test_identity_relative_is_zero():
    """Two identical orientations -> 0 deg knee angle."""
    q = quat_about_axis(X, 37.0)
    rel = relative(q, q)
    ang = angle_about_axis(rel, X)
    assert abs(ang) < 1e-6, f"expected 0, got {ang}"


def test_pure_flexion_reads_back_exactly():
    """Shank rotated 30 deg about flexion axis vs neutral thigh -> 30 deg."""
    q_thigh = quat_about_axis(X, 0.0)
    q_shank = quat_about_axis(X, 30.0)
    rel = relative(q_thigh, q_shank)
    ang = angle_about_axis(rel, X)
    assert abs(ang - 30.0) < 1e-6, f"expected 30, got {ang}"


def test_flexion_survives_whole_leg_orientation():
    """
    The key product-grade property: if the WHOLE leg is rotated in the world
    (person turns, sensor mounted at an angle), the knee angle must be
    unchanged. Apply a big common rotation to both segments; flexion stays 30.
    """
    common = quat_about_axis([0.3, 1.0, 0.5], 65.0)  # arbitrary world rotation
    q_thigh = multiply(common, quat_about_axis(X, 0.0))
    q_shank = multiply(common, quat_about_axis(X, 30.0))
    rel = relative(q_thigh, q_shank)
    ang = angle_about_axis(rel, X)
    assert abs(abs(ang) - 30.0) < 1e-6, f"expected 30, got {ang}"


def test_flexion_ignores_off_axis_rotation():
    """
    Swing-twist should report ONLY flexion. Add a twist about a different
    axis to the shank; the flexion reading about X must stay ~30, not inflate.
    """
    q_thigh = quat_about_axis(X, 0.0)
    q_shank = multiply(quat_about_axis(X, 30.0), quat_about_axis([0, 1, 0], 15.0))
    rel = relative(q_thigh, q_shank)
    ang = angle_about_axis(rel, X)
    assert abs(abs(ang) - 30.0) < 2.0, f"flexion should stay ~30, got {ang}"


def test_conjugate_and_multiply_identity():
    """q (x) q^-1 == identity quaternion."""
    q = normalize(np.array([0.5, 0.2, -0.7, 0.4]))
    ident = multiply(q, conjugate(q))
    assert np.allclose(ident, [1, 0, 0, 0], atol=1e-9)


def test_vectorized_matches_scalar():
    """Array path must equal looping the scalar path (used for whole sessions)."""
    angs = np.array([0, 10, 45, 90, 120, 5])
    q_thigh = np.tile(quat_about_axis(X, 0.0), (len(angs), 1))
    q_shank = np.array([quat_about_axis(X, a) for a in angs])
    rel = relative(q_thigh, q_shank)
    got = np.abs(angle_about_axis(rel, X))
    assert np.allclose(got, angs, atol=1e-6), f"{got} vs {angs}"


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