"""
test_relative_orientation.py
Known-answer tests for the knee relative-orientation layer. Every case is a
rotation whose result we can work out by hand, so a pass is real evidence the
layer is correct — the same standard as test_quaternion_math.py.

Run:  python -m pytest test_relative_orientation.py -v
  or: python test_relative_orientation.py
"""

import numpy as np

from biomechanics.quaternion_math import multiply, normalize
from biomechanics.relative_orientation import (
    knee_relative, remove_offset, canonicalize,
)


def quat_about_axis(axis, deg):
    """Build a quaternion for a rotation of `deg` degrees about `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
X = np.array([1.0, 0.0, 0.0])   # knee flexion axis convention


def test_knee_relative_identity_when_segments_equal():
    """Thigh and shank in the same orientation -> no knee rotation (identity)."""
    q = quat_about_axis(X, 41.0)
    rel = knee_relative(q, q)
    assert np.allclose(canonicalize(rel), IDENTITY, atol=1e-9), rel


def test_knee_relative_pure_flexion_is_the_shank_rotation():
    """Neutral thigh, shank flexed 30 deg about X -> relative == that 30 deg rotation."""
    q_thigh = quat_about_axis(X, 0.0)
    q_shank = quat_about_axis(X, 30.0)
    rel = knee_relative(q_thigh, q_shank)
    assert np.allclose(rel, quat_about_axis(X, 30.0), atol=1e-9), rel


def test_knee_relative_invariant_to_whole_leg_world_rotation():
    """
    The product-grade property, checked at the QUATERNION level (before any
    angle is taken): apply a common world rotation to both segments and the
    knee relative quaternion is exactly unchanged.
    """
    common = quat_about_axis([0.3, 1.0, 0.5], 65.0)
    q_thigh = quat_about_axis(X, 0.0)
    q_shank = quat_about_axis(X, 30.0)
    rel_plain = knee_relative(q_thigh, q_shank)
    rel_world = knee_relative(multiply(common, q_thigh), multiply(common, q_shank))
    assert np.allclose(rel_plain, rel_world, atol=1e-9), f"{rel_plain} vs {rel_world}"


def test_remove_offset_identity_neutral_is_passthrough():
    """Before calibration (neutral == identity), the reading passes through unchanged."""
    q_rel = quat_about_axis(X, 30.0)
    out = remove_offset(q_rel, IDENTITY)
    assert np.allclose(out, normalize(q_rel), atol=1e-12), out


def test_remove_offset_cancels_its_own_neutral():
    """A reading equal to the captured neutral maps to identity -> 0 deg."""
    q_neutral = quat_about_axis([0.1, 0.2, 1.0], 12.0)   # arbitrary mounting offset
    out = remove_offset(q_neutral, q_neutral)
    assert np.allclose(canonicalize(out), IDENTITY, atol=1e-9), out


def test_remove_offset_recovers_the_extra_rotation():
    """
    If the raw reading is the neutral followed by an extra flexion,
    q_rel = q_neutral (x) q_extra, then removing the offset returns q_extra.
    """
    q_neutral = quat_about_axis([0.2, 1.0, 0.3], 20.0)
    q_extra = quat_about_axis(X, 35.0)
    q_rel = multiply(q_neutral, q_extra)
    out = remove_offset(q_rel, q_neutral)
    assert np.allclose(canonicalize(out), canonicalize(q_extra), atol=1e-9), out


def test_canonicalize_forces_positive_w_and_preserves_rotation():
    """canonicalize(-q) == canonicalize(q), and the representative has w >= 0."""
    q = quat_about_axis([0.4, -0.7, 0.5], 200.0)   # >180 deg so raw w < 0
    c = canonicalize(q)
    assert c[0] >= 0.0, c
    assert np.allclose(canonicalize(-q), c, atol=1e-12), (canonicalize(-q), c)


def test_remove_offset_broadcasts_single_neutral_over_batch():
    """One captured neutral applied across a whole session of readings."""
    angs = np.array([0, 15, 45, 90, 120])
    q_neutral = quat_about_axis([0.3, 0.2, 1.0], 8.0)
    q_rel = np.array([multiply(q_neutral, quat_about_axis(X, a)) for a in angs])
    out = remove_offset(q_rel, q_neutral)                 # (5,4)
    expect = np.array([quat_about_axis(X, a) for a in angs])
    assert np.allclose(canonicalize(out), canonicalize(expect), atol=1e-9), out


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
