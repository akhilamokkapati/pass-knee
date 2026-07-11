"""
quaternion_math.py
PASS Biomechanics - quaternion primitives.

Small, pure, dependency-light quaternion helpers used by the orientation and
joint-angle modules. Kept separate so each function can be unit-tested in
isolation against rotations whose answers we can hand-check.

Quaternion convention throughout PASS: (w, x, y, z), unit norm, Hamilton
product, right-handed. This matches the BNO085 game-rotation-vector output
(real, i, j, k) and the synthetic source.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm. Works on (4,) or (N,4)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n


def conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (= inverse for unit quaternions). (4,) or (N,4)."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b. Supports broadcasting over leading N axis."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def relative(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """
    Orientation of frame B expressed relative to frame A: q_a^-1 (x) q_b.

    In PASS terms: pass thigh as A and shank as B, and you get the rotation
    that carries the thigh frame onto the shank frame - i.e. the knee joint
    rotation, independent of how the whole leg is oriented in the world.
    """
    return multiply(conjugate(normalize(q_a)), normalize(q_b))


def angle_about_axis(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Signed rotation angle (degrees) of quaternion q about a given unit axis,
    using swing-twist decomposition.

    Why this and not 2*acos(w): the total quaternion angle mixes flexion with
    any ab/adduction or internal rotation. For a knee we want ONLY the flexion
    component - the rotation about the mediolateral axis. Swing-twist isolates
    exactly the 'twist' about the chosen axis and discards the rest.

    axis : (3,) unit vector of the flexion axis in the joint frame.
    Returns scalar (for (4,) q) or (N,) array.
    """
    q = normalize(np.asarray(q, dtype=float))
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # vector (xyz) part of q
    v = q[..., 1:]
    # projection of the rotation-vector part onto the axis
    proj = np.tensordot(v, axis, axes=([-1], [0]))   # (...,) dot products
    # twist quaternion = (w, proj*axis), then renormalized
    tw_w = q[..., 0]
    tw_vec = proj[..., None] * axis
    twist = np.concatenate(
        [tw_w[..., None], tw_vec], axis=-1
    )
    twist = normalize(twist)

    # signed angle of the twist about the axis
    # angle = 2*atan2( sign * |vec part| , w )
    w = np.clip(twist[..., 0], -1.0, 1.0)
    vec = twist[..., 1:]
    sin_half = np.linalg.norm(vec, axis=-1)
    # sign from whether vec aligns with +axis or -axis
    sign = np.sign(np.tensordot(vec, axis, axes=([-1], [0])))
    sign = np.where(sign == 0, 1.0, sign)
    angle = 2.0 * np.arctan2(sin_half * sign, w)
    return np.degrees(angle)


def to_euler_pitch(q: np.ndarray) -> np.ndarray:
    """
    Convenience: extract pitch (rotation about y) in degrees from (w,x,y,z).
    Provided for compatibility/plotting; the joint-angle path uses
    angle_about_axis, not this.
    """
    q = normalize(np.asarray(q, dtype=float))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    return np.degrees(np.arcsin(sinp))


if __name__ == "__main__":
    print("PASS quaternion_math ready")