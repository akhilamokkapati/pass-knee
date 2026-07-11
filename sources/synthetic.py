"""
synthetic.py
PASS data source — synthetic knee kinematics.

Purpose: develop and validate the whole chain before the BNO085 hardware
arrives, and give us a source whose answer we KNOW exactly.

The design point that makes this trustworthy is an INDEPENDENT FORWARD MODEL:

  1. choose a ground-truth flexion angle theta(t) from a smooth rep profile;
  2. construct the two segment quaternions directly from theta (forward model),
     so their relative rotation IS theta about the flexion axis;
  3. set knee_angle_deg = theta, the ground truth — the engine is never called
     here.

The biomechanics engine runs the other direction (quaternions -> angle). Because
the emitted knee_angle_deg comes from the forward model and NOT from the engine,
a round-trip check (engine recovers knee_angle_deg from the quaternions) compares
two independent implementations — it is not the engine graded against itself.

Interface (shared by every PASS source):
  stream()           -> infinite generator of Packet, one sample at a time
  get_data(duration) -> Capture of arrays

Quaternion convention (inherited): (w, x, y, z), unit norm, Hamilton, right-handed.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from biomechanics.quaternion_math import multiply, normalize
from biomechanics.joint_angles import DEFAULT_FLEXION_AXIS
from .schema import Packet, Capture


def axis_angle_quat(axis, deg: float) -> np.ndarray:
    """Quaternion (w,x,y,z) for a rotation of `deg` degrees about `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(deg) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


class SyntheticSource:
    """
    Emits synthetic thigh/shank quaternions for a smooth knee flexion profile,
    plus the independently-known ground-truth angle.

    rate_hz       : sample rate (BNO085 game-rotation-vector is ~100 Hz).
    min/max_angle : flexion sweep of the profile, in degrees.
    rep_period_s  : duration of one flexion-extension cycle.
    leg_pose      : whole-leg world orientation applied to BOTH segments, so the
                    quaternions look like real world-oriented IMU output while
                    their relative rotation stays exactly the flexion. Defaults
                    to a non-trivial pose (never identity) to exercise the
                    engine's world-invariance.
    axis          : flexion axis used to BUILD the shank rotation; kept equal to
                    the engine default so a correct engine recovers theta.
    noise_deg     : optional Gaussian noise added to the angle used to build the
                    quaternions ONLY. knee_angle_deg stays the clean ground truth,
                    which is the whole point of the independent forward model.
    seed          : RNG seed for reproducible noise.
    """

    def __init__(
        self,
        rate_hz: float = 100.0,
        min_angle_deg: float = 0.0,
        max_angle_deg: float = 60.0,
        rep_period_s: float = 2.0,
        leg_pose: np.ndarray | None = None,
        axis=DEFAULT_FLEXION_AXIS,
        noise_deg: float = 0.0,
        seed: int | None = None,
    ):
        self.rate_hz = float(rate_hz)
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)
        self.rep_period_s = float(rep_period_s)
        # arbitrary but fixed non-identity standing pose (person turned + tilted)
        if leg_pose is None:
            leg_pose = axis_angle_quat([0.3, 1.0, 0.2], 55.0)
        self.leg_pose = normalize(np.asarray(leg_pose, dtype=float))
        self.axis = np.asarray(axis, dtype=float)
        self.noise_deg = float(noise_deg)
        self._rng = np.random.default_rng(seed)

    # --- forward model -----------------------------------------------------

    def _angle_at(self, t_s: float) -> float:
        """Ground-truth flexion angle at time t (smooth min<->max rep cycle)."""
        span = self.max_angle_deg - self.min_angle_deg
        phase = 2.0 * np.pi * t_s / self.rep_period_s
        return self.min_angle_deg + span * (1.0 - np.cos(phase)) / 2.0

    def _segments(self, build_angle_deg: float):
        """Construct (q_thigh, q_shank) whose relative rotation is build_angle
        about the flexion axis, expressed in the fixed leg pose."""
        flex = axis_angle_quat(self.axis, build_angle_deg)
        q_thigh = self.leg_pose
        q_shank = multiply(self.leg_pose, flex)
        return q_thigh, q_shank

    def _packet(self, seq: int) -> Packet:
        t_s = seq / self.rate_hz
        theta = self._angle_at(t_s)                       # ground truth
        build_angle = theta
        if self.noise_deg > 0.0:
            build_angle = theta + self._rng.normal(0.0, self.noise_deg)
        q_thigh, q_shank = self._segments(build_angle)
        return Packet(
            seq=seq,
            t_ms=int(round(seq * 1000.0 / self.rate_hz)),
            knee_angle_deg=theta,                         # NOT engine-derived
            quat_thigh=q_thigh,
            quat_shank=q_shank,
        )

    # --- source interface --------------------------------------------------

    def stream(self, realtime: bool = False) -> Iterator[Packet]:
        """Yield packets one at a time, forever. Set realtime=True to pace at
        rate_hz (for the live plot); default is as-fast-as-possible for tests."""
        seq = 0
        period = 1.0 / self.rate_hz
        while True:
            yield self._packet(seq)
            if realtime:
                time.sleep(period)
            seq += 1

    def get_data(self, duration_s: float) -> Capture:
        """Return a Capture of arrays spanning duration_s (starting at t=0)."""
        n = int(round(duration_s * self.rate_hz))
        packets = [self._packet(i) for i in range(n)]
        return Capture(
            seq=np.array([p.seq for p in packets], dtype=int),
            t_ms=np.array([p.t_ms for p in packets], dtype=int),
            knee_angle_deg=np.array([p.knee_angle_deg for p in packets], dtype=float),
            quat_thigh=np.array([p.quat_thigh for p in packets], dtype=float),
            quat_shank=np.array([p.quat_shank for p in packets], dtype=float),
        )


if __name__ == "__main__":
    src = SyntheticSource()
    cap = src.get_data(2.0)   # one full flexion-extension rep
    print(f"PASS synthetic source ready - {len(cap.seq)} samples, "
          f"angle range {cap.knee_angle_deg.min():.1f}..{cap.knee_angle_deg.max():.1f} deg")
