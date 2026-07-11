"""
joint_angles.py
PASS Biomechanics - the knee flexion-angle metric.

This is the single, authoritative definition of the scalar the knee module
exists to produce: knee_angle_deg. It takes the clean knee relative quaternion
from relative_orientation.py and reduces it to one signed number using the
swing-twist extraction in quaternion_math.angle_about_axis. Nothing downstream
(ROM, velocity, reps, dashboards) redefines the flexion metric; they all read
it from here.

Two conventions are decided HERE and nowhere else:

  SIGN   Positive degrees == flexion (the knee bending); negative degrees ==
         hyperextension. By the right-hand rule this is a positive rotation of
         the shank frame about +axis relative to the thigh frame.

  AXIS   The mediolateral flexion axis is a NAMED PARAMETER, not a hardcoded
         constant, because the true axis depends on how the IMU physically sits
         on the limb - a mounting/calibration quantity we pin later. It defaults
         to +x (DEFAULT_FLEXION_AXIS). Reversing the axis negates the reading,
         which is exactly why the sign has to be a calibratable input.

Convention (inherited): (w, x, y, z), unit norm, Hamilton, right-handed.
Accepts (4,) -> scalar or (N,4) -> (N,).
"""

from __future__ import annotations

import numpy as np

from .quaternion_math import angle_about_axis
from .relative_orientation import canonicalize

# The default mediolateral flexion axis. Documented, overridable, and the one
# place a caller looks to see what "no calibration specified" assumes.
DEFAULT_FLEXION_AXIS = (1.0, 0.0, 0.0)


def knee_flexion_angle(q_rel: np.ndarray,
                       axis: np.ndarray = DEFAULT_FLEXION_AXIS) -> np.ndarray:
    """
    Signed knee flexion angle in degrees from the knee relative quaternion.

    q_rel : knee relative quaternion(s) from relative_orientation.knee_relative
            (optionally after remove_offset for calibration). (4,) or (N,4).
    axis  : unit flexion axis in the joint frame; defaults to +x. This is the
            calibratable mounting direction - its orientation sets the sign.

    Returns a scalar for a single quaternion, or (N,) for a batch. Positive is
    flexion, negative is hyperextension.

    q_rel is canonicalized (w >= 0) first so a sensor reporting -q instead of q
    - physically the same rotation - cannot flip the reported angle.
    """
    return angle_about_axis(canonicalize(q_rel), np.asarray(axis, dtype=float))


if __name__ == "__main__":
    print("PASS joint_angles ready")
