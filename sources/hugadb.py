"""
hugadb.py
PASS data source - HuGaDB offline dataset (real human lower-limb IMU).

Second source behind the SAME interface as SyntheticSource (stream / get_data,
Packet / Capture). Its purpose is to validate the knee engine on REAL human
motion, not just synthetic rotations.

WHY MADGWICK LIVES HERE (and only here)
---------------------------------------
The BNO085 fuses accel+gyro into quaternions ON-CHIP, so the live path never
runs a fusion filter. HuGaDB is RAW accelerometer + gyroscope with no on-chip
fusion, so this offline path must fuse it itself. We use the `ahrs` library's
Madgwick AHRS (IMU form: gyro + accel), which outputs scalar-first (w,x,y,z)
quaternions - already our convention. Fusion runs once at load and is cached.

NO GROUND-TRUTH ANGLE
---------------------
HuGaDB has no labelled knee angle, so knee_angle_deg is set to NaN (honest: no
reference). The engine still computes the angle from the fused quaternions
downstream. HuGaDB's validation is therefore PHYSIOLOGICAL PLAUSIBILITY on real
sit-to-stand motion, not a numerical round-trip.

FLEXION AXIS (mounting-specific, determined empirically)
--------------------------------------------------------
HuGaDB's shin/thigh IMUs are mounted in the dataset's own frame, so knee flexion
is NOT about our synthetic +x default. A spike on HuGaDB_v2_various_01_00 (a
sit-to-stand recording), using a standing window as the straight-leg neutral,
gave the de-neutraled relative rotation vector during sitting as
[x, y, z] = [-11, -59, -1.7] deg - i.e. the flexion is dominated by -Y (total
knee angle 60.1 deg, of which the Y-twist is ~59 deg; the -11 on X is off-axis
ab/adduction that swing-twist correctly discards). Standing read 2.7 deg, sitting
60.1 deg. Hence HUGADB_FLEXION_AXIS = (0, -1, 0), so sitting reads as positive
flexion. Override via the axis argument to the engine if a file differs.

Units (HuGaDB v2, int16): accel +/-2g, gyro +/-2000 deg/s. Sample rate ~56.3 Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from ahrs.filters import Madgwick

from .schema import Packet, Capture

# --- HuGaDB file layout (0-indexed fields incl. the leading pandas index) ---
# 1..6 right_foot, 7..12 right_shin, 13..18 right_thigh, ... 39 activity
_SHIN_ACC = (7, 8, 9)
_SHIN_GYR = (10, 11, 12)
_THIGH_ACC = (13, 14, 15)
_THIGH_GYR = (16, 17, 18)
_ACTIVITY = 39

# int16 -> physical units
_ACC_G = 2.0 / 32768.0                       # +/-2 g  (direction only matters to Madgwick)
_GYR_RADS = np.deg2rad(2000.0 / 32768.0)     # +/-2000 deg/s -> rad/s

HUGADB_SAMPLE_RATE_HZ = 56.3                  # HuGaDB nominal rate
HUGADB_FLEXION_AXIS = (0.0, -1.0, 0.0)        # empirical, see module docstring


class HuGaDBSource:
    """
    Load one HuGaDB v2 CSV, fuse the right thigh & shin IMUs to quaternions with
    Madgwick, and expose them through the standard PASS source interface.

    filepath : path to a HuGaDB_v2_*.csv file.
    fs_hz    : sample rate for fusion and timestamps (HuGaDB ~56.3 Hz).
    """

    def __init__(self, filepath: str | Path, fs_hz: float = HUGADB_SAMPLE_RATE_HZ):
        self.filepath = Path(filepath)
        self.fs_hz = float(fs_hz)

        num_cols = _SHIN_ACC + _SHIN_GYR + _THIGH_ACC + _THIGH_GYR
        data = np.genfromtxt(self.filepath, delimiter=",", skip_header=1,
                             usecols=num_cols, dtype=float)
        activity = np.genfromtxt(self.filepath, delimiter=",", skip_header=1,
                                usecols=(_ACTIVITY,), dtype=str)
        self.activity = np.char.strip(activity)

        shin_acc = data[:, 0:3] * _ACC_G
        shin_gyr = data[:, 3:6] * _GYR_RADS
        thigh_acc = data[:, 6:9] * _ACC_G
        thigh_gyr = data[:, 9:12] * _GYR_RADS

        self.quat_thigh = self._fuse(thigh_gyr, thigh_acc)
        self.quat_shank = self._fuse(shin_gyr, shin_acc)
        self.n = self.quat_thigh.shape[0]

    def _fuse(self, gyr_rads: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """Madgwick IMU fusion of one segment -> (N,4) w-first quaternions."""
        return Madgwick(gyr=gyr_rads, acc=acc, frequency=self.fs_hz).Q

    # --- source interface --------------------------------------------------

    def _packet(self, i: int) -> Packet:
        return Packet(
            seq=i,
            t_ms=int(round(i * 1000.0 / self.fs_hz)),
            knee_angle_deg=float("nan"),          # no reference angle in HuGaDB
            quat_thigh=self.quat_thigh[i],
            quat_shank=self.quat_shank[i],
        )

    def stream(self) -> Iterator[Packet]:
        """Yield the recording one packet at a time (finite: the file ends)."""
        for i in range(self.n):
            yield self._packet(i)

    def get_data(self, duration_s: float | None = None) -> Capture:
        """Return a Capture of arrays. duration_s=None returns the whole file;
        otherwise the first duration_s seconds. Includes the activity labels."""
        n = self.n if duration_s is None else min(self.n, int(round(duration_s * self.fs_hz)))
        idx = np.arange(n)
        return Capture(
            seq=idx,
            t_ms=np.round(idx * 1000.0 / self.fs_hz).astype(int),
            knee_angle_deg=np.full(n, np.nan),
            quat_thigh=self.quat_thigh[:n],
            quat_shank=self.quat_shank[:n],
            activity=self.activity[:n],
        )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).resolve().parents[1] / "hugadb" / "HuGaDB_v2_various_01_00.csv")
    src = HuGaDBSource(path)
    cap = src.get_data()
    print(f"PASS hugadb ready - {cap.seq.size} samples, "
          f"activities={sorted(set(cap.activity))}")
