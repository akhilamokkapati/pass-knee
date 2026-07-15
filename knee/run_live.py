r"""
run_live.py
PASS knee module - live run on the real BNO085 serial link (hardware day).

Sequences the already-tested pieces into one guided flow:

    SerialSource (live COM port)
      -> straight-leg zero        (calibrate.calibrate_from_quaternions)
      -> measured flexion axis    (axis_calibration.calibrate_flexion_axis)
      -> live scrolling knee angle (live_plot.LiveKneePlot, causal low-pass)

WHY THE TWO CALIBRATION POSES
-----------------------------
The two IMUs sit on the limb at an arbitrary orientation, so we do NOT assume the
mounting or the flexion axis - we MEASURE them:
  * straight-leg pose  -> the neutral relative orientation that reads 0 deg;
  * bent pose          -> the neutral->bent rotation whose axis IS the flexion axis.
That is what makes a straight leg read ~0 and a real bend track correctly.

All the math is imported and unit-tested. This file only captures the two windows
and hands the results to the plot; it adds no new engine logic.

USAGE (run in a terminal, with the Arduino Serial Monitor CLOSED so the port is free):
    cd knee
    ..\.venv\Scripts\python run_live.py --port COM5
    ..\.venv\Scripts\python run_live.py --port COM5 --hold 3
"""

from __future__ import annotations

import argparse

import numpy as np

from sources.serial_source import SerialSource
from calibrate import calibrate_from_quaternions
from axis_calibration import calibrate_flexion_axis
from live_plot import LiveKneePlot
from filters import DEFAULT_CUTOFF_HZ


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PASS knee live run on the real BNO085 serial link")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200, help="baud (matches firmware)")
    ap.add_argument("--hold", type=float, default=2.5,
                    help="seconds to hold each calibration pose")
    ap.add_argument("--window", type=float, default=10.0, help="live scroll window (s)")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF_HZ,
                    help="low-pass cutoff (Hz)")
    ap.add_argument("--fps", type=float, default=30.0, help="plot redraw rate")
    args = ap.parse_args()

    source = SerialSource(port=args.port, baud=args.baud)

    print("# PASS knee live run")
    print(f"# port {args.port} @ {args.baud} baud. "
          "Make sure the Arduino Serial Monitor is CLOSED (it locks the port).")

    # 1. Straight-leg pose: this window is both the zero (q_neutral) and the
    #    neutral for the axis measurement.
    input("\n>> Stand with your leg STRAIGHT and hold still, then press Enter... ")
    straight = source.get_data(args.hold)
    neu = calibrate_from_quaternions(straight.quat_thigh, straight.quat_shank)
    print(f"#   straight-leg captured: n={neu.n_samples}  "
          f"residual_rms={neu.residual_rms_deg:.2f} deg  (smaller = held stiller)")

    # 2. Bent pose: the neutral->bent rotation gives the flexion axis.
    input(">> Now BEND your knee to about 60 deg and hold still, then press Enter... ")
    bent = source.get_data(args.hold)
    axis_cal = calibrate_flexion_axis(straight.quat_thigh, straight.quat_shank,
                                      bent.quat_thigh, bent.quat_shank)
    print(f"#   flexion axis = {np.round(axis_cal.flexion_axis, 3)}")
    print(f"#   bend {axis_cal.bend_angle_deg:.1f} deg  "
          f"confidence {axis_cal.axis_confidence:.2f}  reliable={axis_cal.reliable}")
    if not axis_cal.reliable:
        print("#   WARN weak calibration: bend further (>= 30 deg), hold stiller, and "
              "keep the motion about ONE axis. Ctrl-C and re-run for a clean axis.")

    # 3. Live scrolling knee angle, zeroed and on the measured axis, causal-filtered.
    print("\n# Opening the live plot. Bend and extend your knee to see it track.")
    print("# Close the plot window to stop.")
    LiveKneePlot(source, window_s=args.window, target_fps=args.fps,
                 cutoff_hz=args.cutoff, axis=axis_cal.flexion_axis,
                 q_neutral=neu.q_neutral).run()


if __name__ == "__main__":
    main()
