# PASS: Knee Module

**Patient Assessment Sensing System (PASS)** is a modular lower-limb wearable that
makes physiotherapy quantitative: it captures objective knee kinematics, gives
patients real-time feedback, and produces clinically useful data for physiotherapists.

PASS is **not tied to any single condition**. The same knee-flexion measurements apply
across many rehabilitation contexts: post-surgical recovery (e.g. ACL reconstruction,
knee replacement), sports and injury rehab, general musculoskeletal physiotherapy, and
neurological rehabilitation (including post-stroke).

This repository is the **knee module**: a quaternion-native engine that turns two
IMU orientations (thigh + shank) into a validated knee-flexion signal and the
clinical metrics derived from it.

> Course project for SUTD 30.007 Engineering Design Innovation, Team 01 *Super Strokers*.
> This repo is the knee subsystem (sole software author).

## What it measures

From two IMUs on the thigh and shank, this module produces a knee-flexion angle and
these **defensible, validated** metrics:

- **Range of motion (ROM)**: core metric, quaternion-native, target **±2.5°**
- **Angular velocity**: derivative of the angle
- **Repetitions**: adaptive peak detection with confidence indicators
- **Max flexion / max extension**
- **Rep consistency**: variation across rep peaks

Loading, gait phases and muscle activity are *out of scope* for the knee module; they
belong to the feet (FSR), hip and actuation modules of the full PASS platform.

## Architecture

A **data-source abstraction**: synthetic, live-serial (BNO085) and the HuGaDB offline
dataset all expose the *same* interface and feed the *same* biomechanics engine.
Only the source changes; the engine never does.

```
BNO085 -> XIAO ESP32-C3 -> serial -> source -> biomechanics engine -> metrics -> plot / dashboard
```

Knee angle is computed **quaternion-native** via **swing-twist decomposition** to
isolate pure flexion about the knee's mediolateral axis (no Euler pitch-subtraction,
no gimbal lock). Straight-leg calibration removes the mounting offset.

## Layout

| Path | What it is |
|------|------------|
| `biomechanics/` | Pure quaternion math: primitives, relative orientation, joint angles |
| `calibrate.py`, `axis_calibration.py` | Straight-leg zero + live flexion-axis measurement |
| `filters.py` | Butterworth low-pass (zero-phase offline + causal streaming) |
| `sources/` | `synthetic`, `serial_source`, `hugadb` behind one `stream()` / `get_data()` interface |
| `metrics.py`, `repetitions.py` | ROM, velocity, max flex/ext, rep detection |
| `run_capture.py`, `live_plot.py` | Capture-to-graph accuracy report + real-time scrolling plot |
| `dashboard/` | Streamlit dashboard |
| `firmware/knee_imu_serial.ino` | XIAO ESP32-C3 + 2x BNO085 sketch emitting the serial CSV contract |
| `test_*.py` | Test-first known-answer / property tests |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest -q
```

### HuGaDB dataset (optional)

The HuGaDB v2 CSVs are **not** included in this repo (third-party dataset, ~230 MB).
To run the HuGaDB source and its tests, download HuGaDB and place the v2 CSV files
in `./hugadb/`. Everything else (synthetic + serial paths) runs without it.

## Hardware

- **Compute:** Seeed XIAO ESP32-C3 (S3 compatible; SDA=D4, SCL=D5)
- **Sensors:** 2x BNO085 IMU (thigh + shank), I2C at 100 kHz, addresses 0x4A / 0x4B
- The BNO085 outputs fused quaternions on-chip (game rotation vector), so no host-side
  fusion is needed on the live path. HuGaDB (raw accel/gyro) is the only Madgwick-fused path.

## Status

Engine, sources, calibration, filtering, metrics, firmware and live plot are built and
**test-first verified**. Validated on real HuGaDB sit-to-stand (standing ~0°, sitting
~61°, peak ~64°). Live-hardware validation follows once the BNO085 sensors are mounted.
