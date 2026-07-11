"""
test_metrics.py
Known-answer tests for the direct angle metrics, plus validation on the
synthetic source and real HuGaDB sit-to-stand ROM.

Run:  python -m pytest test_metrics.py -v
"""

import numpy as np

from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle
from calibrate import calibrate_from_quaternions
from sources.synthetic import SyntheticSource
from sources.hugadb import HUGADB_FLEXION_AXIS
from metrics import (
    range_of_motion, max_flexion, max_extension,
    angular_velocity, peak_angular_velocity, summarize,
)


def test_rom_and_extremes_known_answer():
    a = np.array([0.0, 30.0, 60.0, 10.0])
    assert range_of_motion(a) == 60.0
    assert max_flexion(a) == 60.0
    assert max_extension(a) == 0.0


def test_max_extension_reports_hyperextension():
    a = np.array([-8.0, 0.0, 45.0])
    assert max_extension(a) == -8.0
    assert range_of_motion(a) == 53.0


def test_angular_velocity_constant_slope():
    """A linear ramp of 5 deg/s reads back as ~5 deg/s everywhere."""
    fs = 100.0
    t = np.arange(0, 2, 1 / fs)
    a = 5.0 * t                          # 5 deg per second
    v = angular_velocity(a, fs)
    assert np.allclose(v, 5.0, atol=1e-6)


def test_angular_velocity_sign_follows_motion():
    fs = 100.0
    up = angular_velocity(np.arange(0, 50, 0.5), fs)
    down = angular_velocity(np.arange(50, 0, -0.5), fs)
    assert np.all(up > 0) and np.all(down < 0)


def test_peak_angular_velocity_sine_known_answer():
    """For A*sin(2*pi*f*t), peak |velocity| = A*2*pi*f (deg/s)."""
    fs, A, f = 200.0, 40.0, 0.5
    t = np.arange(0, 4, 1 / fs)
    a = A * np.sin(2 * np.pi * f * t)
    assert np.isclose(peak_angular_velocity(a, fs), A * 2 * np.pi * f, rtol=2e-3)


def test_empty_signal_is_nan_not_crash():
    assert np.isnan(range_of_motion(np.array([])))
    assert np.isnan(peak_angular_velocity(np.array([]), 100.0))
    assert summarize(np.array([]), 100.0).n_samples == 0


def test_rom_on_synthetic_source():
    """A 0..60 deg synthetic profile has ROM ~60 deg."""
    cap = SyntheticSource(min_angle_deg=0.0, max_angle_deg=60.0).get_data(4.0)
    ang = knee_flexion_angle(knee_relative(cap.quat_thigh, cap.quat_shank))
    assert abs(range_of_motion(ang) - 60.0) < 0.5


def test_rom_on_hugadb_sit_to_stand(hugadb_source):
    """Real sit-to-stand ROM lands in a plausible band (~55-80 deg)."""
    cap = hugadb_source.get_data()
    stand = np.where(cap.activity == "standing")[0]
    neutral = calibrate_from_quaternions(
        cap.quat_thigh[stand], cap.quat_shank[stand]).q_neutral
    ang = knee_flexion_angle(remove_offset(knee_relative(cap.quat_thigh, cap.quat_shank),
                                           neutral), axis=HUGADB_FLEXION_AXIS)
    m = summarize(ang, fs_hz=hugadb_source.fs_hz)
    assert 55.0 < m.range_of_motion_deg < 80.0, m.range_of_motion_deg
    assert 55.0 < m.max_flexion_deg < 75.0, m.max_flexion_deg


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"run via: pytest {__file__}")
