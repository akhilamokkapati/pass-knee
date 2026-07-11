"""
serial_source.py
PASS data source — live serial link to the XIAO + BNO085 firmware.

Built BEFORE the sensors arrive so hardware day is "flash, wire, run" rather than
"write a parser while debugging hardware". It parses the firmware's CSV packet
lines and exposes them through the SAME interface as every other PASS source.

FIRMWARE PACKET LINE (the contract the .ino must emit), 11 comma-separated
fields, newline-terminated:

    seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz

  seq            uint    sample counter
  t_ms           uint    device milliseconds
  knee_angle_deg float   the firmware's ON-DEVICE angle  (CROSS-CHECK ONLY)
  qt*            float   thigh quaternion (w,x,y,z), BNO085 game rotation vector
  qs*            float   shank quaternion (w,x,y,z)

QUATERNIONS ARE THE SOURCE OF TRUTH
-----------------------------------
The raw quaternions are emitted into the Capture so the biomechanics engine
recomputes the knee angle through the validated swing-twist path. The firmware's
on-device knee_angle_deg is carried along ONLY as a cross-check (it lands in the
Capture's knee_angle_deg reference slot); it is never what we trust. Comparing
engine-recovered vs firmware angle is a diagnostic, not an accuracy claim.

HARDWARE-DECOUPLED
------------------
The source reads from a line iterable. Tests and replay pass an in-memory list of
simulated firmware strings; real hardware passes a pyserial port, which is opened
LAZILY (so `import serial` and a device are only needed for an actual live run).
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from .schema import Packet, Capture

FIRMWARE_FIELDS = 11              # seq, t_ms, angle, 4 thigh, 4 shank
DEFAULT_BAUD = 115200             # XIAO ESP32-C3 USB CDC


def parse_packet_line(line: str) -> Packet | None:
    """Parse one firmware CSV line into a Packet, or return None if the line is
    blank, a comment, or malformed (wrong field count / non-numeric). Tolerant
    by design: a garbled line is skipped, never fatal."""
    if line is None:
        return None
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) != FIRMWARE_FIELDS:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    return Packet(
        seq=int(vals[0]),
        t_ms=int(vals[1]),
        knee_angle_deg=vals[2],                       # firmware angle: cross-check
        quat_thigh=np.array(vals[3:7], dtype=float),
        quat_shank=np.array(vals[7:11], dtype=float),
    )


def format_packet_line(packet: Packet) -> str:
    """Inverse of parse_packet_line: render a Packet as a firmware CSV line. Used
    to synthesize firmware lines from other sources for offline testing/replay."""
    q = lambda a: ",".join(f"{v:.6f}" for v in np.asarray(a, dtype=float))
    return (f"{int(packet.seq)},{int(packet.t_ms)},{packet.knee_angle_deg:.4f},"
            f"{q(packet.quat_thigh)},{q(packet.quat_shank)}")


class SerialSource:
    """
    Live serial source. Provide either `line_source` (any iterable of decoded
    str lines, for tests/replay) or `port` (a serial device path, opened lazily
    with pyserial for a real run).

    port         : e.g. "COM5" (Windows) / "/dev/ttyACM0". Ignored if line_source
                   is given.
    baud         : serial baud rate (XIAO USB CDC default 115200).
    line_source  : iterable of str lines; when set, no serial port is opened.
    """

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD,
                 line_source: Iterable[str] | None = None):
        self.port = port
        self.baud = int(baud)
        self.line_source = line_source
        self.n_malformed = 0          # lines seen that did not parse

    def _raw_lines(self) -> Iterator[str]:
        """Yield decoded lines from the replay iterable or a live serial port."""
        if self.line_source is not None:
            yield from self.line_source
            return
        if self.port is None:
            raise ValueError("SerialSource needs either a line_source or a port")
        import serial                                 # lazy: only for real hardware
        with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue                          # timeout with no data
                yield raw.decode("ascii", errors="ignore")

    def stream(self) -> Iterator[Packet]:
        """Yield packets one at a time; malformed lines are skipped and counted."""
        for line in self._raw_lines():
            pkt = parse_packet_line(line)
            if pkt is None:
                if line and line.strip() and not line.strip().startswith("#"):
                    self.n_malformed += 1
                continue
            yield pkt

    def get_data(self, duration_s: float | None = None) -> Capture:
        """Collect packets into a Capture. Stops when the line source ends or,
        for a live/timed capture, once duration_s of device time has elapsed.

        quat_thigh/quat_shank are the raw quaternions (engine input, the truth);
        knee_angle_deg holds the firmware's on-device angle as a cross-check.
        """
        packets: list[Packet] = []
        t0 = None
        for pkt in self.stream():
            if t0 is None:
                t0 = pkt.t_ms
            packets.append(pkt)
            if duration_s is not None and (pkt.t_ms - t0) >= duration_s * 1000.0:
                break

        return Capture(
            seq=np.array([p.seq for p in packets], dtype=int),
            t_ms=np.array([p.t_ms for p in packets], dtype=int),
            knee_angle_deg=np.array([p.knee_angle_deg for p in packets], dtype=float),
            quat_thigh=np.array([p.quat_thigh for p in packets], dtype=float).reshape(-1, 4),
            quat_shank=np.array([p.quat_shank for p in packets], dtype=float).reshape(-1, 4),
            activity=None,                            # serial has no activity labels
        )


if __name__ == "__main__":
    print("PASS serial_source ready — firmware line format:")
    print("  seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz")
