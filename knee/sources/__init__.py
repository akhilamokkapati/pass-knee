"""
sources - PASS data-source abstraction.

Every source (synthetic now; serial live sensor and HuGaDB offline later) exposes
the SAME interface and emits the SAME packet schema, so the biomechanics engine
never changes - only where the data comes from does.

  stream()            -> yields Packet objects one at a time (real-time / live plot)
  get_data(duration)  -> returns a Capture of arrays (capture + offline analysis)

Packet schema (matches firmware): seq, t_ms, knee_angle_deg,
quat_thigh[4] (w,x,y,z), quat_shank[4] (w,x,y,z).
"""

from .schema import Packet, Capture
from .synthetic import SyntheticSource, axis_angle_quat
from .hugadb import HuGaDBSource
from .serial_source import SerialSource, parse_packet_line, format_packet_line

__all__ = [
    "Packet", "Capture", "SyntheticSource", "axis_angle_quat", "HuGaDBSource",
    "SerialSource", "parse_packet_line", "format_packet_line",
]
