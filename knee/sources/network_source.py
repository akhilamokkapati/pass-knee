"""
network_source.py
PASS data source - wireless UDP link to the XIAO + BNO085 firmware.

Same packet contract as the wired link (sources/serial_source.py), just delivered
over WiFi instead of USB. The firmware broadcasts one CSV line per sample as a UDP
datagram; this source listens on the port and exposes the packets through the SAME
stream() / get_data() interface as every other PASS source, so run_live.py, the
dashboard and the engine all work unchanged.

FIRMWARE PACKET LINE (identical to the serial contract, parsed by the same
parse_packet_line):
    seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz

USAGE (laptop joined to the XIAO SoftAP "PASS-knee"):
    from sources.network_source import NetworkSource
    cap = NetworkSource().get_data(3)          # 3 s of live wireless data

HARDWARE-DECOUPLED
------------------
Like SerialSource, a `line_source` (any iterable of str lines) can be injected for
tests/replay, so no socket or device is needed to exercise the parsing path.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from .schema import Packet, Capture
from .serial_source import parse_packet_line

DEFAULT_UDP_PORT  = 5005
DEFAULT_BIND_HOST = "0.0.0.0"     # every interface (receives the unit's packets)
DEFAULT_AP_HOST   = "192.168.4.1" # XIAO SoftAP gateway; where we send the "hello"
HELLO             = b"PASS-hello" # announces us so the unit unicasts the stream back


class NetworkSource:
    """
    Live UDP source. Binds a datagram socket and yields packets parsed from the
    firmware's broadcast lines. Mirrors SerialSource.

    port        : UDP port the firmware broadcasts to (must match the firmware).
    host        : bind address; 0.0.0.0 receives on every interface.
    timeout_s   : socket read timeout, so a quiet link does not block forever.
    line_source : iterable of str lines; when set, no socket is opened (tests/replay).
    """

    def __init__(self, port: int = DEFAULT_UDP_PORT, host: str = DEFAULT_BIND_HOST,
                 timeout_s: float = 1.0, ap_host: str = DEFAULT_AP_HOST,
                 line_source: Iterable[str] | None = None):
        self.port = int(port)
        self.host = host
        self.timeout_s = float(timeout_s)
        self.ap_host = ap_host          # the XIAO SoftAP gateway, for the "hello"
        self.line_source = line_source
        self.n_malformed = 0

    def _raw_lines(self) -> Iterator[str]:
        """Yield decoded lines from the replay iterable, or from inbound UDP
        datagrams (one line per packet, split defensively just in case)."""
        if self.line_source is not None:
            yield from self.line_source
            return
        import socket, time                            # stdlib, no extra dependency
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(self.timeout_s)
        ap = (self.ap_host, self.port)
        try:
            sock.sendto(HELLO, ap)                     # announce us to the unit
            last_hello = time.monotonic()
            while True:
                try:
                    data, _addr = sock.recvfrom(2048)
                except socket.timeout:
                    # re-announce until data flows (covers a unit that booted late)
                    if time.monotonic() - last_hello > 1.0:
                        try:
                            sock.sendto(HELLO, ap)
                        except OSError:
                            pass
                        last_hello = time.monotonic()
                    continue
                for line in data.decode("ascii", errors="ignore").splitlines():
                    yield line
        finally:
            sock.close()

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
        """Collect packets into a Capture, stopping after duration_s of device
        time (or when the line source ends). Same shape as SerialSource.get_data."""
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
            activity=None,                             # network has no activity labels
        )


if __name__ == "__main__":
    print("PASS network_source ready - listening for the firmware UDP broadcast:")
    print(f"  bind {DEFAULT_BIND_HOST}:{DEFAULT_UDP_PORT}")
    print("  line format: seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz")
