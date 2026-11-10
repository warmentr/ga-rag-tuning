"""GPU energy logging by sampling `nvidia-smi` power draw in a background thread.

Measures GPU board power only (not CPU, RAM, or the rest of the system), so
it is a lower bound on total energy. Returns None values if nvidia-smi is
unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time


def _read_power_watts() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return sum(float(x) for x in out.stdout.split())  # sum over GPUs
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class GpuPowerMeter:
    """Context manager: `with GpuPowerMeter() as m: ...; m.wh, m.seconds`."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.available = shutil.which("nvidia-smi") is not None
        self.joules = 0.0
        self.samples = 0
        self.seconds = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            w = _read_power_watts()
            now = time.perf_counter()
            if w is not None:
                self.joules += w * (now - last)
                self.samples += 1
            last = now
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.seconds = time.perf_counter() - self._t0
        return False

    @property
    def wh(self) -> float | None:
        return self.joules / 3600 if self.available and self.samples else None
