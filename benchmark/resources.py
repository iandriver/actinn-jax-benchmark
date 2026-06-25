"""Wall-clock + peak-memory monitoring for a timed code region."""

import os
import threading
import time

import psutil


class ResourceMonitor:
    """Context manager measuring elapsed wall time and peak RSS (MB).

    Peak RSS samples this process plus any child processes (so BLAS / subprocess
    memory is counted), on a background thread.

        with ResourceMonitor() as m:
            ...work...
        m.elapsed     # seconds
        m.peak_mb     # peak resident memory in MB
    """

    def __init__(self, poll=0.05):
        self.poll = poll
        self._peak = 0
        self._running = False
        self.elapsed = 0.0
        self.peak_mb = 0.0

    def _rss(self):
        rss = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss

    def _sample(self):
        while self._running:
            try:
                self._peak = max(self._peak, self._rss())
            except psutil.Error:
                pass
            time.sleep(self.poll)

    def __enter__(self):
        self.proc = psutil.Process(os.getpid())
        self._peak = self.proc.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        self._running = False
        self._thread.join(timeout=1.0)
        self.peak_mb = self._peak / 1e6
        return False
