"""Host-RAM watchdog that stops training before the machine runs out of memory.

Why a background thread instead of a per-step check: the biggest allocation
spikes happen where the training loop never gets a chance to poll — DataLoader
worker spawn (``num_workers`` x ``prefetch_factor`` batches materialise at
once), the test dataloader spin-up, and the generation-based metric loops.
A daemon thread samples system ``MemAvailable`` on a fixed interval and reacts
in two tiers:

- **soft (abort)**: raise a flag. ``MemoryGuard.raise_if_tripped()`` is called
  at train-step boundaries by the trainer, which then tears the dataloaders
  down, saves an emergency checkpoint and exits with a non-zero status.
- **hard (kill)**: the main thread is either blocked or already too starved to
  save anything. The watchdog itself terminates the child processes (the
  DataLoader workers) and calls ``os._exit`` immediately.

Measurement is system-wide ``MemAvailable`` in absolute GB, never process RSS:
the worker processes share copy-on-write pages, so summing RSS massively
over-counts, and memory pressure from *other* applications is what actually
kills the machine.

This is best-effort user-space protection. The kernel-enforced equivalent is a
cgroup memory limit (``systemd-run --user --scope -p MemoryMax=...``).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

import psutil

_BYTES_PER_GB = 1024.0 ** 3

# Exit code used for a memory-guard abort. Distinct from 1 so callers/scripts
# can tell "stopped by the memory guard" apart from an ordinary crash.
MEMORY_ABORT_EXIT_CODE = 17


class MemoryLimitExceeded(RuntimeError):
    """Raised on the main thread when the soft memory threshold was crossed."""

    def __init__(self, available_gb: float, threshold_gb: float) -> None:
        super().__init__(
            f"System available memory {available_gb:.2f} GB fell below the "
            f"abort threshold {threshold_gb:.2f} GB"
        )
        self.available_gb = available_gb
        self.threshold_gb = threshold_gb


def available_gb() -> float:
    """System-wide available (reclaimable-inclusive) memory in GB."""
    return psutil.virtual_memory().available / _BYTES_PER_GB


def total_gb() -> float:
    """Total system memory in GB."""
    return psutil.virtual_memory().total / _BYTES_PER_GB


def _kill_children(timeout: float = 3.0) -> int:
    """SIGTERM then SIGKILL every descendant of this process.

    Used on the hard path to free the DataLoader workers' memory immediately.
    Deliberately *not* ``os.killpg``: under a non-interactive shell script there is
    no job control, so the process group also contains the driving shell.
    """
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return 0
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    return len(children)


class MemoryGuard:
    """Background sampler of system available memory with soft/hard thresholds.

    ``abort_gb`` must be greater than ``kill_gb``: the soft threshold needs
    enough headroom left for the graceful path (tearing down workers and
    staging the safetensors save in host memory).
    """

    def __init__(
        self,
        *,
        abort_gb: float,
        kill_gb: float,
        poll_interval_s: float = 1.0,
        reader: Callable[[], float] = available_gb,
        on_kill: Optional[Callable[[float], None]] = None,
    ) -> None:
        if abort_gb <= kill_gb:
            raise ValueError(
                f"memory_guard.abort_available_gb ({abort_gb}) must be greater "
                f"than kill_available_gb ({kill_gb})"
            )
        self.abort_gb = float(abort_gb)
        self.kill_gb = float(kill_gb)
        self.poll_interval_s = float(poll_interval_s)
        self._reader = reader
        self._on_kill = on_kill
        self._tripped = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped_available_gb = 0.0
        self.min_available_gb = float("inf")

    def start(self) -> "MemoryGuard":
        if self._thread is not None:
            return self
        print(
            f"[memory_guard] armed: abort < {self.abort_gb:.1f} GB, "
            f"hard kill < {self.kill_gb:.1f} GB available "
            f"(total {total_gb():.1f} GB, poll {self.poll_interval_s:.1f}s)"
        )
        self._thread = threading.Thread(
            target=self._loop, name="memory-guard", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.poll_interval_s + 1.0)
        self._thread = None

    def __enter__(self) -> "MemoryGuard":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            self.poll_once()

    def poll_once(self) -> float:
        """Sample once and react. Returns the observed available GB."""
        avail = self._reader()
        self.min_available_gb = min(self.min_available_gb, avail)
        if avail < self.kill_gb:
            self._hard_kill(avail)
        elif avail < self.abort_gb and not self._tripped.is_set():
            self._tripped_available_gb = avail
            self._tripped.set()
            print(
                f"\n[memory_guard] available memory {avail:.2f} GB < "
                f"{self.abort_gb:.2f} GB — requesting graceful abort "
                f"at the next step boundary.",
                flush=True,
            )
        return avail

    def _hard_kill(self, avail: float) -> None:
        print(
            f"\n[memory_guard] HARD LIMIT: available memory {avail:.2f} GB < "
            f"{self.kill_gb:.2f} GB. Killing worker processes and exiting "
            f"immediately to keep the machine alive.",
            flush=True,
        )
        if self._on_kill is not None:
            try:
                self._on_kill(avail)
            except Exception:  # pragma: no cover - never block the kill path
                pass
        killed = _kill_children()
        print(f"[memory_guard] terminated {killed} child process(es).", flush=True)
        os._exit(MEMORY_ABORT_EXIT_CODE)

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    def raise_if_tripped(self) -> None:
        """Called from the training loop at a safe point; raises on abort."""
        if self._tripped.is_set():
            raise MemoryLimitExceeded(self._tripped_available_gb, self.abort_gb)


def build_memory_guard(cfg: Any) -> Optional[MemoryGuard]:
    """Construct a ``MemoryGuard`` from ``cfg.train.memory_guard`` (or None)."""
    gcfg = getattr(getattr(cfg, "train", None), "memory_guard", None)
    if gcfg is None or not bool(getattr(gcfg, "enabled", False)):
        return None
    return MemoryGuard(
        abort_gb=float(getattr(gcfg, "abort_available_gb", 6.0)),
        kill_gb=float(getattr(gcfg, "kill_available_gb", 2.5)),
        poll_interval_s=float(getattr(gcfg, "poll_interval_s", 1.0)),
    )


def describe_children() -> str:
    """Live descendants, for the teardown diagnostic.

    A count alone is ambiguous: right after the dataloaders are released, a
    worker that has not finished exiting yet, or multiprocessing's resource
    tracker, still shows up. Naming them makes "1 left" readable instead of
    alarming.
    """
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return "unknown"
    if not children:
        return "0"
    names = []
    for child in children:
        try:
            names.append(child.name())
        except psutil.Error:
            names.append("?")
    return f"{len(children)} ({', '.join(sorted(set(names)))})"
