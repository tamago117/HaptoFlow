"""Cooperative SIGINT / SIGTERM handling for the training loop.

Saving a checkpoint from inside a signal handler is not safe — the handler can
fire between any two bytecodes, e.g. halfway through an optimizer step. Instead
the handler only records the signal; the training loop calls
``InterruptHandler.raise_if_requested()`` at the same safe points the memory
guard uses (train-step boundaries, epoch end, val/test loss batches) and raises
``TrainingInterrupted`` there.

A *second* signal restores the default disposition and re-raises immediately, so
an impatient Ctrl+C still works.

On the way out the process must die *by the signal*, not via ``sys.exit``:
shells rely on 130/143 to tell a user-requested stop from an ordinary failure.
``reraise_after_cleanup`` does that.
"""
from __future__ import annotations

import os
import signal
import sys
from types import FrameType
from typing import Any, Optional

_SIGNAL_NAMES = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}


class TrainingInterrupted(RuntimeError):
    """Raised at a safe point after SIGINT/SIGTERM was received."""

    def __init__(self, signum: int) -> None:
        name = _SIGNAL_NAMES.get(signum, f"signal {signum}")
        super().__init__(f"Training interrupted by {name}")
        self.signum = signum


class InterruptHandler:
    """Context manager installing cooperative SIGINT/SIGTERM handlers."""

    def __init__(self) -> None:
        self.signum: Optional[int] = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "InterruptHandler":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # Not the main thread, or the platform lacks the signal.
                self._previous.pop(sig, None)
        return self

    def __exit__(self, *exc: Any) -> None:
        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
        self._previous.clear()

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        if self.signum is not None:
            # Second request — the user is not waiting for a clean save.
            signal.signal(signum, signal.SIG_DFL)
            print(
                f"\n[interrupt] second {_SIGNAL_NAMES.get(signum, signum)} — "
                f"exiting now without saving.",
                file=sys.stderr, flush=True,
            )
            os.kill(os.getpid(), signum)
            return
        self.signum = signum
        print(
            f"\n[interrupt] {_SIGNAL_NAMES.get(signum, signum)} received — will save a "
            f"resumable checkpoint at the next safe point. "
            f"Send it again to exit immediately without saving.",
            file=sys.stderr, flush=True,
        )

    @property
    def requested(self) -> bool:
        return self.signum is not None

    def raise_if_requested(self) -> None:
        if self.signum is not None:
            raise TrainingInterrupted(self.signum)


def reraise_after_cleanup(signum: int) -> None:
    """Terminate by ``signum`` so callers observe death-by-signal (130/143).

    Flushes first: dying by a signal skips the interpreter's exit-time flush, so
    anything still sitting in a block-buffered stdout (a redirect or a ``| tee``)
    would be lost — including the "resume with:" hint printed just before.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - a broken pipe must not block the exit
            pass
    try:
        signal.signal(signum, signal.SIG_DFL)
    except (ValueError, OSError):
        pass
    os.kill(os.getpid(), signum)
    # Fallback for platforms where os.kill returns.
    sys.exit(128 + signum)
