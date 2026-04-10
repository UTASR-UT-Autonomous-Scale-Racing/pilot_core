"""
BaseWorker — abstract base class for all pipeline worker processes.

Every worker follows the same lifecycle:
    1. ``setup()``   — called once *inside* the child process
    2. ``step()``    — called in a tight loop until stopped
    3. ``teardown()``— cleanup before the process exits

The orchestrator (start.py) spawns each worker as a
``multiprocessing.Process(target=worker.run)``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Dict


class BaseWorker:
    """Abstract worker that all pipeline workers inherit from.

    Parameters
    ----------
    name : str
        Human-readable name (also used as the process name).
    **kwargs
        Arbitrary configuration forwarded to the subclass.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.config: Dict[str, Any] = kwargs
        self._stop_event = mp.Event()
        self._logger: logging.Logger | None = None

    # --- abstract interface ---

    def setup(self) -> None:
        """Initialise resources *inside the child process*.

        Override in subclasses to open devices, load models, attach to
        shared-memory buffers, etc.
        """
        raise NotImplementedError

    def step(self) -> None:
        """Execute one unit of work (one frame, one sensor read, …).

        Override in subclasses.  Must be non-blocking or short-lived.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Release resources before exit.  Override as needed."""

    # --- lifecycle ---

    def run(self) -> None:
        """Entry point executed inside the child process.

        Do **not** override this unless you need a fundamentally different
        loop structure.
        """
        # Configure per-process logging
        self._logger = logging.getLogger(f"worker.{self.name}")
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s %(levelname)s {self.name}(pid=%(process)d)] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        self._logger.addHandler(handler)

        # Ignore SIGINT in workers — the orchestrator handles shutdown via _stop_event
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # OpenCV keeps its own thread pool; match the orchestrator's cap.
        try:
            import cv2
            cv2.setNumThreads(int(os.environ.get("OMP_NUM_THREADS", "0")))
        except (ImportError, ValueError):
            pass

        self._logger.info("Starting up …")
        try:
            self.setup()
            self._logger.info("Setup complete — entering run loop")
            while not self._stop_event.is_set():
                try:
                    self.step()
                except Exception:
                    self._logger.exception("Exception in step()")
                    time.sleep(0.01)  # back off briefly on error
        except Exception:
            self._logger.exception("Fatal error during setup()")
        finally:
            self._logger.info("Tearing down …")
            try:
                self.teardown()
            except Exception:
                self._logger.exception("Exception in teardown()")
            self._logger.info("Exited.")

    def stop(self) -> None:
        """Signal the worker to finish its current iteration and exit."""
        self._stop_event.set()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            # Fallback for use before run() — e.g. in orchestrator process
            return logging.getLogger(f"worker.{self.name}")
        return self._logger
