"""
Base class for the pipeline workers.

start.py spawns each one as multiprocessing.Process(target=worker.run).
Inside the child, setup() runs once, then step() in a tight loop until
stop() is called, then teardown().
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Dict


class BaseWorker:
    """name is used for logging and the process name; any kwargs end up in
    self.config for the subclass to pick apart."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.config: Dict[str, Any] = kwargs
        self._stop_event = mp.Event()
        self._logger: logging.Logger | None = None

    def setup(self) -> None:
        """Open devices, load models, attach to shm buffers etc.

        Runs in the child process, not in __init__, so nothing unpicklable
        has to be pickled over to the child.
        """
        raise NotImplementedError

    def step(self) -> None:
        """One unit of work (a frame, a sensor read...). Keep it short, the
        stop event is only checked between calls."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional cleanup hook."""

    def run(self) -> None:
        """Process entry point. Subclasses shouldn't normally override this."""
        # per-process logger
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

        # Ignore SIGINT here; the orchestrator does shutdown through _stop_event
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # OpenCV has its own thread pool, match the orchestrator's cap
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
                    time.sleep(0.01)  # brief backoff so a persistent error doesn't spin
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
            # before run(), i.e. still in the orchestrator process
            return logging.getLogger(f"worker.{self.name}")
        return self._logger
