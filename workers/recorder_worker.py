"""
RecorderWorker — reads from all buffers and persists data to disk.

Replaces ``StateVectorProcessor._continuous_save_to_file()`` and
``saveRgbdFrame()`` from the old pipeline.  Data is written to a SQLite
database and/or binary frame files.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict

import cv2

from .base_worker import BaseWorker
from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    PoseBuffer,
    SegmentationBuffer,
)


class RecorderWorker(BaseWorker):
    """Worker process for recording pipeline data to disk.

    Config kwargs
    -------------
    buffers : dict[str, buffer]
        Named buffers: ``"frames"``, ``"imu"``, ``"segments"``, ``"pose"``, ``"control"``.
    output_dir : str
        Directory to write recordings into.
    """

    def __init__(self, name: str = "recorder", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._imu_buf: IMUBuffer | None = None
        self._seg_buf: SegmentationBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._ctrl_buf: ControlBuffer | None = None
        self._output_dir = ""
        self._db_conn = None
        self._db_cursor = None
        self._last_frame_seq = 0
        self._last_imu_seq = 0
        self._last_ctrl_seq = 0

    def setup(self) -> None:
        bufs: Dict[str, Any] = self.config["buffers"]
        self._frame_buf = bufs.get("frames")
        self._imu_buf = bufs.get("imu")
        self._seg_buf = bufs.get("segments")
        self._pose_buf = bufs.get("pose")
        self._ctrl_buf = bufs.get("control")

        self._output_dir = self.config.get("output_dir", "/tmp/pilot_recordings")
        os.makedirs(self._output_dir, exist_ok=True)

        # Create SQLite database
        db_path = os.path.join(
            self._output_dir, f"recording_{int(time.time())}.db"
        )
        self._db_conn = sqlite3.connect(db_path)
        self._db_cursor = self._db_conn.cursor()
        self._create_tables()
        self.logger.info(f"Recording to {db_path}")

    def _create_tables(self) -> None:
        self._db_cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq_no INTEGER,
                produced_ts REAL,
                read_ts REAL,
                rgb_path TEXT,
                depth_path TEXT
            );
            CREATE TABLE IF NOT EXISTS imu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq_no INTEGER,
                produced_ts REAL,
                w REAL, x REAL, y REAL, z REAL,
                ax REAL, ay REAL, az REAL,
                throttle REAL, steering REAL
            );
            CREATE TABLE IF NOT EXISTS control (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq_no INTEGER,
                produced_ts REAL,
                steering REAL,
                throttle REAL
            );
            """
        )
        self._db_conn.commit()

    def step(self) -> None:
        recorded_anything = False

        # Record frames
        if self._frame_buf is not None:
            result = self._frame_buf.read_latest()
            if result is not None:
                header, rgb, depth = result
                if header.seq_no > self._last_frame_seq:
                    self._last_frame_seq = header.seq_no
                    ts = int(header.produced_ts * 1000)
                    rgb_path = os.path.join(self._output_dir, f"rgb_{ts}.png")
                    depth_path = os.path.join(self._output_dir, f"depth_{ts}.bin")
                    cv2.imwrite(rgb_path, rgb)
                    depth.tofile(depth_path)
                    self._db_cursor.execute(
                        "INSERT INTO frames (seq_no, produced_ts, read_ts, rgb_path, depth_path) VALUES (?,?,?,?,?)",
                        (header.seq_no, header.produced_ts, header.read_ts, rgb_path, depth_path),
                    )
                    recorded_anything = True

        # Record IMU
        if self._imu_buf is not None:
            result = self._imu_buf.read_latest()
            if result is not None:
                header, vec = result
                if header.seq_no > self._last_imu_seq:
                    self._last_imu_seq = header.seq_no
                    self._db_cursor.execute(
                        "INSERT INTO imu (seq_no, produced_ts, w, x, y, z, ax, ay, az, throttle, steering) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (header.seq_no, header.produced_ts, *vec.tolist()),
                    )
                    recorded_anything = True

        # Record control
        if self._ctrl_buf is not None:
            result = self._ctrl_buf.read_latest()
            if result is not None:
                header, steering, throttle = result
                if header.seq_no > self._last_ctrl_seq:
                    self._last_ctrl_seq = header.seq_no
                    self._db_cursor.execute(
                        "INSERT INTO control (seq_no, produced_ts, steering, throttle) VALUES (?,?,?,?)",
                        (header.seq_no, header.produced_ts, steering, throttle),
                    )
                    recorded_anything = True

        if recorded_anything:
            self._db_conn.commit()
        else:
            time.sleep(0.01)  # no new data — back off

    def teardown(self) -> None:
        if self._db_conn is not None:
            self._db_conn.commit()
            self._db_conn.close()
            self.logger.info("Recording database closed")
