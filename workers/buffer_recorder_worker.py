"""
Records one shared-memory buffer to disk as raw bytes plus a small binary
header. Spawn one of these per buffer you want recorded.

Everything is little-endian. Each sampled slot becomes a fixed-size record
header followed by the payload:

    [8  bytes]  uint64   seq_no
    [8  bytes]  float64  produced_ts
    [8  bytes]  float64  read_ts  (wall-clock when this worker read the slot)
    [4  bytes]  uint32   payload_size
    [2  bytes]  uint16   schema_version
    [2  bytes]  uint16   dtype_tag
    [payload_size bytes]  raw payload bytes  (identical to the shm slot payload)

Files are named <buffer_name>_<session_ts>.bin and start with a 64-byte
header:

    [8  bytes]  magic     b"PLTREC\\x00\\x01"
    [2  bytes]  uint16    schema_version
    [2  bytes]  uint16    dtype_tag
    [4  bytes]  uint32    sample_k  (configured sample interval)
    [4  bytes]  uint32    top_n     (max distinct samples kept)
    [8  bytes]  float64   session_start_ts
    [36 bytes]  reserved  (zero-filled)

Only slots with seq_no % sample_k == 0 get written, so sample_k just
decimates. After top_n records we seek back to the first record and
overwrite from there. NB: the file is not truncated, so older tail records
can still be sitting past the current write position. top_n=0 means no cap.
"""

from __future__ import annotations

import os
import struct
import time
from typing import Any

from .base_worker import BaseWorker
from shared_buffer.ring_buffer import SharedRingBuffer
from shared_buffer.schema import SlotHeader, SCHEMA_VERSION

FILE_MAGIC = b"PLTREC\x00\x01"             # 8 bytes
# magic(8) schema(2) dtype(2) sample_k(4) top_n(4) session_ts(8) reserved(30) = 58 bytes
FILE_HEADER_FMT = "<8sHHIId30s"
# on-disk size; struct is zero-padded by 6 bytes
FILE_HEADER_SIZE = 64

# seq_no, produced_ts, read_ts, payload_size, schema_ver, dtype_tag
RECORD_HEADER_FMT = "<QddIHH"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)  # 32 bytes


class BufferRecorderWorker(BaseWorker):
    """Records a single shared-memory buffer to disk.

    Config kwargs:
      buffer       a SharedRingBuffer, or a typed wrapper (FrameBuffer etc.),
                   in which case we use its ._rb
      buffer_name  goes in the output filename
      output_dir   where the .bin goes (default /tmp/pilot_recordings)
      sample_k     record every k-th seq_no (default 1, i.e. everything)
      top_n        records before wrapping, 0 = unlimited
      dtype_tag    DTypeTag written to the file header; defaults to the
                   buffer's own
    """

    def __init__(self, name: str = "buf_recorder", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._rb: SharedRingBuffer | None = None
        self._buffer_name: str = ""
        self._output_dir: str = ""
        self._sample_k: int = 1
        self._top_n: int = 0
        self._dtype_tag: int = 0
        self._fp = None
        self._last_seq: int = 0
        self._records_written: int = 0
        self._data_start_offset: int = 0
        self._session_ts: float = 0.0

    def setup(self) -> None:
        buf = self.config["buffer"]
        # typed wrappers keep the ring buffer in ._rb
        if hasattr(buf, "_rb"):
            self._rb = buf._rb
        else:
            self._rb = buf

        self._buffer_name = self.config.get("buffer_name", self.name)
        self._output_dir = self.config.get("output_dir", "/tmp/pilot_recordings")
        self._sample_k = max(1, int(self.config.get("sample_k", 1)))
        self._top_n = max(0, int(self.config.get("top_n", 0)))
        self._dtype_tag = int(self.config.get("dtype_tag", self._rb.dtype_tag))

        os.makedirs(self._output_dir, exist_ok=True)

        self._session_ts = time.time()
        fname = f"{self._buffer_name}_{int(self._session_ts)}.bin"
        path = os.path.join(self._output_dir, fname)
        self._fp = open(path, "wb")
        self._write_file_header()
        self._data_start_offset = self._fp.tell()

        self.logger.info(
            f"Recording {self._buffer_name} → {path}  "
            f"(sample_k={self._sample_k}, top_n={self._top_n})"
        )

    def step(self) -> None:
        result = self._rb.read_latest()
        if result is None:
            time.sleep(0.005)
            return

        header, payload = result

        if header.seq_no <= self._last_seq:
            time.sleep(0.001)
            return

        self._last_seq = header.seq_no

        if header.seq_no % self._sample_k != 0:
            return

        # hit top_n, rewind and overwrite from the first record
        if self._top_n > 0 and self._records_written >= self._top_n:
            self._fp.seek(self._data_start_offset)
            self._records_written = 0

        self._write_record(header, payload)
        self._records_written += 1

    def teardown(self) -> None:
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self.logger.info(
                f"Recorder {self._buffer_name}: "
                f"{self._records_written} records written"
            )

    def _write_file_header(self) -> None:
        struct_size = struct.calcsize(FILE_HEADER_FMT)
        header_bytes = struct.pack(
            FILE_HEADER_FMT,
            FILE_MAGIC,
            SCHEMA_VERSION,
            self._dtype_tag,
            self._sample_k,
            self._top_n,
            self._session_ts,
            b"\x00" * 30,
        )
        header_bytes += b"\x00" * (FILE_HEADER_SIZE - struct_size)
        self._fp.write(header_bytes)

    def _write_record(self, header: SlotHeader, payload: bytes) -> None:
        rec_hdr = struct.pack(
            RECORD_HEADER_FMT,
            header.seq_no,
            header.produced_ts,
            header.read_ts,
            len(payload),
            header.schema_version,
            header.dtype_tag,
        )
        self._fp.write(rec_hdr)
        self._fp.write(payload)

    # for reading recordings back

    @staticmethod
    def read_file_header(fp) -> dict:
        """Decode the 64-byte file header into a dict (magic, schema_version,
        dtype_tag, sample_k, top_n, session_ts)."""
        raw = fp.read(FILE_HEADER_SIZE)
        struct_size = struct.calcsize(FILE_HEADER_FMT)
        magic, schema, dtype, sample_k, top_n, session_ts, _reserved_b = (
            struct.unpack(FILE_HEADER_FMT, raw[:struct_size])
        )
        return {
            "magic": magic,
            "schema_version": schema,
            "dtype_tag": dtype,
            "sample_k": sample_k,
            "top_n": top_n,
            "session_ts": session_ts,
        }

    @staticmethod
    def read_record(fp) -> tuple[dict, bytes] | None:
        """(header_dict, payload) for the next record, or None at EOF / on a
        truncated record."""
        raw_hdr = fp.read(RECORD_HEADER_SIZE)
        if len(raw_hdr) < RECORD_HEADER_SIZE:
            return None
        seq_no, produced_ts, read_ts, payload_size, schema_ver, dtype_tag = (
            struct.unpack(RECORD_HEADER_FMT, raw_hdr)
        )
        payload = fp.read(payload_size)
        if len(payload) < payload_size:
            return None
        return {
            "seq_no": seq_no,
            "produced_ts": produced_ts,
            "read_ts": read_ts,
            "payload_size": payload_size,
            "schema_version": schema_ver,
            "dtype_tag": dtype_tag,
        }, payload
