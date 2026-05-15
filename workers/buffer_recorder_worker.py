"""
Records shared-memory buffers to disk, one file series <name>_<session_ms>_<part>.bin
per buffer. Each file: 64-byte header (FILE_HEADER_FMT), then records of a 32-byte
header (RECORD_HEADER_FMT; timestamps on time.monotonic()) + the raw slot payload,
decodable with shared_buffer.buffer_class_for(...).decode. Little-endian; files are
never overwritten and a truncated last record is ignored by read_record().
"""

from __future__ import annotations

import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from .base_worker import BaseWorker
from shared_buffer.ring_buffer import SharedRingBuffer
from shared_buffer.schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, SlotHeader

FILE_MAGIC = b"PLTREC\x00\x01"
# magic(8) schema(2) dtype(2) sample_k(4) top_n(4) session_ts(8) part(4)
# session_mono(8) name(16) reserved(8) = 64 bytes; pre-v2 files have zero part/mono/name
FILE_HEADER_FMT = "<8sHHIIdId16s8s"
FILE_HEADER_SIZE = 64
assert struct.calcsize(FILE_HEADER_FMT) == FILE_HEADER_SIZE

# seq_no, produced_ts, read_ts, payload_size, schema_ver, dtype_tag
RECORD_HEADER_FMT = "<QddIHH"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)  # 32 bytes

ON_FULL_POLICIES = ("rotate", "stop")
DEFAULT_KEEP_FILES = 2


class RecordingFormatError(ValueError):
    """Not a recording file, or a schema version this code can't read."""


@dataclass
class _Stream:
    """Per-buffer recording state."""
    name: str
    rb: SharedRingBuffer
    sample_k: int
    top_n: int
    on_full: str
    keep_files: int
    dtype_tag: int
    fp: Any = None
    path: str = ""
    part: int = -1
    last_seq: int = 0
    received: int = 0
    dropped: int = 0
    records_written: int = 0   # total, all parts
    records_in_file: int = 0
    stopped: bool = False
    files: Deque[str] = field(default_factory=deque)


def _ring_of(buf) -> SharedRingBuffer:
    return buf._rb if hasattr(buf, "_rb") else buf


class BufferRecorderWorker(BaseWorker):
    """Records shared-memory buffers to disk, one file series per buffer.

    Config kwargs:
      buffers         dict name -> buffer, or name -> dict with "buffer" plus
                      optional per-buffer "sample_k", "top_n", "on_full",
                      "keep_files". A buffer is a SharedRingBuffer or a typed
                      wrapper (FrameBuffer, StateBuffer, LidarBuffer, ...).
      buffer, buffer_name
                      single-buffer form (same as buffers={buffer_name: buffer})
      output_dir      where the .bin files go (default /tmp/pilot_recordings)
      sample_k        default: record every k-th received sample (1 = all)
      top_n           default: records per file, 0 = unlimited
      on_full         when top_n is hit: "rotate" to a new part (default) or "stop"
      keep_files      newest parts kept with on_full="rotate", 0 = all (default 2)
      flush_interval  seconds between flushes (default 1.0)
      dtype_tag       single-buffer form only: override the DTypeTag in the file header
    """

    def __init__(self, name: str = "buf_recorder", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._streams: Dict[str, _Stream] = {}
        self._output_dir: str = ""
        self._session_ts: float = 0.0
        self._session_mono: float = 0.0
        self._flush_interval: float = 1.0
        self._next_flush: float = 0.0

    # setup / config

    def _stream_configs(self) -> Dict[str, Dict[str, Any]]:
        cfg = self.config
        buffers = cfg.get("buffers")
        if buffers is None:
            if "buffer" not in cfg:
                raise ValueError("BufferRecorderWorker needs buffers={name: buffer} or buffer=...")
            single = {"buffer": cfg["buffer"]}
            if cfg.get("dtype_tag") is not None:
                single["dtype_tag"] = cfg["dtype_tag"]
            buffers = {cfg.get("buffer_name", self.name): single}
        out = {}
        for bname, spec in buffers.items():
            if spec is None:
                continue
            if not isinstance(spec, dict):
                spec = {"buffer": spec}
            if spec.get("buffer") is None:
                continue
            out[bname] = spec
        return out

    def setup(self) -> None:
        cfg = self.config
        self._output_dir = cfg.get("output_dir", "/tmp/pilot_recordings")
        self._flush_interval = float(cfg.get("flush_interval", 1.0))
        os.makedirs(self._output_dir, exist_ok=True)
        self._session_ts = time.time()
        self._session_mono = time.monotonic()
        self._next_flush = self._session_mono + self._flush_interval

        for bname, spec in self._stream_configs().items():
            rb = _ring_of(spec["buffer"])
            on_full = spec.get("on_full", cfg.get("on_full", "rotate"))
            if on_full not in ON_FULL_POLICIES:
                raise ValueError(f"on_full must be one of {ON_FULL_POLICIES}, got {on_full!r}")
            st = _Stream(
                name=bname,
                rb=rb,
                sample_k=max(1, int(spec.get("sample_k", cfg.get("sample_k", 1)))),
                top_n=max(0, int(spec.get("top_n", cfg.get("top_n", 0)))),
                on_full=on_full,
                keep_files=max(0, int(spec.get("keep_files", cfg.get("keep_files", DEFAULT_KEEP_FILES)))),
                dtype_tag=int(spec.get("dtype_tag", rb.dtype_tag)),
            )
            self._streams[bname] = st
            self._open_next_part(st)
            self.logger.info(
                f"Recording {bname} -> {st.path} (sample_k={st.sample_k}, top_n={st.top_n}, "
                f"on_full={st.on_full}, keep_files={st.keep_files})"
            )
        if not self._streams:
            self.logger.warning("BufferRecorderWorker: no buffers to record")

    # main loop

    def step(self) -> None:
        got = 0
        for st in self._streams.values():
            if st.stopped:
                continue
            records = st.rb.read_since(st.last_seq, stamp=True)
            got += len(records)
            for header, payload in records:
                if st.last_seq and header.seq_no > st.last_seq + 1:
                    st.dropped += header.seq_no - st.last_seq - 1
                st.last_seq = header.seq_no
                st.received += 1
                if st.received % st.sample_k != 0:
                    continue
                self._record(st, header, payload)
                if st.stopped:
                    break

        now = time.monotonic()
        if now >= self._next_flush:
            self._flush_all()
            self._next_flush = now + self._flush_interval
        if not got:
            time.sleep(0.002)

    def teardown(self) -> None:
        for st in self._streams.values():
            self._close_file(st)
            msg = (
                f"Recorder {st.name}: {st.records_written} records written "
                f"({st.received} received, {st.dropped} dropped in the ring, "
                f"{st.part + 1} file(s))"
            )
            if st.dropped:
                self.logger.warning(msg + " - increase the buffer's slots if drops matter")
            else:
                self.logger.info(msg)

    @property
    def session_id(self) -> str:
        return str(int(self._session_ts * 1000))

    # stats (tests / monitoring)

    @property
    def streams(self) -> Dict[str, _Stream]:
        return self._streams

    @property
    def _records_written(self) -> int:
        return sum(st.records_written for st in self._streams.values())

    # file handling

    def _record(self, st: _Stream, header: SlotHeader, payload: bytes) -> None:
        if st.top_n > 0 and st.records_in_file >= st.top_n:
            if st.on_full == "stop":
                self.logger.info(f"Recorder {st.name}: top_n={st.top_n} reached, stopping")
                self._close_file(st)
                st.stopped = True
                return
            self._open_next_part(st)
        try:
            st.fp.write(struct.pack(
                RECORD_HEADER_FMT,
                header.seq_no,
                header.produced_ts,
                header.read_ts,
                len(payload),
                header.schema_version,
                header.dtype_tag,
            ))
            st.fp.write(payload)
        except OSError:
            self.logger.exception(f"Recorder {st.name}: write failed, no longer recording this buffer")
            self._close_file(st)
            st.stopped = True
            return
        st.records_in_file += 1
        st.records_written += 1

    def _open_next_part(self, st: _Stream) -> None:
        self._close_file(st)
        st.part += 1
        st.records_in_file = 0
        st.path = os.path.join(
            self._output_dir, f"{st.name}_{self.session_id}_{st.part:04d}.bin"
        )
        st.fp = open(st.path, "xb")  # never clobber an existing recording
        st.fp.write(struct.pack(
            FILE_HEADER_FMT,
            FILE_MAGIC,
            SCHEMA_VERSION,
            st.dtype_tag,
            st.sample_k,
            st.top_n,
            self._session_ts,
            st.part,
            self._session_mono,
            st.name.encode()[:16],
            b"",
        ))
        st.files.append(st.path)
        while st.keep_files and len(st.files) > st.keep_files:
            old = st.files.popleft()
            try:
                os.remove(old)
            except OSError:
                self.logger.warning(f"Recorder {st.name}: could not delete {old}")

    @staticmethod
    def _close_file(st: _Stream) -> None:
        if st.fp is not None:
            try:
                st.fp.flush()
                st.fp.close()
            finally:
                st.fp = None

    def _flush_all(self) -> None:
        for st in self._streams.values():
            if st.fp is not None:
                try:
                    st.fp.flush()
                except OSError:
                    self.logger.exception(f"Recorder {st.name}: flush failed")

    # for reading recordings back

    @staticmethod
    def read_file_header(fp) -> dict:
        """Decode and validate the 64-byte file header.

        Returns a dict (magic, schema_version, dtype_tag, sample_k, top_n,
        session_ts, part, session_mono, buffer_name). Raises
        RecordingFormatError on a bad magic, an unsupported schema version or
        a truncated header.
        """
        raw = fp.read(FILE_HEADER_SIZE)
        if len(raw) < FILE_HEADER_SIZE:
            raise RecordingFormatError(f"file header truncated ({len(raw)} of {FILE_HEADER_SIZE} bytes)")
        (magic, schema, dtype, sample_k, top_n, session_ts, part, session_mono,
         name, _reserved) = struct.unpack(FILE_HEADER_FMT, raw)
        if magic != FILE_MAGIC:
            raise RecordingFormatError(f"not a pilot recording (magic {magic!r})")
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise RecordingFormatError(
                f"unsupported recording schema version {schema} (supported: {SUPPORTED_SCHEMA_VERSIONS})"
            )
        return {
            "magic": magic,
            "schema_version": schema,
            "dtype_tag": dtype,
            "sample_k": sample_k,
            "top_n": top_n,
            "session_ts": session_ts,
            "part": part,
            "session_mono": session_mono,
            "buffer_name": name.rstrip(b"\x00").decode(errors="replace"),
        }

    @staticmethod
    def read_record(fp) -> Optional[tuple]:
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
