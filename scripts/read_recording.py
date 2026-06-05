#!/usr/bin/env python3
"""Read recordings written by BufferRecorderWorker (start.py --recording).

    python scripts/read_recording.py /tmp/pilot_recordings
    python scripts/read_recording.py /tmp/pilot_recordings --buffer imu --dump 5
    from scripts.read_recording import iter_records, load_buffer
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_buffer import buffer_class_for  # noqa: E402
from shared_buffer.schema import DTypeTag  # noqa: E402
from workers.buffer_recorder_worker import BufferRecorderWorker  # noqa: E402


@dataclass
class RecordingFile:
    path: str
    buffer: str
    session: str
    part: int
    header: Dict[str, Any]


@dataclass
class Record:
    buffer: str
    seq_no: int
    produced_ts: float     # time.monotonic() of the sample (same clock for all buffers of a run)
    read_ts: float         # time.monotonic() when the recorder read it
    dtype_tag: int
    payload: bytes
    value: Any = None      # decode(payload), i.e. what read_latest() returns after the header
    session: str = ""      # session id from the file name


def _parse_name(path: str):
    """<buffer>_<session>_<part>.bin -> (buffer, session, part); also accepts
    single-part <buffer>_<session>.bin names."""
    stem = os.path.basename(path)[: -len(".bin")]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1], 0
    return stem, "", 0


def list_files(path: str, buffer: Optional[str] = None, session: Optional[str] = None) -> List[RecordingFile]:
    """Recording files under ``path`` (a directory or a single .bin), sorted
    by (buffer, session, part), i.e. chronologically per buffer. Files that
    aren't valid recordings are skipped with a warning."""
    if os.path.isdir(path):
        paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".bin")]
    else:
        paths = [path]
    out = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                header = BufferRecorderWorker.read_file_header(f)
        except (OSError, ValueError) as e:
            print(f"warning: skipping {p}: {e}", file=sys.stderr)
            continue
        name, sess, part = _parse_name(p)
        name = header["buffer_name"] or name
        if header["schema_version"] >= 2:
            part = header["part"]
        if buffer is not None and name != buffer:
            continue
        if session is not None and sess != session:
            continue
        out.append(RecordingFile(p, name, sess, part, header))
    out.sort(key=lambda r: (r.buffer, r.session, r.part))
    return out


def _decoder(rf: RecordingFile):
    cls = buffer_class_for(rf.header["dtype_tag"], rf.buffer)
    return cls.decode if cls is not None else None


def iter_file(rf: RecordingFile, decode: bool = True, stats: Optional[dict] = None) -> Iterator[Record]:
    """Records of one file, in file order. If ``stats`` is given,
    stats["truncated_bytes"] is increased by any partial record at the end."""
    dec = _decoder(rf) if decode else None
    with open(rf.path, "rb") as f:
        BufferRecorderWorker.read_file_header(f)
        while True:
            end = f.tell()
            rec = BufferRecorderWorker.read_record(f)
            if rec is None:
                break
            h, payload = rec
            yield Record(
                buffer=rf.buffer,
                seq_no=h["seq_no"],
                produced_ts=h["produced_ts"],
                read_ts=h["read_ts"],
                dtype_tag=h["dtype_tag"],
                payload=payload,
                value=dec(payload) if dec is not None else None,
                session=rf.session,
            )
        if stats is not None:
            stats["truncated_bytes"] = stats.get("truncated_bytes", 0) + os.path.getsize(rf.path) - end


def iter_records(
    path: str, buffer: Optional[str] = None, session: Optional[str] = None, decode: bool = True,
) -> Iterator[Record]:
    """Every record under ``path``, buffer by buffer, each buffer in
    chronological order (seq_no order across its files)."""
    for rf in list_files(path, buffer, session):
        yield from iter_file(rf, decode)


def load_buffer(path: str, buffer: str, session: Optional[str] = None, decode: bool = True) -> List[Record]:
    """All records of one buffer as a list, sorted by produced_ts."""
    recs = list(iter_records(path, buffer, session, decode))
    recs.sort(key=lambda r: (r.produced_ts, r.seq_no))
    return recs


def summarize(path: str, buffer: Optional[str] = None, session: Optional[str] = None) -> Dict[str, dict]:
    """Per-buffer summary: files, records, seq range, time span, rate."""
    summary: Dict[str, dict] = {}
    for rf in list_files(path, buffer, session):
        s = summary.setdefault(rf.buffer, {
            "files": 0, "records": 0, "bytes": 0, "first_ts": None, "last_ts": None,
            "first_seq": None, "last_seq": None, "sample_k": rf.header["sample_k"],
            "dtype": DTypeTag(rf.header["dtype_tag"]).name if rf.header["dtype_tag"] in DTypeTag._value2member_map_ else str(rf.header["dtype_tag"]),
            "sessions": set(), "truncated_bytes": 0, "out_of_order": 0,
        })
        s["files"] += 1
        s["sessions"].add(rf.session)
        for rec in iter_file(rf, decode=False, stats=s):
            s["records"] += 1
            s["bytes"] += len(rec.payload)
            if s["last_seq"] is not None and rec.seq_no <= s["last_seq"]:
                s["out_of_order"] += 1
            s["first_ts"] = rec.produced_ts if s["first_ts"] is None else min(s["first_ts"], rec.produced_ts)
            s["last_ts"] = rec.produced_ts if s["last_ts"] is None else max(s["last_ts"], rec.produced_ts)
            if s["first_seq"] is None:
                s["first_seq"] = rec.seq_no
            s["last_seq"] = rec.seq_no
    for s in summary.values():
        span = (s["last_ts"] - s["first_ts"]) if s["records"] else 0.0
        s["duration_s"] = span
        s["rate_hz"] = (s["records"] - 1) / span if s["records"] > 1 and span > 0 else 0.0
        s["sessions"] = sorted(s["sessions"])
    return summary


def print_summary(summary: Dict[str, dict]) -> None:
    if not summary:
        print("no recordings found")
        return
    print(f"{'buffer':<14}{'dtype':<14}{'files':>6}{'records':>9}{'MB':>9}{'span s':>9}{'rate Hz':>9}  seq range   notes")
    for name, s in sorted(summary.items()):
        notes = []
        if s["sample_k"] > 1:
            notes.append(f"sample_k={s['sample_k']}")
        if len(s["sessions"]) > 1:
            notes.append(f"{len(s['sessions'])} sessions")
        if s["truncated_bytes"]:
            notes.append(f"{s['truncated_bytes']} B truncated tail")
        if s["out_of_order"]:
            notes.append(f"{s['out_of_order']} out-of-order records")
        seq = f"{s['first_seq']}-{s['last_seq']}" if s["records"] else "-"
        print(f"{name:<14}{s['dtype']:<14}{s['files']:>6}{s['records']:>9}{s['bytes'] / 1e6:>9.1f}"
              f"{s['duration_s']:>9.1f}{s['rate_hz']:>9.1f}  {seq:<11} {', '.join(notes)}")


def _describe(value: Any) -> str:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    vals = value if isinstance(value, tuple) else (value,)
    out = []
    for v in vals:
        if np is not None and isinstance(v, np.ndarray):
            out.append(f"array{v.shape} {v.dtype}" if v.size > 9 else np.array2string(v, precision=3))
        else:
            out.append(repr(v))
    return ", ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="recording directory or a single .bin file")
    ap.add_argument("--buffer", help="only this buffer (frame, imu, control, state, ...)")
    ap.add_argument("--session", help="only this session id (the number in the file names)")
    ap.add_argument("--dump", type=int, default=0, metavar="N", help="also print the first N decoded records per buffer")
    args = ap.parse_args(argv)

    print_summary(summarize(args.path, args.buffer, args.session))
    if args.dump:
        shown: Dict[str, int] = {}
        for rf in list_files(args.path, args.buffer, args.session):
            for rec in iter_file(rf):
                if shown.get(rec.buffer, 0) >= args.dump:
                    break
                shown[rec.buffer] = shown.get(rec.buffer, 0) + 1
                print(f"{rec.buffer} seq={rec.seq_no} t={rec.produced_ts:.3f}: {_describe(rec.value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
