"""
SharedRingBuffer concurrency: a reader should never get a header paired with a
payload from some other record (older, or half written).
"""

from __future__ import annotations

import multiprocessing as mp
import uuid

from shared_buffer.ring_buffer import SharedRingBuffer
from shared_buffer.schema import DTypeTag

PAYLOAD = 512 * 1024  # large enough that a copy is not instantaneous
SLOTS = 2


def _reader(name: str, ready, stop, out) -> None:
    rb = SharedRingBuffer(name=name, slots=SLOTS, slot_payload_size=PAYLOAD, create=False)
    ready.set()
    torn = reads = 0
    while not stop.is_set():
        result = rb.read_latest()
        if result is None:
            continue
        header, payload = result
        reads += 1
        expected = header.seq_no % 256
        # first/middle/last byte all have to match this header's record
        if (payload[0], payload[PAYLOAD // 2], payload[-1]) != (expected,) * 3:
            torn += 1
    rb.close()
    out.put((torn, reads))


def test_reader_never_sees_torn_payload():
    ctx = mp.get_context("spawn")
    name = f"rb_torn_{uuid.uuid4().hex[:8]}"
    rb = SharedRingBuffer(
        name=name, slots=SLOTS, slot_payload_size=PAYLOAD,
        dtype_tag=DTypeTag.UNKNOWN, create=True,
    )
    try:
        ready, stop, out = ctx.Event(), ctx.Event(), ctx.Queue()
        p = ctx.Process(target=_reader, args=(name, ready, stop, out))
        p.start()
        assert ready.wait(timeout=30), "reader process failed to start"
        for seq in range(1, 4001):
            rb.write(bytes([seq % 256]) * PAYLOAD)
        stop.set()
        torn, reads = out.get(timeout=30)
        p.join(timeout=10)
        assert reads > 0
        assert torn == 0, f"{torn}/{reads} reads returned a payload not matching their header"
    finally:
        rb.unlink()
