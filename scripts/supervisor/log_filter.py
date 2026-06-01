#!/usr/bin/env python3
"""stdin -> stdout line filter for the Unity player log: drops managed stack-trace lines
("   at ...") and rate-limits repeated messages (the sim logs a 'Connection refused'
storm, thousands per second, while no server is listening on :4567)."""

import re
import sys
import time

WINDOW = 10.0   # s; a repeated message is printed at most once per window, with a count
STAMP = re.compile(r"^\d\d/\d\d/\d{4} \d\d:\d\d:\d\d\|")

last: dict[str, tuple[float, int]] = {}
for line in sys.stdin:
    if line.startswith(("  ", "\t")) and line.lstrip().startswith("at "):
        continue
    key = STAMP.sub("", line.rstrip())
    now = time.monotonic()
    seen, n = last.get(key, (0.0, 0))
    if now - seen < WINDOW:
        last[key] = (seen, n + 1)
        continue
    suffix = f"   [x{n} more in the last {WINDOW:.0f} s]" if n else ""
    sys.stdout.write(line.rstrip() + suffix + "\n")
    sys.stdout.flush()
    last[key] = (now, 0)
    if len(last) > 5000:
        last.clear()
