"""
SafetySupervisor — the last stage between a controller and the actuators.

Every output tick it turns the newest command into the command actually
sent, applying (in order):

1. **Arming** — while disarmed: neutral throttle, centred steering.
2. **Watchdog** — if the newest command is older than ``command_timeout``:
   neutral throttle, steering held (the car coasts to a stop).
3. **Speed cap** — forward throttle tapers linearly to zero over the last
   ``cap_margin`` m/s below ``speed_cap`` and is zero above it (the car
   coasts).  The taper absorbs the lag of a pose-derived speed estimate.
   Without a speed estimate, forward throttle is limited to
   ``max_throttle_no_speed``.  Independently of any estimate, forward
   throttle never exceeds ``speed_cap / speed_per_throttle`` — the
   steady-state throttle for the cap — so estimator lag cannot let the car
   settle above it (requires a calibrated ``speed_per_throttle``).
4. **Slew limits** on steering and throttle.

Works in signed units (steering +1 = left, throttle 0 = neutral).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Command


@dataclass
class SafetyConfig:
    command_timeout: float = 0.150   # s
    speed_cap: float = 1.5           # m/s
    cap_margin: float = 0.2          # m/s — forward throttle tapers to 0 over this band
    speed_per_throttle: Optional[float] = 3.0   # m/s per unit throttle (calibrate); None disables the bound
    max_throttle: float = 1.0        # forward throttle cap
    max_throttle_no_speed: float = 0.4
    steering_rate: float = 7.6       # signed units/s (~max_steering_vel / max_steering_angle)
    throttle_rate: float = 4.0       # signed units/s


@dataclass
class SafetyStatus:
    armed: bool
    watchdog_tripped: bool
    speed_limited: bool


class SafetySupervisor:
    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self._out = Command(0.0, 0.0)
        self._last_t: Optional[float] = None
        self.status = SafetyStatus(False, False, False)

    @property
    def output(self) -> Command:
        return Command(self._out.steering, self._out.throttle)

    def step(
        self,
        now: float,
        command: Optional[Command],
        command_time: Optional[float],
        armed: bool,
        speed: Optional[float] = None,
    ) -> Command:
        c = self.cfg
        dt = 0.0 if self._last_t is None else max(now - self._last_t, 0.0)
        self._last_t = now

        stale = command is None or command_time is None or (now - command_time) > c.command_timeout
        speed_limited = False

        if not armed:
            target = Command(0.0, 0.0)
        elif stale:
            target = Command(self._out.steering, 0.0)
        else:
            target = command.clipped()
            if speed is None:
                fwd_cap = min(c.max_throttle, c.max_throttle_no_speed)
            else:
                taper = np.clip((c.speed_cap - speed) / max(c.cap_margin, 1e-6), 0.0, 1.0)
                fwd_cap = c.max_throttle * float(taper)
            if c.speed_per_throttle:
                fwd_cap = min(fwd_cap, c.speed_cap / c.speed_per_throttle)
            if target.throttle > fwd_cap:
                target.throttle = fwd_cap
                speed_limited = True

        if not armed:
            # Disarming is immediate — no slew on the way to neutral
            self._out = Command(0.0, 0.0)
        else:
            self._out = Command(
                _slew(self._out.steering, target.steering, c.steering_rate * dt),
                # Dropping to neutral / coasting is never slew-limited
                target.throttle if (target.throttle == 0.0 or abs(target.throttle) < abs(self._out.throttle)
                                    and np.sign(target.throttle) == np.sign(self._out.throttle))
                else _slew(self._out.throttle, target.throttle, c.throttle_rate * dt),
            )
        self.status = SafetyStatus(armed, armed and stale, speed_limited)
        return self.output


def _slew(current: float, target: float, max_step: float) -> float:
    return float(current + np.clip(target - current, -max_step, max_step))
