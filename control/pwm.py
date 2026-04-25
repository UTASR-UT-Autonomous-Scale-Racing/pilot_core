"""
PWM mapping for the steering servo and the ESC.

ControlBuffer values in [0, 1] map linearly onto each channel's [min, max]
pulse width in us. Below neutral is left / backward, above is right /
forward. Widths come from the `pwm` config section; defaults are calibrated
for our car. Throttle 0.5 lands exactly on neutral. Steering neutral is a bit
off the midpoint, and that's intentional.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PWMConfig:
    steer_min: int = 1105        # µs, full left
    steer_max: int = 1825        # µs, full right
    steer_neutral: int = 1461    # µs, straight
    throttle_min: int = 1199     # µs, full backward
    throttle_max: int = 1595     # µs, full forward
    throttle_neutral: int = 1397  # µs, neutral (the car coasts)


class PWMMapper:
    def __init__(self, cfg: PWMConfig | None = None) -> None:
        self.cfg = cfg or PWMConfig()

    def to_pwm(self, steering: float, throttle: float) -> tuple[int, int]:
        """ControlBuffer [0, 1] values -> (steering_us, throttle_us)."""
        if not (0.0 <= steering <= 1.0 and 0.0 <= throttle <= 1.0):
            raise ValueError(f"PWM inputs must be in [0, 1], got ({steering}, {throttle})")
        c = self.cfg
        return (
            int(c.steer_min + steering * (c.steer_max - c.steer_min)),
            int(c.throttle_min + throttle * (c.throttle_max - c.throttle_min)),
        )

    @property
    def neutral(self) -> tuple[int, int]:
        return self.cfg.steer_neutral, self.cfg.throttle_neutral
