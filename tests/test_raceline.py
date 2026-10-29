"""Minimum-curvature racing line (control/raceline.py) and TrackModel.set_path."""

from __future__ import annotations

import numpy as np
import pytest

from control import TrackModel
from control.raceline import min_curvature_line
from control.sim import oval_track, twisty_track
from control.speed import speed_profile


def _lap_estimate(track: TrackModel) -> float:
    v = speed_profile(track, v_max=12.0, a_lat_max=8.0, a_accel=5.0, a_decel=5.0)
    ds = np.diff(np.concatenate([track.s, [track.length]]))
    return float((ds / v).sum())


@pytest.mark.parametrize("make", [oval_track, twisty_track])
def test_line_stays_inside_and_is_faster(make):
    outer, inner, _ = make()
    centre = TrackModel(outer, inner)
    margin = 0.3
    line = min_curvature_line(centre, outer, inner, margin=margin)
    raced = TrackModel(outer, inner)
    raced.set_path(line)

    assert raced.half_width.min() > margin - 0.08     # smoothing may shave a few cm
    assert raced.length < centre.length
    # minimising total curvature need not lower the peak (an ellipse's ends), but the
    # speed planner's lap time on the line must improve
    assert _lap_estimate(raced) < _lap_estimate(centre)


def test_set_path_keeps_direction_and_walls():
    outer, inner, _ = twisty_track()
    track = TrackModel(outer, inner)
    heading0 = track.heading[0]
    line = min_curvature_line(track, outer, inner, margin=0.3)
    track.set_path(line)
    p = track.project(*line[0])
    # same travel direction as the centre-line it came from
    assert abs(np.angle(np.exp(1j * (p.path_heading - heading0)))) < np.pi / 2
    # half_width is now measured from the line, so it is smaller on one side somewhere
    assert track.half_width.min() < TrackModel(outer, inner).half_width.min()
