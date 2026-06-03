#!/usr/bin/env python3
"""Evaluate a PPO run (rl/checkpoints/PPO_N) and pure_pursuit in rl/bicycle_env.BicycleTrackEnv.

    python rl/eval_bicycle.py --run PPO_7 [--step N] [--episodes 10] [--json out.json]
    python rl/eval_bicycle.py --run PPO_7 --onnx policy.onnx   # the exported graph instead

Per track (default: the saved AutoDRIVE practice + compete maps), episodes start at the
sim's spawn pose (jittered by the env's start noise) and run until a lap, a wall hit, being
stuck, or the time limit. Pure pursuit (v_max 4.0, config_autodrive.yaml gains, as in bench
Stage A/B) is run in the same env on the true pose, and also through
control.sim.run_closed_loop on the same plant (SimPlant) for the benchmark-style numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional
from unittest import mock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rl.bicycle_env import DEFAULT_MAPS, BicycleTrackEnv, SimPlant, SimPlantConfig, load_map  # noqa: E402

PILOT_CONFIG = os.path.join(REPO_ROOT, "config_autodrive.yaml")


def run_episodes(env: BicycleTrackEnv, act: Callable[[np.ndarray, BicycleTrackEnv], np.ndarray],
                 track: str, n: int, seed: int = 0, reset_fn: Optional[Callable] = None,
                 start="spawn") -> dict:
    """n episodes on one map dir with act(obs, env) -> action. Aggregate metrics."""
    laps, times, coll, stuck, xt, spd = [], [], 0, 0, [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep, options={"track": track, "start": start})
        if reset_fn:
            reset_fn()
        e2, v = [], []
        while True:
            obs, _, term, trunc, info = env.step(act(obs, env))
            e2.append(info["xtrack"] ** 2)
            v.append(info["speed"])
            if term or trunc:
                break
        laps.append(bool(info["lap_completed"]))
        if info["lap_completed"]:
            times.append(info["lap_time"])
        coll += bool(info["collided"])
        stuck += bool(info["stuck"])
        xt.append(np.mean(e2))
        spd.append(np.mean(v))
    return {
        "episodes": n,
        "lap_rate": float(np.mean(laps)),
        "lap_time_mean": float(np.mean(times)) if times else None,
        "lap_time_best": float(np.min(times)) if times else None,
        "collisions": int(coll),
        "stuck": int(stuck),
        "xtrack_rms": float(np.sqrt(np.mean(xt))),
        "speed_mean": float(np.mean(spd)),
    }


def sb3_actor(model, vec_normalize) -> Callable:
    def act(obs, env):
        o = vec_normalize.normalize_obs(obs[None]) if vec_normalize is not None else obs[None]
        return model.predict(o, deterministic=True)[0][0]
    return act


def onnx_actor(onnx_path: str, outer, inner) -> Callable:
    """Runs the exported graph through the on-car session setup (control.policy.PolicyFollower)."""
    from control.policy import PolicyFollower

    pf = PolicyFollower(onnx_path, outer, inner)

    def act(obs, env):
        return pf.session.run(None, {pf._input: obs[None].astype(np.float32)})[0][0]
    return act


def pilot_cfg(path: str = PILOT_CONFIG) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def pp_speed_cfg(cfg: dict, v_max: float) -> dict:
    speed = {k: v for k, v in (cfg.get("track_control", {}).get("speed") or {}).items()}
    speed["v_max"] = v_max
    return speed


def pure_pursuit_actor(map_dir, v_max: float = 4.0, cfg: Optional[dict] = None):
    """(act, reset) running the on-car pure_pursuit TrackFollower on the env's true pose.
    map_dir: saved map dir or (outer, inner)."""
    from control.sim import make_follower_factory
    from control.types import VehicleParams, VehicleState

    cfg = cfg or pilot_cfg()
    params = VehicleParams.load(os.path.join(REPO_ROOT, cfg.get("vehicle_params", "vehicle_params_roboracer.yaml")))
    outer, inner = load_map(map_dir) if isinstance(map_dir, str) else map_dir
    follower = make_follower_factory("pure_pursuit", params, None, pp_speed_cfg(cfg, v_max))(outer, inner)

    def act(obs, env):
        x, y, yaw, v = env.true_pose()
        cmd = follower.step(VehicleState(t=env._t, x=x, y=y, yaw=yaw, v=v))
        return np.array([cmd.steering, max(cmd.throttle, 0.0)], np.float32)
    return act, follower.reset


def pure_pursuit_closed_loop(map_dir: str, safety: dict, v_max: float = 4.0, duration: float = 60.0,
                             seed: int = 0, cfg: Optional[dict] = None):
    """control.sim.run_closed_loop (pose sensor + EKF + SafetySupervisor at 50 Hz) on SimPlant."""
    import control.sim as csim
    from control.safety import SafetyConfig
    from control.types import VehicleParams

    cfg = cfg or pilot_cfg()
    params = VehicleParams.load(os.path.join(REPO_ROOT, cfg.get("vehicle_params", "vehicle_params_roboracer.yaml")))
    outer, inner = load_map(map_dir)
    factory = csim.make_follower_factory("pure_pursuit", params, None, pp_speed_cfg(cfg, v_max))
    with mock.patch.object(csim, "BicyclePlant", SimPlant):
        return csim.run_closed_loop(factory, outer, inner, params, duration=duration,
                                    plant=SimPlantConfig(), safety=SafetyConfig(**safety),
                                    control_rate=float(cfg.get("control_rate_hz", 50)), seed=seed)


def make_eval_env(env_kwargs: dict, bicycle_kwargs: Optional[dict] = None, max_steps: int = 600,
                  safety: Optional[dict] = None) -> BicycleTrackEnv:
    kw = dict(env_kwargs)
    kw.update(bicycle_kwargs or {})
    kw["max_steps"] = max_steps          # 10 Hz units: 60 s, room for a full compete lap
    if safety is not None:
        kw["safety"] = safety
    return BicycleTrackEnv(**kw)


def _fmt(name: str, r: dict) -> str:
    lt = f"{r['lap_time_mean']:.2f}" if r["lap_time_mean"] is not None else "-"
    lb = f"{r['lap_time_best']:.2f}" if r["lap_time_best"] is not None else "-"
    return (f"| {name} | {100 * r['lap_rate']:.0f}% | {lt} | {lb} | {r['collisions']}/{r['episodes']} | "
            f"{r['stuck']} | {r['xtrack_rms']:.3f} | {r['speed_mean']:.2f} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run name under rl/checkpoints/")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--onnx", help="evaluate this exported ONNX instead of the SB3 checkpoint")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--maps", nargs="*", default=list(DEFAULT_MAPS))
    ap.add_argument("--no-pp", action="store_true", help="skip the pure_pursuit baselines")
    ap.add_argument("--random", type=int, default=0,
                    help="also N held-out generated tracks (seeded apart from training), 1 episode each")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    from rl.checkpoints import find_checkpoint

    run_dir = os.path.join(REPO_ROOT, "rl", "checkpoints", args.run)
    with open(os.path.join(run_dir, "config.json")) as f:
        env_kwargs = json.load(f)["ENV_KWARGS"]
    bicycle_kwargs = {}
    bpath = os.path.join(run_dir, "bicycle_pretrain.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            bicycle_kwargs = json.load(f).get("BICYCLE_KWARGS", {})
    safety = env_kwargs["safety"]

    actor_for = None
    if args.onnx:
        label = f"policy (ONNX {os.path.basename(args.onnx)})"
        actor_for = lambda m: onnx_actor(args.onnx, *(load_map(m) if isinstance(m, str) else m))   # noqa: E731
    else:
        import pickle
        from stable_baselines3 import PPO
        model_path, vec_path = find_checkpoint(run_dir, args.step)
        model = PPO.load(model_path, device="cpu")
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
        vec.training = False
        label = f"policy ({os.path.basename(model_path)})"
        act = sb3_actor(model, vec)
        actor_for = lambda m: act   # noqa: E731

    results, rows = {}, []
    stage_a = dict(safety, speed_cap=4.5)   # bench Stage A: speed_cap = v_max + 0.5
    for m in args.maps:
        name = os.path.basename(os.path.normpath(m))
        env = make_eval_env(env_kwargs, bicycle_kwargs)
        r = run_episodes(env, actor_for(m), m, args.episodes)
        results[f"{name}/policy"] = r
        rows.append(_fmt(f"{name}: {label}, speed_cap {safety['speed_cap']}", r))
        if not args.no_pp:
            for cap_name, saf in ((f"speed_cap {safety['speed_cap']}", safety), ("speed_cap 4.5 (Stage A)", stage_a)):
                env = make_eval_env(env_kwargs, bicycle_kwargs, safety=saf)
                act_pp, reset_pp = pure_pursuit_actor(m)
                r = run_episodes(env, act_pp, m, args.episodes, reset_fn=reset_pp)
                results[f"{name}/pure_pursuit_env/{cap_name}"] = r
                rows.append(_fmt(f"{name}: pure_pursuit v_max 4.0 in env, {cap_name}", r))
                cl = [pure_pursuit_closed_loop(m, saf, seed=s) for s in range(min(args.episodes, 3))]
                lt = [t for c in cl for t in c.lap_times]
                results[f"{name}/pure_pursuit_run_closed_loop/{cap_name}"] = {
                    "laps_in_60s": float(np.mean([c.laps for c in cl])),
                    "lap_time_mean": float(np.mean(lt)) if lt else None,
                    "lap_time_first": float(np.mean([c.lap_times[0] for c in cl if c.lap_times])) if lt else None,
                    "xtrack_rms": float(np.mean([c.xtrack_rms for c in cl])),
                    "off_track_fraction": float(np.mean([c.off_track_fraction for c in cl])),
                    "speed_mean": float(np.mean([c.mean_speed for c in cl])),
                }
    if args.random:
        from rl.bicycle_env import generate_track
        rng = np.random.default_rng(987_654_321)
        tracks = [generate_track(rng) for _ in range(args.random)]
        env = make_eval_env(env_kwargs, bicycle_kwargs)
        agg = {"policy": [], "pp": []}
        for k, (o, i) in enumerate(tracks):
            act = actor_for((o, i)) if args.onnx else actor_for(None)
            agg["policy"].append(run_episodes(env, act, (o, i), 1, seed=k, start="first"))
            if not args.no_pp:
                act_pp, reset_pp = pure_pursuit_actor((o, i))
                agg["pp"].append(run_episodes(env, act_pp, (o, i), 1, seed=k, reset_fn=reset_pp, start="first"))
        for who, rs in agg.items():
            if not rs:
                continue
            lt = [r["lap_time_mean"] for r in rs if r["lap_time_mean"]]
            r = {"episodes": len(rs), "lap_rate": float(np.mean([r["lap_rate"] for r in rs])),
                 "lap_time_mean": float(np.mean(lt)) if lt else None, "lap_time_best": None,
                 "collisions": sum(r["collisions"] for r in rs), "stuck": sum(r["stuck"] for r in rs),
                 "xtrack_rms": float(np.sqrt(np.mean([r["xtrack_rms"] ** 2 for r in rs]))),
                 "speed_mean": float(np.mean([r["speed_mean"] for r in rs]))}
            results[f"random{args.random}/{who}"] = r
            rows.append(_fmt(f"{args.random} held-out random tracks: {label if who == 'policy' else 'pure_pursuit v_max 4.0 in env, speed_cap ' + str(safety['speed_cap'])}", r))
    print("| run | laps | lap time mean (s) | best (s) | collisions | stuck | xtrack RMS (m) | mean speed (m/s) |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    for k, v in results.items():
        if "run_closed_loop" in k:
            print(k, json.dumps(v))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
