import csv, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
C = {"imu_ekf": "#2a78d6", "pose_ekf": "#eb6834", "slam_raw": "#1baf7a"}
LBL = {"imu_ekf": "Rear IMU + SLAM (ImuEKF, new)", "pose_ekf": "SLAM poses only (PoseEKF, car today)", "slam_raw": "Latest SLAM pose, no filter"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True})

r = list(csv.DictReader(open("/workspace/bench/results/fig_bo0/ticks.csv")))
t0 = float(r[0]["t"])
t = np.array([float(x["t"]) for x in r]) - t0
f = lambda k: np.array([float(x[k]) if x[k] not in ("", "nan") else np.nan for x in r])
err = {n: np.hypot(f(n + "_x") - f("truth_x"), f(n + "_y") - f("truth_y")) for n in C}
gaps = [(3.0, 3.3), (9.0, 9.3)]

fig, ax = plt.subplots(4, 1, figsize=(11, 13), gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1]})
fig.suptitle("Dual IMU in the sim: rear-of-car IMU + ORB-SLAM3 vs the car's current estimator\n"
             "rendered race laps, two 0.3 s camera blackouts (grey), what the controller would see at each 100 Hz tick",
             fontsize=12, color=INK, x=0.01, ha="left")
def shade(a, lo, hi):
    for g0, g1 in gaps:
        if g1 > lo and g0 < hi:
            a.axvspan(g0, g1, color="#d9d8d4", alpha=0.7, lw=0)
for n in ("slam_raw", "pose_ekf", "imu_ekf"):
    ax[0].plot(t, err[n], color=C[n], label=LBL[n], lw=1.6 if n != "imu_ekf" else 2.2)
shade(ax[0], 0, 33)
ax[0].set_ylabel("position error (m)"); ax[0].set_xlim(0, t[-1]); ax[0].set_ylim(0, 4.5)
ax[0].set_title("Position error, whole run", loc="left", color=INK, fontsize=11)
ax[0].legend(loc="upper right")
m = (t > 2.6) & (t < 4.6)
for n in ("slam_raw", "pose_ekf", "imu_ekf"):
    ax[1].plot(t[m], err[n][m], color=C[n], lw=1.6 if n != "imu_ekf" else 2.4)
shade(ax[1], 2.6, 4.6)
ax[1].set_ylabel("position error (m)"); ax[1].set_xlim(2.6, 4.6); ax[1].set_ylim(0, 2.3)
ax[1].set_title("Zoom on the first blackout: the IMU carries the pose; SLAM's bad first poses after it are rejected", loc="left", color=INK, fontsize=11)
ax[1].annotate("blackout", (3.15, 2.1), ha="center", color=INK2)
ax[1].annotate("SLAM's first poses after the blackout\nare ~1.2 m off; ImuEKF rejects them", (3.75, 1.45), color=INK2, fontsize=9)
m = (t > 5) & (t < 13)
ax[2].plot(t[m], np.degrees(f("pose_ekf_r")[m]), color=C["pose_ekf"], lw=1.4)
ax[2].plot(t[m], np.degrees(f("imu_ekf_r")[m]), color=C["imu_ekf"], lw=2.2)
ax[2].plot(t[m], np.degrees(f("truth_yaw_rate")[m]), color=INK, lw=1.2, ls="--", label="truth")
shade(ax[2], 5, 13)
ax[2].set_ylabel("yaw rate (deg/s)"); ax[2].set_xlim(5, 13); ax[2].set_ylim(-200, 200)
ax[2].set_title("Yaw rate: gyro (blue) vs differenced poses (orange); truth dashed", loc="left", color=INK, fontsize=11)
ax[2].legend(loc="upper right")
ax[3].plot(t[m], f("pose_ekf_v")[m], color=C["pose_ekf"], lw=1.4)
ax[3].plot(t[m], f("imu_ekf_v")[m], color=C["imu_ekf"], lw=2.2)
ax[3].plot(t[m], f("truth_speed")[m], color=INK, lw=1.2, ls="--", label="truth")
shade(ax[3], 5, 13)
ax[3].set_ylabel("speed (m/s)"); ax[3].set_xlabel("time (s)"); ax[3].set_xlim(5, 13); ax[3].set_ylim(0, 8)
ax[3].set_title("Speed", loc="left", color=INK, fontsize=11)
ax[3].legend(loc="upper left")
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("/workspace/bench/results/fig_dual_imu.png", dpi=110)

# SLAM comparison: two charts, one axis each
S = json.load(open("/workspace/bench/results/slam_summary.json"))
seqs = list(S)
systems = [("ORB-SLAM3 CUDA", "#2a78d6"), ("cuVSLAM", "#eb6834"), ("cuVSLAM odometry", "#1baf7a")]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("RGB-D + IMU on the rendered race laps: ORB-SLAM3 CUDA (our fork) vs cuVSLAM, real-time, 8 cores, L40S",
             fontsize=12, color=INK, x=0.01, ha="left")
x = np.arange(len(seqs)); w = 0.26
for k, (sname, col) in enumerate(systems):
    vals = [S[s][sname]["ate"] for s in seqs]
    b = ax[0].bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=sname + (" (median of %s runs)" % "/".join(str(S[s][sname]["n"]) for s in seqs) if sname.startswith("ORB") else ""))
    for xi, v in zip(x + (k - 1) * w, vals):
        ax[0].annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=9, color=INK2, xytext=(0, 2), textcoords="offset points")
    p95 = [S[s][sname]["p95"] for s in seqs]
    ax[1].bar(x + (k - 1) * w, p95, w - 0.03, color=col, label=sname)
    for xi, v in zip(x + (k - 1) * w, p95):
        ax[1].annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=9, color=INK2, xytext=(0, 2), textcoords="offset points")
ax[0].set_xticks(x, seqs); ax[0].set_ylabel("position error after alignment, ATE RMSE (m)")
ax[0].set_title("Accuracy (lower is better)", loc="left", color=INK, fontsize=11); ax[0].legend(loc="upper left")
ax[1].set_yscale("log"); ax[1].set_xticks(x, seqs); ax[1].set_ylabel("tracking call p95 (ms, log scale)")
ax[1].set_title("Latency p95 (lower is better)", loc="left", color=INK, fontsize=11); ax[1].legend(loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("/workspace/bench/results/fig_slam_compare.png", dpi=110)
print("ok")
