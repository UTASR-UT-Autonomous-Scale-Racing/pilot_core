#!/bin/bash
#SBATCH --job-name=ad_race
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=1:40:00
#SBATCH --exclude=rack13-08
#SBATCH --output=/scratch/dandebru/pilot_core/logs/%x_%j.out
# AutoDRIVE sim on a Slurm GPU node, no Docker or supervisor: Xvfb, the sim in a
# relaunch loop, the display tools from scripts/cluster/bin. Runs STAGES from this file.
#   sbatch [--export=ALL,STAGES="map race"] scripts/cluster/autodrive_job.sh
BASE=${AD_BASE:-/scratch/dandebru/pilot_core}
cd "$BASE/repo" || exit 1
export AUTODRIVE_ROOT=$BASE/root AUTODRIVE_LOG_DIR=$BASE/root/logs
mkdir -p "$AUTODRIVE_ROOT/autodrive_sim" "$AUTODRIVE_LOG_DIR" runs
for v in compete practice; do ln -sfn "$BASE/sim/$v" "$AUTODRIVE_ROOT/autodrive_sim/$v"; done
export PATH=$BASE/repo/scripts/cluster/bin:$BASE/venv/bin:/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin:$PATH
export DISPLAY=:$((100 + ${SLURM_JOB_ID:-0} % 800)) OMP_NUM_THREADS=2
TAG=${TAG:-$(date +%m%d)}
mkdir -p "runs/$TAG"; : > "$AUTODRIVE_LOG_DIR/autodrive_sim.log"
[ -f "$AUTODRIVE_ROOT/autodrive_sim/sim.env" ] || echo SIM_VARIANT=compete > "$AUTODRIVE_ROOT/autodrive_sim/sim.env"

Xvfb "$DISPLAY" -screen 0 1280x720x24 +extension GLX +render -noreset & XP=$!
sleep 3
STOP=/tmp/ad_stop_${SLURM_JOB_ID:-0}
( while [ ! -e "$STOP" ]; do bash scripts/supervisor/autodrive_sim.sh >> "$AUTODRIVE_LOG_DIR/autodrive_sim.log" 2>&1; sleep 1; done ) &
trap 'touch "$STOP"; pkill -f "AutoDRIVE Simulator.x86_64"; kill $XP' EXIT

variant() {   # switch build (restarts the sim) and wait for it
    timeout -k 10 300 python - "$1" <<'PY'
import sys
sys.path.insert(0, "scripts")
import autodrive_bench as b
b.ensure_sim_variant(sys.argv[1])
PY
}
wait_sim() {  # give up rather than sit out the job's time limit
    local t=0
    until grep -q UnloadTime "$AUTODRIVE_LOG_DIR/autodrive_sim.log"; do
        sleep 2; t=$((t + 2))
        [ $t -ge 180 ] && { echo "sim not up after ${t} s"; tail -5 "$AUTODRIVE_LOG_DIR/autodrive_sim.log"; exit 1; }
    done
    timeout -k 10 120 python -c "import sys; sys.path.insert(0, 'scripts'); import autodrive_sim_ctl as c; c.wait_scene()" || exit 1
}
ultra() { python scripts/autodrive_sim_ctl.py menu; sleep 1; python scripts/autodrive_sim_ctl.py quality; sleep 2
          python scripts/autodrive_sim_ctl.py quality; sleep 3; python scripts/autodrive_sim_ctl.py close-menu; }
bench() {     # name, args
    local n=$1; shift
    timeout -k 20 700 python scripts/autodrive_bench.py run --out "runs/$TAG/$n" "$@" > "runs/$TAG/$n.json" 2>&1
    echo "== $n rc=$? $(tr -d ' \n' < "runs/$TAG/$n.json" | cut -c1-400)"
}
wait_sim
import -window root "runs/$TAG/start.png" || echo "no screenshot"

for stage in ${STAGES:-map race}; do
  case $stage in
  map)      # segmentation mapping lap on the compete build, then the racing line
    variant compete
    bench map --laps 1 --timeout 400 --set enable_slam=true --set use_saved_map=false \
        --set mapping.boundary_save_path=maps/autodrive_compete_seg
    python scripts/make_raceline.py maps/autodrive_compete_seg --margin 0.5 || exit 1 ;;
  sweep)    # mapping-lap settings, first to close a lap without a hit wins: a hit respawns
            # the car and the wall map is scribbled over around that corner
    variant compete
    D=track_control.mapping_drive; M=maps/autodrive_compete_seg; n=0; found=
    for v in "$D.wall_margin=0.6 $D.wall_gain=1.5 seg_scan.memory_s=3" \
             "$D.wall_margin=0.6 $D.wall_gain=1.5 $D.bubble=0.6 seg_scan.memory_s=3" \
             "$D.wall_margin=0.7 $D.wall_gain=2.5 $D.speed=0.5 $D.min_speed=0.3 seg_scan.memory_s=4" \
             "$D.wall_margin=0.7 $D.wall_gain=2.5 $D.bubble=0.65 $D.speed=0.5 $D.min_speed=0.3 seg_scan.memory_s=4" \
             "$D.wall_margin=0.8 $D.wall_gain=3 $D.bubble=0.7 $D.speed=0.4 $D.min_speed=0.3 seg_scan.memory_s=5"; do
        n=$((n + 1)); rm -rf "$M.try"; sets=
        for kv in $v; do sets="$sets --set $kv"; done
        echo "-- sweep$n $v"
        bench sweep$n --laps 1 --timeout 200 --set enable_slam=true --set use_saved_map=false \
            --set mapping.boundary_save_path=$M.try $sets
        hits=$(python -c "import json; print(json.load(open('runs/$TAG/sweep$n.json')).get('collisions', -1))" 2>/dev/null)
        if [ "$hits" = 0 ] && [ -f "$M.try/outer_line.npy" ]; then found=$n; break; fi
    done
    [ -n "$found" ] || { echo "no clean mapping lap"; exit 1; }
    rm -rf "$M"; mv "$M.try" "$M"; echo "sweep: variant $found"
    python scripts/make_raceline.py "$M" --margin 0.5 || exit 1 ;;
  segmap)   # map from recorded laps ($SEG_RUNS, recording dirs), hits or not; stops if the line
            # it gives runs near a wall. The LiDAR map is only a reference in its report
    timeout -k 10 900 python scripts/seg_map.py $SEG_RUNS --out maps/autodrive_compete_seg \
        --reference maps/autodrive_compete || exit 1
    timeout -k 10 1200 python scripts/make_raceline.py maps/autodrive_compete_seg --margin ${MARGIN:-0.65} \
        | tee "runs/$TAG/raceline.txt" || exit 1
    python - "runs/$TAG/raceline.txt" <<'PY' || { echo "racing line too close to a wall"; exit 1; }
import re, sys
c = float(re.search(r"closest wall ([0-9.]+)", open(sys.argv[1]).read()).group(1))
print("closest wall", c)
sys.exit(c < 0.3)
PY
    ;;
  race)     # the final, as the config stands, 3 runs
    variant compete
    for i in 1 2 3; do bench race$i --config config_autodrive_compete.yaml --laps 10 --timeout 420; done ;;
  start)    # as the competition container starts: sim restarted, auto_arm, no keys
    variant compete
    bench compstart --config config_autodrive_compete.yaml --competition-start --laps 10 --timeout 420 ;;
  ultra)
    variant compete; ultra
    bench ultra --config config_autodrive_compete.yaml --laps 10 --timeout 420 ;;
  live)     # in-loop mapping lap on the occupancy walls, then 10 laps on that map, 3 times on
            # $VARIANT (practice | compete). Each map run goes to maps/autodrive_<variant>_live<n>
    V=${VARIANT:-compete}; variant "$V"
    for i in 1 2 3; do
        M=maps/autodrive_${V}_live$i; rm -rf "$M"
        bench map_$i --config config_autodrive_compete.yaml --laps 1 --timeout 600 \
            --set enable_slam=true --set use_saved_map=false --set mapping.boundary_save_path=$M \
            --set track_control.raceline_path=null
        [ -f "$M/outer_line.npy" ] || { echo "map_$i: no map"; continue; }
        timeout -k 10 1200 python scripts/make_raceline.py "$M" --margin ${MARGIN:-0.65} \
            | tee "runs/$TAG/raceline_$i.txt" || continue
        bench race_$i --config config_autodrive_compete.yaml --laps 10 --timeout 420 \
            --set saved_map_path=$M --set track_control.raceline_path=$M/raceline.npy
    done ;;
  degraded) # the final on the saved map with the pose degraded like SLAM's, 3 runs
    variant compete
    for i in 1 2 3; do bench degraded$i --config config_autodrive_compete.yaml --laps 10 --timeout 420 \
        --set sim_pose_degrade.enabled=true --set sim_pose_degrade.seed=$i; done ;;
  qual)     # qualifier on the practice build's segmentation map
    variant practice
    bench qual --config config_autodrive_compete.yaml --laps 10 --timeout 420 \
        --set saved_map_path=maps/autodrive_practice_seg \
        --set track_control.raceline_path=maps/autodrive_practice_seg/raceline.npy ;;
  esac
done
echo JOB_DONE
