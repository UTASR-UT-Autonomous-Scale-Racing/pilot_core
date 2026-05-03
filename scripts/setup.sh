#!/usr/bin/env bash
# Initialise the submodules the quick start needs (firmware), at
# their pinned commits. ORB_SLAM3 (~930 MB) is opt-in.
#   scripts/setup.sh               # firmware
#   scripts/setup.sh --with-slam   # also ORB_SLAM3 (then build it, docs/slam.md)
set -euo pipefail
cd "$(dirname "$0")/.."

with_slam=0
for arg in "$@"; do
    case "$arg" in
        --with-slam) with_slam=1 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

wanted=(firmware)
[ "$with_slam" -eq 1 ] && wanted+=(ORB_SLAM3)

# only the submodules this checkout actually declares
modules=()
for name in "${wanted[@]}"; do
    if git config -f .gitmodules --get "submodule.${name}.path" >/dev/null 2>&1; then
        modules+=("$(git config -f .gitmodules --get "submodule.${name}.path")")
    fi
done

if [ "${#modules[@]}" -eq 0 ]; then
    echo "No submodules to initialise."
    exit 0
fi

echo "Initialising: ${modules[*]}"
git submodule update --init --recursive -- "${modules[@]}"

if [ "$with_slam" -eq 0 ]; then
    echo "ORB_SLAM3 skipped (SLAM needs it: scripts/setup.sh --with-slam, then docs/slam.md)."
fi
