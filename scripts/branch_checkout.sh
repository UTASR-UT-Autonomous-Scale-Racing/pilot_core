#!/usr/bin/env bash
# Check out the same branch in pilot_core and every component submodule.
#   scripts/branch_checkout.sh [branch]   (default: main)
set -euo pipefail
cd "$(dirname "$0")/.."

branch="${1:-main}"

for repo in pilot_ingest pilot_vision pilot_controller firmware; do
    # Submodules have a .git *file*, not a directory
    if [ -e "$repo/.git" ]; then
        echo "Checking out '$branch' in $repo ..."
        git -C "$repo" fetch origin "$branch"
        git -C "$repo" checkout "$branch"
    fi
done

echo "Checking out '$branch' in pilot_core ..."
git fetch origin "$branch"
git checkout "$branch"
