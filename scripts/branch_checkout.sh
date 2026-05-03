#!/usr/bin/env bash
# Put pilot_core and each submodule on the same branch
#   scripts/branch_checkout.sh [branch]   (default: main)
set -euo pipefail
cd "$(dirname "$0")/.."

branch="${1:-main}"

for repo in firmware; do
    # submodules have a .git file, not a dir
    if [ -e "$repo/.git" ]; then
        echo "Checking out '$branch' in $repo ..."
        git -C "$repo" fetch origin "$branch"
        git -C "$repo" checkout "$branch"
    fi
done

echo "Checking out '$branch' in pilot_core ..."
git fetch origin "$branch"
git checkout "$branch"
