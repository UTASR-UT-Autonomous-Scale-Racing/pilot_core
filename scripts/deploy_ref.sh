# Sourced by jetson_run.sh and start.sh: check out what the car runs.
#   PILOT_REF=<tag|branch|commit>   that ref
#   otherwise                       the newest car-v* release tag, or main fast-forwarded if there is none
# Submodules follow the commits that ref pins. Refuses to run with uncommitted changes.
update_repo() {
    if ! git diff --quiet --ignore-submodules || ! git diff --cached --quiet --ignore-submodules; then
        echo "Error: pilot_core has uncommitted changes; commit or stash them first." >&2
        exit 1
    fi
    git fetch --quiet --tags origin
    local ref="${PILOT_REF:-$(git tag -l 'car-v*' --sort=-v:refname | head -n 1)}"
    if [ -n "$ref" ]; then
        git checkout --quiet --detach "$ref"
    else
        git checkout --quiet main
        git pull --quiet --ff-only
        ref=main
    fi
    git submodule sync --quiet
    git submodule update --init --recursive
    echo "pilot_core at $ref ($(git rev-parse --short HEAD)); submodules:"
    git submodule status
}

# start.py's --config from the arguments, else config.yaml
config_from_args() {
    local cfg="" i
    local -a a=("$@")
    for ((i = 0; i < ${#a[@]}; i++)); do
        case "${a[$i]}" in
            --config) cfg="${a[$((i + 1))]:-}" ;;
            --config=*) cfg="${a[$i]#--config=}" ;;
        esac
    done
    echo "${cfg:-config.yaml}"
}
