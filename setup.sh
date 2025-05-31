# Only clone repositories if they do not exist
for repo in pilot_ingest pilot_vision pilot_control; do
    if [ ! -d "$(pwd)/$repo" ]; then
        echo "$repo does not exist. Cloning..."
        git clone "https://github.com/UTASR-UT-Autonomous-Scale-Racing/$repo.git" "$(pwd)/$repo"
    fi
done

echo "Repositories downloaded if not present."
