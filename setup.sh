#chmod +x setup.sh

# Load GitHub token
TOKEN=$(cat ~/Documents/git_token.git)

# Set GitHub username and organization name
userName="utasrCar"
orgName="UTASR-UT-Autonomous-Scale-Racing"

# Only clone repositories if they do not exist
for repo in pilot_ingest pilot_vision pilot_controller pilot_core; do
    if [ ! -d "$(pwd)/$repo" ]; then
        echo "$repo does not exist. Cloning..."
        git clone "https://$userName:$TOKEN@github.com/$orgName/$repo.git" "$(pwd)/$repo"
    fi
done

echo "All repositories are up to date."
