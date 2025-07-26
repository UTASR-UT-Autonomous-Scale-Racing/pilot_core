#chmod +x setup.sh

# Load GitHub token
TOKEN=$(cat ~/Documents/git_token.git)

# Set GitHub username and organization name
userName="utasrCar"
orgName="UTASR-UT-Autonomous-Scale-Racing"

# Only clone repositories if they do not exist
for repo in pilot_ingest pilot_vision pilot_controller; do
    if [ ! -d "$(pwd)/$repo" ]; then
        echo "Cloning $repo ..."
        git clone "https://$userName:$TOKEN@github.com/$orgName/$repo.git" "$(pwd)/$repo"
    else
        echo "$repo already exists."
    fi
done
