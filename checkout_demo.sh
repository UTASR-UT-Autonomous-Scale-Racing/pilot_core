#chmod +x checkout_demo.sh

# Load GitHub token
TOKEN=$(cat ~/Documents/git_token.git)

# Set GitHub username and organization name
userName="utasrCar"
orgName="UTASR-UT-Autonomous-Scale-Racing"


if [ "$1" = "-demo" ]; then
    for repo in pilot_ingest pilot_vision pilot_controller pilot_core; do
        if [ -d "$(pwd)/$repo/.git" ]; then
            echo "Checking out 'demo' branch in $repo..."
            git -C "$(pwd)/$repo" fetch origin demo
            git -C "$(pwd)/$repo" checkout demo
        fi
    done
fi