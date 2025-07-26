#chmod +x branch_checkout.sh

branch="${1:-main}"
branch="${branch:1}"
# Switch to the main branch in each repository
for repo in pilot_ingest pilot_vision pilot_controller; do
    if [ -d "$(pwd)/$repo/.git" ]; then
        echo "Checking out $branch branch in $repo..."
        git -C "$(pwd)/$repo" fetch origin $branch
        git -C "$(pwd)/$repo" checkout $branch
    fi
done
# Switch to the main branch in pilot_core
echo "Checking out '$branch' branch in pilot_core..."
git -C "$(pwd)" fetch origin $branch
git -C "$(pwd)" checkout $branch