#chmod +x checkout_demo.sh

# Switch to the demo branch in each repository
for repo in pilot_ingest pilot_vision pilot_controller; do
    if [ -d "$(pwd)/$repo/.git" ]; then
        echo "Checking out 'demo' branch in $repo..."
        git -C "$(pwd)/$repo" fetch origin demo
        git -C "$(pwd)/$repo" checkout demo
    fi
# Switch to the demo branch in pilot_core
echo "Checking out 'demo' branch in pilot_core..."
git -C "$(pwd)" fetch origin demo
git -C "$(pwd)" checkout demo
done