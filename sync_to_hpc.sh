#!/bin/bash

# Local directory to sync
LOCAL_DIR="/Users/rubenmaarek/Desktop/Master-Thesis-NUS/variational-martingale-solver"
# Remote directory to sync to
REMOTE_DIR="e1124919@atlas9.nus.edu.sg:/home/svu/e1124919/Desktop/variational-martingale-solver"

# Use rsync to sync directories, excluding unnecessary files and directories
rsync -av \
  --exclude 'model' \
  --exclude 'Old' \
  --exclude 'result_app' \
  --exclude 'test' \
  --exclude '.DS_Store' \
  --exclude '.git/' \
  --exclude '.dvc/' \
  --exclude 'July_Notebook.ipynb' \
  --exclude 'monte_carlo_simulation.png' \
  --exclude 'README.md' \
  --exclude '.python-version' \
  "$LOCAL_DIR/" "$REMOTE_DIR/"

# Output the result
if [ $? -eq 0 ]; then
    echo "Directory synced successfully."
else
    echo "Error syncing directory."
fi
