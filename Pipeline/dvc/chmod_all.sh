#!/bin/bash

# Find all .sh and .pbs files in the current directory and subdirectories
# and apply chmod +x to make them executable
find . -type f \( -name "*.sh" -o -name "*.pbs" \) -exec chmod +x {} +

echo "All .sh and .pbs files have been made executable."
