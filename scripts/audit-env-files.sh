#!/bin/bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Audit script to find all .env files (excluding .env.example)
set -euo pipefail

OMNI_HOME="${OMNI_HOME:-.}"
cd "$OMNI_HOME"

echo "Scanning for rogue .env files..."
echo ""

found=0
while IFS= read -r file; do
    found=$((found + 1))
    # Check if file is tracked in git
    repo_dir=$(dirname "$file")
    if git -C "$repo_dir" rev-parse --git-dir > /dev/null 2>&1; then
        repo_root=$(git -C "$repo_dir" rev-parse --show-toplevel)
        rel_path=$(python3 -c "import os.path; print(os.path.relpath('$file', '$repo_root'))")
        if git -C "$repo_root" ls-files --error-unmatch "$rel_path" > /dev/null 2>&1; then
            status="⛔ COMMITTED"
        else
            status="⚠️  UNTRACKED"
        fi
    else
        status="❓ NOT IN GIT REPO"
    fi
    printf "%s  %s\n" "$status" "$file"
done < <(find "$OMNI_HOME" -type f -name ".env*" ! -name ".env.example" ! -name ".env.*.example" 2>/dev/null | sort)

if [ $found -eq 0 ]; then
    echo "✅ No rogue .env files found"
    exit 0
else
    echo ""
    echo "⚠️  Found $found rogue .env file(s)"
    exit 1
fi
