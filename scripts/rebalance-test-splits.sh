#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# scripts/rebalance-test-splits.sh
# Downloads latest test durations from CI and merges into .test_durations.
# Run locally after CI has stored durations, then commit the updated file.
#
# Usage:
#   ./scripts/rebalance-test-splits.sh
#
# Prerequisites:
#   - gh CLI authenticated
#   - At least one successful CI run with --store-durations
#
# OMN-6480: Rebalance test splits by execution time [F20]

set -euo pipefail

REPO="OmniNode-ai/omnibase_infra"
WORKFLOW="ci.yml"
SPLITS=10

echo "Downloading test duration artifacts from latest successful CI run..."
RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --status success --limit 1 --json databaseId --jq '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
    echo "ERROR: No successful CI runs found for $WORKFLOW"
    exit 1
fi

echo "Using run ID: $RUN_ID"

mkdir -p /tmp/test-durations
for split in $(seq 1 "$SPLITS"); do
    echo "  Downloading split $split/$SPLITS..."
    gh run download "$RUN_ID" --repo "$REPO" \
        --name "test-durations-split-${split}" \
        --dir "/tmp/test-durations/split-${split}" 2>/dev/null || {
        echo "  WARNING: No duration artifact for split $split (may not have run yet)"
    }
done

echo ""
echo "Merging durations..."

# pytest-split merges automatically when --store-durations is used.
# Find the largest (most complete) durations file across all splits.
BEST=$(find /tmp/test-durations -name ".test_durations" -exec ls -S {} + 2>/dev/null | head -1)

if [ -n "$BEST" ]; then
    cp "$BEST" .test_durations
    LINES=$(wc -l < .test_durations | tr -d ' ')
    echo "Updated .test_durations from $BEST ($LINES entries)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the .test_durations file"
    echo "  2. Commit: git add .test_durations && git commit -m 'chore: rebalance test split durations'"
    echo "  3. Push to trigger a CI run with balanced splits"
else
    echo "ERROR: No duration files found in any split."
    echo "Ensure CI has run with --store-durations at least once."
    exit 1
fi

# Cleanup
rm -rf /tmp/test-durations
