#!/usr/bin/env bash
# ONEX Nightly Producer - Generate incremental diff for daily review
set -euo pipefail

# Configuration
EXCLUDES_REGEX='(^|/)(archive|archived|deprecated|dist|build|node_modules|venv|\.venv|\.mypy_cache|\.pytest_cache)(/|$)'
INCLUDE_EXT='py|pyi|yaml|yml|toml|json|ini|cfg|sh|bash|ps1|psm1|go|rs|ts|tsx|js|jsx|proto|sql|md'

# Marker file to track last successful run
PREV_FILE=".onex_nightly_prev"
REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR=".onex_nightly/$REPO_NAME/$TIMESTAMP"

echo "ONEX Nightly Producer for $REPO_NAME"
echo "Timestamp: $TIMESTAMP"

# Fetch latest changes
echo "Fetching latest from origin..."
git fetch origin

# Initialize or read previous SHA
if [[ ! -f "$PREV_FILE" ]]; then
  echo "First run - initializing with current main"
  echo "$(git rev-parse origin/main)" > "$PREV_FILE"
  echo "Run baseline producer first for initial review"
  exit 0
fi

PREV_SHA="$(cat "$PREV_FILE")"
HEAD_SHA="$(git rev-parse origin/main)"

# Check for changes
if [[ "$PREV_SHA" == "$HEAD_SHA" ]]; then
  echo "No changes since last run ($PREV_SHA)"
  exit 0
fi

echo "Changes detected:"
echo "  Previous: $PREV_SHA"
echo "  Current:  $HEAD_SHA"

# Create output directory
mkdir -p "$OUT_DIR"

# Generate file list for filtering
echo "Analyzing changed files..."
git diff --name-only "$PREV_SHA...$HEAD_SHA" \
  | while read -r file; do
    if [[ ! "$file" =~ $EXCLUDES_REGEX ]] && [[ "$file" =~ \.(${INCLUDE_EXT})$ ]]; then
      echo "$file"
    fi
  done > "$OUT_DIR/changed_files.list"

CHANGED_COUNT=$(wc -l < "$OUT_DIR/changed_files.list" 2>/dev/null || echo "0")

if [[ "$CHANGED_COUNT" -eq 0 ]]; then
  echo "No relevant files changed (all excluded or wrong extension)"
  # Still update marker since we processed successfully
  echo "$HEAD_SHA" > "$PREV_FILE"
  exit 0
fi

echo "Found $CHANGED_COUNT relevant changed files"

# Generate git outputs
echo "Generating diff outputs..."

# Stats for all changes
git diff --stat "$PREV_SHA...$HEAD_SHA" -- $(cat "$OUT_DIR/changed_files.list") > "$OUT_DIR/nightly.stats" || true

# Name status for all changes
git diff --name-status "$PREV_SHA...$HEAD_SHA" -- $(cat "$OUT_DIR/changed_files.list") > "$OUT_DIR/nightly.names" || true

# Unified diff with minimal context
git diff -U1 --no-color "$PREV_SHA...$HEAD_SHA" -- $(cat "$OUT_DIR/changed_files.list") > "$OUT_DIR/nightly.diff" || true

DIFF_SIZE=$(wc -c < "$OUT_DIR/nightly.diff")
echo "Diff size: $DIFF_SIZE bytes"

# Check if diff exceeds size limit (500KB for nightly)
MAX_DIFF_SIZE=$((500*1024))
if [[ $DIFF_SIZE -gt $MAX_DIFF_SIZE ]]; then
  echo "WARNING: Diff exceeds ${MAX_DIFF_SIZE} bytes, truncating..."
  head -c $MAX_DIFF_SIZE "$OUT_DIR/nightly.diff" > "$OUT_DIR/nightly.diff.truncated"
  mv "$OUT_DIR/nightly.diff.truncated" "$OUT_DIR/nightly.diff"
  echo "TRUNCATED=true" >> "$OUT_DIR/metadata.json"
fi

# Generate metadata
cat > "$OUT_DIR/metadata.json" <<EOF
{
  "repo": "$REPO_NAME",
  "timestamp": "$TIMESTAMP",
  "prev_sha": "$PREV_SHA",
  "head_sha": "$HEAD_SHA",
  "range": "${PREV_SHA}...${HEAD_SHA}",
  "changed_files": $CHANGED_COUNT,
  "diff_size": $DIFF_SIZE,
  "truncated": ${TRUNCATED:-false}
}
EOF

# Generate commit log for context
git log --oneline "$PREV_SHA...$HEAD_SHA" > "$OUT_DIR/commits.log"

echo "Nightly diff generation complete!"
echo "  Output: $OUT_DIR"
echo "  Files changed: $CHANGED_COUNT"
echo "  Commits included: $(wc -l < "$OUT_DIR/commits.log")"

# DO NOT update marker yet - only after successful agent run
echo ""
echo "Next step: Run agent review on $OUT_DIR"
echo "After success: echo '$HEAD_SHA' > $PREV_FILE"