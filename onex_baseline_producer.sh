#!/usr/bin/env bash
set -euo pipefail

# Configuration
EXCLUDES_REGEX='(^|/)(archive|archived|deprecated|dist|build|node_modules|venv|\.venv|\.mypy_cache|\.pytest_cache)(/|$)'
INCLUDE_EXT='py|pyi|yaml|yml|toml|json|ini|cfg|sh|bash|ps1|psm1|go|rs|ts|tsx|js|jsx|proto|sql|md'
BYTES_PER_SHARD=$((200*1024))  # 200KB per shard
EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904

# Get repository information
REPO_NAME="omnibase_infra"  # Using the actual repo name from src/
OUT_DIR=".onex_baseline/$REPO_NAME/$(date -u +%Y%m%dT%H%M%SZ)"

echo "Creating baseline review for repository: $REPO_NAME"
echo "Output directory: $OUT_DIR"

# Create output directories
mkdir -p "$OUT_DIR/shards"

# Get list of files excluding archived content
echo "Gathering file list (excluding archived content)..."
git ls-files -z \
  | tr -d '\r' \
  | xargs -0 -I{} bash -c '[[ "{}" =~ '"$EXCLUDES_REGEX"' ]] || echo "{}"' \
  | grep -E "\.($INCLUDE_EXT)$" \
  > "$OUT_DIR/files.list" || true

# Count files
FILE_COUNT=$(wc -l < "$OUT_DIR/files.list")
echo "Found $FILE_COUNT files to review"

# Generate git names (all files are 'Added' for baseline)
echo "Generating git names..."
awk '{print "A\t"$0}' "$OUT_DIR/files.list" > "$OUT_DIR/nightly.names"

# Generate unified diff against empty tree
echo "Generating unified diff..."
# Use two dots (..) for proper diff range syntax
git diff -U0 --no-color "$EMPTY_TREE" HEAD -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.diff" 2>/dev/null || true

# Generate stats
echo "Generating diff stats..."
# Use two dots (..) for proper diff range syntax
git diff --stat "$EMPTY_TREE" HEAD -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.stats" 2>/dev/null || true

# Check if diff was generated
if [[ ! -s "$OUT_DIR/nightly.diff" ]]; then
    echo "Warning: No diff generated. Checking for files..."
    if [[ -s "$OUT_DIR/files.list" ]]; then
        echo "Files exist but diff is empty. This may indicate all files are binary or empty."
    fi
fi

# Split diff into shards
echo "Splitting diff into shards..."
# First split by file boundaries
csplit -s -f "$OUT_DIR/tmpdiff_" -b "%03d" "$OUT_DIR/nightly.diff" '/^diff --git /' '{*}' 2>/dev/null || true

# Now combine into size-limited shards
shard_idx=0
shard_bytes=0
> "$OUT_DIR/manifest.tsv"

for part in "$OUT_DIR"/tmpdiff_*; do
    if [[ ! -f "$part" ]]; then
        continue
    fi

    bytes=$(wc -c < "$part")

    # Start new shard if current would exceed limit or if this is the first file
    if (( shard_bytes + bytes > BYTES_PER_SHARD || shard_bytes == 0 )); then
        (( shard_idx++ ))
        shard_file="$OUT_DIR/shards/diff_shard_${shard_idx}.diff"
        > "$shard_file"
        shard_bytes=0
        echo "Created shard $shard_idx"
    fi

    # Add file to current shard
    cat "$part" >> "$shard_file"
    shard_bytes=$((shard_bytes + bytes))

    # Record in manifest
    echo -e "$shard_idx\t$bytes\t$(basename "$part")" >> "$OUT_DIR/manifest.tsv"
done

# Clean up temporary files
rm -f "$OUT_DIR"/tmpdiff_*

# Generate summary
echo ""
echo "Baseline generation complete!"
echo "================================"
echo "Repository: $REPO_NAME"
echo "Files reviewed: $FILE_COUNT"
echo "Shards created: $shard_idx"
echo "Output directory: $OUT_DIR"
echo ""
echo "Files generated:"
ls -lh "$OUT_DIR/"*.* 2>/dev/null | awk '{print "  - "$9": "$5}'
echo ""
echo "Shards:"
ls -lh "$OUT_DIR/shards/"*.diff 2>/dev/null | awk '{print "  - "$9": "$5}'
echo ""
echo "Next step: Run the baseline review agent on these inputs"