#!/usr/bin/env bash
# ONEX Baseline Producer - Generate sharded baseline inputs for Opus review
set -euo pipefail

# Configuration
EXCLUDES_REGEX='(^|/)(archive|archived|deprecated|dist|build|node_modules|venv|\.venv|\.mypy_cache|\.pytest_cache)(/|$)'
INCLUDE_EXT='py|pyi|yaml|yml|toml|json|ini|cfg|sh|bash|ps1|psm1|go|rs|ts|tsx|js|jsx|proto|sql|md'
BYTES_PER_SHARD=$((200*1024))  # 200KB per shard
EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904

# Get repo name and create output directory
REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR=".onex_baseline/$REPO_NAME/$TIMESTAMP"
mkdir -p "$OUT_DIR/shards"

echo "ONEX Baseline Producer for $REPO_NAME"
echo "Output directory: $OUT_DIR"

# Generate file list (excluding archived code)
echo "Generating file list..."
git ls-files -z \
  | tr -d '\r' \
  | xargs -0 -I{} bash -c '[[ "{}" =~ '"$EXCLUDES_REGEX"' ]] || echo "{}"' \
  | grep -E "\.(${INCLUDE_EXT})$" \
  > "$OUT_DIR/files.list"

FILE_COUNT=$(wc -l < "$OUT_DIR/files.list")
echo "Found $FILE_COUNT files to review"

# Generate git names output (all files as Added)
awk '{print "A\t"$0}' "$OUT_DIR/files.list" > "$OUT_DIR/nightly.names"

# Generate unified diff
echo "Generating unified diff..."
git diff -U0 --no-color "$EMPTY_TREE"...HEAD -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.diff" || true

# Generate stats
git diff --stat "$EMPTY_TREE"...HEAD -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.stats" || true

# Split diff into shards
echo "Sharding diff (max ${BYTES_PER_SHARD} bytes per shard)..."
csplit -s -f "$OUT_DIR/tmpdiff_" -b "%03d" "$OUT_DIR/nightly.diff" '/^diff --git /' '{*}' 2>/dev/null || true

# Combine parts into size-limited shards
shard_idx=0
shard_bytes=0
> "$OUT_DIR/manifest.tsv"

for part in "$OUT_DIR"/tmpdiff_*; do
  if [[ ! -f "$part" ]]; then continue; fi

  bytes=$(wc -c < "$part")

  # Start new shard if needed
  if (( shard_bytes + bytes > BYTES_PER_SHARD || shard_bytes == 0 )); then
    (( shard_idx++ ))
    shard_file="$OUT_DIR/shards/diff_shard_${shard_idx}.diff"
    > "$shard_file"
    shard_bytes=0
    echo "Creating shard $shard_idx: $shard_file"
  fi

  # Add to current shard
  cat "$part" >> "$shard_file"
  shard_bytes=$((shard_bytes + bytes))

  # Log to manifest
  echo -e "$shard_idx\t$bytes\t$(basename "$part")" >> "$OUT_DIR/manifest.tsv"
done

# Clean up temp files
rm -f "$OUT_DIR"/tmpdiff_*

# Generate metadata
cat > "$OUT_DIR/metadata.json" <<EOF
{
  "repo": "$REPO_NAME",
  "timestamp": "$TIMESTAMP",
  "empty_tree": "$EMPTY_TREE",
  "head_sha": "$(git rev-parse HEAD)",
  "file_count": $FILE_COUNT,
  "shard_count": $shard_idx,
  "bytes_per_shard": $BYTES_PER_SHARD
}
EOF

echo "Baseline generation complete!"
echo "  Files reviewed: $FILE_COUNT"
echo "  Shards created: $shard_idx"
echo "  Output: $OUT_DIR"