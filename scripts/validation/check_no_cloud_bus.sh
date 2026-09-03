#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# check_no_cloud_bus.sh — CI invariant: block cloud bus (port 29092) references
#
# Usage:
#   bash scripts/check_no_cloud_bus.sh [TARGET_DIR]
#
# TARGET_DIR defaults to the current directory.
#
# Scans git-tracked files for port 29092 references.
# Exits 1 if any unsuppressed references are found.
#
# Suppression: add "# cloud-bus-ok OMN-XXXX" on the same line.
#   Bare "# cloud-bus-ok" (without OMN- ticket) does NOT suppress.
#
# Excluded paths: CLAUDE.md, MEMORY.md, CHANGELOG.md, docs/history/,
#   docs/historical-planning/, docs/deep-dives/, docs/plans/, docs/archive/,
#   docs/velocity-reports/, docs/reference/, docs/architecture/,
#   docker-compose.e2e*, check_no_cloud_bus.sh (self-exclusion)
#
# File extensions scanned: *.py *.ts *.tsx *.js *.sh *.yml *.yaml *.toml

set -euo pipefail

TARGET_DIR="${1:-.}"

if ! cd "$TARGET_DIR" 2>/dev/null; then
    echo "ERROR: Cannot access directory: $TARGET_DIR"
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: $TARGET_DIR is not inside a git repository"
    exit 1
fi

# File extensions to scan
EXTENSIONS=(py ts tsx js sh yml yaml toml)

# Build git ls-files pattern list
PATTERNS=()
for ext in "${EXTENSIONS[@]}"; do
    PATTERNS+=("*.${ext}")
done

# Paths to exclude (relative to repo root)
EXCLUDE_PATHS=(
    "CLAUDE.md"
    "MEMORY.md"
    "CHANGELOG.md"
    "docs/history/"
    "docs/historical-planning/"
    "docs/deep-dives/"
    "docs/plans/"
    "docs/archive/"
    "docs/velocity-reports/"
    "docs/reference/"
    "docs/architecture/"
)

# Also exclude self and e2e compose files
EXCLUDE_SELF="check_no_cloud_bus"
EXCLUDE_E2E="docker-compose.e2e"

pathspecs=("${PATTERNS[@]}")
for excl in "${EXCLUDE_PATHS[@]}"; do
    pathspecs+=(":!${excl}**")
    pathspecs+=(":!**/${excl}**")
done
pathspecs+=(":!*${EXCLUDE_SELF}*")
pathspecs+=(":!*${EXCLUDE_E2E}*")

violations=0
matches="$(git grep -n '29092' -- "${pathspecs[@]}" 2>/dev/null || true)"

if [ -z "$matches" ]; then
    exit 0
fi

while IFS= read -r line_content; do
    file="${line_content%%:*}"
    rest="${line_content#*:}"
    line_num="${rest%%:*}"
    line_text="${rest#*:}"

    # Check for valid suppression: "# cloud-bus-ok OMN-" followed by digits
    if echo "$line_text" | grep -qE '#\s*cloud-bus-ok\s+OMN-[0-9]+'; then
        continue
    fi

    violations=$((violations + 1))
    echo "VIOLATION: $file:$line_num: $line_text"
done <<< "$matches"

if [ "$violations" -gt 0 ]; then
    echo ""
    echo "Found $violations unsuppressed cloud bus (29092) reference(s)."
    echo "To suppress, add on the same line: # cloud-bus-ok OMN-XXXX"
    echo "(bare '# cloud-bus-ok' without ticket ID does NOT suppress)"
    exit 1
fi

exit 0
