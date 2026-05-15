#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/ci/reject-node-implementations.sh [--all|--staged]

Reject unallowlisted Python files under src/omnibase_infra/nodes/*/handlers/.
Business-domain node implementations belong in omnimarket. Any infra-owned
handler that remains in omnibase_infra must be listed in
scripts/ci/infra-node-allowlist.txt with an inline justification comment.
USAGE
}

mode="all"
case "${1:-}" in
  ""|--all)
    mode="all"
    ;;
  --staged)
    mode="staged"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

allowlist_path="scripts/ci/infra-node-allowlist.txt"
handler_re='^src/omnibase_infra/nodes/[^/]+/handlers/.+\.py$'

if [[ ! -f "$allowlist_path" ]]; then
  echo "ERROR: missing $allowlist_path" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

allowlist_source="$allowlist_path"
if [[ "$mode" == "staged" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  index_allowlist="$tmpdir/infra-node-allowlist.from-index.txt"
  if git show ":$allowlist_path" > "$index_allowlist" 2>/dev/null; then
    allowlist_source="$index_allowlist"
  fi
fi

allowlist_entries="$tmpdir/allowlist.txt"
allowlist_errors="$tmpdir/allowlist-errors.txt"
: > "$allowlist_entries"
: > "$allowlist_errors"

line_number=0
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line_number=$((line_number + 1))
  line="${raw_line%$'\r'}"

  trimmed="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
    continue
  fi

  if [[ "$line" != *"#"* ]]; then
    echo "$allowlist_path:$line_number missing inline justification comment" >> "$allowlist_errors"
    continue
  fi

  path_part="${line%%#*}"
  comment_part="${line#*#}"
  path="$(printf '%s' "$path_part" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  comment="$(printf '%s' "$comment_part" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  if [[ -z "$path" ]]; then
    echo "$allowlist_path:$line_number missing handler path before comment" >> "$allowlist_errors"
    continue
  fi
  if [[ -z "$comment" ]]; then
    echo "$allowlist_path:$line_number missing justification after #" >> "$allowlist_errors"
    continue
  fi
  if ! [[ "$path" =~ $handler_re ]]; then
    echo "$allowlist_path:$line_number path is not an infra node handler: $path" >> "$allowlist_errors"
    continue
  fi

  printf '%s\n' "$path" >> "$allowlist_entries"
done < "$allowlist_source"

if [[ -s "$allowlist_errors" ]]; then
  echo "ERROR: invalid infra node handler allowlist:" >&2
  cat "$allowlist_errors" >&2
  exit 1
fi

sort -u "$allowlist_entries" -o "$allowlist_entries"
duplicates="$(sed -n 's/[[:space:]]*#.*$//p' "$allowlist_path" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | sed '/^$/d' | sort | uniq -d)"
if [[ -n "$duplicates" ]]; then
  echo "ERROR: duplicate entries in $allowlist_path:" >&2
  echo "$duplicates" >&2
  exit 1
fi

candidate_paths="$tmpdir/candidates.txt"
if [[ "$mode" == "staged" ]]; then
  git diff --cached --name-only --diff-filter=ACMR \
    | grep -E "$handler_re" \
    | sort -u > "$candidate_paths" || true
else
  if [[ -d src/omnibase_infra/nodes ]]; then
    find src/omnibase_infra/nodes -path '*/handlers/*.py' -type f \
      | sort -u > "$candidate_paths"
  else
    : > "$candidate_paths"
  fi
fi

violations="$tmpdir/violations.txt"
comm -23 "$candidate_paths" "$allowlist_entries" > "$violations"

if [[ -s "$violations" ]]; then
  echo "ERROR: Node handlers belong in omnimarket, not omnibase_infra." >&2
  echo "Unallowlisted infra node handler path(s):" >&2
  sed 's/^/  - /' "$violations" >&2
  echo >&2
  echo "Move business-domain node implementations to omnimarket, or add a justified" >&2
  echo "infra-only exception to $allowlist_path in the same PR." >&2
  exit 1
fi

if [[ "$mode" == "all" ]]; then
  stale="$tmpdir/stale.txt"
  comm -13 "$candidate_paths" "$allowlist_entries" > "$stale"
  if [[ -s "$stale" ]]; then
    echo "ERROR: stale infra node handler allowlist entry/entries:" >&2
    sed 's/^/  - /' "$stale" >&2
    echo "Remove stale entries from $allowlist_path." >&2
    exit 1
  fi
fi

echo "PASS: infra node handler ownership allowlist is satisfied."
