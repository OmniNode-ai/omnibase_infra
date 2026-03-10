#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# check_migration_freeze.sh — Block new migrations while .migration_freeze exists.
#
# Usage:
#   Pre-commit: ./scripts/check_migration_freeze.sh           (checks staged files)
#   CI:         ./scripts/check_migration_freeze.sh --ci       (checks diff vs base branch)
#
# Freeze age policy (requires freeze_date= field in .migration_freeze):
#   WARNING at 30+ days since freeze_date
#   ERROR (exit 1) at 60+ days since freeze_date
#
# Exit codes:
#   0 — No freeze active, or no new migrations detected
#   1 — Freeze violation: new migration files added, OR freeze expired (60+ days)

set -euo pipefail

FREEZE_FILE=".migration_freeze"
MIGRATIONS_DIR="docker/migrations"

# If no freeze file, nothing to enforce.
if [ ! -f "$FREEZE_FILE" ]; then
    echo "No migration freeze active — skipping check."
    exit 0
fi

echo "Migration freeze is ACTIVE ($FREEZE_FILE exists)"

# ── Freeze age gate ──────────────────────────────────────────────────────────
# Parse freeze_date= field from .migration_freeze (format: freeze_date=YYYY-MM-DD)
FREEZE_DATE_RAW=$(grep -E '^freeze_date=' "$FREEZE_FILE" | head -1 | cut -d= -f2 | tr -d '[:space:]' || true)

if [ -n "$FREEZE_DATE_RAW" ]; then
    # Cross-platform date arithmetic: macOS uses date -j, Linux uses date -d
    if date --version >/dev/null 2>&1; then
        # GNU date (Linux)
        FREEZE_EPOCH=$(date -d "$FREEZE_DATE_RAW" +%s 2>/dev/null || echo "")
        NOW_EPOCH=$(date +%s)
    else
        # BSD date (macOS)
        FREEZE_EPOCH=$(date -j -f "%Y-%m-%d" "$FREEZE_DATE_RAW" +%s 2>/dev/null || echo "")
        NOW_EPOCH=$(date +%s)
    fi

    if [ -n "$FREEZE_EPOCH" ] && [ -n "$NOW_EPOCH" ]; then
        FREEZE_AGE_DAYS=$(( (NOW_EPOCH - FREEZE_EPOCH) / 86400 ))
        echo "Freeze age: ${FREEZE_AGE_DAYS} day(s) (since ${FREEZE_DATE_RAW})"

        if [ "$FREEZE_AGE_DAYS" -ge 60 ]; then
            echo ""
            echo "ERROR: Migration freeze has EXPIRED!"
            echo "Freeze date: ${FREEZE_DATE_RAW} (${FREEZE_AGE_DAYS} days ago)"
            echo "Freezes are automatically invalidated after 60 days."
            echo ""
            echo "Action required:"
            echo "  1. Either lift the freeze (remove $FREEZE_FILE) if the DB boundary work is complete"
            echo "  2. Or renew the freeze by updating freeze_date= with a justification comment"
            echo "  3. Track in the freeze ticket (see ticket= field in $FREEZE_FILE)"
            echo ""
            exit 1
        elif [ "$FREEZE_AGE_DAYS" -ge 30 ]; then
            echo ""
            echo "WARNING: Migration freeze is approaching expiry!"
            echo "Freeze date: ${FREEZE_DATE_RAW} (${FREEZE_AGE_DAYS} days ago)"
            echo "This freeze will become an ERROR in $(( 60 - FREEZE_AGE_DAYS )) day(s)."
            echo "Review the freeze status and update $FREEZE_FILE if still needed."
            echo ""
            # Warning only — do not exit; continue migration checks below
        fi
    else
        echo "Warning: could not parse freeze_date='${FREEZE_DATE_RAW}' — skipping age check." >&2
    fi
else
    echo "Warning: no freeze_date= field found in $FREEZE_FILE — skipping age check." >&2
fi
# ── End freeze age gate ──────────────────────────────────────────────────────

MODE="${1:-precommit}"

if [ "$MODE" = "--ci" ]; then
    # CI mode: compare against base branch.
    # On push events (not PRs), GITHUB_BASE_REF is empty; defaults to 'main'.
    # This means on direct pushes to main, origin/main...HEAD diff is typically
    # empty — the check is effectively a no-op, which is correct (the freeze
    # should only block PR merges, not post-merge pushes).
    BASE_BRANCH="${GITHUB_BASE_REF:-main}"
    # Defensive fetch: ensure origin/<base> ref is up-to-date even if
    # the CI runner's checkout didn't fully resolve it.
    git fetch origin "${BASE_BRANCH}" --quiet 2>/dev/null || echo "Warning: could not fetch origin/${BASE_BRANCH} (may already be available from checkout)" >&2
    # Detect added (A) or renamed (R) files in the migrations directory.
    # Modified (M) files are intentionally allowed — fixing existing
    # migrations (rollback bug fixes, comment tweaks) is safe during freeze.
    NEW_MIGRATIONS=$(git diff --name-status "origin/${BASE_BRANCH}...HEAD" -- "$MIGRATIONS_DIR" \
        | grep -E '^[AR]' | awk '{print $NF}' || true)
else
    # Pre-commit mode: check staged files
    NEW_MIGRATIONS=$(git diff --cached --name-status -- "$MIGRATIONS_DIR" \
        | grep -E '^[AR]' | awk '{print $NF}' || true)
fi

if [ -n "$NEW_MIGRATIONS" ]; then
    echo ""
    echo "ERROR: Migration freeze violation!"
    echo "Blocked: new migration files (A=added) or renames (R) while $FREEZE_FILE exists."
    echo "Allowed: modifications (M) to existing migrations (bug fixes, comments)."
    echo ""
    echo "Violating files:"
    echo "$NEW_MIGRATIONS" | sed 's/^/  /'
    echo ""
    echo "See $FREEZE_FILE for details on the active freeze."
    exit 1
fi

echo "No new migrations detected — freeze check passed."
exit 0
