#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-14865 (fan-out of the OMN-14854 omniclaude canary): Required-Check
# Skip-Vector Guard -- local (pre-commit) run.
#
# DRY, NOT a re-implementation (root CLAUDE.md Rule #7a): omniclaude is the
# canonical source for validate_no_required_check_skip_vectors.py (its own
# workflow comment says so explicitly -- "all repos call this via uses:
# instead of running their own copy"). This wrapper imports that SAME script,
# resolved from the OMNI_HOME sibling clone -- mirroring the established
# omnibase_infra pattern for shifting a hosted cross-repo gate left (see
# scripts/ci/check_deploy_scope_dod.py, which resolves omniclaude's
# deploy-gate validator the same way). Local and CI (the reusable workflow
# fetched via sparse-checkout) verdicts can never diverge because both
# execute the identical file.
#
# FAIL-LOUD (root CLAUDE.md Rule #8): if OMNI_HOME is unset or the sibling
# omniclaude clone does not contain the validator, this hook hard-errors
# (exit 1) with a remediation message -- it never degrades to a green skip.
#
# Honest drift note: CI resolves the validator at the pinned omniclaude SHA
# in required-check-skip-guard-caller.yml; this local hook imports whatever
# is currently checked out in $OMNI_HOME/omniclaude (tracking dev). A
# same-session `git -C "$OMNI_HOME/omniclaude" pull` keeps them converged.

set -euo pipefail

if REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" && [[ -n "$REPO_ROOT" ]]; then
    :
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -z "${OMNI_HOME:-}" ]]; then
    echo "ERROR: OMNI_HOME is not set. required-check-skip-guard needs the" >&2
    echo "canonical validator from the omniclaude sibling clone at" >&2
    # Deliberately single-quoted/literal, not an expansion.
    # shellcheck disable=SC2016
    echo '$OMNI_HOME/omniclaude -- set OMNI_HOME to the omni_home path.' >&2
    exit 1
fi

VALIDATOR="${OMNI_HOME}/omniclaude/.github/actions/required-check-skip-guard/validate_no_required_check_skip_vectors.py"

if [[ ! -f "${VALIDATOR}" ]]; then
    echo "ERROR: canonical validator not found at ${VALIDATOR}." >&2
    echo "Ensure \$OMNI_HOME/omniclaude is a checked-out clone containing" >&2
    echo ".github/actions/required-check-skip-guard/validate_no_required_check_skip_vectors.py" >&2
    echo "(git -C \"\$OMNI_HOME/omniclaude\" pull --ff-only)." >&2
    exit 1
fi

PY="${PYTHON_BIN:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH for required-check-skip-guard" >&2
    exit 1
fi

exec "$PY" "${VALIDATOR}" \
    --manifest "${REPO_ROOT}/.github/required-checks.yaml" \
    --workflows-dir "${REPO_ROOT}/.github/workflows"
