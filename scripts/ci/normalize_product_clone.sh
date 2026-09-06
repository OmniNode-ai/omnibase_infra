#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

#
# OMN-17975. Make a product clone's freshness ESTABLISHABLE, which is the
# precondition dod_verify requires before it will execute a behaviour check in
# that tree.
#
# `EvidenceCollector._compute_product_clone_resolution` (omnimarket,
# src/omnimarket/nodes/node_dod_verify/services/evidence_collector.py:4396-4460)
# decides a clone's freshness in three steps, and each one must succeed:
#
#   1. git rev-parse --abbrev-ref --symbolic-full-name @{upstream}
#   2. git fetch <remote> <branch>
#   3. git rev-list --count HEAD..<upstream>   -> must be 0
#
# Step 1 fails outright on a DETACHED HEAD ("fatal: HEAD does not point to a
# branch"), the resolution is UNKNOWN, `_product_clone_unverifiable_cause`
# (:4523-4537) maps UNKNOWN to PRODUCT_CLONE_FRESHNESS_UNKNOWN, and the check is
# refused UNEXECUTED (:3186-3208). That refusal is correct — a verdict from a
# tree whose contents cannot be placed relative to the work under adjudication is
# not evidence about it. The defect is the tree, not the gate.
#
# Every clone the evidence-autoclose sweep materialised was detached: the reused
# `actions/checkout` trees are pinned to a resolved SHA by construction. Since
# OMN-16434 auto-mints exactly one cwd-anchored behaviour proof onto every new
# companion, and every such cwd resolves into one of those trees, the single
# behaviour-proving check on every candidate was refused and
# `behavior_proving_count` was 0 for the entire corpus.
#
# Measured on the runner, dispatch dry-run 33997616996, 2026-09-05T23:05:36Z:
#
#   [skipped] dod-occ-diff-derived-behavior-proof: PRODUCT_CLONE_NOT_FRESH: the
#   check's cwd names /home/runner/work/omnibase_infra/omnimarket, whose
#   freshness is unknown (HEAD a138c1e4c79a..., upstream <none>, behind None; no
#   remote-tracking upstream is configured for HEAD, so freshness cannot be
#   established (fatal: HEAD does not point to a branch)). The command was NOT
#   executed.
#
# This lives in a script rather than inline in the workflow so its behaviour is
# testable against real git repositories rather than asserted by reading YAML:
# tests/ci/test_autoclose_product_clone_freshness_omn17975.py.
#
# Usage:
#   normalize_product_clone.sh [--no-move] <clone_dir> <remote_url> <branch>
#
#   --no-move   Establish freshness WITHOUT moving the working tree. Required
#               for the job's own `github.workspace` checkout: fast-forwarding
#               it mid-job would change the source the gate venv was synced
#               against, and the post-sweep purity assertion would then be
#               measuring a different tree than the one it installed. If that
#               tree is behind, this refuses (exit 3) and leaves it exactly as
#               it was, so the resulting PRODUCT_CLONE_NOT_FRESH refusal stands
#               honestly instead of being laundered.
#
# Exit codes:
#   0  FRESH   — @{upstream} resolves and HEAD is at the upstream tip.
#   1  a named, fail-closed refusal (not a repository, fetch failed, ...).
#   3  BEHIND  — --no-move only: freshness is knowable and the tree is behind.
#
# Deliberately NOT here: anything resembling
# DOD_VERIFY_ALLOW_STALE_PRODUCT_CLONE. That flag marks every verdict it touches
# un-attributable to a verified-fresh tree; the point of this script is to make
# the tree actually fresh so the flag is never needed.

set -uo pipefail

no_move=0
if [ "${1:-}" = "--no-move" ]; then
  no_move=1
  shift
fi

if [ "$#" -ne 3 ]; then
  echo "NORMALIZE_USAGE_ERROR: expected [--no-move] <clone_dir> <remote_url> <branch>, got $# argument(s)" >&2
  exit 1
fi

clone="$1"
remote_url="$2"
branch="$3"

if [ ! -d "${clone}" ]; then
  echo "NOT_A_GIT_REPOSITORY: ${clone} does not exist" >&2
  exit 1
fi

if ! git -C "${clone}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "NOT_A_GIT_REPOSITORY: ${clone} is not inside a git repository" >&2
  exit 1
fi

# `origin` may be absent (a `cp -a` of a tree someone stripped), present with a
# different URL, or already correct. Converge rather than branching on which.
if git -C "${clone}" remote get-url origin >/dev/null 2>&1; then
  git -C "${clone}" remote set-url origin "${remote_url}"
else
  git -C "${clone}" remote add origin "${remote_url}"
fi

# An explicit refspec, not the remote's configured one. A copied
# `actions/checkout` tree carries whatever narrow refspec that action wrote, and
# a plain `git fetch origin <branch>` under it can leave
# refs/remotes/origin/<branch> unwritten — which would make step 3 above compare
# HEAD against a ref that does not exist. Naming the destination ref makes the
# outcome independent of the inherited config.
#
# --depth 1 keeps this affordable on the shallow clones the sweep creates; the
# collector's own later `git fetch <remote> <branch>` is then a no-op.
if ! fetch_err="$(git -C "${clone}" fetch --quiet --depth 1 origin \
      "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>&1)"; then
  echo "FETCH_FAILED: git fetch origin ${branch} in ${clone}: ${fetch_err}" >&2
  exit 1
fi

if ! upstream_sha="$(git -C "${clone}" rev-parse "refs/remotes/origin/${branch}" 2>/dev/null)"; then
  echo "UPSTREAM_REF_MISSING: refs/remotes/origin/${branch} is absent in ${clone} after a successful fetch" >&2
  exit 1
fi

head_sha="$(git -C "${clone}" rev-parse HEAD 2>/dev/null || true)"
if [ -z "${head_sha}" ]; then
  echo "HEAD_UNRESOLVABLE: ${clone} has no resolvable HEAD" >&2
  exit 1
fi

if [ "${no_move}" -eq 1 ] && [ "${head_sha}" != "${upstream_sha}" ]; then
  # Knowable AND behind. Report both facts and change nothing: the caller asked
  # for freshness without a tree move, and there is no way to have both.
  echo "BEHIND ${clone}: HEAD ${head_sha} is not at origin/${branch} ${upstream_sha}; left untouched (--no-move)."
  exit 3
fi

# `-B` is idempotent: it creates the branch or repoints it, whether HEAD was
# detached, on this branch already, or on some other one.
if ! checkout_err="$(git -C "${clone}" checkout --quiet -B "${branch}" "${upstream_sha}" 2>&1)"; then
  echo "CHECKOUT_FAILED: git checkout -B ${branch} in ${clone}: ${checkout_err}" >&2
  exit 1
fi

# Set tracking explicitly. `checkout -B <branch> <sha>` branches from a commit,
# not from a remote-tracking ref, so it configures no upstream on its own — and
# an upstream that does not resolve is the entire defect this script exists to
# remove.
if ! upstream_err="$(git -C "${clone}" branch --set-upstream-to="origin/${branch}" "${branch}" 2>&1)"; then
  echo "SET_UPSTREAM_FAILED: ${clone}: ${upstream_err}" >&2
  exit 1
fi

# Read back the collector's own three-step predicate rather than trusting that
# the commands above did what they were asked. A step that reports success
# without measuring its result is how this defect survived 24 scheduled runs.
resolved_upstream="$(git -C "${clone}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
if [ -z "${resolved_upstream}" ] || [ "${resolved_upstream#*/}" = "${resolved_upstream}" ]; then
  echo "UPSTREAM_UNRESOLVED_AFTER_NORMALISE: ${clone} still has no <remote>/<branch> upstream" >&2
  exit 1
fi

behind="$(git -C "${clone}" rev-list --count "HEAD..${resolved_upstream}" 2>/dev/null || echo "")"
if [ -z "${behind}" ] || [ "${behind}" != "0" ]; then
  echo "STILL_BEHIND_AFTER_NORMALISE: ${clone} is ${behind:-unknown} commit(s) behind ${resolved_upstream}" >&2
  exit 1
fi

echo "FRESH ${clone}: HEAD $(git -C "${clone}" rev-parse HEAD) on ${branch} tracking ${resolved_upstream}, behind 0"
