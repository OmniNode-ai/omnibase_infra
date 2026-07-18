#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# merge_sweep_common.sh — canonical bash helpers for merge-controller shell loops
# (OMN-14761 / F-22).
#
# Source this; do NOT execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/merge_sweep_common.sh"
#
# Purpose: replace hand-typed, zsh-fragile probe loops (the F-22 friction: zsh
# newline-splitting and quoted `repo pr` loops that produced invalid `gh`
# repository names) with a single canonical, shellcheck-clean bash surface —
# the authoritative REPOS array, a safe iterator, an opt-in strict-mode switch,
# and a heredoc-based merge-queue probe.
#
# Provides:
#   REPOS[]                     canonical OmniNode-ai repositories (PR-bearing)
#   msc_strict                  opt-in `set -euo pipefail` (call at script top)
#   for_each_repo CMD [ARGS...] run `CMD [ARGS...] <repo>` for each repo
#   msc_mergequeue_probe REPO   print dev/main merge-queue ids via a heredoc'd
#                               GraphQL query (no shell word-splitting hazard)

# Guard against direct execution — this is a library.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "merge_sweep_common.sh is a library; source it, do not run it." >&2
  exit 2
fi

# Canonical OmniNode-ai repositories that carry PRs / merge-sweep state.
# Mirrors scripts/pull-all.sh; this is the single source of truth for shell loops.
REPOS=(
  omniclaude
  omnibase_compat
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnidash
  omnigemini
  omniintelligence
  omnimarket
  omnimemory
  omninode_infra
  omniweb
  onex_change_control
)

# msc_strict — opt-in strict mode for a merge-controller script. Kept as a
# function (not a top-level `set`) so interactive `source` of this library for
# its REPOS array does not turn on `set -u`/`set -e` in the operator's shell.
msc_strict() {
  set -euo pipefail
}

# for_each_repo CMD [ARGS...] — run `CMD [ARGS...] <repo>` for each canonical repo,
# passing the repo as the final argument. Returns the first non-zero exit unless
# MERGE_SWEEP_CONTINUE_ON_ERROR=1, in which case it runs all repos and returns the
# last non-zero exit (0 if all succeeded).
for_each_repo() {
  local _cmd="${1:?for_each_repo requires a command}"
  shift
  local _repo
  local _rc=0
  for _repo in "${REPOS[@]}"; do
    # NB: capture the command's real exit in the else-branch. `if ! cmd; then
    # rc=$?` would record 0 because `!` negates the status for the if-test.
    if "${_cmd}" "$@" "${_repo}"; then
      :
    else
      _rc=$?
      if [ "${MERGE_SWEEP_CONTINUE_ON_ERROR:-0}" != "1" ]; then
        return "${_rc}"
      fi
    fi
  done
  return "${_rc}"
}

# msc_mergequeue_probe REPO — print the dev/main merge-queue node ids for REPO.
# Uses a quoted heredoc for the GraphQL body so the query text is never subject
# to shell word-splitting or newline-splitting (the exact F-22 hazard). Requires
# `gh`. Prints raw `gh api graphql` JSON on stdout; returns gh's exit code.
msc_mergequeue_probe() {
  local repo="${1:?msc_mergequeue_probe requires a repository name}"
  local query
  query="$(
    cat <<'GRAPHQL'
query($owner:String!,$name:String!){
  repository(owner:$owner,name:$name){
    dev:mergeQueue(branch:"dev"){ id }
    main:mergeQueue(branch:"main"){ id }
  }
}
GRAPHQL
  )"
  gh api graphql -f owner="OmniNode-ai" -f name="${repo}" -f query="${query}"
}
