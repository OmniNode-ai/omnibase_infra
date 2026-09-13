#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Warm the cloud pre-push host's caches (OMN-16634) so a dispatched governed
# suite pays no cold start: for each repo, ship a git bundle (HEAD + tags,
# same transport contract as the picker's remote leg -- OMN-17240), clone it
# into <workroot>/warm/<repo>, and run `uv sync --all-extras` there. The uv
# package cache this populates (~/.cache/uv) is what makes the picker's
# per-run `uv sync` fast; the warm clones themselves double as the per-repo
# transport proof (collect-only) when run with --prove.
#
# ZERO credentials travel: the bundle is built from the local canonical clone
# and shipped over the picker's existing SSH trust. The cloud host never
# holds a GitHub credential.
#
# Usage, from a lab host with $OMNI_HOME set:
#   deploy/prepush-cloud/warm_cache.sh <ssh-target> [--prove] [repo ...]
# Default repos: omnibase_infra omnibase_core omnimarket
set -euo pipefail

TARGET="${1:?usage: warm_cache.sh <ssh-target> [--prove] [repo ...]}"
shift
PROVE=0
if [ "${1:-}" = "--prove" ]; then
  PROVE=1
  shift
fi
REPOS=("$@")
[ "${#REPOS[@]}" -gt 0 ] || REPOS=(omnibase_infra omnibase_core omnimarket)
OMNI_HOME="${OMNI_HOME:?set OMNI_HOME to the workspace root}"
# Remote-side paths. REMOTE_UV expands on the REMOTE shell ($HOME escaped
# locally); REMOTE_WARM is relative to the remote user's home, which is the
# cwd of every non-interactive ssh/scp session used below.
REMOTE_UV="\$HOME/.local/bin/uv"
REMOTE_WARM="onex-prepush/warm"

for repo in "${REPOS[@]}"; do
  src="${OMNI_HOME}/${repo}"
  [ -d "$src/.git" ] || {
    echo "warm_cache: ${src} is not a git clone" >&2
    exit 1
  }
  bundle="$(mktemp -t "warm-${repo}-XXXXXX").bundle"
  echo "warm_cache: bundling ${repo} (HEAD + tags)"
  if [ "$(git -C "$src" rev-parse --is-shallow-repository 2> /dev/null)" = "true" ]; then
    git -C "$src" fetch --unshallow --tags
  fi
  git -C "$src" bundle create "$bundle" HEAD --tags
  echo "warm_cache: shipping $(du -h "$bundle" | cut -f1 | tr -d ' ') to ${TARGET}"
  ssh -n -o BatchMode=yes "$TARGET" "mkdir -p ${REMOTE_WARM}"
  scp -q -o BatchMode=yes "$bundle" "${TARGET}:${REMOTE_WARM}/${repo}.bundle"
  rm -f "$bundle"
  echo "warm_cache: clone + uv sync on remote (${repo})"
  ssh -n -o BatchMode=yes "$TARGET" \
    "set -e; cd ${REMOTE_WARM}; rm -rf ${repo}; git clone -q ${repo}.bundle ${repo}; cd ${repo}; ${REMOTE_UV} sync --all-extras > /dev/null 2>&1 && echo 'uv sync OK' || { echo 'uv sync FAILED'; exit 1; }"
  if [ "$PROVE" = "1" ]; then
    echo "warm_cache: transport proof (collect-only) for ${repo}"
    ssh -n -o BatchMode=yes "$TARGET" \
      "cd ${REMOTE_WARM}/${repo} && ${REMOTE_UV} run pytest tests/ --collect-only -q --ignore=tests/integration 2>/dev/null | tail -2"
  fi
done
echo "warm_cache: done"
