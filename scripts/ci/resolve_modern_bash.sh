#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# resolve_modern_bash.sh -- OMN-15617: explicit, PATH-order-independent
# resolution of a bash interpreter new enough for `declare -A`.
#
# Root cause: on stickybeatz-studio (.200, the rule-11a default gate host),
# non-interactive ssh sessions resolve `bash` / `env bash` to the system
# 3.2.57 shell (no associative arrays) even though a modern bash 5.x sits at
# /opt/homebrew/bin/bash -- it simply is not first on PATH for that session
# class. Callers that spawn `bash <script-using-declare--A>` via bare "bash"
# silently run the wrong interpreter and fail with a bash syntax error deep
# inside the script, not a resolvable "wrong bash" diagnostic.
#
# This script performs the resolution ONCE, explicitly, independent of PATH
# order, and is the single source of truth callers (the pre-push canary and
# the runner-monitor.sh pytest harness) both use -- so they can never drift.
#
# MUST remain runnable under bash 3.2 itself (no `declare -A`, no `[[ ]]`
# reliance beyond what 3.2 supports) -- interpreter resolution cannot
# presuppose the thing it is resolving.
#
# Output contract: on success, prints the resolved interpreter's absolute
# path to stdout and exits 0. On failure, prints nothing to stdout, prints a
# pointed ERROR + REMEDIATION message to stderr, and exits 1. Never a silent
# fallback and never a quiet skip -- an unresolvable modern bash is a hard
# failure the caller must surface.
#
# Env overrides:
#   OMNIBASE_INFRA_BASH_BIN       explicit interpreter path to try FIRST.
#   OMNIBASE_INFRA_MIN_BASH_MAJOR minimum BASH_VERSINFO[0] required (default 5).

set -euo pipefail

MIN_MAJOR="${OMNIBASE_INFRA_MIN_BASH_MAJOR:-5}"

# Prints the candidate's BASH_VERSINFO[0] on stdout, or nothing + nonzero exit
# if the candidate is not an executable bash at all.
bash_major_version() {
  local bin="$1"
  [ -x "$bin" ] || return 1
  "$bin" -c 'printf "%s" "${BASH_VERSINFO[0]:-}"' 2> /dev/null
}

seen=""
try_candidate() {
  local bin="$1" major
  case " ${seen} " in
    *" ${bin} "*) return 1 ;;
  esac
  seen="${seen} ${bin}"
  major="$(bash_major_version "$bin")" || return 1
  [ -n "$major" ] || return 1
  [ "$major" -ge "$MIN_MAJOR" ] || return 1
  printf '%s\n' "$bin"
  return 0
}

# Ordered candidate list: explicit override, the two brew prefixes (Apple
# Silicon + Intel), then every "bash" found walking $PATH -- so a modern bash
# installed somewhere non-standard is still picked up without PATH surgery.
CANDIDATES=""
if [ -n "${OMNIBASE_INFRA_BASH_BIN:-}" ]; then
  CANDIDATES="${CANDIDATES} ${OMNIBASE_INFRA_BASH_BIN}"
fi
CANDIDATES="${CANDIDATES} /opt/homebrew/bin/bash /usr/local/bin/bash"

old_ifs="$IFS"
IFS=':'
for _dir in $PATH; do
  [ -n "$_dir" ] || continue
  CANDIDATES="${CANDIDATES} ${_dir}/bash"
done
IFS="$old_ifs"

for _candidate in $CANDIDATES; do
  if try_candidate "$_candidate"; then
    exit 0
  fi
done

{
  printf 'ERROR: no bash interpreter >= %s found (declare -A requires bash>=4; OMN-15617 requires >=%s).\n' "$MIN_MAJOR" "$MIN_MAJOR"
  printf 'Checked: OMNIBASE_INFRA_BASH_BIN, /opt/homebrew/bin/bash, /usr/local/bin/bash, and every "bash" on PATH.\n'
  printf 'REMEDIATION: install a modern bash (e.g. `brew install bash`) and/or set OMNIBASE_INFRA_BASH_BIN to its absolute path.\n'
} >&2
exit 1
