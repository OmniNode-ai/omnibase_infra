#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Deploy-agent launcher: bash-source the operator env store, then exec the agent.
#
# WHY THIS EXISTS (OMN-18073)
#
# The dev-lane unit used to declare `EnvironmentFile=<the operator env store>`.
# systemd's env-file parser does not implement bash ANSI-C $'...' quoting: given
# a line written in that form it keeps the literal $' / ' wrapper and drops every
# backslash escape, so each \n collapses to the bare letter n. The agent then
# hands dict(os.environ) to `docker compose`, whose interpolation writes the
# mangled value into every container the deploy creates.
#
# Measured on the lab host 2026-09-09, values identified by length and sha256-12
# only: the store's line yields 1679 bytes through bash `source` (parses as a
# PEM) and 1682 bytes through systemd's EnvironmentFile= (still $'...'-wrapped,
# zero backslashes). scripts/deploy-runtime.sh has always bash-`source`d the very
# same file and decoded it correctly, which is why only the agent path produced
# broken containers.
#
# This launcher makes the agent's process environment the bash-decoded one, by
# doing exactly what deploy-runtime.sh does. It is the transport fix; the
# executor's UndecodedAnsiCQuotingError guard stays as the fail-closed backstop
# for any path that still arrives mangled.
#
# WHY THE PROTECTED LIST IS NOT OPTIONAL
#
# systemd applies a unit's directives in file order, so the `Environment=` lines
# that follow `EnvironmentFile=` WIN over the store. A naive `set -a; source`
# inverts that: the store would win over the unit.
#
# That is not hypothetical. Measured on the lab host 2026-09-09, the store
# defines both KAFKA_BOOTSTRAP_SERVERS and KAFKA_ENVIRONMENT, and its
# KAFKA_ENVIRONMENT is `local` while the dev unit declares `dev`. A naive source
# would silently relabel the dev agent's lane. DEPLOY_AGENT_ENV_PROTECTED names
# the variables the UNIT owns; their values are snapshotted before the source and
# restored after it, which reproduces systemd's ordering exactly.
#
# The unit must list every name it declares. `tests/unit/
# test_deploy_agent_launcher_omn18073.py` derives the expected set from the unit
# file itself and fails if a new `Environment=` line is added without extending
# the list, so the invariant cannot rot.
#
# CONTRACT
#
#   DEPLOY_AGENT_ENV_FILE       required, readable  -- the operator env store
#   DEPLOY_AGENT_PYTHON         required, executable -- the canonical venv python
#   DEPLOY_AGENT_ENV_PROTECTED  required, non-empty  -- whitespace-separated names
#
# All three are REQUIRED and have no defaults. An unset, unreadable or
# malformed input exits 2 and names the reason on stderr; it never falls back to
# an un-sourced environment, because that is the failure mode this script exists
# to remove. No value is ever printed.
#
# This script exports DEPLOY_AGENT_LAUNCHER (its own absolute path) so
# executor.self_update() can re-exec THROUGH it rather than through the bare
# interpreter. os.execv inherits the caller's environment, so an interpreter
# re-exec carries the process's original environment forward forever -- which is
# why the agent kept a stale env across four self-updates on 2026-09-09 while its
# code advanced normally. Re-execing the launcher re-reads the store.

set -euo pipefail

die() {
	printf '[deploy-agent-launch] ERROR: %s\n' "$1" >&2
	exit 2
}

[ -n "${DEPLOY_AGENT_ENV_FILE:-}" ] || die "DEPLOY_AGENT_ENV_FILE is required and has no default"
[ -n "${DEPLOY_AGENT_PYTHON:-}" ] || die "DEPLOY_AGENT_PYTHON is required and has no default"
[ -n "${DEPLOY_AGENT_ENV_PROTECTED:-}" ] || die "DEPLOY_AGENT_ENV_PROTECTED is required and has no default"

[ -r "$DEPLOY_AGENT_ENV_FILE" ] || die "env file is not readable: $DEPLOY_AGENT_ENV_FILE"
[ -x "$DEPLOY_AGENT_PYTHON" ] || die "interpreter is not executable: $DEPLOY_AGENT_PYTHON"

# Snapshot the unit-owned names. Each is validated before it reaches `eval`:
# a name that is not a shell identifier is a malformed unit, which refuses.
__restore_names=""
for __name in $DEPLOY_AGENT_ENV_PROTECTED; do
	case "$__name" in
	[!A-Za-z_]* | *[!A-Za-z0-9_]*)
		die "DEPLOY_AGENT_ENV_PROTECTED contains a non-identifier name"
		;;
	esac
	eval "__is_set=\${${__name}+x}"
	if [ -n "${__is_set:-}" ]; then
		eval "__saved_${__name}=\${${__name}}"
		__restore_names="${__restore_names} ${__name}"
	fi
done

# `set -a` so every assignment in the store is exported, exactly as
# scripts/deploy-runtime.sh does. `set +u` only for the duration of the source:
# the store is the operator's file, not ours, and a line referencing an unset
# variable must not abort the agent. Errexit stays armed -- a store that fails to
# parse is a refusal, not something to run past.
set -a
set +u
# shellcheck disable=SC1090
. "$DEPLOY_AGENT_ENV_FILE"
set -u
set +a

# Restore systemd's ordering: the unit's own declarations win over the store.
for __name in $__restore_names; do
	eval "export ${__name}=\${__saved_${__name}}"
done

DEPLOY_AGENT_LAUNCHER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
export DEPLOY_AGENT_LAUNCHER

exec "$DEPLOY_AGENT_PYTHON" -m deploy_agent "$@"
