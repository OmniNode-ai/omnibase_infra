#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# reconcile-workspace-venvs.sh (OMN-17190)
# ----------------------------------------------------------------------------
# Bring every locally-installed venv back into agreement with the canonical
# clones and their dependency files. Nothing about the local install is
# hand-built, and nothing about repairing it is hand-run.
#
# Operator direction (2026-08-30): "Why is anything hand built? ... omnimarket
# has a dependency file that specifies what should be installed. We need a
# process that either (1) disconnects the local installation from the canonical
# clones, or (2) automatically pulls the clones whenever a PR is merged and
# refreshes the venv."  Decision: option 2. Dev-tip dogfooding is the point, so
# the local install TRACKS the clones and this reconciler is what closes the
# gap, automatically, on a tick and at session start.
#
# ============================================================================
# THE CORE FACT THIS SCRIPT EXISTS TO ENCODE
# ============================================================================
# The `onex` CLI venv has TWO governed layers, and reconciling only the first
# one actively BREAKS the second:
#
#   Layer 1 (lock-governed):  omnibase_infra/uv.lock, applied by `uv sync --frozen`
#   Layer 2 (composed):       omnimarket + its --no-deps companions, applied by
#                             scripts/install-node-skill-package.sh
#
# `omnimarket` is DELIBERATELY absent from omnibase_infra's pyproject.toml and
# uv.lock. The layer graph is compat -> core -> spi -> infra and omnimarket sits
# ABOVE infra (it depends on omnibase-infra >=0.38.3,<0.39.0); declaring it
# would invert the graph and publish a cycle in the omnibase-infra wheel. It is
# composed at runtime through `onex.nodes` entry points instead.
#
# The consequence is sharp and has already bitten once: a bare
# `uv sync --frozen` against this venv UNINSTALLS omnimarket (verified live
# 2026-08-30 -- `uv sync --frozen --check` proposes uninstalling 14 packages,
# of which omnimarket and its companions are 11), after which every
# `onex skill` / `onex node` / `onex delegate` dispatch dies on the OMN-14060
# guard's "omnimarket is NOT INSTALLED from git" refusal. The OMN-15620 venv
# purity repair did exactly this and bricked the CLI.
#
# `uv sync --inexact` is the primitive this shape actually needs: it applies
# every pin in the lock but does NOT remove packages the lock does not mention.
# So layer 1 stops being destructive to layer 2, and the two stop being an
# ordering puzzle.
#
# The order is then chosen for a different reason -- PROVIDER FIRST, LOCK
# SECOND -- because the provider co-install is itself capable of moving
# lock-governed pins. It carries a hardcoded `COMPAT_PIN="omnibase-compat==0.5.5"`
# and installs it `--no-deps`, silently downgrading the locked 0.5.6. That is
# OMN-16262, and it is not theoretical: it was reproduced on this Mac on
# 2026-08-30 by this very script, and the downgrade broke the `occ` CLI
# extension so completely that the `onex` binary would not start at all
# ("No module named 'omnibase_compat.contracts.pr_occ_stamp'"). Ending on the
# lock pass repairs that downgrade structurally, rather than leaving the
# reconciler to inherit the bug.
#
# ============================================================================
# WHY THE PROVIDER LAYER PINS TO THE LOCAL CLONE HEAD, NOT origin/dev
# ============================================================================
# install-node-skill-package.sh defaults to resolving omnimarket's ref from a
# live `git ls-remote ... dev`. That default is wrong for reconciliation and is
# the OMN-16366 defect: the drift guard compares the INSTALLED commit against
# the LOCAL clone's checked-out HEAD, so installing from a remote tip the clone
# has not fast-forwarded to leaves the venv *ahead* of the clone -- still
# drifted, still refused, now in the direction nobody looks for. This script
# therefore always passes an explicit OMNIMARKET_REF of the local clone HEAD.
# Advancing the clone is a separate concern and a separate actor (pull-all.sh,
# or the periodic tick that calls this script after a successful ff-only pull).
#
# ============================================================================
# THIS RECONCILES ONE CLI VENV -- SO THERE MUST BE ONE CLI ENTRY POINT
# ============================================================================
# Everything below repairs `$OMNI_HOME/omnibase_infra/.venv` and nothing else on
# the CLI surface. That is only useful if the `onex` an operator (or a script,
# or a hook) actually runs IS that venv's entry point. It routinely was not:
# `onex` was documented as an interactive shell alias, aliases do not exist in
# non-interactive shells, and PATH there resolved sibling installs with their
# own omnimarket state (measured 2026-08-30: a `uv tool` shim on pre-self-heal
# omnibase_infra 0.38.11, and a brew-python global install). A reconcile driven
# from one of those repairs this venv and then re-probes a different one, so it
# can never converge.
#
# `scripts/onex` is the fix and the documented entry point; the drift guard now
# refuses deterministically from any other interpreter rather than reconciling
# blind. See docs/runbooks/onex-cli-entrypoint.md.
#
# ============================================================================
# INTERIM BY DESIGN -- the node-based successor
# ============================================================================
# This is a script-level stopgap, authorized for beta. The successor is a
# NodeCompute drift-detect handler -- a pure function of (clone SHA, installed
# SHA, lock hash) -> typed verdict -- behind a NodeEffect reconcile publisher,
# driven by the runtime rather than by a shell tick, emitting its receipt to the
# bus instead of to a local log line. `--check` below is deliberately shaped as
# exactly that pure function, and `--repair` as exactly that effect, so the port
# is a lift rather than a rewrite. Tracked on OMN-17190.
#
# ----------------------------------------------------------------------------
# Usage:
#   reconcile-workspace-venvs.sh [--check] [--verbose] [--omni-home PATH]
#
#     --check       Report the verdict and mutate NOTHING. This is the mode the
#                   SessionStart line and any read-only probe must use.
#     --verbose     Echo each collaborator command before running it.
#     --omni-home   Canonical registry root, overriding $OMNI_HOME. An explicit
#                   argument exists so an in-process caller (the CLI drift
#                   guard) can hand the root it already resolved without
#                   rebuilding an environment for the subprocess.
#
# Env:
#   OMNI_HOME                        (required unless --omni-home is passed --
#                                    no default, CLAUDE.md rule 8)
#   ONEX_RECONCILE_INSTALL_SCRIPT    override the provider co-install script
#                                    (tests only; defaults to the sibling script)
#   CLAUDE_PLUGIN_DATA               optional; when its .venv exists it is a
#                                    hook-venv surface too
#
# Exit codes:
#   0  IN_SYNC (--check) / reconciled successfully (default)
#   1  DRIFT detected (--check only -- never returned by the repair path)
#   2  reconcile FAILED; the message names the exact command to run by hand
#   3  INDETERMINATE configuration (no OMNI_HOME, no canonical clone)
#
# There is NO bypass variable. The OMN-13930 override
# (ONEX_ALLOW_OMNIMARKET_DRIFT) exists on the *guard*, for an operator who
# knowingly accepts results from an unverified build. A reconcile that cannot
# complete is a different thing entirely -- the venv is broken -- and a bypass
# would only move the breakage to the next dispatch.
# ----------------------------------------------------------------------------
set -uo pipefail

readonly EXIT_OK=0
readonly EXIT_DRIFT=1
readonly EXIT_FAILED=2
readonly EXIT_INDETERMINATE=3


MODE="repair"
VERBOSE=0
OMNI_HOME_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE="check" ;;
    --verbose) VERBOSE=1 ;;
    --omni-home)
      shift
      if [[ $# -eq 0 ]]; then
        echo "reconcile-workspace-venvs.sh: --omni-home requires a path" >&2
        exit "$EXIT_INDETERMINATE"
      fi
      OMNI_HOME_ARG="$1"
      ;;
    --omni-home=*) OMNI_HOME_ARG="${1#--omni-home=}" ;;
    -h|--help) sed -n '1,105p' "${BASH_SOURCE[0]}"; exit "$EXIT_OK" ;;
    *) echo "reconcile-workspace-venvs.sh: unknown argument: $1" >&2; exit "$EXIT_INDETERMINATE" ;;
  esac
  shift
done

# The explicit argument wins over the ambient variable: a caller that resolved
# the root already must not be silently overridden by whatever the surrounding
# shell happened to export.
if [[ -n "$OMNI_HOME_ARG" ]]; then
  OMNI_HOME="$OMNI_HOME_ARG"
fi

say() { printf '[reconcile] %s\n' "$*"; }
trace() { [[ "$VERBOSE" -eq 1 ]] && printf '[reconcile]   $ %s\n' "$*"; return 0; }

# --------------------------------------------------------------------------- #
# Configuration: fail fast, never guess (CLAUDE.md rule 8)
# --------------------------------------------------------------------------- #
if [[ -z "${OMNI_HOME:-}" ]]; then
  say "INDETERMINATE: OMNI_HOME is not set and --omni-home was not passed."
  say "  Export OMNI_HOME to the canonical repo registry root, e.g."
  say "    export OMNI_HOME=/path/to/omni_home"
  say "  or pass it explicitly:"
  say "    reconcile-workspace-venvs.sh --omni-home /path/to/omni_home"
  say "  No default is applied: a guessed root would reconcile some other"
  say "  checkout's venv and report success for a venv nobody is running."
  exit "$EXIT_INDETERMINATE"
fi

INFRA_DIR="$OMNI_HOME/omnibase_infra"
MARKET_CLONE="$OMNI_HOME/omnimarket"
CLAUDE_DIR="$OMNI_HOME/omniclaude"

INFRA_VENV="$INFRA_DIR/.venv"
INFRA_PYTHON="$INFRA_VENV/bin/python"

INSTALL_SCRIPT="${ONEX_RECONCILE_INSTALL_SCRIPT:-$INFRA_DIR/scripts/install-node-skill-package.sh}"

if [[ ! -d "$MARKET_CLONE/.git" ]]; then
  say "INDETERMINATE: no canonical omnimarket clone at $MARKET_CLONE"
  say "  The provider layer is reconciled against that clone's HEAD; without it"
  say "  there is no reference commit to reconcile to."
  exit "$EXIT_INDETERMINATE"
fi

UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" ]]; then
  say "INDETERMINATE: \`uv\` is not on PATH; every sync below is a uv operation."
  exit "$EXIT_INDETERMINATE"
fi

# --------------------------------------------------------------------------- #
# Pure observations (the future NodeCompute inputs)
# --------------------------------------------------------------------------- #
# Local `git rev-parse HEAD` only -- never `ls-remote`. Advancing the clone is
# somebody else's job; this script reconciles the venv to whatever is checked
# out, which is precisely what the drift guard compares against.
market_head() {
  git -C "$MARKET_CLONE" rev-parse HEAD 2>/dev/null || true
}

# Whether the lock-governed layer is satisfied. `--inexact` is what makes this
# answerable at all for the CLI venv: without it uv reports every provider
# package as "extraneous" and the venv can never read as conformant. With it,
# uv answers the only question that matters -- is every LOCKED pin installed at
# the locked version -- and stays silent about the composed layer above.
#
# uv is deliberately the authority here rather than a hash this script computes
# and stamps. A stamp records what a previous run BELIEVED; it cannot see a
# package mutated in place afterwards, which is exactly how the OMN-15620
# pollution went unnoticed. This costs one uv invocation and cannot go stale.
lock_layer_ok() {
  local project="$1"
  shift
  (cd "$project" && env -u PYTHONPATH "$UV_BIN" sync --frozen --check --project "$project" "$@" >/dev/null 2>&1)
}

installed_market_commit() {
  [[ -x "$INFRA_PYTHON" ]] || return 0
  env -u PYTHONPATH "$INFRA_PYTHON" - <<'PYEOF' 2>/dev/null || true
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution

try:
    dist = distribution("omnimarket")
except PackageNotFoundError:
    print("")
    sys.exit(0)
raw = dist.read_text("direct_url.json") or ""
try:
    data = json.loads(raw) if raw else {}
except json.JSONDecodeError:
    data = {}
print(data.get("vcs_info", {}).get("commit_id", ""))
PYEOF
}

# --------------------------------------------------------------------------- #
# Hook-venv surface resolution
# --------------------------------------------------------------------------- #
# CLAUDE.md rule 11 and the memory record both name a daemon venv path that
# does NOT exist on this host (`omniclaude/plugins/onex/lib/.venv`,
# `~/.claude/plugins/data/onex-omninode-tools/.venv`). The venv actually
# executing hooks here is the omniclaude PROJECT venv, reached through the
# dev marketplace plugin. Trusting the documented path would reconcile a venv
# nothing runs while the live one drifts, so resolve by probing instead --
# mirroring `plugins/onex/hooks/scripts/common.sh:find_python()`'s precedence.
#
# Every candidate that EXISTS is reconciled. Reconciling all of them is
# strictly safer than choosing one, because whichever find_python() selects at
# runtime will then be in sync. A candidate that does not exist is SKIPPED and
# never created -- creating a plugin venv is repair-plugin-venv.sh's job, and
# doing it from a background tick would be a surprise, not a repair.
hook_venv_projects() {
  local seen=""
  local candidate
  for candidate in "${CLAUDE_PLUGIN_DATA:-}" "$CLAUDE_DIR"; do
    [[ -n "$candidate" ]] || continue
    [[ -d "$candidate/.venv" ]] || continue
    [[ -f "$candidate/uv.lock" ]] || continue
    case " $seen " in *" $candidate "*) continue ;; esac
    seen="$seen $candidate"
    printf '%s\n' "$candidate"
  done
}

# --------------------------------------------------------------------------- #
# --check : verdict only, zero mutation
# --------------------------------------------------------------------------- #
run_check() {
  local drift=0
  local head installed

  head="$(market_head)"
  if [[ -z "$head" ]]; then
    say "INDETERMINATE: could not read HEAD of $MARKET_CLONE"
    exit "$EXIT_INDETERMINATE"
  fi

  if [[ ! -x "$INFRA_PYTHON" ]]; then
    say "DRIFT: cli venv absent ($INFRA_VENV)"
    drift=1
  else
    if ! lock_layer_ok "$INFRA_DIR" --inexact; then
      say "DRIFT: cli venv does not satisfy $INFRA_DIR/uv.lock"
      drift=1
    fi
    installed="$(installed_market_commit)"
    if [[ "$installed" != "$head" ]]; then
      say "DRIFT: cli venv omnimarket ${installed:0:12} != clone HEAD ${head:0:12}"
      drift=1
    fi
  fi

  local project
  while IFS= read -r project; do
    [[ -n "$project" ]] || continue
    trace "uv sync --frozen --check --project $project"
    if ! lock_layer_ok "$project"; then
      say "DRIFT: hook venv $project/.venv is outdated against $project/uv.lock"
      drift=1
    fi
  done < <(hook_venv_projects)

  if [[ "$drift" -eq 1 ]]; then
    say "verdict: DRIFT"
    exit "$EXIT_DRIFT"
  fi
  say "verdict: IN_SYNC (omnimarket ${head:0:12})"
  exit "$EXIT_OK"
}

# --------------------------------------------------------------------------- #
# repair : idempotent, quiet, fatal on failure
# --------------------------------------------------------------------------- #
fail() {
  # A refusal that does not name the command to run is a dead end. Every
  # failure path below prints the exact invocation.
  say "FAILED: $1"
  shift
  local line
  for line in "$@"; do
    say "  $line"
  done
  exit "$EXIT_FAILED"
}

run_repair() {
  local head installed need_lock=0 need_provider=0

  head="$(market_head)"
  if [[ -z "$head" ]]; then
    say "INDETERMINATE: could not read HEAD of $MARKET_CLONE"
    exit "$EXIT_INDETERMINATE"
  fi

  # ---- surface 1: the onex CLI venv (two layers) -------------------------- #
  #
  # Each layer is decided independently, so a tick that runs every 10 minutes
  # does the least work that closes the actual gap. The common case by far --
  # the clone advanced, the lock did not -- is additive `--no-deps` provider
  # work plus a lock pass that finds nothing to do.
  if [[ ! -x "$INFRA_PYTHON" ]]; then
    need_lock=1
    need_provider=1
  else
    lock_layer_ok "$INFRA_DIR" --inexact || need_lock=1
    installed="$(installed_market_commit)"
    [[ "$installed" == "$head" ]] || need_provider=1
  fi

  if [[ "$need_lock" -eq 0 && "$need_provider" -eq 0 ]]; then
    say "cli venv: already in sync (omnimarket ${head:0:12})"
  else
    # PROVIDER FIRST. The co-install can move lock-governed pins (OMN-16262:
    # its hardcoded COMPAT_PIN downgrades omnibase-compat 0.5.6 -> 0.5.5 and
    # breaks the `occ` CLI extension badly enough that `onex` will not start),
    # so the lock pass has to come after it to undo that.
    if [[ "$need_provider" -eq 1 ]]; then
      say "cli venv: reconciling provider layer to omnimarket ${head:0:12}"
      if [[ ! -x "$INSTALL_SCRIPT" ]]; then
        fail "provider co-install script is missing or not executable." \
          "Expected at: $INSTALL_SCRIPT"
      fi
      # OMNIMARKET_REF is set explicitly so the install script's own ls-remote
      # default (OMN-16366 reversed drift) can never apply here.
      trace "OMNIMARKET_REF=$head $INSTALL_SCRIPT --execute $INFRA_PYTHON"
      if ! env OMNIMARKET_REF="$head" OMNI_HOME="$OMNI_HOME" \
          bash "$INSTALL_SCRIPT" --execute "$INFRA_PYTHON"; then
        fail "provider co-install did not complete; omnimarket is not installed." \
          "Every \`onex skill\`/\`onex node\`/\`onex delegate\` dispatch will refuse" \
          "until this succeeds. Run by hand and read the error:" \
          "  OMNIMARKET_REF=$head OMNI_HOME=$OMNI_HOME \\" \
          "    bash $INSTALL_SCRIPT --execute $INFRA_PYTHON" \
          "  (this is scripts/install-node-skill-package.sh)"
      fi
      # The co-install just ran, so the lock pass is mandatory regardless of
      # what the pre-check said -- that is the whole OMN-16262 repair.
      need_lock=1
    fi

    if [[ "$need_lock" -eq 1 ]]; then
      say "cli venv: applying $INFRA_DIR/uv.lock"
      # --frozen: apply the lock, never re-resolve it -- a re-resolution here
      #   would silently move the very pins the lock exists to hold.
      # --inexact: do not remove the composed provider layer, which the lock
      #   correctly does not mention and must not be asked to.
      trace "uv sync --frozen --inexact --project $INFRA_DIR"
      if ! (cd "$INFRA_DIR" && env -u PYTHONPATH "$UV_BIN" sync --frozen --inexact --project "$INFRA_DIR"); then
        fail "cli venv lock sync did not complete." \
          "Run by hand and read the error:" \
          "  cd $INFRA_DIR && env -u PYTHONPATH uv sync --frozen --inexact"
      fi
    fi
    say "cli venv: reconciled"
  fi

  # ---- surface 2: the hook venv(s), lock-governed only -------------------- #
  # Exact (not --inexact) on purpose: a hook venv has no composed layer above
  # its lock, so anything the lock does not mention is cross-repo pollution.
  # This host had omnibase_infra's dev group (pre-commit, import-linter,
  # pytest-xdist, hypothesis, ...) installed into omniclaude/.venv -- the
  # OMN-15620 class, and precisely the sort of thing that makes one venv answer
  # a question differently from another. An exact sync is what removes it.
  local project any_hook=0
  while IFS= read -r project; do
    [[ -n "$project" ]] || continue
    any_hook=1
    trace "uv sync --frozen --check --project $project"
    if lock_layer_ok "$project"; then
      say "hook venv: already in sync ($project/.venv)"
      continue
    fi
    say "hook venv: reconciling $project/.venv to $project/uv.lock"
    trace "uv sync --frozen --project $project"
    if ! (cd "$project" && env -u PYTHONPATH "$UV_BIN" sync --frozen --project "$project"); then
      fail "hook venv lock sync did not complete for $project." \
        "Run by hand and read the error:" \
        "  cd $project && env -u PYTHONPATH uv sync --frozen"
    fi
    say "hook venv: reconciled ($project/.venv)"
  done < <(hook_venv_projects)

  if [[ "$any_hook" -eq 0 ]]; then
    # Not a failure: a host with no hook venv (CI, a fresh clone) has nothing
    # to reconcile on this surface. Creating one is repair-plugin-venv.sh's job.
    say "hook venv: SKIP (no existing hook venv found to reconcile)"
  fi

  exit "$EXIT_OK"
}

if [[ "$MODE" == "check" ]]; then
  run_check
fi
run_repair
