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
# ...AND WHY THAT COMPOSED VENV IS NOT THE CLONE'S OWN .venv (OMN-17819)
# ============================================================================
# Everything above is true about the CLI, and every word of it was applied to
# the WRONG DIRECTORY until OMN-17819. The composed layer used to be installed
# into `$OMNI_HOME/omnibase_infra/.venv` -- which is not only the CLI's venv,
# it is also the canonical clone's PROJECT venv, the one `uv run pytest` in
# that clone executes in. So the same directory was being asked to satisfy two
# contracts that contradict each other:
#
#   the CLI     needs omnimarket present   (OMN-14060 refuses without it)
#   the gate    needs omnimarket ABSENT    (OMN-15620 refuses with it, because
#                                           an undeclared `onex.nodes` provider
#                                           collides with declared providers of
#                                           the same node identity and
#                                           manufactures DUPLICATE_REGISTRATION
#                                           false REDs across the whole suite)
#
# Measured on this Mac 2026-09-03 and again 2026-09-17: `cd
# $OMNI_HOME/omnibase_infra && uv run pytest <anything>` was refused BEFORE
# collection with "Canonical venv is IMPURE: ... omnimarket==0.4.111", and the
# repair the refusal itself recommends (`uv sync`, exact) is the CLI-bricking
# path described above. Worse, the repair did not even hold: this reconciler
# re-composed the provider layer on its next tick (600 s), so `uv sync` bought
# minutes. Three separate lanes were pushed into throwaway worktrees to run
# three tests each.
#
# This is the OMN-16846 collision, which CI already settled on 2026-08-28 in
# exactly these words: "Neither side is wrong on its own -- the purity gate is
# correct and the co-install is required -- so the defect is the collapse of
# two environments into one." CI's fix is `$DISPATCH_VENV` in
# `.github/workflows/evidence-autoclose-sweep.yml`; the measured effect on one
# ticket was `verified=1 failed=3 behavior_proving=0` collapsed versus
# `verified=5 failed=0 behavior_proving=3` separated. This script is the same
# fix for the local workspace, so that the split is a property of the platform
# rather than of one workflow file.
#
#   GATE venv      $OMNI_HOME/omnibase_infra/.venv       lock-governed ONLY.
#                                                        Synced EXACT. Anything
#                                                        the lock does not name
#                                                        is pollution and is
#                                                        removed.
#   DISPATCH venv  $OMNI_HOME/.onex-dispatch-venv        both layers. Synced
#                                                        --inexact, then the
#                                                        provider co-install.
#                                                        `scripts/onex` execs
#                                                        THIS one, and the
#                                                        OMN-17309 floor reads
#                                                        its omnimarket commit.
#
# ORDER MATTERS ACROSS THE TWO SURFACES, not just within one. The dispatch venv
# is reconciled FIRST and the gate venv second. The gate pass is the step that
# removes omnimarket from `.venv`, and doing that before a working dispatch venv
# exists would leave a window -- possibly a permanent one, if the dispatch build
# then fails -- in which no interpreter on this host can run `onex`. A dispatch
# failure therefore refuses without touching the gate venv at all.
#
# The dispatch venv is NOT under `$OMNI_HOME/omnibase_infra/`: a venv inside the
# clone is a venv some future `uv` invocation or purity probe will find while
# answering a question about the clone. It sits beside
# `.onex-workspace-floor.json` and `.onex-workspace-reconcile.json`, which are
# the other pieces of workspace-scoped state this reconciler owns. It is also
# deliberately not `$CLAUDE_PLUGIN_DATA/.venv`: that venv belongs to
# `repair-plugin-venv.sh` (CLAUDE.md rule 11's table), and taking it over here
# would give one directory two owners -- the same mistake one level up.
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
# ...AND WHY THE VERDICT STILL HAS TO MENTION origin/dev (OMN-17295)
# ============================================================================
# Pinning to the clone is right. Reporting only that comparison is not. A clone
# eleven commits behind origin/dev, with a venv installed exactly at its HEAD,
# produced this line unchanged:
#
#     [reconcile] verdict: IN_SYNC (omnimarket 87afb9c33215)
#
# Every word of it is true about the layer this script owns and false about the
# workspace the reader is asking about -- and the SessionStart hook advertises
# this exact command as the way to "settle it now" when no tick has run. That is
# the OMN-17295 defect class in the ticket's own words: a probe that silently
# measures something narrower than it names does not fail loudly, it produces a
# confident wrong answer. OMN-17295 AC5 is "local venv -> origin/dev drift is
# detected on this Mac, not just venv -> clone".
#
# So the verdict now carries a THIRD leg: for every canonical clone named by
# sibling_clone_manifest.sh, HEAD against origin/<branch>. Two properties of it
# are deliberate and are asserted by
# tests/scripts/test_reconcile_clone_origin_drift_omn17295.py:
#
#   1. It NEVER fetches. `--check` exempts itself from the OMN-17366 ownership
#      plan on the grounds that a read-only probe writes nothing -- and a fetch
#      writes, depositing objects, refs and reflogs. On the `.201` root cron
#      that is exactly how 1118 root-owned paths ended up inside operator-owned
#      clones. The target is read AS LAST FETCHED and the report says so;
#      reconcile-host.sh is the actor that fetches, under an ownership plan.
#   2. It NEVER advances a clone. There is exactly one clone reconciler
#      (scripts/runtime_build/reconcile_deploy_clones.sh, OMN-17291) and one
#      venv reconciler (this file), composed by reconcile-host.sh. A
#      fast-forward added here would be the third implementation that
#      composition exists to prevent, and AC5 says to coordinate with OMN-17291
#      rather than duplicate its reconciler.
#
# Detection is therefore verdict-bearing in `--check` and report-only in the
# repair path: the repair path's exit code answers for the surfaces this script
# actually writes, and a clone it is forbidden to touch is not one of them.
# Convergence is preserved because the command the message names is
# reconcile-host.sh, not this script's own repair mode.
#
# ============================================================================
# WHAT THE VERDICT COVERS, AND WHAT IT ONLY REPORTS (OMN-18663)
# ============================================================================
# Two surfaces were invisible to the verdict until OMN-18663, in opposite ways.
#
# 1. The DISPATCH venv's lock leg asked uv a question the repair path never
#    answers. The repair syncs with `--python "$base_python"`; the check did
#    not, so uv resolved the interpreter from `.python-version` (3.12) while the
#    venv is correctly on the brew 3.13 rule 11 requires, and answered "would
#    replace this environment". The result was a permanent DRIFT verdict that no
#    repair could clear, with repair ticks reporting success into it. The check
#    now passes the same interpreter argument the repair does.
#
# 2. The plugin CLI venv (rule 11's CLI-venv row) was omitted ENTIRELY and
#    silently, because `hook_venv_projects` requires a `uv.lock` and that venv
#    is built from a requirements.txt. It is still not owned here -- taking it
#    over would give one directory two owners -- but it is now named in both
#    modes, its lock layer reported as unowned with the script that does own it,
#    and its provider layer read back against the canonical clone. A surface
#    this script cannot repair is a surface it must still be honest about.
#
# A THIRD SURFACE, AND IT IS NOT A VENV (OMN-18260). The lane-identity
# `prepare-commit-msg` hook is host state that lives inside each clone's git
# directory, and until now nothing converged it: it was armed on the operator's
# Mac because somebody ran `lane_identity reconcile --execute` by hand, and
# unarmed and unmentioned everywhere else. Measured over 2026-09-11..09-18, 39
# of 4,105 commits across six repositories carried the lane trailer (0.95%),
# which is the ceiling on every downstream gate that reads a lane out of a
# commit. This script now arms it and READS IT BACK per clone, in both modes.
# Unlike the CLI venv above, this surface IS written here, so an unarmed clone
# is verdict-bearing rather than merely reported. What is never written is a
# worktree file, a ref or the index -- the rule that this script never advances
# a clone is untouched.
#
# ============================================================================
# INTERIM BY DESIGN -- the node-based successor
# ============================================================================
# movement-proof-delegated-to: scripts/reconcile-host.sh
#
# ============================================================================
# THIS SCRIPT DOES NOT PROVE ITS OWN WORK (OMN-17307)
# ============================================================================
# Read the repair path below and note what it does NOT do: after
# `install-node-skill-package.sh` and `uv sync --frozen --inexact` return, it
# exits 0 on their exit STATUS. It never re-reads the venv to confirm the pins
# actually landed. That is not a small gap -- the provider co-install is known
# to move pins nobody asked it to move (OMN-16262: a hardcoded COMPAT_PIN
# downgrading omnibase-compat 0.5.6 -> 0.5.5, which broke the `occ` CLI
# extension badly enough that the `onex` binary would not start), and no exit
# code can see a content change like that.
#
# The readback lives one layer up, in `scripts/reconcile-host.sh`, which
# observes every governed distribution BEFORE and AFTER this script runs and
# compares the result against the lock and the canonical clone HEAD. That is
# deliberate rather than incidental: this script owns REPAIR, the orchestrator
# owns PROOF, and keeping them separate is what stops the repair from being its
# own witness.
#
# So: do not add a success claim here. Call this through reconcile-host.sh.
# The `check_reconciler_movement_proof.py` gate holds the marker above to that
# arrangement.
#
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
#                                [--branch NAME]
#
#     --check       Report the verdict and mutate NOTHING. This is the mode the
#                   SessionStart line and any read-only probe must use.
#     --verbose     Echo each collaborator command before running it.
#     --omni-home   Canonical registry root, overriding $OMNI_HOME. An explicit
#                   argument exists so an in-process caller (the CLI drift
#                   guard) can hand the root it already resolved without
#                   rebuilding an environment for the subprocess.
#     --branch      Branch the canonical clones are expected to track
#                   (default: dev). Only the clone->origin/<branch> observation
#                   reads it; it selects no behaviour and moves no ref.
#
# Env:
#   OMNI_HOME                        (required unless --omni-home is passed --
#                                    no default, CLAUDE.md rule 8)
#   ONEX_RECONCILE_INSTALL_SCRIPT    override the provider co-install script
#                                    (tests only; defaults to the sibling script)
#   ONEX_RECONCILE_UV_BIN            absolute path to `uv`, tried FIRST. Not a
#                                    bypass: it changes which uv runs, never
#                                    whether the sync has to succeed.
#   ONEX_DISPATCH_VENV               absolute path to the dispatch venv,
#                                    overriding $OMNI_HOME/.onex-dispatch-venv.
#                                    `scripts/onex` reads the same variable with
#                                    the same default, so the two cannot point
#                                    at different directories by accident. Not a
#                                    bypass: it relocates the composed venv, it
#                                    never makes the gate venv a legal home for
#                                    the provider layer again.
#   CLAUDE_PLUGIN_DATA               optional; when its .venv exists it is a
#                                    hook-venv surface too
#   ONEX_LANE_IDENTITY_SCRIPT        override the lane-identity module armed on
#                                    each clone (tests only; defaults to
#                                    $OMNI_HOME/omniclaude/scripts/lane_identity.py).
#                                    Not a bypass: a path that does not resolve
#                                    reports the surface as unasked, it never
#                                    reports it as armed.
#
# Exit codes:
#   0  IN_SYNC (--check) / reconciled successfully (default)
#   1  DRIFT detected. Three classes, and the message says which: a venv layer
#      that does not match its target, which THIS script repairs; a canonical
#      clone behind origin/<branch> (OMN-17295), which reconcile-host.sh
#      repairs and this script only reports; or a clone carrying no
#      lane-identity stamping hook (OMN-18260), which this script arms and
#      then reads back. The first two are --check only. The third is returned
#      by the REPAIR path as well, because the repair path writes that surface
#      and its exit code has to answer for what it wrote.
#   2  reconcile FAILED; the message names the exact command to run by hand
#   3  INDETERMINATE configuration (no OMNI_HOME, no canonical clone, no uv,
#      or a surface this process must not write)
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
BRANCH="dev"
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
    --branch)
      shift
      if [[ $# -eq 0 ]]; then
        echo "reconcile-workspace-venvs.sh: --branch requires a name" >&2
        exit "$EXIT_INDETERMINATE"
      fi
      BRANCH="$1"
      ;;
    --branch=*) BRANCH="${1#--branch=}" ;;
    # Through the end of the usage block, not a hand-counted line number: the
    # previous literal stopped inside the header and cut the options, the env
    # table and the exit codes out of `--help` entirely, and it went stale
    # again the moment anything above it grew.
    -h|--help) sed -n '1,/^set -uo pipefail$/p' "${BASH_SOURCE[0]}"; exit "$EXIT_OK" ;;
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

# The GATE venv: the canonical clone's own project environment, the one
# `uv run pytest` executes in and the one the OMN-15620 purity gate judges.
# Lock-governed only. The two names are kept because every caller and test
# already uses them; what changed in OMN-17819 is what they are ALLOWED to
# contain, not where they point.
INFRA_VENV="$INFRA_DIR/.venv"
INFRA_PYTHON="$INFRA_VENV/bin/python"

# The DISPATCH venv: the composed environment `scripts/onex` execs and the
# OMN-17309 floor is read from. Outside the clone on purpose (see the header).
#
# Derived from $OMNI_HOME, which is already resolved fail-fast above -- there is
# no default root and no absolute path literal here, so this resolves correctly
# on a machine whose checkout lives somewhere else (CLAUDE.md rules 6 and 8).
DISPATCH_VENV="${ONEX_DISPATCH_VENV:-$OMNI_HOME/.onex-dispatch-venv}"
DISPATCH_PYTHON="$DISPATCH_VENV/bin/python"

# --------------------------------------------------------------------------- #
# The dispatch venv may never BE a clone's own venv
# --------------------------------------------------------------------------- #
# The default above cannot be a clone's `.venv`, but `ONEX_DISPATCH_VENV` can,
# and pointing it at one would reconstruct the exact OMN-17819 defect by hand:
# the provider layer composed into a project environment, the OMN-15620 purity
# gate refusing every `uv run pytest` there, and the refusal's own recommended
# repair breaking the CLI. The override exists to RELOCATE the composed venv,
# never to re-collapse the two.
#
# `-e`, not `-d`, on the `.git` test: in a worktree `.git` is a FILE, and a
# `-d` test would wave through exactly the per-ticket worktrees lanes spend all
# day inside.
refuse_dispatch_venv_inside_a_clone() {
  local parent
  parent="${DISPATCH_VENV%/*}"

  # The gate venv is refused UNCONDITIONALLY, before any `.git` test. Whether
  # that directory happens to be a git checkout right now is irrelevant: it is
  # the environment `uv run pytest` executes in and the OMN-15620 purity gate
  # judges, and that is true of a tarball, a CI workspace and a fresh clone
  # alike. Gating this on `.git` made the refusal miss the single case the whole
  # ticket is about whenever the directory was not a repository.
  if [[ "$DISPATCH_VENV" == "$INFRA_VENV" ]]; then
    say "INDETERMINATE: refusing to compose the provider layer into $DISPATCH_VENV."
    say "  That is the canonical clone's GATE venv -- the one \`uv run pytest\`"
    say "  runs in and the OMN-15620 purity gate judges. Composing an undeclared"
    say "  \`onex.nodes\` provider into it is the OMN-17819 defect itself."
    say "  The dispatch venv must live OUTSIDE every clone. Unset"
    say "  ONEX_DISPATCH_VENV to use the default ($OMNI_HOME/.onex-dispatch-venv),"
    say "  or point it at a path that is not a clone's .venv."
    exit "$EXIT_INDETERMINATE"
  fi

  [[ "${DISPATCH_VENV##*/}" == ".venv" ]] || return 0
  [[ -e "$parent/.git" ]] || return 0

  say "INDETERMINATE: refusing to compose the provider layer into $DISPATCH_VENV."
  say "  That is the project venv of the git clone at $parent. A composed"
  say "  layer there is undeclared in that project's lock, so its own purity"
  say "  checks and test runs would be refused -- the OMN-17819 defect, moved."
  say "  The dispatch venv must live OUTSIDE every clone. Unset"
  say "  ONEX_DISPATCH_VENV to use the default ($OMNI_HOME/.onex-dispatch-venv),"
  say "  or point it at a path that is not a clone's .venv."
  exit "$EXIT_INDETERMINATE"
}
refuse_dispatch_venv_inside_a_clone

# --------------------------------------------------------------------------- #
# The dispatch venv's base interpreter (CLAUDE.md rule 11)
# --------------------------------------------------------------------------- #
# macOS Sonoma+ grants Local Network access per binary path and signature, not
# per Python version. An adhoc-signed uv-managed interpreter never surfaces the
# privacy dialog and its LAN connections fail silently with EHOSTUNREACH; the
# brew interpreter carries the grant. The dispatch venv is what `scripts/onex`
# execs, and `onex` talks to the LAN services on the lab host, so on macOS it
# must be built on the brew binary at its literal resolved path -- never
# `$(brew --prefix)`, which a launchd or cron PATH cannot resolve.
#
# The requirement is macOS-only, deliberately. Rule 11 scopes itself to the
# `local_macos_claude_hooks` profile and says in terms that it does not apply to
# CI runners, containers, or the lab host. This same reconciler runs from the
# lab host's cron, where there is no brew prefix and no privacy gate to satisfy;
# requiring one there would refuse every tick for a constraint that does not
# exist on that platform.
#
# Historical note, so this is not mis-read as a regression: the venv the
# OMN-17819 split replaced was ALREADY uv-managed -- `omnibase_infra/.venv` read
# `home = <uv data dir>/cpython-3.12-macos-aarch64-none/bin`. The CLI never had
# the grant on this host. The split is what made the interpreter fixable in one
# place rather than entangled with the clone's own project environment.
DISPATCH_BASE_PYTHON_CANDIDATES=(
  "/opt/homebrew/bin/python3.13"  # Apple Silicon
  "/usr/local/bin/python3.13"     # Intel
)

# Echoes the interpreter `uv` must build the dispatch venv on, or nothing when
# this platform has no such requirement. A macOS host with no brew interpreter
# echoes nothing too, and the caller refuses -- naming every path it looked at,
# because "not on PATH" was read as "not installed" once already (OMN-17335).
# Whether this host has the LAN-grant constraint at all.
#
# Both overrides can only ADD the requirement, never remove it -- there is no
# value of either that turns it off, and `dispatch_base_python` refuses rather
# than falling back when it cannot satisfy one. That is deliberate: a variable
# that switched a safety requirement off would be the bypass rule 17 forbids.
# What they buy is a behaviour that is exercised by the merge gate: CI runs on
# Linux, so without a seam the entire rule-11 enforcement would be untestable
# there and would ship unproven (rule 5 -- opt-in verification never gets
# adopted).
dispatch_requires_base_python() {
  [[ "${ONEX_DISPATCH_REQUIRE_BASE_PYTHON:-0}" == "1" ]] && return 0
  [[ -n "${ONEX_DISPATCH_BASE_PYTHON:-}" ]] && return 0
  [[ "$(uname -s)" == "Darwin" ]]
}

dispatch_base_python() {
  dispatch_requires_base_python || return 0
  if [[ -n "${ONEX_DISPATCH_BASE_PYTHON:-}" ]]; then
    # Validated, not trusted. Echoing a path that is not there hands uv an
    # argument it cannot use and turns a clear "that interpreter does not
    # exist" into whatever uv says three steps later -- and on a shim-backed
    # test host, into no error at all.
    if [[ ! -x "$ONEX_DISPATCH_BASE_PYTHON" ]]; then
      say "INDETERMINATE: ONEX_DISPATCH_BASE_PYTHON names $ONEX_DISPATCH_BASE_PYTHON,"
      say "  which is not an executable file. The dispatch venv is what"
      say "  \`scripts/onex\` execs; building it on an interpreter that is not"
      say "  there would leave this host with no CLI."
      exit "$EXIT_INDETERMINATE"
    fi
    printf '%s' "$ONEX_DISPATCH_BASE_PYTHON"
    return 0
  fi
  local candidate
  for candidate in $(dispatch_base_python_candidates); do
    [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
  done
}

# The candidates, as a whitespace-separated list. `ONEX_DISPATCH_BASE_PYTHON_
# CANDIDATES` (colon-separated) replaces the built-ins.
#
# This narrows WHICH interpreter qualifies; it never changes WHETHER one is
# required, so it is not an opt-out. It exists because the refusal path -- no
# acceptable interpreter anywhere -- is otherwise untestable on any host that
# has brew installed, which is every developer Mac. Two earlier fixtures in this
# suite already read host state that way and had to be repaired; this is the
# same lesson applied to the code instead of to the test.
dispatch_base_python_candidates() {
  if [[ -n "${ONEX_DISPATCH_BASE_PYTHON_CANDIDATES:-}" ]]; then
    printf '%s' "${ONEX_DISPATCH_BASE_PYTHON_CANDIDATES//:/ }"
    return 0
  fi
  printf '%s' "${DISPATCH_BASE_PYTHON_CANDIDATES[*]}"
}

# Resolve a directory through every symlink, with shell builtins only: macOS
# ships no GNU `realpath`, and the comparison below cannot be a string match --
# `/opt/homebrew/bin/python3.13` is a symlink into the Cellar while the venv
# records `/opt/homebrew/opt/python@3.13/bin`, two spellings of one directory.
real_dir() { (cd "$1" 2>/dev/null && pwd -P) || true; }

# The directory an executable REALLY lives in, following the file's own symlinks
# first. This is not the same as resolving its parent directory, and the
# difference is the whole comparison: `/opt/homebrew/bin` is a real directory
# full of symlinks, so resolving IT yields `/opt/homebrew/bin`, while the
# interpreter inside it points at `/opt/homebrew/Cellar/python@3.13/<v>/bin` --
# which is what a venv built on it records, through the third spelling
# `/opt/homebrew/opt/python@3.13/bin`.
#
# Resolving the parent refused a venv that had just been rebuilt correctly, on
# the live host, with every package installed and the interpreter exactly right.
# The readback caught it, which is what a readback is for; the comparison it fed
# was the thing that was wrong. The symlink walk is the same idiom
# `scripts/onex` uses to resolve itself.
real_file_dir() {
  local f="$1" link
  while [[ -L "$f" ]]; do
    link="$(readlink "$f")"
    case "$link" in
      /*) f="$link" ;;
      *) f="${f%/*}/$link" ;;
    esac
  done
  real_dir "${f%/*}"
}

# The `home` line of a venv's own pyvenv.cfg: the interpreter it was built on,
# as recorded by the builder rather than as assumed by us.
venv_base_home() {
  local cfg="$1/pyvenv.cfg" line
  [[ -f "$cfg" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      home*=*)
        line="${line#*=}"
        printf '%s' "${line# }"
        return 0
        ;;
    esac
  done < "$cfg"
}

# The installation an interpreter belongs to: its path with the trailing
# `bin/<exe>` removed, resolved. `/opt/homebrew/bin/python3.13` -> `/opt/homebrew`.
python_install_root() {
  local p="${1%/*}"
  [[ "${p##*/}" == "bin" ]] && p="${p%/*}"
  real_dir "$p"
}

# Whether the existing dispatch venv is already on the required interpreter.
# True when no interpreter is required (non-macOS) so the caller reads the same
# on every platform.
#
# CONTAINMENT, NOT DIRECTORY EQUALITY -- and that is the third attempt at this
# predicate, so the two measured failures are recorded here rather than left for
# someone to rediscover:
#
#   1. Comparing the two PARENT DIRECTORIES. `/opt/homebrew/bin` is a real
#      directory full of symlinks, so it resolves to itself, while a venv built
#      on the interpreter inside it records `/opt/homebrew/opt/python@3.13/bin`.
#      Never equal.
#   2. Fully resolving the interpreter FILE. Measured on this host, brew's
#      `python3.13` resolves through the Cellar and into the framework bundle:
#         /opt/homebrew/Cellar/python@3.13/3.13.3/Frameworks/Python.framework/Versions/3.13/bin
#      while the venv records
#         /opt/homebrew/Cellar/python@3.13/3.13.3/bin
#      Both are correct paths to one installation, and neither directory equals
#      the other.
#
# Directory equality was simply the wrong question. What rule 11 asks is whether
# the venv's base interpreter comes from the BREW INSTALLATION -- which holds the
# macOS Local Network grant -- rather than from uv's managed store. Both brew
# spellings resolve under `/opt/homebrew`; a uv-managed interpreter resolves
# under the uv data directory. Containment answers exactly that and is immune to
# how many symlinks either side happens to traverse.
# The second argument is the venv to judge, defaulting to the live dispatch
# venv. An atomic rebuild has to ask this about the STAGED venv, before that
# one becomes live -- see swap_dispatch_venv below.
dispatch_interpreter_ok() {
  local want="$1" venv="${2:-$DISPATCH_VENV}" home want_root have_dir
  [[ -n "$want" ]] || return 0
  home="$(venv_base_home "$venv")"
  [[ -n "$home" ]] || return 1
  want_root="$(python_install_root "$want")"
  have_dir="$(real_dir "$home")"
  [[ -n "$want_root" && -n "$have_dir" ]] || return 1
  # Equal, or underneath. The trailing slash stops `/opt/homebrew-other` from
  # matching `/opt/homebrew`.
  [[ "$have_dir" == "$want_root" || "$have_dir" == "$want_root"/* ]]
}

# --------------------------------------------------------------------------- #
# Atomic rebuild: stage a sibling, prove it, then rename it in (OMN-17819)
# --------------------------------------------------------------------------- #
# A rebuild used to happen IN PLACE, on the one path every lane's `onex` execs.
# `uv sync` removes what the lock does not mention before it installs, so for
# the length of a rebuild -- and for the whole of a FAILED one -- the dispatch
# venv on disk is a half-built environment with no omnimarket in it, and every
# `onex skill` / `node` / `delegate` in every concurrent lane refuses. That is
# not hypothetical: on 2026-09-17 a wrong interpreter predicate rebuilt and then
# refused its own readback on a ~600s tick, and the provider layer was missing
# from the shared venv for over an hour while lanes failed against it.
#
# So a rebuild is built at a SIBLING path and renamed into place only once it
# has been read back and proven good. The live venv keeps serving until the
# rename, and a rebuild that fails leaves it untouched rather than gutted.
#
# RELOCATABLE IS NOT OPTIONAL HERE, and it is why this is not merely a `mv`. A
# venv's console scripts carry an ABSOLUTE shebang naming the venv they were
# installed into -- 101 of them in this one, `onex` among them -- so a sibling
# build renamed into place would yield a venv whose every entry point names a
# directory that no longer exists. That is worse than the in-place rebuild it
# replaces. `uv venv --relocatable` writes a `/bin/sh` wrapper that resolves the
# interpreter from the script's own location instead, and `uv sync` preserves
# that property for packages installed afterwards. Both halves were measured on
# a real uv before this was written rather than assumed, and the second one is
# read back off disk below because `uv sync` recreating the environment would
# silently drop it.
venv_is_relocatable() {
  local cfg="$1/pyvenv.cfg"
  [[ -f "$cfg" ]] || return 1
  grep -qE '^[[:space:]]*relocatable[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$cfg"
}

# Rename a proven staged venv over the live one: two renames on one filesystem,
# keeping the old generation until the new one is in place, so a failed second
# rename is rolled back rather than leaving the host with no venv at all.
swap_dispatch_venv() {
  local staging="$1" live="$2" previous="$2.previous"
  rm -rf "$previous"
  if [[ -e "$live" ]] && ! mv "$live" "$previous"; then
    fail "could not move the live dispatch venv aside; it is UNTOUCHED and" \
      "still serving, and the gate venv was NOT touched." \
      "The proven replacement is at:" \
      "  $staging"
  fi
  if ! mv "$staging" "$live"; then
    [[ -e "$previous" ]] && mv "$previous" "$live"
    fail "could not move the rebuilt dispatch venv into place; the previous" \
      "one has been restored and the gate venv was NOT touched." \
      "The replacement is at:" \
      "  $staging"
  fi
  rm -rf "$previous"
}

INSTALL_SCRIPT="${ONEX_RECONCILE_INSTALL_SCRIPT:-$INFRA_DIR/scripts/install-node-skill-package.sh}"

if [[ ! -d "$MARKET_CLONE/.git" ]]; then
  say "INDETERMINATE: no canonical omnimarket clone at $MARKET_CLONE"
  say "  The provider layer is reconciled against that clone's HEAD; without it"
  say "  there is no reference commit to reconcile to."
  exit "$EXIT_INDETERMINATE"
fi

# --------------------------------------------------------------------------- #
# Surface ownership, and resolving the one tool that does all the writing
# --------------------------------------------------------------------------- #
# OMN-17335. The first live `.201` run of this reconciler refused with
# "`uv` is not on PATH", and the ticket filed against it concluded uv was not
# installed on the host at all. That conclusion was wrong, and the probe that
# produced it is the same shape as the defect this whole epic exists to close:
#
#     $ command -v uv                      -> (nothing)      # non-interactive PATH
#     $ /home/<owner>/.local/bin/uv --version -> uv 0.11.5    # it is installed
#
# A non-interactive shell never sources the profile that puts `~/.local/bin` on
# PATH, and the cron.d unit sets an explicit minimal PATH, so PATH-only
# resolution CANNOT see a user-local install by construction. The absence of a
# name on PATH was read as the absence of the tool -- an empty result treated as
# evidence of absence. So resolution is now an ordered list of candidates, and a
# refusal names every one of them.
#
# The second defect only became visible while fixing the first. The cron unit
# runs as ROOT; the venv and every file in it are owned by the operator user.
# Putting uv on root's PATH would have "fixed" the refusal by having root write
# root-owned files into a user-owned venv -- after which the owner's own
# reconcile fails on permissions. That trades a loud, correct failure for a
# quiet, latent one, which is strictly worse than the bug. So every operation
# that WRITES runs as the owner of the surface, or does not run.
#
# Both rules are enforced mechanically by scripts/check_reconciler_privilege.py
# (pre-commit + CI), not by this comment.

CURRENT_USER="$(id -un)"
UV_BIN=""
UV_SEARCHED=()

# The ownership mechanics live in ONE place (OMN-17366). This script established
# the rule for the venv surface; `reconcile-host.sh` then needed the identical
# rule for the clone surface, and a second copy of a privilege guard is a copy
# that drifts -- with the drifting half being the one nobody is watching. So
# `rp_surface_owner`, `rp_user_home`, `rp_plan_privileges` and `as_owner` are
# sourced, and what stays here is POLICY: which surface, which message, which
# exit code.
_VENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PRIVILEGE_LIB="$_VENV_SCRIPT_DIR/reconcile_privilege_lib.sh"
if [[ ! -f "$_PRIVILEGE_LIB" ]]; then
  say "INDETERMINATE: privilege library missing at $_PRIVILEGE_LIB"
  say "  Without it there is no way to know who owns the venv, and writing as"
  say "  whoever this process happens to be is the defect it exists to prevent."
  exit "$EXIT_INDETERMINATE"
fi
# shellcheck source=./reconcile_privilege_lib.sh
source "$_PRIVILEGE_LIB"

# The clone set is read from the SAME manifest reconcile-host.sh and
# ensure_runner_clones.sh read (OMN-15137). A hand-maintained list here would be
# the fourth copy, and the copy nobody updates is the one that silently stops
# covering a repo.
_CLONE_MANIFEST_SH="$_VENV_SCRIPT_DIR/runtime_build/sibling_clone_manifest.sh"
if [[ ! -f "$_CLONE_MANIFEST_SH" ]]; then
  say "INDETERMINATE: clone manifest missing at $_CLONE_MANIFEST_SH"
  say "  Without it there is no way to know which clones this workspace is"
  say "  supposed to carry, and guessing the set would report a clean bill of"
  say "  health for whichever repo the guess left out."
  exit "$EXIT_INDETERMINATE"
fi
# shellcheck source=./runtime_build/sibling_clone_manifest.sh
source "$_CLONE_MANIFEST_SH"

# Aliases for the names the rest of this script and its tests already use. The
# library owns the mechanism; these two are this script's view of it.
SURFACE_OWNER=""
SURFACE_OWNER_HOME=""

# Decide, once, who the package operations must run as. Sets RUN_AS to the
# command prefix that gets there -- empty when this process already IS the
# owner, which is the case on every developer machine.
plan_privileges() {
  local surface="$INFRA_DIR" rc=0
  [[ -d "$INFRA_VENV" ]] && surface="$INFRA_VENV"

  rp_plan_privileges "$surface" || rc=$?
  SURFACE_OWNER="$RP_OWNER"
  SURFACE_OWNER_HOME="$RP_OWNER_HOME"

  case "$rc" in
    0)
      [[ ${#RUN_AS[@]} -eq 0 ]] || \
        say "writing as $SURFACE_OWNER (owner of $surface); this process is $CURRENT_USER"
      return 0
      ;;
    1)
      say "INDETERMINATE: cannot read the owner of $surface."
      say "  Every package operation below writes into that tree. Without knowing"
      say "  who owns it there is no way to write as the right user, and writing"
      say "  as the wrong one leaves a venv its owner can no longer reconcile."
      exit "$EXIT_INDETERMINATE"
      ;;
  esac

  # --check writes nothing, so an ownership mismatch is not an obstacle to it.
  # Refusing here would break the read-only SessionStart probe for any user who
  # can read the tree, and a read-only probe that refuses teaches people to stop
  # running it.
  #
  # This exemption is specific to THIS surface and must not be copied to
  # reconcile-host.sh, whose check mode fetches -- and a fetch writes. See the
  # note above `plan_clone_privileges` there.
  if [[ "$MODE" == "check" ]]; then
    RUN_AS=()
    return 0
  fi

  if [[ "$rc" -eq 3 ]]; then
    # HOME must be explicit: `runuser` without `-l` keeps root's HOME, so uv
    # would try to write its cache into /root/.cache as an unprivileged user and
    # fail on permissions -- a confusing failure two layers from its cause.
    say "INDETERMINATE: $surface is owned by $SURFACE_OWNER, whose home directory"
    say "  could not be resolved. Dropping privileges without a HOME points uv's"
    say "  cache at root's, which the dropped user cannot write."
    exit "$EXIT_INDETERMINATE"
  fi

  say "INDETERMINATE: $surface is owned by $SURFACE_OWNER, but this process runs"
  say "  as $CURRENT_USER and cannot become that user."
  say "  Writing anyway would put $CURRENT_USER-owned files inside a"
  say "  $SURFACE_OWNER-owned venv, after which $SURFACE_OWNER's own reconcile"
  say "  fails on permissions. A loud refusal now beats a silent breakage later."
  say "  Run this as $SURFACE_OWNER, or as root on a host that has runuser."
  exit "$EXIT_INDETERMINATE"
}

# Ordered candidates, every one of them named in the refusal. There is
# deliberately NO hardcoded system path here: /usr/local/bin and /opt/homebrew/bin
# are already on any PATH that includes them, so the PATH candidate covers them,
# and a hardcoded absolute path would also make these tests non-hermetic.
resolve_uv() {
  local candidate owner_home
  UV_SEARCHED=()

  if [[ -n "${ONEX_RECONCILE_UV_BIN:-}" ]]; then
    UV_SEARCHED+=("$ONEX_RECONCILE_UV_BIN  (ONEX_RECONCILE_UV_BIN)")
    if [[ -x "$ONEX_RECONCILE_UV_BIN" ]]; then
      UV_BIN="$ONEX_RECONCILE_UV_BIN"
      return 0
    fi
  fi

  candidate="$(command -v uv 2>/dev/null || true)"
  UV_SEARCHED+=("${candidate:-<not found>}  (PATH=${PATH})")
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    UV_BIN="$candidate"
    return 0
  fi

  # The case that actually broke `.201`: uv installed under the surface owner's
  # home, invisible to a cron PATH.
  owner_home="$SURFACE_OWNER_HOME"
  if [[ -n "$owner_home" ]]; then
    candidate="$owner_home/.local/bin/uv"
    UV_SEARCHED+=("$candidate  (home of $SURFACE_OWNER)")
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      return 0
    fi
  else
    UV_SEARCHED+=("<home of $SURFACE_OWNER could not be resolved>")
  fi

  return 1
}

plan_privileges

if ! resolve_uv; then
  say "INDETERMINATE: no usable \`uv\`; every sync below is a uv operation."
  say "  Searched, in order:"
  for _candidate in "${UV_SEARCHED[@]}"; do
    say "    - $_candidate"
  done
  say "  Install uv, or set ONEX_RECONCILE_UV_BIN to an absolute uv path."
  say "  Note that 'not on PATH' is NOT 'not installed': a cron unit's PATH"
  say "  cannot reach a user-local install (OMN-17335)."
  exit "$EXIT_INDETERMINATE"
fi
trace "uv resolved to $UV_BIN"

# --------------------------------------------------------------------------- #
# Pure observations (the future NodeCompute inputs)
# --------------------------------------------------------------------------- #
# Local `git rev-parse HEAD` only -- never `ls-remote`. Advancing the clone is
# somebody else's job; this script reconciles the venv to whatever is checked
# out, which is precisely what the drift guard compares against.
market_head() {
  git -C "$MARKET_CLONE" rev-parse HEAD 2>/dev/null || true
}

# The clone -> origin/<branch> leg (OMN-17295). Prints one row per present
# canonical clone and returns non-zero when at least one of them cannot be
# asserted current against the branch it tracks.
#
# Read-only in the strongest sense available: `rev-parse` and `rev-list`, never
# `fetch` and never `ls-remote`. See the header -- a fetch would void this
# script's OMN-17366 check-mode ownership exemption, and the network call would
# make every test here non-hermetic. The consequence is stated in the output
# rather than hidden: the target is whatever the last fetch left, so this leg
# can only ever prove staleness it can already see. reconcile-host.sh fetches
# first and is the surface that closes that gap.
clone_branch_report() {
  local repo clone head target behind stale=0 printed=0

  for repo in "${SIBLING_CLONE_MANIFEST[@]}"; do
    clone="$OMNI_HOME/$repo"
    # `-e`, not `-d`: a worktree's `.git` is a FILE. A `-d` test here would skip
    # every worktree silently, which is the shape of gap this leg exists to end.
    [[ -e "$clone/.git" ]] || continue

    head="$(git -C "$clone" rev-parse HEAD 2>/dev/null || true)"
    [[ -n "$head" ]] || continue

    # No remote at all: there is no branch this clone tracks, so there is
    # nothing for it to be stale against. Manufacturing a failure here would
    # fire on every workspace where the question is not even askable.
    #
    # `config --get`, not `remote get-url`: the two answer the same question,
    # but `git remote` is a WRITE verb to check_reconciler_privilege.py's
    # matcher (`git remote add`/`set-url` mutate config) and it matches on the
    # verb, not the subcommand. Narrowing that matcher to let `get-url` through
    # would weaken a gate to admit a read -- and `config --get` is already named
    # in its comments as read-only plumbing it deliberately does not guard. Use
    # the primitive the gate blesses rather than teaching the gate an exception.
    git -C "$clone" config --get remote.origin.url >/dev/null 2>&1 || continue

    if [[ "$printed" -eq 0 ]]; then
      say "clone surface vs origin/$BRANCH (as last fetched; this script does not fetch):"
      printed=1
    fi

    target="$(git -C "$clone" rev-parse --verify --quiet "refs/remotes/origin/$BRANCH" 2>/dev/null || true)"
    if [[ -z "$target" ]]; then
      # An unknown target renders as unproven, never as fresh -- the same rule
      # the SessionStart hook applies to a verdict whose age it cannot read.
      say "  $repo: HEAD ${head:0:12}  origin/$BRANCH UNFETCHED (no tracking ref; never fetched here)"
      stale=1
      continue
    fi

    behind="$(git -C "$clone" rev-list --count "HEAD..$target" 2>/dev/null || true)"
    [[ "$behind" =~ ^[0-9]+$ ]] || behind=0
    say "  $repo: HEAD ${head:0:12}  origin/$BRANCH ${target:0:12}  behind by $behind commit(s)"
    [[ "$behind" -eq 0 ]] || stale=1
  done

  [[ "$stale" -eq 0 ]]
}

# The one message both modes share, so the two can never describe the same
# state differently.
say_clone_stale_remedy() {
  say "  The venv layers are reconciled to the CLONE, so a clone behind"
  say "  origin/$BRANCH means the installed code is behind what is merged --"
  say "  even when every venv layer matches its target exactly."
  say "  This script deliberately never advances a clone: there is exactly one"
  say "  clone reconciler (scripts/runtime_build/reconcile_deploy_clones.sh)."
  say "  Fetch, fast-forward and prove all three surfaces with:"
  say "    bash $INFRA_DIR/scripts/reconcile-host.sh --omni-home $OMNI_HOME --branch $BRANCH"
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
  lock_layer_ok_in "$project" "" "$@"
}

# The same question, asked of a venv that is NOT the project's default one.
# `UV_PROJECT_ENVIRONMENT` is how uv is told which environment a project sync
# targets; it is the same primitive `.github/workflows/evidence-autoclose-
# sweep.yml` uses to compose CI's dispatch venv, so the local split and the CI
# split are the same mechanism rather than two lookalikes (OMN-16846/OMN-17819).
lock_layer_ok_in() {
  local project="$1" venv="$2"
  shift 2
  if [[ -n "$venv" ]]; then
    (cd "$project" && as_owner env -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$venv" \
      "$UV_BIN" sync --frozen --check --project "$project" "$@" >/dev/null 2>&1)
  else
    # `-u UV_PROJECT_ENVIRONMENT` is not decoration. uv reads that variable from
    # the ambient environment, so a caller that has one set -- a CI job, a shell
    # inside an activated venv, a `uv run` parent -- silently redirects a sync
    # that means "the project's own venv" to somewhere else entirely. Measured
    # on this repo's CI, where the runner exports it: the gate pass targeted
    # /home/runner/work/omnibase_infra/omnibase_infra/.venv and the canonical
    # clone's venv was never purified at all, while the run still exited 0.
    (cd "$project" && as_owner env -u PYTHONPATH -u UV_PROJECT_ENVIRONMENT "$UV_BIN" sync --frozen --check --project "$project" "$@" >/dev/null 2>&1)
  fi
}

# Read from the DISPATCH venv (OMN-17819): that is where the provider layer
# lives and where `scripts/onex` runs from, so it is the only interpreter whose
# omnimarket commit answers "which build would a dispatch actually use".
# Reading the gate venv here would report the commit of a package that is
# supposed to be absent from it.
installed_market_commit() {
  [[ -x "$DISPATCH_PYTHON" ]] || return 0
  env -u PYTHONPATH "$DISPATCH_PYTHON" - <<'PYEOF' 2>/dev/null || true
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
# The plugin CLI venv: a surface this script does NOT own (OMN-18663)
# --------------------------------------------------------------------------- #
# `hook_venv_projects` above skips a candidate that has no `uv.lock`, and it
# does so SILENTLY. On this Mac exactly one directory is in that state, and it
# is the one that matters most: the plugin CLI venv from CLAUDE.md rule 11's
# table, which serves `onex` for every lane reaching the CLI through the plugin
# rather than through `omnibase_infra/scripts/onex`. It is built from a
# requirements.txt by `repair-plugin-venv.sh`, so it has no lock for this
# script to apply, and the header above records the deliberate decision NOT to
# take it over -- one directory, one owner.
#
# Not owning it is a defensible boundary. Not MENTIONING it is not. Measured
# 2026-09-18T15:45Z: the reconciler reported zero failures while that venv sat
# one omnimarket bump behind the canonical clone and the drift guard inside it
# refused every `onex delegate` through the plugin path. A verdict that is
# silent about a surface reads, to the operator and to the SessionStart line, as
# a verdict that covered it.
#
# So the census below names it either way. Its LOCK layer is reported as
# unowned, naming the script that does own it. Its PROVIDER layer is READ BACK
# against the canonical clone -- the same comparison the in-process guard makes
# -- and that half is verdict-bearing in `--check`, because a drifted provider
# layer there blocks dispatch just as surely as one in a venv this script writes.
#
# Resolved WITHOUT depending on $CLAUDE_PLUGIN_DATA being exported. A cron or
# SessionStart tick does not carry it, which is precisely how this surface
# stayed invisible; rule 11 documents the default path, so the default is used
# when the variable is absent.
DEFAULT_CLI_VENV_DIR="$HOME/.claude/plugins/data/onex-omninode-tools"

cli_venv_candidates() {
  local seen=""
  local candidate
  for candidate in "${CLAUDE_PLUGIN_DATA:-}" "$DEFAULT_CLI_VENV_DIR"; do
    [[ -n "$candidate" ]] || continue
    [[ -x "$candidate/.venv/bin/python" ]] || continue
    # A candidate WITH a lock is reconciled by hook_venv_projects; this census
    # is for the ones that fall through it.
    [[ -f "$candidate/uv.lock" ]] && continue
    case " $seen " in *" $candidate "*) continue ;; esac
    seen="$seen $candidate"
    printf '%s\n' "$candidate"
  done
}

# Report every unowned CLI venv. Returns 1 when any of them is drifted against
# the canonical clone, so the caller can decide what that means for its verdict.
report_unowned_cli_venvs() {
  local head="$1"
  local candidate python_bin drifted=0
  local readback="$_VENV_SCRIPT_DIR/venv_readback.py"
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    python_bin="$candidate/.venv/bin/python"
    say "CLI venv (not owned here): $candidate/.venv"
    say "  lock layer : NOT RECONCILED -- no uv.lock; owned by scripts/repair-plugin-venv.sh"
    if [[ ! -f "$readback" ]]; then
      say "  provider   : UNREADABLE -- no readback at $readback"
      drifted=1
      continue
    fi
    # RUN the readback on the dispatch interpreter, not on the venv being read.
    # The venv under inspection is the one suspected of being broken, and a
    # broken interpreter that exits 0 without running the program would report
    # "in sync" about itself. venv_readback.py probes its --python target in a
    # subprocess, so the reader and the read need not be the same interpreter.
    if [[ ! -x "$DISPATCH_PYTHON" ]]; then
      say "  provider   : UNREADABLE -- no dispatch interpreter to run the readback"
      drifted=1
      continue
    fi
    if env -u PYTHONPATH "$DISPATCH_PYTHON" "$readback" \
        --python "$python_bin" --clone "$MARKET_CLONE" --ref "$head" \
        --label "unowned CLI venv census (OMN-18663)" >/dev/null 2>&1; then
      say "  provider   : in sync (omnimarket ${head:0:12})"
    else
      say "  provider   : DRIFT -- does not carry omnimarket ${head:0:12}"
      say "  Every \`onex delegate\` through the plugin path refuses until this is"
      say "  converged. This script does not write that venv; the sanctioned"
      say "  remedy is:"
      say "    OMNI_HOME=$OMNI_HOME bash $_VENV_SCRIPT_DIR/check-omnimarket-venv-drift.sh \\"
      say "      --repair $python_bin"
      drifted=1
    fi
  done < <(cli_venv_candidates)
  return "$drifted"
}

# --------------------------------------------------------------------------- #
# The lane-identity hook surface (OMN-18260)
# --------------------------------------------------------------------------- #
# WHY A VENV RECONCILER ARMS A GIT HOOK. It is the same surface, asked about a
# different file. This script already exists because a host's convergence must
# not depend on somebody remembering to run a command, and the lane-identity
# stamping hook was in exactly that state: armed on this Mac because a lane ran
# `lane_identity reconcile --execute` by hand, unarmed and unmentioned anywhere
# else. Measured 2026-09-18 over 2026-09-11..09-18, 39 of 4,105 commits across
# six repositories carried the `Onex-Lane` trailer -- 0.95% -- which is the
# ceiling on the OMN-18262 pre-push refusal and the OMN-18263 check, both of
# which read the lane out of the trailer and can never see an unstamped commit.
#
# WHAT IS WRITTEN, AND WHY IT IS NOT THE CLONE MUTATION THIS SCRIPT FORBIDS.
# The arming verb writes `<git-common-dir>/hooks/prepare-commit-msg` in each
# clone and a dispatch entry in the shared `core.hooksPath` guard directory. It
# never touches a worktree file, a ref, or the index, so the header's rule --
# this script never advances a clone, there is exactly one clone reconciler --
# is untouched. What it converges is HOST state that lives inside the clone's
# git directory, which is the same class of thing as a venv.
#
# ONE INSTALL PER CLONE COVERS EVERY WORKTREE. Git worktrees share the main
# clone's hooks directory, so the 400-odd worktrees under `omni_worktrees/`
# need no install of their own. Stating it here because the obvious reading of
# "arm every clone" is that the worktrees are a second, uncovered surface, and
# they are not.
#
# THE RESOLUTION IS NOT REIMPLEMENTED HERE. Which clones exist, whether a hook
# is installed, and whether git will actually dispatch it are all answered by
# `lane_identity.py status`, the same module the arming verb and the hook
# itself use. A second reader of that question in bash would be a second place
# for it to drift, with the drifting half being the one nobody watches.
LANE_IDENTITY_SCRIPT="${ONEX_LANE_IDENTITY_SCRIPT:-$OMNI_HOME/omniclaude/scripts/lane_identity.py}"

# Run the module on the DISPATCH interpreter, for the reason
# `report_unowned_cli_venvs` runs the readback there: it is the interpreter this
# workspace already proves, and it is not the thing under inspection.
lane_identity_runnable() {
  [[ -f "$LANE_IDENTITY_SCRIPT" ]] && [[ -x "$DISPATCH_PYTHON" ]]
}

# NOT ASKABLE is not the same as ARMED, and neither is silence. A host whose
# clone set does not include omniclaude -- a deploy runner, whose set is
# `sibling_clone_manifest.sh` and does not name it -- cannot be asked this
# question at all, so manufacturing a failure there would fire on every host
# where the question does not arise. It still prints a line naming the path it
# looked for, because the failure this whole leg exists to end is a surface
# nobody mentions.
lane_hook_not_askable() {
  if [[ ! -f "$LANE_IDENTITY_SCRIPT" ]]; then
    say "lane-identity hooks: NOT ASKABLE -- no $LANE_IDENTITY_SCRIPT in this workspace"
    say "  The stamping hook ships in omniclaude, which is not in this host's"
    say "  clone manifest. Nothing is armed here and nothing is claimed to be."
    return 0
  fi
  return 1
}

# Print the per-clone hook state and return non-zero when any clone is unarmed.
#
# The exit status is the module's own: 1 when at least one clone is not armed,
# 2 when it checked NO clone at all. The second matters as much as the first --
# a sweep that inspected nothing reads exactly like a clean bill of health, and
# `lane_identity status` refuses to report a pass on an empty sweep for that
# reason (CLAUDE.md rule 16). Both are findings here.
#
# stderr is merged into the output rather than discarded: the module prints the
# unarmed count and the remedy there, and a verification sweep that throws away
# its own stderr is the shape of the false zero rule 16 is about.
lane_hook_report() {
  local out rc=0 line
  lane_hook_not_askable && return 0
  if [[ ! -x "$DISPATCH_PYTHON" ]]; then
    say "lane-identity hooks: UNREADABLE -- no dispatch interpreter at $DISPATCH_PYTHON"
    say "  The hook state cannot be read without one, and an unreadable surface"
    say "  is reported as unproven rather than as armed."
    return 1
  fi
  out="$(env -u PYTHONPATH "$DISPATCH_PYTHON" "$LANE_IDENTITY_SCRIPT" status 2>&1)" || rc=$?
  say "lane-identity hooks (prepare-commit-msg, one install per clone):"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    say "  $line"
  done <<<"$out"
  return "$rc"
}

say_lane_hook_remedy() {
  say "  A clone with no stamping hook produces commits carrying no lane, and a"
  say "  commit carrying no lane is one the branch-claim resolution can never"
  say "  compare against a claim holder. Arm every clone with:"
  say "    OMNI_HOME=$OMNI_HOME $DISPATCH_PYTHON $LANE_IDENTITY_SCRIPT reconcile --execute"
}

# The repair half: arm, then READ BACK. The arming verb exiting 0 is not
# evidence that the surface is armed (OMN-17307, a repair reporting its own exit
# status as proof), so the verdict below comes from a fresh status read and
# never from the install's return code.
arm_lane_hooks() {
  lane_hook_not_askable && return 0
  if [[ ! -x "$DISPATCH_PYTHON" ]]; then
    say "lane-identity hooks: UNREADABLE -- no dispatch interpreter at $DISPATCH_PYTHON"
    return 1
  fi
  say "lane-identity hooks: arming every canonical clone"
  # `as_owner`, and `cd` into a directory the dropped user can read: the hook
  # files land inside clones owned by the surface owner, and a root-owned hook
  # in a user-owned clone is the OMN-17335 hazard in a different directory.
  if ! (cd "$OMNI_HOME" && as_owner env -u PYTHONPATH "$DISPATCH_PYTHON" \
      "$LANE_IDENTITY_SCRIPT" reconcile --execute); then
    # Reported, never fatal on its own: the readback below is what answers, and
    # a non-zero here can be the ledger-backfill half failing while every hook
    # installed correctly.
    say "lane-identity hooks: the arming pass reported a failure; the readback decides"
  fi
  lane_hook_report
}

# --------------------------------------------------------------------------- #
# --check : verdict only, zero mutation
# --------------------------------------------------------------------------- #
run_check() {
  local drift=0
  local head installed

  # Reported FIRST, because it is the leg that decides what the other two mean:
  # a venv proven equal to a clone that is itself behind dev is proven equal to
  # the wrong thing (OMN-17295).
  if ! clone_branch_report; then
    say "DRIFT: a canonical clone is not at origin/$BRANCH"
    say_clone_stale_remedy
    drift=1
  fi

  head="$(market_head)"
  if [[ -z "$head" ]]; then
    say "INDETERMINATE: could not read HEAD of $MARKET_CLONE"
    exit "$EXIT_INDETERMINATE"
  fi

  # ---- the DISPATCH venv: both layers (OMN-17819) ------------------------- #
  if [[ ! -x "$DISPATCH_PYTHON" ]]; then
    say "DRIFT: dispatch venv absent ($DISPATCH_VENV)"
    drift=1
  else
    # Reported in check mode as well as repaired, because `--check` is what the
    # SessionStart line and every read-only probe run: a verdict that stayed
    # silent about the interpreter would print "in sync" over a CLI whose LAN
    # calls to the lab host fail silently (CLAUDE.md rule 11).
    local base_python
    base_python="$(dispatch_base_python)"
    # Built exactly as the repair path builds it, and used for the lock check
    # below so the two cannot ask uv different questions (OMN-18663).
    local -a dispatch_python_arg=()
    [[ -n "$base_python" ]] && dispatch_python_arg=(--python "$base_python")
    if dispatch_requires_base_python && [[ -z "$base_python" ]]; then
      say "DRIFT: no brew Python to build the dispatch venv on; looked at:"
      say "  $(dispatch_base_python_candidates)"
      drift=1
    elif ! dispatch_interpreter_ok "$base_python"; then
      say "DRIFT: dispatch venv is on the wrong interpreter -- built on"
      say "  $(venv_base_home "$DISPATCH_VENV"), required ${base_python%/*}"
      say "  (CLAUDE.md rule 11: the macOS Local Network grant is per binary path,"
      say "  so a uv-managed interpreter's LAN calls fail silently)"
      drift=1
    fi

    # THE SAME QUESTION THE REPAIR ANSWERS, flag for flag (OMN-18663).
    #
    # `--inexact` was already here and is right: the composed provider layer is
    # deliberately absent from the lock. The INTERPRETER was not, and that is a
    # difference uv acts on. The repair syncs this venv with
    # `--python "$base_python"` (the brew interpreter rule 11 requires for the
    # macOS LAN grant); a check that omits it lets uv resolve the interpreter
    # its own way -- from `.python-version`, which pins 3.12 in this project
    # while the venv is correctly built on 3.13. uv then answers a question
    # nobody asked, "would you replace this environment", and says yes.
    #
    # Measured on this host, single variable, everything else identical:
    #
    #   uv sync --frozen --check --inexact                       -> exit 1
    #   uv sync --frozen --check --inexact --python <brew 3.13>   -> exit 0
    #                                                               "Would make
    #                                                                no changes"
    #
    # So the pre-OMN-18663 check reported DRIFT for a dispatch venv that was
    # exactly what the repair had just built, permanently: no repair could ever
    # clear it, and two consecutive repair ticks reported success into it
    # (2026-09-18T15:45Z). A check that a correct repair cannot satisfy is not a
    # strict check, it is a broken one -- it trains the reader to disbelieve the
    # verdict, which is worse than not having it.
    if ! lock_layer_ok_in "$INFRA_DIR" "$DISPATCH_VENV" --inexact \
        "${dispatch_python_arg[@]}"; then
      say "DRIFT: dispatch venv does not satisfy $INFRA_DIR/uv.lock"
      drift=1
    fi
    installed="$(installed_market_commit)"
    if [[ "$installed" != "$head" ]]; then
      say "DRIFT: dispatch venv omnimarket ${installed:0:12} != clone HEAD ${head:0:12}"
      drift=1
    fi
  fi

  # ---- the GATE venv: lock-governed ONLY ---------------------------------- #
  # EXACT, deliberately: no `--inexact` here. `--inexact` is what lets a
  # composed layer coexist with a lock, and the whole point of the OMN-17819
  # split is that this venv has no composed layer. An undeclared `onex.nodes`
  # provider in here is what the OMN-15620 gate refuses on, so a check that
  # tolerated it would report IN_SYNC for the exact state that blocks every
  # `uv run pytest` in the canonical clone.
  if [[ ! -x "$INFRA_PYTHON" ]]; then
    say "DRIFT: gate venv absent ($INFRA_VENV)"
    drift=1
  elif ! lock_layer_ok "$INFRA_DIR"; then
    say "DRIFT: gate venv does not match $INFRA_DIR/uv.lock exactly"
    say "  (either a locked pin is missing, or an undeclared distribution is"
    say "  installed -- the second is what the OMN-15620 purity gate refuses on)"
    drift=1
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

  # ---- CLI venvs this script does not own (OMN-18663) --------------------- #
  # Verdict-bearing on the provider layer, for the reason the clone leg above
  # is verdict-bearing: the reader is asking about the workspace, not about the
  # subset of it this script happens to write.
  if ! report_unowned_cli_venvs "$head"; then
    drift=1
  fi

  # ---- the lane-identity hook surface (OMN-18260) ------------------------- #
  # Verdict-bearing, because after this change the repair path WRITES it. An
  # unarmed clone is named by the report above; this is the line that stops the
  # verdict reading IN_SYNC over it.
  if ! lane_hook_report; then
    say "DRIFT: a canonical clone carries no lane-identity stamping hook"
    say_lane_hook_remedy
    drift=1
  fi

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

  # Report-only on this path. The exit code below answers for the surfaces this
  # script WRITES, and a clone it is forbidden to touch is not one of them --
  # so a stale clone must not turn a successful venv reconcile into a failure
  # for pull-all.sh or the CLI drift guard, both of which read that status.
  # Saying nothing, though, is how the venv gets pinned to a stale commit with
  # a clean "reconciled" line over the top of it.
  if ! clone_branch_report; then
    say "NOTE: a canonical clone is not at origin/$BRANCH, so the pin below"
    say "  tracks the clone, not what is merged."
    say_clone_stale_remedy
  fi

  head="$(market_head)"
  if [[ -z "$head" ]]; then
    say "INDETERMINATE: could not read HEAD of $MARKET_CLONE"
    exit "$EXIT_INDETERMINATE"
  fi

  # RUN_AS was planned from the gate venv's owner. A dispatch venv owned by
  # somebody else would be written as the wrong user by that same prefix --
  # the hazard plan_privileges exists to prevent -- so it is refused rather
  # than written, exactly as a foreign-owned hook venv is below. A dispatch
  # venv that does not exist yet has no owner to disagree with and is created
  # by the surface owner like everything else here.
  local dispatch_owner
  if [[ -d "$DISPATCH_VENV" ]]; then
    dispatch_owner="$(rp_surface_owner "$DISPATCH_VENV" || true)"
    if [[ -n "$dispatch_owner" && "$dispatch_owner" != "$SURFACE_OWNER" ]]; then
      say "INDETERMINATE: dispatch venv $DISPATCH_VENV is owned by $dispatch_owner,"
      say "  but the package operations are running as $SURFACE_OWNER (owner of"
      say "  $INFRA_DIR). Reconcile it as $dispatch_owner, or make the two"
      say "  surfaces share an owner."
      exit "$EXIT_INDETERMINATE"
    fi
  fi

  # ---- surface 1: the DISPATCH venv (two layers) -------------------------- #
  #
  # FIRST, before the gate venv, and the ordering is load-bearing: the gate pass
  # below is what REMOVES omnimarket from the clone's `.venv`, and doing that
  # before a working dispatch venv exists would leave this host with no
  # interpreter that can run `onex` at all. Every failure path here exits
  # without reaching the gate venv (OMN-17819).
  #
  # Each layer is decided independently, so a tick that runs every 10 minutes
  # does the least work that closes the actual gap. The common case by far --
  # the clone advanced, the lock did not -- is additive `--no-deps` provider
  # work plus a lock pass that finds nothing to do.
  if [[ ! -x "$DISPATCH_PYTHON" ]]; then
    need_lock=1
    need_provider=1
  else
    lock_layer_ok_in "$INFRA_DIR" "$DISPATCH_VENV" --inexact || need_lock=1
    installed="$(installed_market_commit)"
    [[ "$installed" == "$head" ]] || need_provider=1
  fi

  # The base interpreter, resolved once (CLAUDE.md rule 11). Empty off macOS,
  # where the LAN-grant constraint does not exist.
  local base_python
  base_python="$(dispatch_base_python)"
  if dispatch_requires_base_python && [[ -z "$base_python" ]]; then
    fail "no brew Python found to build the dispatch venv on; the gate venv was NOT touched." \
      "The dispatch venv is what \`scripts/onex\` execs, and on macOS the Local" \
      "Network grant is per binary path -- a uv-managed interpreter's LAN" \
      "connections to the lab host fail silently with EHOSTUNREACH." \
      "Looked at, in order:" \
      "  $(dispatch_base_python_candidates)" \
      "Install it (brew install python@3.13), or name one explicitly with" \
      "ONEX_DISPATCH_BASE_PYTHON=<absolute path>."
  fi

  local -a dispatch_python_arg=()
  [[ -n "$base_python" ]] && dispatch_python_arg=(--python "$base_python")

  # An existing dispatch venv on the WRONG interpreter is drift, not a state to
  # leave alone. Without this the rule-11 requirement would hold only for hosts
  # that happened to build the venv after it landed, and every host that already
  # had one would keep a silently LAN-blind CLI forever. `uv sync --python`
  # recreates the environment when the interpreter differs, so the rebuild is
  # uv's single code path rather than an `rm -rf` in a reconciler.
  # A BOOLEAN, separate from the message. An earlier draft used the recorded
  # `home` string as the flag, which is empty in exactly the drift case that
  # matters most -- a venv with no readable pyvenv.cfg -- so it printed
  # "Rebuilding." and then skipped both the rebuild and its readback. A hand-run
  # reproduction caught that; no test would have, because every fixture wrote a
  # pyvenv.cfg.
  local rebuild=0 rebuild_home=""
  if [[ -x "$DISPATCH_PYTHON" ]] && ! dispatch_interpreter_ok "$base_python"; then
    rebuild=1
    rebuild_home="$(venv_base_home "$DISPATCH_VENV")"
    say "dispatch venv: interpreter drift -- built on ${rebuild_home:-<unreadable pyvenv.cfg>},"
    say "  required ${base_python%/*} (CLAUDE.md rule 11). Rebuilding."
    need_lock=1
    need_provider=1
  fi

  # WHERE THIS PASS WRITES. A rebuild is staged at a sibling and renamed in at
  # the end; everything else writes straight to the live venv, because an
  # ADDITIVE provider or lock pass never leaves it unusable and staging one
  # would buy nothing. A venv that does not exist YET is also built in place --
  # there is no live environment to protect, and no lane can be using it.
  local target_venv="$DISPATCH_VENV" target_python="$DISPATCH_PYTHON" staging=""
  if [[ "$rebuild" -eq 1 ]]; then
    staging="$DISPATCH_VENV.rebuilding"
    target_venv="$staging"
    target_python="$staging/bin/python"
    # A leftover from an earlier failed rebuild is scrap, not state.
    rm -rf "$staging"
    say "dispatch venv: staging the rebuild at $staging"
    # Created HERE rather than left to the `uv sync` below, because only
    # `uv venv` takes --relocatable and `uv sync` has no equivalent -- measured:
    # UV_VENV_RELOCATABLE is not honoured by `uv sync`, which writes absolute
    # shebangs. Without this step the rename would break every console script.
    trace "uv venv --relocatable ${dispatch_python_arg[*]} $staging"
    if ! (cd "$INFRA_DIR" && as_owner env -u PYTHONPATH \
        "$UV_BIN" venv --relocatable "${dispatch_python_arg[@]}" "$staging"); then
      fail "could not stage the dispatch venv rebuild; the live venv is" \
        "UNTOUCHED and still serving, and the gate venv was NOT touched." \
        "Run by hand and read the error:" \
        "  cd $INFRA_DIR && env -u PYTHONPATH \\" \
        "    uv venv --relocatable ${dispatch_python_arg[*]} $staging"
    fi
  fi

  # What a refusal from here on can HONESTLY claim survived. A staged rebuild
  # writes nowhere near the live venv, so every failure below leaves it serving
  # -- and saying only "the gate venv was NOT touched" would leave the reader
  # believing the CLI is gone and reaching for a repair that is not needed.
  # A first-ever build has no live venv to make that claim about.
  local intact_note="the gate venv was NOT touched."
  if [[ -n "$staging" ]]; then
    intact_note="the live venv is UNTOUCHED and still serving, and the gate venv was NOT touched."
  fi

  # A dispatch venv that does not exist yet has no provider layer to install
  # into. Build the lock layer first in that one case, so `uv` creates the
  # environment and the co-install has an interpreter to target. This is the
  # ONLY situation in which the lock pass precedes the provider pass; the
  # OMN-16262 ordering (provider first, lock second) still holds afterwards,
  # because the co-install below forces a second lock pass.
  if [[ ! -x "$DISPATCH_PYTHON" || "$rebuild" -eq 1 ]]; then
    say "dispatch venv: creating $target_venv from $INFRA_DIR/uv.lock${base_python:+ on $base_python}"
    trace "UV_PROJECT_ENVIRONMENT=$target_venv uv sync --frozen --inexact --project $INFRA_DIR ${dispatch_python_arg[*]}"
    if ! (cd "$INFRA_DIR" && as_owner env -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$target_venv" \
        "$UV_BIN" sync --frozen --inexact --project "$INFRA_DIR" "${dispatch_python_arg[@]}"); then
      fail "dispatch venv could not be created; $intact_note" \
        "Without it there is no interpreter for \`onex\` to exec, so this" \
        "refuses rather than purifying the gate venv and leaving the host with" \
        "no CLI at all. Run by hand and read the error:" \
        "  cd $INFRA_DIR && env -u PYTHONPATH UV_PROJECT_ENVIRONMENT=$target_venv \\" \
        "    uv sync --frozen --inexact ${dispatch_python_arg[*]}"
    fi

    # Prove the interpreter MOVED, by reading the venv back. `uv sync --python`
    # exiting 0 is not evidence that it rebuilt on the interpreter asked for --
    # that is the OMN-17307 defect class, a repair reporting its own exit status
    # as proof. A silent failure here would leave a LAN-blind CLI reading as
    # reconciled.
    if ! dispatch_interpreter_ok "$base_python" "$target_venv"; then
      fail "dispatch venv is still not on the required interpreter; $intact_note" \
        "  required : ${base_python%/*}" \
        "  recorded : $(venv_base_home "$target_venv")" \
        "Remove $target_venv and re-run, or name the interpreter explicitly" \
        "with ONEX_DISPATCH_BASE_PYTHON=<absolute path>."
    fi
  fi

  if [[ "$need_lock" -eq 0 && "$need_provider" -eq 0 ]]; then
    say "dispatch venv: already in sync (omnimarket ${head:0:12})"
  else
    # PROVIDER FIRST. The co-install can move lock-governed pins (OMN-16262:
    # its hardcoded COMPAT_PIN downgrades omnibase-compat 0.5.6 -> 0.5.5 and
    # breaks the `occ` CLI extension badly enough that `onex` will not start),
    # so the lock pass has to come after it to undo that.
    if [[ "$need_provider" -eq 1 ]]; then
      say "dispatch venv: reconciling provider layer to omnimarket ${head:0:12}"
      if [[ ! -x "$INSTALL_SCRIPT" ]]; then
        fail "provider co-install script is missing or not executable." \
          "Expected at: $INSTALL_SCRIPT"
      fi
      # OMNIMARKET_REF is set explicitly so the install script's own ls-remote
      # default (OMN-16366 reversed drift) can never apply here.
      trace "OMNIMARKET_REF=$head $INSTALL_SCRIPT --execute $target_python"
      # PATH carries the resolved uv down to the child (OMN-17383). The
      # co-install calls bare `uv`, and it inherits the cron PATH -- which is
      # exactly the PATH that cannot reach a user-local install, so on `.201`
      # this step died with "uv: command not found" even after resolve_uv() had
      # already located the binary in the parent. The interpreter is PASSED
      # DOWN rather than re-resolved: two independent resolutions can disagree,
      # and then the version the parent proved usable is not the one that runs.
      #
      # `cd "$OMNI_HOME"` for the same reason, one variable over (OMN-17800).
      # `runuser` changes UID, GID and HOME; it does NOT change the inherited
      # working directory. Under the `.201` root cron that directory is /root,
      # mode 0700 -- so the child became the operator while still standing in
      # root's home, and uv, which discovers configuration by walking UP from
      # the working directory, died on:
      #
      #   error: failed to open file `/root/uv.toml`: Permission denied (os error 13)
      #
      # every hour for 67 consecutive ticks. The co-install is the FIRST step of
      # the venv repair and forces the lock pass after it, so one unreadable
      # directory took out all three venv surfaces at once.
      #
      # $OMNI_HOME rather than $INFRA_DIR, unlike the sync calls below: this is a
      # `uv pip install`, and from inside the project uv would newly discover
      # omnibase_infra/pyproject.toml's `[tool.uv] override-dependencies` -- the
      # layer beneath, which this install exists NOT to re-resolve (hence its own
      # --no-deps). The workspace root carries no uv configuration on any host,
      # so this changes where the child stands without changing what it installs.
      if ! (cd "$OMNI_HOME" && as_owner env OMNIMARKET_REF="$head" OMNI_HOME="$OMNI_HOME" \
          PATH="$(dirname "$UV_BIN"):$PATH" \
          bash "$INSTALL_SCRIPT" --execute "$target_python"); then
        fail "provider co-install did not complete; omnimarket is not installed." \
          "Every \`onex skill\`/\`onex node\`/\`onex delegate\` dispatch will refuse" \
          "until this succeeds. $intact_note Run by hand and" \
          "read the error:" \
          "  OMNIMARKET_REF=$head OMNI_HOME=$OMNI_HOME \\" \
          "    bash $INSTALL_SCRIPT --execute $target_python" \
          "  (this is scripts/install-node-skill-package.sh)"
      fi
      # The co-install just ran, so the lock pass is mandatory regardless of
      # what the pre-check said -- that is the whole OMN-16262 repair.
      need_lock=1
    fi

    if [[ "$need_lock" -eq 1 ]]; then
      say "dispatch venv: applying $INFRA_DIR/uv.lock"
      # --frozen: apply the lock, never re-resolve it -- a re-resolution here
      #   would silently move the very pins the lock exists to hold.
      # --inexact: do not remove the composed provider layer, which the lock
      #   correctly does not mention and must not be asked to. This flag belongs
      #   to THIS venv only; the gate venv below is synced exact (OMN-17819).
      trace "UV_PROJECT_ENVIRONMENT=$target_venv uv sync --frozen --inexact --project $INFRA_DIR ${dispatch_python_arg[*]}"
      if ! (cd "$INFRA_DIR" && as_owner env -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$target_venv" \
          "$UV_BIN" sync --frozen --inexact --project "$INFRA_DIR" "${dispatch_python_arg[@]}"); then
        fail "dispatch venv lock sync did not complete; $intact_note" \
          "Run by hand and read the error:" \
          "  cd $INFRA_DIR && env -u PYTHONPATH UV_PROJECT_ENVIRONMENT=$target_venv \\" \
          "    uv sync --frozen --inexact"
      fi
    fi
    say "dispatch venv: reconciled"
  fi

  # PROVE THE STAGED VENV BEFORE IT BECOMES THE LIVE ONE. `uv` exiting 0 is not
  # evidence (OMN-17307, a repair reporting its own exit status as proof), and
  # the interpreter readback above ran BEFORE the provider and lock passes --
  # either of which can recreate the environment and drop what makes the rename
  # safe. Both properties are therefore read back off disk here, at the last
  # moment before the swap. Every refusal on this path leaves the live venv
  # serving, which is the whole point of staging.
  if [[ -n "$staging" ]]; then
    if ! dispatch_interpreter_ok "$base_python" "$staging"; then
      fail "the staged dispatch venv is not on the required interpreter; the" \
        "live venv is UNTOUCHED and still serving, and the gate venv was NOT" \
        "touched." \
        "  required : ${base_python%/*}" \
        "  recorded : $(venv_base_home "$staging")" \
        "The staged build is at $staging; remove it and re-run."
    fi
    if ! venv_is_relocatable "$staging"; then
      fail "the staged dispatch venv is not relocatable, so renaming it into" \
        "place would leave every console script -- \`onex\` among them --" \
        "naming $staging, which is about to stop existing. The live venv is" \
        "UNTOUCHED and still serving, and the gate venv was NOT touched." \
        "Expected 'relocatable = true' in $staging/pyvenv.cfg."
    fi
    say "dispatch venv: swapping the proven rebuild into $DISPATCH_VENV"
    swap_dispatch_venv "$staging" "$DISPATCH_VENV"
  fi

  # ---- surface 2: the GATE venv, lock-governed ONLY (OMN-17819) ------------ #
  # EXACT. This is the step that removes an undeclared `onex.nodes` provider
  # from the canonical clone's own environment, which is what unblocks every
  # `uv run pytest` there. It runs only after the dispatch venv above is proven
  # good, because it is also the step that takes omnimarket away from whatever
  # used to be running out of this directory.
  #
  # The previous revision of this script synced this venv `--inexact` and
  # composed the provider layer INTO it. That is the OMN-17819 defect: the same
  # directory cannot satisfy OMN-14060 (omnimarket must be present) and
  # OMN-15620 (it must be absent) at once. Do not restore `--inexact` here --
  # it would silently re-admit exactly the pollution this pass exists to remove,
  # and the refusal it causes appears at `pytest_configure`, far from here.
  local gate_needs_sync=0
  if [[ ! -x "$INFRA_PYTHON" ]]; then
    gate_needs_sync=1
  elif ! lock_layer_ok "$INFRA_DIR"; then
    gate_needs_sync=1
  fi

  if [[ "$gate_needs_sync" -eq 1 ]]; then
    say "gate venv: applying $INFRA_DIR/uv.lock exactly"
    trace "uv sync --frozen --project $INFRA_DIR"
    if ! (cd "$INFRA_DIR" && as_owner env -u PYTHONPATH -u UV_PROJECT_ENVIRONMENT \
        "$UV_BIN" sync --frozen --project "$INFRA_DIR"); then
      fail "gate venv lock sync did not complete." \
        "Until it does, \`uv run pytest\` in $INFRA_DIR may be refused by the" \
        "OMN-15620 purity gate. Run by hand and read the error:" \
        "  cd $INFRA_DIR && env -u PYTHONPATH uv sync --frozen"
    fi
    say "gate venv: reconciled"
  else
    say "gate venv: already lock-pure"
  fi

  # ---- surface 3: the hook venv(s), lock-governed only -------------------- #
  # Exact (not --inexact) on purpose: a hook venv has no composed layer above
  # its lock, so anything the lock does not mention is cross-repo pollution.
  # This host had omnibase_infra's dev group (pre-commit, import-linter,
  # pytest-xdist, hypothesis, ...) installed into omniclaude/.venv -- the
  # OMN-15620 class, and precisely the sort of thing that makes one venv answer
  # a question differently from another. An exact sync is what removes it.
  local project any_hook=0
  local hook_owner
  while IFS= read -r project; do
    [[ -n "$project" ]] || continue
    any_hook=1
    # RUN_AS was planned from the CLI venv's owner. A hook venv owned by someone
    # else would be written as the wrong user by that same prefix, which is the
    # exact hazard plan_privileges exists to prevent -- so it is refused rather
    # than written. Silently skipping it would be the OMN-17291 condition: a
    # surface nobody reconciles and nobody is told about.
    hook_owner="$(rp_surface_owner "$project/.venv" || true)"
    if [[ -n "$hook_owner" && "$hook_owner" != "$SURFACE_OWNER" ]]; then
      say "INDETERMINATE: hook venv $project/.venv is owned by $hook_owner, but the"
      say "  package operations are running as $SURFACE_OWNER (owner of $INFRA_DIR)."
      say "  Reconcile it as $hook_owner, or make the two surfaces share an owner."
      exit "$EXIT_INDETERMINATE"
    fi
    trace "uv sync --frozen --check --project $project"
    if lock_layer_ok "$project"; then
      say "hook venv: already in sync ($project/.venv)"
      continue
    fi
    say "hook venv: reconciling $project/.venv to $project/uv.lock"
    trace "uv sync --frozen --project $project"
    if ! (cd "$project" && as_owner env -u PYTHONPATH -u UV_PROJECT_ENVIRONMENT \
        "$UV_BIN" sync --frozen --project "$project"); then
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

  # ---- CLI venvs this script does not own (OMN-18663) --------------------- #
  # REPORT-ONLY here, on exactly the terms the clone leg is report-only: the
  # exit code answers for the surfaces this script WRITES, and a venv it is
  # forbidden to write is not one of them. What changes is that a drifted one is
  # now named, with its remedy, instead of leaving a reader to infer from
  # "reconciled" that every venv on the host was covered.
  report_unowned_cli_venvs "$(market_head)" || true

  # ---- surface 4: the lane-identity hook on every clone (OMN-18260) ------- #
  # Unlike the CLI-venv census above, this one is NOT report-only: this script
  # writes it, so its exit code answers for it. A repair that armed nothing and
  # exited 0 would be the state this ticket was reopened over.
  if ! arm_lane_hooks; then
    say "DRIFT: the lane-identity hook readback still names an unarmed clone"
    say_lane_hook_remedy
    exit "$EXIT_DRIFT"
  fi

  exit "$EXIT_OK"
}

if [[ "$MODE" == "check" ]]; then
  run_check
fi
run_repair
