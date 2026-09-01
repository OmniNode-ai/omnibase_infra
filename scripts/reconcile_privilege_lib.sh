# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# shellcheck shell=bash
#
# reconcile_privilege_lib.sh -- write as the owner of the surface, or refuse.
# ============================================================================
#
# THE RULE, IN ONE LINE
#   A reconciler that mutates a surface must run that mutation as the user who
#   OWNS the surface. If it cannot become that user, it refuses. It never writes
#   as whoever it happens to be.
#
# WHY THIS IS A LIBRARY AND NOT A SECOND COPY
#   OMN-17335 established the rule for the venv surface and implemented it
#   inline in `reconcile-workspace-venvs.sh`. OMN-17366 then hit the identical
#   defect on the CLONE surface -- and the fix for it must be the same guard,
#   not a parallel one. Two implementations of a privilege rule drift, and the
#   half that drifts is the half nobody is looking at. So the mechanics live
#   here, once, and both reconcilers source them.
#
#   What stays with each caller is POLICY: which surface it is about, what its
#   refusal message says, and which exit code it uses. Those legitimately
#   differ. The mechanics -- who owns this, what is their home, how do I become
#   them -- do not.
#
# THE DAMAGE THIS PREVENTS, OBSERVED
#   `.201` runs the reconcile cron as root against `/data/omninode`, every file
#   of which is owned by the operator. The clone surface wrote as root for as
#   long as it existed, and by 2026-09-01 had deposited root-owned objects into
#   operator-owned clones:
#
#       omnibase_infra 572   omnimarket 261   omnibase_compat 150
#       omnibase_core  119   omnibase_spi  16          (1118 total)
#
#   The failure mode is nasty because it is INTERMITTENT: a plain operator
#   `git fetch` fails only when it needs to write near an object root owns --
#
#       error: insufficient permission for adding an object to repository
#       database .git/objects
#       fatal: failed to write object
#
#   -- so the surface looks healthy right up until it does not, and the cause is
#   an hour of cron ticks in the past rather than anything the operator just did.
#
# NO BYPASS
#   There is deliberately no environment variable that says "write anyway".
#   Writing anyway is the defect.
#
# movement-proof: this file moves no surface and reconciles nothing. It decides
#   WHO a write runs as; it performs none. The readback that proves a surface
#   actually moved belongs to the callers, and both of them do it through
#   scripts/reconcile_verify_movement.py (OMN-17307). Adding a proof here would
#   be a second opinion about a surface this file never touched.
#
# CONTRACT
#   Callers must have CURRENT_USER set (or let this file derive it) and must
#   treat a non-zero rp_plan_privileges as fatal. Sourcing this file defines:
#
#     rp_surface_owner  <path>            -> echoes the owning username
#     rp_user_home      <user>            -> echoes that user's home directory
#     rp_plan_privileges <surface-path>   -> sets RP_OWNER / RP_OWNER_HOME / RUN_AS
#     as_owner          <cmd...>          -> runs cmd as RP_OWNER
#
#   rp_plan_privileges return codes, so each caller can phrase its own refusal:
#     0  RUN_AS is set (possibly empty -- we already ARE the owner). Proceed.
#     1  the surface's owner could not be read at all.
#     2  the owner differs and this process cannot become them.
#     3  the owner differs, we are root, but their home is unresolvable.
# ----------------------------------------------------------------------------

: "${CURRENT_USER:=$(id -un)}"

# The command prefix that reaches the owner. Empty means "already them", which
# is the case on every developer machine and whenever the operator runs this by
# hand. Declared here so `set -u` callers can reference it before planning.
declare -a RUN_AS=()
RP_OWNER=""
RP_OWNER_HOME=""

# Owner of the nearest existing ancestor of a path. GNU and BSD `stat` disagree
# on the flag, and this runs on both Linux hosts and macOS.
rp_surface_owner() {
  local path="$1"
  while [[ -n "$path" && "$path" != "/" && ! -e "$path" ]]; do
    path="$(dirname "$path")"
  done
  [[ -e "$path" ]] || return 1
  stat -c '%U' "$path" 2>/dev/null && return 0   # GNU
  stat -f '%Su' "$path" 2>/dev/null && return 0  # BSD
  return 1
}

# Home directory of a user, WITHOUT the `eval echo ~user` trick (which expands
# whatever the name happens to contain). $HOME is used only for the current
# user, where it is authoritative and where a test can control it.
rp_user_home() {
  local user="$1" home=""
  if [[ "$user" == "$CURRENT_USER" && -n "${HOME:-}" ]]; then
    printf '%s' "$HOME"
    return 0
  fi
  home="$(getent passwd "$user" 2>/dev/null | cut -d: -f6)"
  if [[ -z "$home" ]]; then
    home="$(dscl . -read "/Users/$user" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
  fi
  [[ -n "$home" ]] || return 1
  printf '%s' "$home"
}

# Decide, once, who the writes must run as.
rp_plan_privileges() {
  local surface="$1"

  RP_OWNER="$(rp_surface_owner "$surface")" || return 1
  RP_OWNER_HOME="$(rp_user_home "$RP_OWNER" || true)"

  if [[ "$RP_OWNER" == "$CURRENT_USER" ]]; then
    RUN_AS=()
    return 0
  fi

  if [[ "$(id -u)" -eq 0 ]] && command -v runuser >/dev/null 2>&1; then
    # HOME is set explicitly: `runuser` without `-l` keeps root's HOME, so a
    # dropped tool would try to write its cache into /root and fail on
    # permissions -- a confusing failure two layers from its cause. git is the
    # same story: with root's HOME it reads root's .gitconfig and credentials.
    [[ -n "$RP_OWNER_HOME" ]] || return 3
    RUN_AS=(runuser -u "$RP_OWNER" -- env "HOME=$RP_OWNER_HOME")
    return 0
  fi

  return 2
}

# Run a command as the owner of the surface being written. Every mutation in a
# reconciler goes through here; check_reconciler_privilege.py fails the build if
# one does not.
as_owner() {
  if [[ ${#RUN_AS[@]} -eq 0 ]]; then
    "$@"
  else
    "${RUN_AS[@]}" "$@"
  fi
}
