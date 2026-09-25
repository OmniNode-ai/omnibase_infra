#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

# git-maintenance.sh — Clean up merged remote branches and finished worktrees
#
# Usage:
#   git-maintenance.sh [--dry-run] [--execute] [--delete-remote-branches] [--prune-worktrees]
#
# Modes:
#   --dry-run                 Report only (the default). Removes nothing.
#   --execute                 Act on the phases that were explicitly selected.
#   --delete-remote-branches  Phase 1: delete remote branches merged into origin/main.
#                             Without this flag, phase 1 only reports, even with --execute.
#   --prune-worktrees         Phase 2: remove finished worktrees under WORKTREE_ROOT.
#                             Requires WORKTREE_ROOT and ONEX_LEDGER_PATH to be set
#                             explicitly; there is no derived default for a destructive step.
#
# The governed, claim-aware and content-keyed worktree pruner is the morning
# worktree prune (omniclaude scripts/worktree_auto_prune.py). Prefer it. Phase 2
# here is a narrower fallback, and it is safe by construction (OMN-19396):
#
#   - a worktree is removable only when EVERY check below passes, and a check
#     that cannot run counts as a failure (fail closed, never "reads as clean"):
#       * `git status --porcelain --ignored` succeeds and shows no change, and no
#         ignored file outside the regenerable caches (.venv, node_modules, ...);
#       * HEAD is reachable from a remote-tracking ref (nothing unpushed);
#       * the owning clone holds no stash for the worktree's branch;
#       * `gh` reports no open pull request for the branch;
#       * the rolling ledger holds no open CLAIM naming the ticket dir, the
#         branch or the ticket id;
#   - a secrets-shaped file (.env, a key, settings.local.json) is never
#     regenerable, wherever it sits: a `.env` inside an ignored `dist/` keeps the
#     worktree although `git status --ignored` names only `dist/` (OMN-19539);
#   - before any removal, the worktree's diff and its untracked and ignored-but-
#     not-regenerable files are saved under $OMNI_HOME/.onex_state by the shared
#     helper omniclaude/scripts/worktree_removal_snapshot.py; a missing helper
#     or a failed save keeps the worktree (OMN-19539, operator ruling 2026-09-25);
#   - one kept worktree keeps its whole ticket dir untouched;
#   - removal is `git worktree remove` without `--force`, one worktree at a time.
#     Nothing is ever deleted recursively: non-git content (files, hidden dirs,
#     full clones, symlinks) is left in place and reported, and a ticket dir is
#     removed only by `rmdir` once it is empty.
#
# Phase 1 is gated the same way: it deletes only with --execute AND
# --delete-remote-branches, never deletes `dev`, the remote's default branch or
# a long-lived branch, and keeps any branch checked out in a worktree, with an
# open pull request, or named by an open ledger CLAIM.
#
# Branch deletion strategy: merge-based (not age-based) — only branches whose commits
# are reachable from origin/main are candidates.
# ASSUMPTION: origin/main is the canonical merge sink. Repos with different branch-flow
# policy (e.g., staged integration branches, release trains) should add protection
# patterns or perform repo-specific review before destructive execution.

# No machine-path defaults (rules 6 and 8, OMN-19396): OMNI_HOME is required.
OMNI_HOME="${OMNI_HOME:?OMNI_HOME must be set to the registry root; git-maintenance.sh has no default}"
WORKTREE_ROOT="${WORKTREE_ROOT:-}"
ONEX_LEDGER_PATH="${ONEX_LEDGER_PATH:-}"
DRY_RUN=true
PRUNE_WORKTREES=false
DELETE_REMOTE_BRANCHES=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --execute) DRY_RUN=false ;;
    --dry-run) DRY_RUN=true ;;
    --prune-worktrees) PRUNE_WORKTREES=true ;;
    --delete-remote-branches) DELETE_REMOTE_BRANCHES=true ;;
    *) echo "git-maintenance.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

refuse() {
  echo "git-maintenance.sh: REFUSED: $*" >&2
  exit 2
}

# --- OMN-19396: explicit scope for the destructive worktree phase ---------------
# A derived default root once pointed this phase at a real directory and it
# removed 119 ticket dirs. The root must be named by the caller, every time.
if [ "$PRUNE_WORKTREES" = true ]; then
  [ -n "$WORKTREE_ROOT" ] || refuse "--prune-worktrees needs WORKTREE_ROOT set explicitly; there is no default root"
  [ -d "$WORKTREE_ROOT" ] || refuse "WORKTREE_ROOT is not a directory: $WORKTREE_ROOT"
  [ -n "$ONEX_LEDGER_PATH" ] || refuse "--prune-worktrees needs ONEX_LEDGER_PATH set, so open CLAIMs can be honoured"
  [ -r "$ONEX_LEDGER_PATH" ] && [ -f "$ONEX_LEDGER_PATH" ] || refuse "ONEX_LEDGER_PATH is not a readable file: $ONEX_LEDGER_PATH"
fi
if [ "$DELETE_REMOTE_BRANCHES" = true ] && [ "$DRY_RUN" = false ]; then
  [ -n "$ONEX_LEDGER_PATH" ] || refuse "--delete-remote-branches needs ONEX_LEDGER_PATH set, so open CLAIMs can be honoured"
  [ -r "$ONEX_LEDGER_PATH" ] && [ -f "$ONEX_LEDGER_PATH" ] || refuse "ONEX_LEDGER_PATH is not a readable file: $ONEX_LEDGER_PATH"
fi

# Ignored paths that are regenerable caches. Any OTHER ignored path (.env,
# settings.local.json, scratch output) keeps the worktree: `git worktree remove`
# deletes ignored files, and those are not recoverable from git.
is_regenerable_ignored() {
  local p="${1%/}"
  case "$(basename "$p")" in
    .venv|venv|node_modules|__pycache__|.pytest_cache|.mypy_cache|.ruff_cache) return 0 ;;
    .coverage|.coverage.*|coverage.xml|htmlcov|.tox|.nox|dist|build|*.egg-info|.eggs) return 0 ;;
    *.pyc|.DS_Store|.hypothesis|.turbo|.next) return 0 ;;
  esac
  return 1
}

# Secrets-shaped names are never regenerable, wherever they sit (OMN-19539).
# The same set as the shared snapshot helper's, matched case-insensitively.
SECRET_SHAPED_FIND_EXPR=(
  -iname '.env' -o -iname '.env.*' -o -iname '*.env' -o -iname '.envrc'
  -o -iname '*.pem' -o -iname '*.key' -o -iname '*.p12' -o -iname '*.pfx'
  -o -iname 'id_rsa*' -o -iname 'id_ed25519*' -o -iname 'id_ecdsa*'
  -o -iname '.netrc' -o -iname '.npmrc' -o -iname '.pypirc' -o -iname '.pgpass'
  -o -iname 'credentials' -o -iname 'credentials.json' -o -iname '*.tfvars'
  -o -iname 'settings.local.json'
)

# Print the first secrets-shaped file under a directory. Returns non-zero when
# the directory cannot be fully read, which the caller treats as "keep".
first_secret_shaped_under() {
  local dir="$1" found rc=0
  found=$(find "$dir" \( "${SECRET_SHAPED_FIND_EXPR[@]}" \) -print) || rc=$?
  [ "$rc" -eq 0 ] || return "$rc"
  printf '%s' "$found" | head -1
}

SNAPSHOT_HELPER="$OMNI_HOME/omniclaude/scripts/worktree_removal_snapshot.py"

# Print the open CLAIM rows (by timestamp) that mention any of the given keys as
# a whole token. A CLAIM is closed by a TERMINAL whose closes-CLAIM token ends
# with the CLAIM's timestamp, or, when a TERMINAL carries no closes-CLAIM, by any
# later TERMINAL of the same lane. Returns non-zero if the ledger cannot be read.
open_claims_for() {
  [ -n "$ONEX_LEDGER_PATH" ] && [ -r "$ONEX_LEDGER_PATH" ] || return 3
  # Keys never hold spaces (ticket dir names, branch names, ticket ids), so a
  # space-joined list crosses into awk safely.
  local keys="$*"
  # grep exits 1 on a ledger with no CLAIM or TERMINAL row; that is not an error.
  { grep -E '^[^|]+\| (CLAIM|TERMINAL) \|' "$ONEX_LEDGER_PATH" || [ "$?" -eq 1 ]; } | awk -v keys="$keys" '
    function mentions(row,   i, k, pos, rest, before, after, off) {
      for (i = 1; i <= nkeys; i++) {
        k = K[i]; rest = row; off = 0
        while ((pos = index(rest, k)) > 0) {
          before = (off + pos > 1) ? substr(row, off + pos - 1, 1) : ""
          after = substr(row, off + pos + length(k), 1)
          if (before !~ /[A-Za-z0-9_.-]/ && after !~ /[A-Za-z0-9_.-]/) return 1
          off += pos; rest = substr(row, off + 1)
        }
      }
      return 0
    }
    function field(row, name,   re, m) {
      re = "(^|\\| )" name "=[^ |]+"
      if (match(row, re)) { m = substr(row, RSTART, RLENGTH); sub(/^\| /, "", m); sub(name "=", "", m); return m }
      return ""
    }
    BEGIN { nkeys = split(keys, K, " ") }
    {
      split($0, cells, " \\| "); ts = cells[1]; gsub(/ /, "", ts); type = cells[2]; lane = field($0, "lane")
      if (type == "CLAIM") {
        if (mentions($0)) { open[ts] = lane }
      } else if (type == "TERMINAL") {
        closes = field($0, "closes-CLAIM")
        if (closes != "") {
          for (t in open) if (length(closes) >= length(t) && substr(closes, length(closes) - length(t) + 1) == t) delete open[t]
        } else if (lane != "") {
          for (t in open) if (open[t] == lane) delete open[t]
        }
      }
    }
    END { for (t in open) print t " lane=" open[t] }
  '
}

# Keys a ledger CLAIM would use for a ticket dir or a branch: the name itself and
# any OMN ticket id inside it, upper-cased.
claim_keys() {
  local s
  for s in "$@"; do
    [ -n "$s" ] || continue
    printf '%s\n' "$s"
    printf '%s\n' "$s" | grep -oiE 'omn-[0-9]+' | tr '[:lower:]' '[:upper:]' || true
  done | sort -u
}

# Number of open pull requests whose head is <branch>, run inside <dir> so gh
# resolves the repository from its remote. Non-zero when gh cannot answer.
open_pr_count() {
  local dir="$1" branch="$2"
  command -v gh >/dev/null || { echo "gh is not installed" >&2; return 3; }
  (cd "$dir" && gh pr list --head "$branch" --state open --json number --jq 'length')
}

# --- OMN-18826: worktree removal reporting -----------------------------------
#
# `git worktree remove` WITHOUT `--force` is not atomic. Its order of operations
# (verified against git 2.50.1) is: refuse if the tree is dirty; delete the
# admin directory under <clone>/.git/worktrees/<id>; THEN walk the working tree
# unlinking files, continuing past every failure and only returning non-zero at
# the end. A removal that fails partway through the walk leaves a tree with most
# of its files gone, a stale `.git` pointer, and NO admin directory. A later
# removal over that debris fails with "contains modified or untracked files",
# which points at the wrong cause entirely.
#
# This helper therefore reports what git said and what the tree looks like
# afterwards. It never prints a guessed cause, and it never suppresses stderr
# (Operating Rule 16). It does not add `--force`: the removal policy is not this
# script's to set, and a failure here is a signal, not an obstacle.
#
# Prints its report to stdout. Returns 0 on a complete removal, 1 otherwise.
remove_one_worktree() {
  local wt="$1" clone_dir="$2" admin_dir="$3"
  local head_oid="" stderr_file="" rc=0 survivors=0 saved=""

  # Save before removing (OMN-19539). A missing helper or a failed save keeps
  # the worktree; nothing is removed unsaved.
  if [ ! -f "$SNAPSHOT_HELPER" ]; then
    echo "  [kept] $wt: pre-removal snapshot helper missing ($SNAPSHOT_HELPER)"
    return 1
  fi
  saved=$(python3 "$SNAPSHOT_HELPER" "$wt" --reason git-maintenance.sh) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [kept] $wt: pre-removal snapshot failed (exit $rc): $saved"
    return 1
  fi
  echo "  [saved] $wt: $saved"

  # Record what a reconstruction would need BEFORE touching the tree, because a
  # failed removal destroys the index and the admin directory that hold it.
  head_oid=$(git -C "$wt" rev-parse HEAD || echo "unknown")

  stderr_file=$(mktemp)
  git -C "$clone_dir" worktree remove "$wt" 2>"$stderr_file" || rc=$?

  if [ "$rc" -eq 0 ]; then
    rm -f "$stderr_file"
    echo "  [removed] $wt"
    return 0
  fi

  echo "  [remove-failed] $wt (exit $rc, head $head_oid)"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "    git: $line"
  done < "$stderr_file"
  rm -f "$stderr_file"

  if [ ! -e "$wt" ]; then
    echo "    classification: removed-despite-error: the path no longer exists"
    return 1
  fi

  survivors=$(find "$wt" -mindepth 1 -type f | wc -l | tr -d ' ') || true

  if [ -n "$admin_dir" ] && [ ! -d "$admin_dir" ]; then
    echo "    classification: half-removed: admin dir deleted, $survivors files remain"
    echo "    this tree is debris, not peer work; reconstruct from head $head_oid"
  else
    echo "    classification: not-removed: tree intact, $survivors files remain"
  fi
  return 1
}
# -----------------------------------------------------------------------------

# Local checks for one linked worktree (the open pull request check is separate,
# see check_no_open_pr). Prints "[keep] <wt>: <reason>"
# lines and returns 1 when it must stay. On success prints nothing and sets
# WT_BRANCH, WT_CLONE and WT_ADMIN for the caller.
check_worktree_removable() {
  local wt="$1" out err rc ahead stashes line p kept=0
  WT_BRANCH=""; WT_CLONE=""; WT_ADMIN=""

  WT_ADMIN=$(sed -n 's/^gitdir: //p' "$wt/.git" | head -1)
  case "$WT_ADMIN" in
    */.git/worktrees/*) WT_CLONE="${WT_ADMIN%%/.git/worktrees/*}" ;;
    *) echo "  [keep] $wt: .git pointer does not name a clone's worktree admin dir"; return 1 ;;
  esac

  err=$(mktemp)
  rc=0
  out=$(git -C "$wt" status --porcelain --ignored 2>"$err") || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [keep] $wt: git status failed (exit $rc), so the tree is NOT treated as clean"
    while IFS= read -r line; do [ -n "$line" ] && echo "    git: $line"; done < "$err"
    rm -f "$err"; return 1
  fi
  rm -f "$err"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      '!! '*)
        p="${line#!! }"
        if ! is_regenerable_ignored "$p"; then
          echo "  [keep] $wt: holds ignored file not recoverable from git: $p"; kept=1; break
        fi
        if [ -d "$wt/${p%/}" ]; then
          rc=0
          secret=$(first_secret_shaped_under "$wt/${p%/}") || rc=$?
          if [ "$rc" -ne 0 ]; then
            echo "  [keep] $wt: cannot read all of $p (exit $rc), so it is not treated as regenerable"; kept=1; break
          fi
          if [ -n "$secret" ]; then
            echo "  [keep] $wt: holds ignored file not recoverable from git: ${secret#"$wt"/} (a secrets-shaped file inside $p)"; kept=1; break
          fi
        fi
        ;;
      *) echo "  [keep] $wt: has uncommitted changes"; kept=1; break ;;
    esac
  done <<< "$out"
  [ "$kept" -eq 0 ] || return 1

  rc=0
  ahead=$(git -C "$wt" rev-list --count HEAD --not --remotes) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [keep] $wt: cannot compare HEAD with remote-tracking refs (exit $rc)"; return 1
  fi
  if [ "$ahead" != "0" ]; then
    echo "  [keep] $wt: $ahead commit(s) on no remote-tracking ref (unpushed)"; return 1
  fi

  WT_BRANCH=$(git -C "$wt" symbolic-ref -q --short HEAD || true)

  rc=0
  stashes=$(git -C "$WT_CLONE" stash list --format=%gs) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [keep] $wt: cannot read the clone's stash list (exit $rc)"; return 1
  fi
  local stash_branch="${WT_BRANCH:-(no branch)}"
  while IFS= read -r line; do
    case "$line" in
      "WIP on $stash_branch: "*|"On $stash_branch: "*)
        echo "  [keep] $wt: the clone holds a stash for $stash_branch"; return 1 ;;
    esac
  done <<< "$stashes"

  return 0
}

# The remote check, run last so the GitHub API is asked only about worktrees
# every local check already cleared. Returns 1 when the worktree must stay.
check_no_open_pr() {
  local wt="$1" branch="$2" prs rc=0
  [ -n "$branch" ] || return 0
  prs=$(open_pr_count "$wt" "$branch") || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [keep] $wt: cannot check for an open pull request on $branch (exit $rc)"; return 1
  fi
  if [ "$prs" != "0" ]; then
    echo "  [keep] $wt: $branch has an open pull request"; return 1
  fi
  return 0
}

echo "=== Git Maintenance ==="
echo "Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'EXECUTE')"
echo ""

TOTAL_DELETED=0

# Phase 1: Delete remote branches merged into origin/main
for repo_dir in "$OMNI_HOME"/*/; do
  repo=$(basename "$repo_dir")
  [ -d "$repo_dir/.git" ] || [ -f "$repo_dir/.git" ] || continue

  echo "--- $repo ---"

  # Refresh remote refs before evaluating merged state. git's own error is shown.
  git -C "$repo_dir" fetch --prune origin || { echo "  [warn] fetch failed, skipping $repo"; continue; }

  merged_err=$(mktemp)
  merged_rc=0
  merged=$(git -C "$repo_dir" branch -r --merged origin/main 2>"$merged_err") || merged_rc=$?
  if [ "$merged_rc" -ne 0 ]; then
    echo "  [warn] cannot list branches merged into origin/main (exit $merged_rc), skipping $repo"
    while IFS= read -r line; do [ -n "$line" ] && echo "    git: $line"; done < "$merged_err"
    rm -f "$merged_err"; continue
  fi
  rm -f "$merged_err"

  # The remote's default branch and every branch checked out in a worktree of
  # this clone are never candidates.
  default_branch=$(git -C "$repo_dir" symbolic-ref -q --short refs/remotes/origin/HEAD || true)
  default_branch="${default_branch#origin/}"
  wt_list_rc=0
  wt_list=$(git -C "$repo_dir" worktree list --porcelain) || wt_list_rc=$?
  if [ "$wt_list_rc" -ne 0 ]; then
    echo "  [warn] cannot list this clone's worktrees (exit $wt_list_rc), skipping $repo"; continue
  fi
  checked_out=$(printf '%s\n' "$wt_list" | sed -n 's|^branch refs/heads/||p')

  stale=0
  while IFS= read -r branch; do
    branch=$(printf '%s' "$branch" | sed 's/^ *//')
    [ -z "$branch" ] && continue
    case "$branch" in *" -> "*) continue ;; esac
    case "$branch" in origin/*) ;; *) continue ;; esac
    branch="${branch#origin/}"

    # Skip protected branches and long-lived naming patterns
    case "$branch" in
      main|master|dev|develop|HEAD|gh-pages) continue ;;
      release/*|support/*|infra/*) continue ;;  # long-lived branch patterns
    esac
    if [ -n "$default_branch" ] && [ "$branch" = "$default_branch" ]; then
      continue
    fi
    if printf '%s\n' "$checked_out" | grep -qxF -- "$branch"; then
      echo "  [keep] origin/$branch: checked out in a worktree of this clone"
      continue
    fi

    if [ "$DRY_RUN" = true ] || [ "$DELETE_REMOTE_BRANCHES" = false ]; then
      echo "  [would delete] origin/$branch"
      stale=$((stale + 1))
      continue
    fi

    rc=0
    prs=$(open_pr_count "$repo_dir" "$branch") || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  [keep] origin/$branch: cannot check for an open pull request (exit $rc)"; continue
    fi
    if [ "$prs" != "0" ]; then
      echo "  [keep] origin/$branch: has an open pull request"; continue
    fi
    rc=0
    # shellcheck disable=SC2046
    claims=$(open_claims_for $(claim_keys "$branch")) || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  [keep] origin/$branch: cannot read the ledger (exit $rc)"; continue
    fi
    if [ -n "$claims" ]; then
      echo "  [keep] origin/$branch: open ledger CLAIM $(printf '%s' "$claims" | head -1)"; continue
    fi

    if git -C "$repo_dir" push origin --delete "$branch"; then
      echo "  [deleted] origin/$branch"
      stale=$((stale + 1))
    else
      echo "  [delete-failed] origin/$branch"
    fi
  done <<< "$merged"

  if [ "$DRY_RUN" = false ] && [ "$DELETE_REMOTE_BRANCHES" = false ] && [ "$stale" -gt 0 ]; then
    echo "  $stale merged branches reported, none deleted (pass --delete-remote-branches to delete)"
  else
    echo "  $stale merged branches $([ "$DRY_RUN" = true ] && echo 'would be' || echo '') deleted"
  fi
  TOTAL_DELETED=$((TOTAL_DELETED + stale))
done

# Phase 2: Clean finished worktrees
if [ "$PRUNE_WORKTREES" = true ]; then
  echo ""
  echo "=== Worktree Cleanup ==="
  echo "  root: $WORKTREE_ROOT"
  worktrees_removed=0

  for ticket_dir in "$WORKTREE_ROOT"/*/; do
    # The glob leaves a trailing slash, which doubles in every path built below
    # and lands in the report rows. A reported path has to be one a reader can
    # paste back, so normalise it once here.
    ticket_dir="${ticket_dir%/}"
    [ -d "$ticket_dir" ] || continue
    ticket=$(basename "$ticket_dir")
    if [ -L "$ticket_dir" ]; then
      echo "  [skip] $ticket (a symlink, not a ticket dir)"
      continue
    fi

    keep_ticket=false
    debris=false
    linked=()
    branches=()
    for entry in "$ticket_dir"/* "$ticket_dir"/.[!.]* "$ticket_dir"/..?*; do
      [ -e "$entry" ] || [ -L "$entry" ] || continue
      if [ -L "$entry" ]; then
        echo "  [non-git] $entry: symlink, left in place"
        continue
      fi
      if [ -d "$entry" ] && [ -f "$entry/.git" ]; then
        # OMN-18826: a half-deleted worktree's signature is a `.git` pointer file
        # whose `gitdir:` target no longer exists, because a failed removal
        # deletes the admin directory before it walks the tree. Such a tree is
        # debris from an earlier failed removal, not a lane's in-flight work.
        pointer=$(sed -n 's/^gitdir: //p' "$entry/.git" | head -1)
        if [ -n "$pointer" ] && [ ! -d "$pointer" ]; then
          echo "  [debris] $entry: .git points at a missing admin dir ($pointer) — half-removed tree, not peer work"
          debris=true
          continue
        fi
        linked+=("$entry")
        continue
      fi
      if [ -d "$entry" ] && [ -d "$entry/.git" ]; then
        echo "  [non-git] $entry: a full clone, not a linked worktree; left in place"
        continue
      fi
      echo "  [non-git] $entry: not a git worktree; left in place"
    done

    if [ "$debris" = true ]; then
      # Reported, not removed: what to do with debris is a removal-policy
      # decision this script does not own.
      echo "  [skip] $ticket (half-removed worktree present)"
      continue
    fi
    if [ "${#linked[@]}" -eq 0 ]; then
      echo "  [skip] $ticket (no linked worktree; nothing here is removed)"
      continue
    fi

    clones=()
    admins=()
    for wt in "${linked[@]}"; do
      if ! check_worktree_removable "$wt"; then
        keep_ticket=true
        break
      fi
      branches+=("${WT_BRANCH:-}")
      clones+=("$WT_CLONE")
      admins+=("$WT_ADMIN")
    done
    if [ "$keep_ticket" = false ]; then
      rc=0
      # shellcheck disable=SC2046
      claims=$(open_claims_for $(claim_keys "$ticket" ${branches[@]+"${branches[@]}"})) || rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "  [keep] $ticket: cannot read the ledger (exit $rc)"; keep_ticket=true
      elif [ -n "$claims" ]; then
        echo "  [keep] $ticket: open ledger CLAIM $(printf '%s' "$claims" | head -1)"; keep_ticket=true
      fi
    fi
    if [ "$keep_ticket" = false ]; then
      for i in "${!linked[@]}"; do
        if ! check_no_open_pr "${linked[$i]}" "${branches[$i]}"; then
          keep_ticket=true
          break
        fi
      done
    fi

    if [ "$keep_ticket" = true ]; then
      echo "  [skip] $ticket (kept; see the line above)"
    elif [ "$DRY_RUN" = true ]; then
      echo "  [would remove] $ticket"
      worktrees_removed=$((worktrees_removed + 1))
    else
      # Properly remove worktrees via git (no --force: let failures be safety signals)
      all_removed=true
      for i in "${!linked[@]}"; do
        if ! remove_one_worktree "${linked[$i]}" "${clones[$i]}" "${admins[$i]}"; then
          all_removed=false
        fi
      done
      if [ "$all_removed" = true ]; then
        # rmdir, never a recursive delete: it succeeds only on an empty dir, so
        # anything that was not a git worktree stays where it was.
        if rmdir "$ticket_dir"; then
          echo "  [removed] $ticket"
        else
          echo "  [worktrees-removed] $ticket (non-git content left in place, see the lines above)"
        fi
        worktrees_removed=$((worktrees_removed + 1))
      else
        # Saying "removed" over a failed removal is how the misreport reached the
        # report row in the first place. The per-worktree lines above carry the
        # reason; this line only records that the ticket directory still stands.
        echo "  [incomplete] $ticket (see the per-worktree lines above)"
      fi
    fi
  done
  echo "  $worktrees_removed worktrees $([ "$DRY_RUN" = true ] && echo 'would be' || echo '') removed"
fi

echo ""
echo "Total branches: $TOTAL_DELETED | Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN (use --execute to apply)' || echo 'EXECUTED')"
