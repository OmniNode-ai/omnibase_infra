#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# install-canonical-clone-git-hooks.sh -- install the tracked canonical-clone GIT
# hook family onto a host and point every canonical clone's `core.hooksPath` at
# it (OMN-16497).
#
# This is the GIT-hook family: the commit/push guard (OMN-7018 + OMN-15071), the
# reference-transaction guard (OMN-16497), and the position library they share.
# It is NOT the Claude Code PreToolUse guard -- that one is a user-level hook
# installed by omniclaude's `install-canonical-clone-guard.sh`, and the two are
# deliberately separate: the Claude hook cannot see a non-Claude actor, and this
# one cannot see a working-tree write that makes no ref transaction. Neither
# replaces the other.
#
# Until this existed the family was installed BY HAND. The live copy on this Mac
# was byte-identical to the tracked source, which is luck rather than a property
# -- nothing verified it, and nothing would have reported it if a host had
# drifted or had never been installed at all. An enforcement surface that no
# command can prove is installed is detection (rule 5).
#
# Usage:
#   install-canonical-clone-git-hooks.sh                 # READBACK (default): report, change nothing
#   install-canonical-clone-git-hooks.sh --apply         # sync the hooks dir + point every clone at it
#   install-canonical-clone-git-hooks.sh --apply <repo>… # only the named canonical clones
#
# Registry roots (OMN-19388): the clones installed are the direct children of
# `$OMNI_HOME`, of every entry of `ONEX_REGISTRY_ROOTS` (a colon-separated list
# of absolute directories), and of every root already recorded in the roots
# file. A named <repo> is looked up in every root. The live hooks directory
# stays at `$OMNI_HOME/scripts/git-hooks` for every root.
#
# The roots file, `$OMNI_HOME/scripts/git-hooks/registry-roots`, records that
# union (plus `omni_home=`) beside the live hooks, and the hooks read it, so the
# guard holds for an actor whose environment carries neither variable: launchd,
# `env -i`, a GUI git client. The union only grows: a re-run from a shell that
# lacks ONEX_REGISTRY_ROOTS keeps every recorded root whose directory still
# exists. To retire a root, remove its `root=` line by hand and re-run.
#
# Exit codes:
#   0  every canonical clone is installed and current
#   3  action pending (missing, drifted, or a clone not pointed at the hooks dir)
#   1  error, or a refusal
#
# REFUSES, rather than proceeding, when:
#   * `$OMNI_HOME` is unset -- there is no default and no guess (rule 8)
#   * GIT_DIR / GIT_WORK_TREE / GIT_COMMON_DIR are set in the environment. git
#     exports these into hook processes and they OVERRIDE both `-C` and the cwd
#     for every descendant git call. On 2026-09-13 a leaked GIT_DIR pointed an
#     installer at the SHARED canonical-clone hooks directory and every clone on
#     the machine briefly refused commits from unregistered worktrees. An
#     installer that reads a leaked variable does not install what it reports.
#   * a target is a linked worktree rather than a canonical clone.
#   * an `ONEX_REGISTRY_ROOTS` entry is empty, relative, or not a directory.
#   * the existing roots file holds a line that is not omni_home=/root= with an
#     absolute path.

set -euo pipefail

usage() {
  sed -n '/^# Usage:/,/^# REFUSES/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | sed '$d'
}

refuse() {
  printf 'REFUSED: %s\n' "$1" >&2
  exit 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

apply=0
targets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) apply=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*) refuse "unknown option: $1" ;;
    *) targets+=("$1") ;;
  esac
  shift
done

for leaked in GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_INDEX_FILE; do
  if [[ -n "${!leaked:-}" ]]; then
    refuse "$leaked is set in this environment (${!leaked}). git exports it into hook processes and it overrides both -C and the cwd for every git call below, so this installer cannot prove which repository it would be writing to. Run it outside a git hook, or unset $leaked."
  fi
done

OMNI_HOME="${OMNI_HOME:?OMNI_HOME is not set; there is no default registry path}"
[[ -d "$OMNI_HOME" ]] || refuse "OMNI_HOME=$OMNI_HOME is not a directory"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SRC_DIR="$SCRIPT_DIR/git-hooks"
[[ -d "$SRC_DIR" ]] || refuse "tracked hook source $SRC_DIR is missing"

# The live family lives beside the registry, not inside a clone, so pulling any
# one repository cannot silently change what every repository enforces.
LIVE_DIR="$OMNI_HOME/scripts/git-hooks"
LIVE_HOOKS_DIR="$LIVE_DIR/canonical-clone"

SCRIPTS=(canonical_clone_guard.sh canonical_clone_ref_guard.sh canonical_clone_paths.sh)
# hook type -> script it dispatches to. The shared directory composes by hook
# TYPE: a new family member is a new row here, never a repointing of an old one.
HOOK_TYPES=(
  "pre-commit:canonical_clone_guard.sh"
  "pre-push:canonical_clone_guard.sh"
  "commit-msg:canonical_clone_guard.sh"
  "pre-merge-commit:canonical_clone_guard.sh"
  "reference-transaction:canonical_clone_ref_guard.sh"
)

pending=0
note() { printf '  %s\n' "$1"; }

printf 'tracked source: %s\n' "$SRC_DIR"
printf 'live hooks dir: %s\n' "$LIVE_HOOKS_DIR"
printf '\n[1] hook scripts\n'

for script in "${SCRIPTS[@]}"; do
  src="$SRC_DIR/$script"
  dst="$LIVE_DIR/$script"
  [[ -f "$src" ]] || refuse "tracked $src is missing"
  if [[ ! -f "$dst" ]]; then
    note "MISSING  $script"
    pending=1
    if [[ "$apply" == "1" ]]; then
      mkdir -p "$LIVE_DIR"
      cp "$src" "$dst"
      [[ "$script" == *_paths.sh ]] || chmod 755 "$dst"
      note "installed $script"
    fi
  elif [[ "$(sha256_of "$src")" != "$(sha256_of "$dst")" ]]; then
    note "DRIFTED  $script"
    pending=1
    if [[ "$apply" == "1" ]]; then
      backup="$dst.bak.$(date -u +%Y%m%dT%H%M%SZ)"
      cp "$dst" "$backup"
      cp "$src" "$dst"
      [[ "$script" == *_paths.sh ]] || chmod 755 "$dst"
      note "updated $script (previous copy kept at $backup)"
    fi
  else
    note "ok       $script"
  fi
done

printf '\n[2] hook-type symlinks\n'
for row in "${HOOK_TYPES[@]}"; do
  hook_type="${row%%:*}"
  script="${row#*:}"
  link="$LIVE_HOOKS_DIR/$hook_type"
  want="../$script"
  if [[ -L "$link" ]] && [[ "$(readlink "$link")" == "$want" ]]; then
    note "ok       $hook_type -> $want"
    continue
  fi
  note "MISSING  $hook_type -> $want"
  pending=1
  if [[ "$apply" == "1" ]]; then
    mkdir -p "$LIVE_HOOKS_DIR"
    ln -sfn "$want" "$link"
    note "linked $hook_type"
  fi
done

# Every registry root whose clones the hooks guard: OMNI_HOME first, then each
# ONEX_REGISTRY_ROOTS entry, validated exactly as canonical_clone_paths.sh
# validates it, then every root the roots file already records -- all
# de-duplicated by physical path.
ROOTS_FILE="$LIVE_DIR/registry-roots"
roots=("$OMNI_HOME")
seen=("$(cd "$OMNI_HOME" && pwd -P)")
add_root() {
  local entry="$1" physical known
  physical="$(cd "$entry" && pwd -P)"
  for known in "${seen[@]}"; do
    [[ "$known" == "$physical" ]] && return 0
  done
  roots+=("$entry")
  seen+=("$physical")
}
if [[ -n "${ONEX_REGISTRY_ROOTS+set}" ]]; then
  rest="$ONEX_REGISTRY_ROOTS"
  while :; do
    entry="${rest%%:*}"
    [[ "$entry" == /* ]] || refuse "ONEX_REGISTRY_ROOTS=$ONEX_REGISTRY_ROOTS holds the entry '$entry', which is not an absolute path"
    [[ -d "$entry" ]] || refuse "ONEX_REGISTRY_ROOTS=$ONEX_REGISTRY_ROOTS holds the entry '$entry', which is not an existing directory"
    add_root "$entry"
    [[ "$rest" == *:* ]] || break
    rest="${rest#*:}"
  done
fi
if [[ -f "$ROOTS_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      "" | "#"*) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$key" == "$line" ]] || [[ "$value" != /* ]] ||
      [[ "$key" != "omni_home" && "$key" != "root" ]]; then
      refuse "the roots file $ROOTS_FILE holds the line '$line'; every line must be omni_home=<absolute dir> or root=<absolute dir>. Fix or remove the line and re-run."
    fi
    if [[ "$key" == "root" && -d "$value" ]]; then
      add_root "$value"
    fi
  done <"$ROOTS_FILE"
fi

printf '\n[3] registry roots file %s\n' "$ROOTS_FILE"
desired_roots="$(
  printf '# Written by install-canonical-clone-git-hooks.sh --apply (OMN-19388).\n'
  printf '# The canonical-clone hooks read it, so the guard holds when their environment\n'
  printf '# carries neither OMNI_HOME nor ONEX_REGISTRY_ROOTS. Regenerate; do not hand-edit.\n'
  printf 'omni_home=%s\n' "${seen[0]}"
  printf 'root=%s\n' "${seen[0]}"
  # Sorted after the first, so the order the variable happens to list them in
  # never reads as drift.
  printf '%s\n' "${seen[@]:1}" | sed '/^$/d' | LC_ALL=C sort | sed 's/^/root=/'
)"
if [[ -f "$ROOTS_FILE" ]] && [[ "$(cat "$ROOTS_FILE")" == "$desired_roots" ]]; then
  note "ok       ${#seen[@]} root(s): ${seen[*]}"
else
  if [[ -f "$ROOTS_FILE" ]]; then
    note "DRIFTED  registry-roots (want ${#seen[@]} root(s): ${seen[*]})"
  else
    note "MISSING  registry-roots (an actor with no ONEX_REGISTRY_ROOTS sees only its OMNI_HOME)"
  fi
  pending=1
  if [[ "$apply" == "1" ]]; then
    mkdir -p "$LIVE_DIR"
    tmp_roots="$(mktemp "$LIVE_DIR/.registry-roots.XXXXXX")"
    printf '%s\n' "$desired_roots" >"$tmp_roots"
    chmod 644 "$tmp_roots"
    mv -f "$tmp_roots" "$ROOTS_FILE"
    note "wrote registry-roots"
  fi
fi

named_targets=0
[[ ${#targets[@]} -eq 0 ]] || named_targets=1

for root in "${roots[@]}"; do
  printf '\n[4] canonical clones in %s\n' "$root"
  if [[ "$named_targets" == "0" ]]; then
    targets=()
    while IFS= read -r d; do targets+=("$(basename "$d")"); done < <(
      find "$root" -mindepth 2 -maxdepth 2 -name .git -type d -print0 |
        xargs -0 -n1 dirname | sort
    )
  fi

  for repo in ${targets[@]+"${targets[@]}"}; do
    clone="$root/$repo"
    if [[ ! -d "$clone/.git" ]]; then
      note "SKIP     $repo (not a canonical clone: .git is not a directory)"
      continue
    fi
    current="$(git -C "$clone" config --get core.hooksPath 2>/dev/null || true)"
    if [[ "$current" == "$LIVE_HOOKS_DIR" ]]; then
      note "ok       $repo"
      continue
    fi
    if [[ -z "$current" ]]; then
      note "UNSET    $repo (core.hooksPath is not set -- this clone enforces NOTHING)"
    else
      note "OTHER    $repo (core.hooksPath=$current)"
    fi
    pending=1
    if [[ "$apply" == "1" ]]; then
      git -C "$clone" config core.hooksPath "$LIVE_HOOKS_DIR"
      note "pointed $repo at $LIVE_HOOKS_DIR"
    fi
  done
done

printf '\n'
if [[ "$pending" == "1" && "$apply" != "1" ]]; then
  printf 'ACTION PENDING. Re-run with --apply to install.\n'
  exit 3
fi
if [[ "$pending" == "1" ]]; then
  printf 'APPLIED. Re-run with no arguments to read the result back.\n'
  exit 0
fi
printf 'OK: every canonical clone is installed and current.\n'
exit 0
