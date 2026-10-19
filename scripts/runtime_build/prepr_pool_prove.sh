#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# prepr_pool_prove.sh -- the on-host half of one pre-merge runtime proof (OMN-18893).
#
# Copied to a pool host and run there by scripts/runtime_build/prepr_runtime_pool.py,
# one phase per call, so the driver sees every phase's output and can always reach
# teardown. It is the interim recipes A (omnimarket) and B (omnibase_infra test
# merge) as one parameterised file, the method the prove-* lanes ran by hand.
#
# usage: bash prepr_pool_prove.sh <params.env> <phase>
#   phases: snap-pre clone build probe post-bringup tests teardown snap-post
#
# It only ever touches its own compose project and the work directory named by W
# in the params file. Every other project on the host (a dogfood lane, a runner,
# a declared lane) is read for a positive control and never written.
#
# Two kinds of member (config/lab/prepr_runtime_pool.yaml under the workspace config root):
#   isolated    the laptop-bundle project omnibase-infra-local, with its own
#               Postgres, Redpanda and Valkey on the fixed ports.
#   prepr-slot  a numbered .201 pre-PR slot (omnibase-infra-prepr-N), brought up
#               ONLY through prepr_verify_lane.sh and destroyed ONLY through
#               prepr_teardown_slot.sh, both from the test-merge tree. Neither
#               takes a lane argument; the project is derived from the slot.
#
# Params (sourced): TAG W [INFRA_PR INFRA_HEAD | INFRA_GROUP [INFRA_GROUP_BASE INFRA_GROUP_TREE] | INFRA_REF_SHA] [MARKET_PR MARKET_HEAD | MARKET_GROUP [MARKET_GROUP_BASE MARKET_GROUP_TREE] | MARKET_REF_SHA]
#   MODEL_ENDPOINT [ID_FILES LIVE_GREP GROUP_GREP SQL TESTS SUITE_REPO SUITE_WORKERS DEV_CONTROL_MAX CORE_REF SPI_REF COMPAT_REF
#   DOCKER_CONFIG_MODE SLOT_KIND PREPR_SLOT PROJECT MAIN_PORT EFFECTS_PORT SLOT_PORTS
#   POSITIVE_CONTROL REASON OVERLAY_FILE POST_UP_COMMAND POST_BRINGUP_CMD]
#
# The id and file lists below ($C $V $N $IMGS $FILES and the like) are word-split
# ON PURPOSE, and the macOS hosts run bash 3.2, which has no mapfile; hence:
# shellcheck disable=SC2086,SC2013
set -uo pipefail
export PATH=/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH
# shellcheck disable=SC1090
source "$1"
PHASE=$2
: "${TAG:?}" "${W:?}"
KIND=${SLOT_KIND:-isolated}
if [ "$KIND" = prepr-slot ]; then
  case "${PREPR_SLOT:-}" in 1|2) ;; *) echo "prepr-slot needs PREPR_SLOT 1 or 2"; exit 2;; esac
  P=omnibase-infra-prepr-$PREPR_SLOT; M=omninode-prepr-$PREPR_SLOT-runtime; E=omninode-prepr-$PREPR_SLOT-runtime-effects
else
  P=omnibase-infra-local; M=$P-omninode-runtime; E=$P-runtime-effects
fi
# The driver passes PROJECT too; it must agree with the project derived here, so
# a params file can never aim this script at another project.
[ "${PROJECT:-$P}" = "$P" ] || { echo "refusing: PROJECT=$PROJECT but this member is $P"; exit 2; }
MAIN=${MAIN_PORT:-8085}; EFF=${EFFECTS_PORT:-8086}; PORTS=${SLOT_PORTS:-5436|19092|16379|8085|8086}
PC=${POSITIVE_CONTROL:-omnibase-infra-dogfood}
R=$W/root; T=$W/test
ts() { date -u +%FT%TZ; }
lsn() { if command -v ss >/dev/null 2>&1; then ss -ltn; else lsof -nP -iTCP -sTCP:LISTEN; fi; }
# stage_workspace.sh pins each sibling in a worktree under
# ~/.omnibase/state/deploy_source_trees/<build context path, every character outside
# [A-Za-z0-9._-] as _>, and each worktree points into this run's clone. $W is keyed by
# holder and host, so the next run by the same holder reuses the key; with the clone
# gone, its build refused ("not a git repository" on omnibase_core). Removing only this
# run's key, before the clone and with the work directory, leaves other runs' trees alone.
drop_source_trees() {
  local ctx
  ctx="$(cd "$W" 2>/dev/null && pwd -P || printf '%s' "$W")/root/omnibase_infra"
  rm -rf "${HOME}/.omnibase/state/deploy_source_trees/${ctx//[!A-Za-z0-9._-]/_}"
}

# OMN-19064 AC-1: every container of the declared lanes beside a pre-PR slot, by
# id, start time and restart count. snap-pre and snap-post each take one, so a run
# shows by container id that it recreated or restarted none of them. The project
# list is the slot policy's declared-lane table, passed by the driver.
# --- neighbour-readback begin ---
neighbour_snapshot() {
  local proj ids
  for proj in $(printf '%s' "${NEIGHBOUR_PROJECTS:-}" | tr '|' ' '); do
    ids=$(docker ps -aq --filter "label=com.docker.compose.project=$proj")
    [ -n "$ids" ] || continue
    docker inspect -f "$proj {{.Name}} {{.Id}} {{.State.StartedAt}} {{.RestartCount}}" $ids
  done | sort
}
neighbour_diff() {
  local n d
  n=$(grep -c . "$1" 2>/dev/null); n=${n:-0}
  d=$(diff "$1" "$2" 2>/dev/null | grep -E '^[<>]')
  echo "neighbour-diff=$(printf '%s' "$d" | grep -c .) neighbours=$n"
  printf '%s\n' "$d" | grep . | head -20 | sed 's/^/neighbour-changed /'
}
# --- neighbour-readback end ---

# OMN-19064 AC-2: every host port the slot's project publishes, and those outside
# its reserved block (PORTS, the slot policy's four). A port range is never one of
# the four literals, so it is reported outside as written.
# --- slot-ports-readback begin ---
slot_ports_readback() {
  local pub out p
  pub=$(docker ps --filter "label=com.docker.compose.project=$P" --format '{{.Ports}}' \
    | tr ',' '\n' | sed -n 's/.*:\([0-9][0-9-]*\)->.*/\1/p' | sort -u | tr '\n' ' ')
  out=""
  for p in $pub; do
    case "|$PORTS|" in *"|$p|"*) ;; *) out="$out $p" ;; esac
  done
  pub=$(echo $pub | tr ' ' ','); out=$(echo $out | tr ' ' ',')
  echo "slot-published-ports=${pub:-none} outside-block=${out:-none}"
}
# --- slot-ports-readback end ---

# Image identity is proven from content, never from the revision label (a label
# is evidence of intent, not of what was built; OMN-18893). For each repository
# under test the probe lists every file of its installed package inside the
# running container (IMG_LIST_PY, run by the container's python) and compares
# that listing with the build tree's tracked files under src/<pkg>/
# (PKG_TREE_PY, run by the host's python3). A PR that changes nothing under src/
# is then proven by the whole package reading identical, and a PR whose image
# differs from its tree still fails (OMN-19896). The tests extract both scripts
# from between their heredoc markers and run them.
IFS= read -r -d '' IMG_LIST_PY <<'PY' || :
import hashlib, importlib, os, sys
pkg = importlib.import_module(sys.argv[1])
top = os.path.dirname(pkg.__file__)
base = os.path.dirname(top)
for root, dirs, files in os.walk(top):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in files:
        if name.endswith((".pyc", ".pyo")):
            continue
        path = os.path.join(root, name)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        print(digest + "  " + os.path.relpath(path, base))
PY
IFS= read -r -d '' PKG_TREE_PY <<'PY' || :
import hashlib, os, subprocess, sys
root, pkg, container, listing = sys.argv[1:5]
out = subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "src/" + pkg],
                     capture_output=True, check=False).stdout
tree = {}
for raw in out.split(b"\0"):
    rel = raw.decode("utf-8", "replace")
    if not rel.startswith("src/"):
        continue
    path = os.path.join(root, rel)
    rel = rel[len("src/"):]
    if "/__pycache__/" in rel or rel.endswith((".pyc", ".pyo")):
        continue
    if os.path.islink(path) or not os.path.isfile(path):
        continue
    with open(path, "rb") as fh:
        tree[rel] = hashlib.sha256(fh.read()).hexdigest()
image = {}
try:
    with open(listing, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            digest, sep, rel = line.rstrip("\n").partition("  ")
            if sep and len(digest) == 64:
                image[rel] = digest
except OSError:
    pass
missing = sorted(set(tree) - set(image))
differ = sorted(r for r in tree if r in image and image[r] != tree[r])
extra = sorted(set(image) - set(tree))
# a module the image carries and the tree does not is stale code that can run;
# other extras (a force-included resource) are counted, not held against it
stale = [r for r in extra if r.endswith(".py")]
ok = bool(tree) and not missing and not differ and not stale
print("pkg-tree %s %s tree-files=%d image-files=%d missing=%d differ=%d stale-py=%d extra-other=%d match=%s"
      % (pkg, container, len(tree), len(image), len(missing), len(differ), len(stale),
         len(extra) - len(stale), "yes" if ok else "NO"))
for kind, items in (("missing", missing), ("differ", differ), ("stale-py", stale)):
    for rel in items[:5]:
        print("pkg-tree-diff %s %s %s %s" % (pkg, container, kind, rel))
PY

# What a node's contract says about where it runs (OMN-18893), run by the host's
# python3 over one contract.yaml: prints "<contract name or -> <local|runtime>".
# A node is local only when its contract declares no topic under any event_bus key
# and no runtime profile (top level or descriptor): it is invoked directly, so no
# runtime ever logs it. Anything else, and every contract this cannot read (no
# PyYAML, a parse error, an unexpected shape), is runtime: a node is only waived
# on evidence, never for want of it.
IFS= read -r -d '' NODE_CONTRACT_PY <<'PY' || :
import re, sys
name, kind = "-", "runtime"
try:
    import yaml
    with open(sys.argv[1], encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if isinstance(doc, dict):
        given = doc.get("name")
        if isinstance(given, str) and re.fullmatch(r"[A-Za-z0-9_]+", given):
            name = given
        def empty(value):
            return value is None or value == [] or value == {} or value == ""
        bus = doc.get("event_bus")
        descriptor = doc.get("descriptor")
        profiles = [doc.get("runtime_profiles")]
        if descriptor is not None:
            profiles.append(descriptor.get("runtime_profiles") if isinstance(descriptor, dict) else "?")
        no_topics = bus is None or (
            isinstance(bus, dict) and all(empty(v) for k, v in bus.items() if k != "version"))
        if no_topics and all(empty(p) for p in profiles):
            kind = "local"
except Exception:
    name, kind = "-", "runtime"
print(name, kind)
PY

# The facts a pr-head lab-proof receipt binds (OMN-19566): the proven head, the
# dev sha it was proven against, their merge base, and the digest of the PR's own
# diff. The digest is `git diff --no-color --no-ext-diff <merge-base>..<head>`
# hashed with sha256, byte for byte the command lab_pass_receipt.py's
# compute_pr_diff_digest_from_repo runs, so a verifier elsewhere recomputes the
# same value from the same two shas. The PR's changed paths over the same range
# follow, one pr-changed line each, read as the lab-proof-receipt workflow reads
# them (--name-only --no-renames): they decide which of the profile's
# conditional checks the receipt must carry.
pr_subject() { # dir repo pr head base
  local d=$1 repo=$2 pr=$3 h=$4 base=$5 mb dg
  mb=$(git -C "$d" merge-base "$base" "$h" 2>/dev/null) || { echo "pr-subject-unavailable $repo#$pr no merge base with $base"; return 0; }
  dg=$(git -C "$d" diff --no-color --no-ext-diff "$mb..$h" | shasum -a 256 | cut -d' ' -f1)
  echo "pr-subject $repo#$pr head=$h base=$base merge-base=$mb diff-digest=sha256:$dg"
  git -C "$d" diff --name-only --no-renames "$mb" "$h" | sed "s|^|pr-changed $repo#$pr |"
}

# changed_path_live, when the caller names no LIVE_GREP: one term per runtime node
# the PR changes (src/<repo>/nodes/<node>/...) that still exists at the head, so a
# deleted node is not demanded live. The term is the node's names between dots,
# which a runtime logs only once the node runs: its consumer group
# (<lane>.<repo>.<contract name>.consume, on the Auto-wired subscription and group
# lines, and the node= field) or its own modules' loggers
# (<repo>.nodes.<directory>.handlers...). The contract's name need not be the
# directory's (node_host_reconcile_compute logs as host_reconcile_compute), so the
# term carries both when they differ (OMN-18893). Discovery ("Discovered contract:
# <node> (") and registration lines do not match, so a node that is only discovered
# is not live, and node_x does not match node_x_y. A node whose contract declares no
# topics and no runtime profile runs in no runtime: it gets no term and is listed by
# derive_live_not_applicable instead. Printed for the reader; the judge reads the
# mention lines. The terms are extended regular expressions (grep -E).
changed_nodes() { # repo: each node directory the PR changes that exists at the head
  local repo=$1 n
  for n in $(git -C "$R/$repo" diff --name-only --no-renames origin/dev...HEAD -- "src/$repo/nodes/" | awk -F/ 'NF>=5{print $4}' | sort -u); do
    if [ -f "$R/$repo/src/$repo/nodes/$n/contract.yaml" ]; then echo "$n"; fi
  done
}
node_contract_facts() { # repo node -> "<contract name or -> <local|runtime>"
  python3 -c "$NODE_CONTRACT_PY" "$R/$1/src/$1/nodes/$2/contract.yaml" || echo "- runtime"
}
derive_live_grep() { # repo... (each checked out at the tree under proof in $R)
  local repo n cname kind
  for repo in "$@"; do
    for n in $(changed_nodes "$repo"); do
      read -r cname kind <<<"$(node_contract_facts "$repo" "$n")"
      if [ "$kind" = local ]; then continue; fi
      if [ "$cname" = "-" ] || [ "$cname" = "$n" ]; then printf '[.]%s[.] ' "$n"
      else printf '[.](%s|%s)[.] ' "$n" "$cname"; fi
    done
  done
}
derive_live_not_applicable() { # repo... -> the changed nodes that run in no runtime
  local repo n cname kind
  for repo in "$@"; do
    for n in $(changed_nodes "$repo"); do
      read -r cname kind <<<"$(node_contract_facts "$repo" "$n")"
      if [ "$kind" = local ]; then printf '%s ' "$n"; fi
    done
  done
}

test_merge_head() { # dir repo -> merges FETCH_HEAD into dev (no-ff), falls back to the raw head on conflict
  local d=$1 repo=$2
  if git -C "$d" -c user.name=prover -c user.email=prover@lab.invalid merge -q --no-ff --no-edit FETCH_HEAD >/dev/null 2>&1; then
    echo "$repo test-merge clean $(git -C "$d" rev-parse HEAD) (dev $(git -C "$d" rev-parse origin/dev))"
  else
    git -C "$d" merge --abort
    git -C "$d" switch -q --detach FETCH_HEAD
    echo "$repo test-merge CONFLICTS with dev; building the raw head $(git -C "$d" rev-parse HEAD)"
  fi
}
fetch_pr() { # dir repo pr expected -> merges PR head into dev (no-ff), falls back to raw head on conflict
  local d=$1 repo=$2 pr=$3 exp=$4
  git clone -q --branch dev "https://github.com/OmniNode-ai/$repo.git" "$d"
  git -C "$d" fetch -q origin "pull/$pr/head"
  local h; h=$(git -C "$d" rev-parse FETCH_HEAD)
  echo "$repo#$pr fetched head $h expected $exp match=$([ "$h" = "$exp" ] && echo yes || echo NO)"
  pr_subject "$d" "$repo" "$pr" "$h" "$(git -C "$d" rev-parse origin/dev)"
  test_merge_head "$d" "$repo"
}
fetch_ref() { # dir repo sha -> merges a branch head, named by its full sha, into dev; no PR (OMN-20085)
  local d=$1 repo=$2 sha=$3
  git clone -q --branch dev "https://github.com/OmniNode-ai/$repo.git" "$d"
  git -C "$d" fetch -q origin "$sha"
  local h; h=$(git -C "$d" rev-parse FETCH_HEAD)
  echo "$repo@${sha:0:12} fetched head $h expected $sha match=$([ "$h" = "$sha" ] && echo yes || echo NO)"
  test_merge_head "$d" "$repo"
}

# A group of pull requests is proved on the commit the merge queue would land:
# the group base (dev, or the exact dev sha the calling lane planned on) with
# each member squashed on in queue order, one commit per member. This is how the
# omnibase_infra dev merge queue squashes, and the sequence omnimarket creates
# when its PRs squash onto dev one at a time with no interleaving merge
# (OMN-18893). The author, committer and dates are fixed, so the same base and
# heads give the same commit shas here and in prepr_pool_group.py on the lane's
# machine; the tree hash is what the landed commit is later compared with. A
# member that does not squash cleanly fails the clone: members are chosen with
# disjoint files, so a conflict is a planning error, never something to build
# around.
GROUP_GIT_ENV="GIT_AUTHOR_NAME=lab-pool-group GIT_AUTHOR_EMAIL=lab-pool-group@lab.invalid GIT_COMMITTER_NAME=lab-pool-group GIT_COMMITTER_EMAIL=lab-pool-group@lab.invalid GIT_AUTHOR_DATE=2026-01-01T00:00:00+0000 GIT_COMMITTER_DATE=2026-01-01T00:00:00+0000"
fetch_group() { # dir repo "n:head n:head ..." [base] [expected tree]
  local d=$1 repo=$2 members=$3 base=${4:-} want=${5:-} m n exp h
  git clone -q --branch dev "https://github.com/OmniNode-ai/$repo.git" "$d"
  if [ -n "$base" ]; then
    git -C "$d" switch -q --detach "$base" || { echo "group-base-missing $repo $base"; return 1; }
  fi
  echo "$repo group base $(git -C "$d" rev-parse HEAD) (dev $(git -C "$d" rev-parse origin/dev))"
  for m in $members; do
    n=${m%%:*}; exp=${m#*:}
    git -C "$d" fetch -q origin "pull/$n/head" || { echo "group-fetch-failed $repo#$n"; return 1; }
    h=$(git -C "$d" rev-parse FETCH_HEAD)
    echo "$repo#$n fetched head $h expected $exp match=$([ "$h" = "$exp" ] && echo yes || echo NO)"
    pr_subject "$d" "$repo" "$n" "$h" "$(git -C "$d" rev-parse origin/dev)"
    # git wants an identity for a squash merge too; a host may have none set
    # shellcheck disable=SC2086
    if ! env $GROUP_GIT_ENV git -C "$d" merge -q --squash FETCH_HEAD > "$d.merge.log" 2>&1; then
      echo "group-conflict $repo#$n $(grep -m1 -iE 'conflict|fatal|error' "$d.merge.log")"; return 1
    fi
    # shellcheck disable=SC2086
    env $GROUP_GIT_ENV git -C "$d" commit -q --allow-empty -m "lab-pool group member $repo#$n $h" \
      || { echo "group-commit-failed $repo#$n"; return 1; }
    if git -C "$d" diff --quiet HEAD^ HEAD; then echo "group-empty $repo#$n (its diff is already on the base)"; fi
    echo "group-step $repo#$n commit $(git -C "$d" rev-parse HEAD) tree $(git -C "$d" rev-parse 'HEAD^{tree}')"
  done
  local tree agrees count
  tree=$(git -C "$d" rev-parse 'HEAD^{tree}')
  count=$(echo "$members" | wc -w | tr -d ' ')
  if [ -z "$want" ]; then agrees=unchecked; elif [ "$tree" = "$want" ]; then agrees=yes; else agrees=NO; fi
  echo "group-commit $repo $(git -C "$d" rev-parse HEAD) tree $tree base $(git -C "$d" rev-parse "HEAD~$count") tree-agrees=$agrees"
}

case "$PHASE" in
snap-pre|snap-post)
  out=/tmp/$TAG-$PHASE.txt
  if [ "$KIND" = prepr-slot ]; then
    # .201 runs a dozen declared lanes that rebuild and restart on their own, so a
    # whole-host diff would never read zero. Snapshot the slot's own axes; the
    # shared-server axes are prepr_teardown_slot.sh's readback.
    { ts; uptime
      echo "## ps"; docker ps -a --filter label=com.docker.compose.project=$P --format '{{.Names}} {{.ID}}' | sort
      echo "## volumes"; docker volume ls -q --filter label=com.docker.compose.project=$P | sort
      echo "## networks"; docker network ls --filter label=com.docker.compose.project=$P --format '{{.Name}}' | sort
      echo "## listeners"; lsn | grep -E ":($PORTS)\b" || echo none
    } > "$out"
    neighbour_snapshot > "/tmp/$TAG-$PHASE-neighbours.txt"
    if [ "$PHASE" = snap-post ]; then
      neighbour_diff "/tmp/$TAG-snap-pre-neighbours.txt" "/tmp/$TAG-snap-post-neighbours.txt"
    fi
  else
  { ts; uptime
    echo "## ps"; docker ps -a --format '{{.Names}} {{.ID}} {{.Label "com.docker.compose.project"}}' | sort
    echo "## volumes"; docker volume ls -q | sort
    echo "## networks"; docker network ls --format '{{.Name}}' | sort
    echo "## images"; docker images -q | sort -u
    echo "## listeners"; lsn | grep -E ":($PORTS)\b" || echo none
  } > "$out"
  fi
  if [ "$PHASE" = snap-post ] && [ -f "/tmp/$TAG-snap-pre.txt" ]; then
    D=$(diff <(sed '1,2d' "/tmp/$TAG-snap-pre.txt") <(sed '1,2d' "$out") | grep -E '^[<>]')
    echo "snapshot-diff=$(printf '%s' "$D" | grep -c .)"; printf '%s\n' "$D" | head -20
  fi
  echo "$PHASE lines=$(wc -l < "$out") local-containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') listeners=$(grep -cE ":($PORTS)\b" "$out")"
  ;;
clone)
  drop_source_trees
  echo "clone start $(ts)"; mkdir -p "$R" "$T"
  if [ -n "${INFRA_GROUP:-}" ]; then
    fetch_group "$R/omnibase_infra" omnibase_infra "$INFRA_GROUP" "${INFRA_GROUP_BASE:-}" "${INFRA_GROUP_TREE:-}" || { echo "clone failed: group did not build"; exit 1; }
    # the focused tests run against the group commit itself
    git clone -q "https://github.com/OmniNode-ai/omnibase_infra.git" "$T/omnibase_infra"; git -C "$T/omnibase_infra" fetch -q "$R/omnibase_infra" HEAD; git -C "$T/omnibase_infra" switch -q --detach FETCH_HEAD
  elif [ -n "${INFRA_REF_SHA:-}" ]; then fetch_ref "$R/omnibase_infra" omnibase_infra "$INFRA_REF_SHA"
    git clone -q "https://github.com/OmniNode-ai/omnibase_infra.git" "$T/omnibase_infra"; git -C "$T/omnibase_infra" fetch -q origin "$INFRA_REF_SHA"; git -C "$T/omnibase_infra" switch -q --detach FETCH_HEAD
  elif [ -n "${INFRA_PR:-}" ]; then fetch_pr "$R/omnibase_infra" omnibase_infra "$INFRA_PR" "$INFRA_HEAD"
    git clone -q "https://github.com/OmniNode-ai/omnibase_infra.git" "$T/omnibase_infra"; git -C "$T/omnibase_infra" fetch -q origin "pull/$INFRA_PR/head"; git -C "$T/omnibase_infra" switch -q --detach FETCH_HEAD
  else git clone -q --branch dev https://github.com/OmniNode-ai/omnibase_infra.git "$R/omnibase_infra"; fi
  if [ -n "${MARKET_GROUP:-}" ]; then
    fetch_group "$R/omnimarket" omnimarket "$MARKET_GROUP" "${MARKET_GROUP_BASE:-}" "${MARKET_GROUP_TREE:-}" || { echo "clone failed: group did not build"; exit 1; }
    # the focused tests run against the group commit itself
    git clone -q "https://github.com/OmniNode-ai/omnimarket.git" "$T/omnimarket"; git -C "$T/omnimarket" fetch -q "$R/omnimarket" HEAD; git -C "$T/omnimarket" switch -q --detach FETCH_HEAD
  elif [ -n "${MARKET_REF_SHA:-}" ]; then fetch_ref "$R/omnimarket" omnimarket "$MARKET_REF_SHA"
    git clone -q "https://github.com/OmniNode-ai/omnimarket.git" "$T/omnimarket"; git -C "$T/omnimarket" fetch -q origin "$MARKET_REF_SHA"; git -C "$T/omnimarket" switch -q --detach FETCH_HEAD
  elif [ -n "${MARKET_PR:-}" ]; then fetch_pr "$R/omnimarket" omnimarket "$MARKET_PR" "$MARKET_HEAD"
    git clone -q "https://github.com/OmniNode-ai/omnimarket.git" "$T/omnimarket"; git -C "$T/omnimarket" fetch -q origin "pull/$MARKET_PR/head"; git -C "$T/omnimarket" switch -q --detach FETCH_HEAD
  else git clone -q --branch dev https://github.com/OmniNode-ai/omnimarket.git "$R/omnimarket"; fi
  git clone -q --branch main https://github.com/OmniNode-ai/omnibase_compat.git "$R/omnibase_compat"
  git clone -q --branch dev https://github.com/OmniNode-ai/omnibase_core.git "$R/omnibase_core"
  git clone -q --branch dev https://github.com/OmniNode-ai/omnibase_spi.git "$R/omnibase_spi"
  # The workspace build vendors each sibling from these clones, not from the PR's
  # pin, so a PR that moves a sibling pin names the ref here (CORE_REF=v0.47.24).
  for pair in "omnibase_core:${CORE_REF:-}" "omnibase_spi:${SPI_REF:-}" "omnibase_compat:${COMPAT_REF:-}"; do
    repo=${pair%%:*}; ref=${pair#*:}; [ -n "$ref" ] || continue
    git -C "$R/$repo" fetch -q --tags origin "$ref" && git -C "$R/$repo" switch -q --detach FETCH_HEAD
    echo "$repo pinned to $ref = $(git -C "$R/$repo" rev-parse HEAD)"
  done
  for d in "$R"/* "$T"/*; do [ -d "$d" ] && echo "$d $(git -C "$d" rev-parse HEAD)"; done
  echo "clone done $(ts)"
  ;;
build)
  if [ "$KIND" = prepr-slot ]; then
    # The sanctioned .201 entrypoint, from the test-merge tree. Its siblings
    # resolve from the ticket directory ($R), where clone put every one.
    cd "$R/omnibase_infra" || exit 1
    if [ -n "${OVERLAY_FILE:-}" ]; then
      # A run that names an overlay swaps the slot's Bifrost overlay: the slot
      # mounts ${ONEX_LANE_OVERLAY_DIR}/<project>.bifrost.yaml, so a private
      # directory holding only that file replaces the workspace one.
      mkdir -p "$W/lane-overlay"
      cp "$OVERLAY_FILE" "$W/lane-overlay/$P.bifrost.yaml" || { echo "build rc=1 overlay copy failed"; exit 1; }
      export ONEX_LANE_OVERLAY_DIR="$W/lane-overlay"
      echo "overlay mounted as $P.bifrost.yaml sha256=$(shasum -a 256 "$OVERLAY_FILE" | cut -c1-12)"
    fi
    echo "slot $PREPR_SLOT bring-up start $(ts) GIT_SHA=$(git rev-parse HEAD)"
    OMNI_HOME="$R" bash scripts/runtime_build/prepr_verify_lane.sh --slot "$PREPR_SLOT" \
      --worktree "$R/omnibase_infra" --reason "${REASON:?REASON is required on a prepr slot}" \
      --descriptor-out "$W/descriptor.json" > "$W/build.log" 2>&1
    src=$?; echo "prepr_verify_lane rc=$src $(ts)"; grep '^\[prepr-verify-lane\]' "$W/build.log" | tail -6
    case $src in
      0) echo "build rc=0"; echo "up rc=0"; : > "$W/migrated" ;;
      8) # another holder owns the slot: never tear down what this run did not start
         : > "$W/foreign-slot"; echo "build rc=8 slot lock held by another process" ;;
      9) if grep -q 'migration failed' "$W/build.log"; then
           echo "build rc=0"; echo "slot-migration FAILED"; echo "up rc=9"
         else echo "build rc=9 slot provisioning failed"; fi ;;
      11|12) echo "build rc=0"; echo "up rc=$src" ;;
      *) echo "build rc=$src" ;;
    esac
    [ "$src" = 0 ] || exit 1
    exit 0
  fi
  export DEPLOY_SOURCE_REFS_OUT=$W/refs.json
  cd "$R/omnibase_infra" || exit 1
  echo "stage start $(ts)"
  OMNI_HOME="$R" CONSUMER_LOCK="$R/omnimarket/uv.lock" bash scripts/runtime_build/stage_workspace.sh \
    --repo-ref omnibase_core="$(git -C "$R/omnibase_core" rev-parse HEAD)" \
    --repo-ref omnibase_compat="$(git -C "$R/omnibase_compat" rev-parse HEAD)" \
    --repo-ref omnimarket="$(git -C "$R/omnimarket" rev-parse HEAD)" 2>&1 | tail -5
  echo "stage rc=${PIPESTATUS[0]} $(ts)"
  make local-env LOCAL_ENV_FILE="$W/local.env" LOCAL_OVERLAY_FILE="$W/local.bifrost.yaml" 2>&1 | tail -2
  # the model rung comes from the pool config (config/lab/prepr_runtime_pool.yaml under the workspace config root)
  : "${MODEL_ENDPOINT:?MODEL_ENDPOINT is required}"
  sed -i.bak -E "s|endpoint_url: &model_endpoint \"[^\"]*\"|endpoint_url: \\&model_endpoint \"${MODEL_ENDPOINT}\"|" "$W/local.bifrost.yaml"
  grep -n 'model_endpoint "' "$W/local.bifrost.yaml"
  if [ -n "${OVERLAY_FILE:-}" ]; then
    # the named overlay replaces the generated one whole, after the pool's model endpoint
    cp "$OVERLAY_FILE" "$W/local.bifrost.yaml" || { echo "build rc=1 overlay copy failed"; exit 1; }
    echo "overlay replaces local.bifrost.yaml sha256=$(shasum -a 256 "$OVERLAY_FILE" | cut -c1-12)"
  fi
  uv run -q python -m omnibase_infra.docker.catalog.cli generate local --env-file "$W/local.env" 2>&1 | tail -2
  RV=$(grep -m1 '^version' pyproject.toml | sed -E 's/.*"(.*)".*/\1/')
  if [ "${DOCKER_CONFIG_MODE:-host}" = isolated ]; then
    # A macOS host whose docker credsStore lives in the login keychain cannot
    # build from a non-interactive ssh session (the keychain will not unlock:
    # drain-runtime-token-83 on .200, 2026-09-27). Build with a private config
    # that has no credential store, keeping the plugin, context and buildx dirs.
    DC="$W/docker-config"; mkdir -p "$DC"
    python3 -c 'import json,os,sys
src=os.path.expanduser("~/.docker/config.json")
try:
    cfg=json.load(open(src))
except (OSError, ValueError):
    cfg={}
for k in ("credsStore","credHelpers","auths"):
    cfg.pop(k, None)
# With no auths at all the CLI detects the platform default store again (the
# keychain on macOS), so keep one empty entry: it holds no credential and pins
# the plain file store (reproduced on a macOS lab host, 2026-10-16).
cfg["auths"]={"lab.invalid":{}}
json.dump(cfg, open(sys.argv[1],"w"))' "$DC/config.json"
    for d in cli-plugins contexts buildx; do [ -e "$HOME/.docker/$d" ] && ln -s "$HOME/.docker/$d" "$DC/$d"; done
    export DOCKER_CONFIG="$DC"; echo "docker config isolated at $DC (no credential store)"
  fi
  echo "build start $(ts) RUNTIME_VERSION=$RV GIT_SHA=$(git rev-parse HEAD)"
  docker compose -f docker/docker-compose.generated.yml --env-file docker/runtime-policy.env --env-file "$W/local.env" build \
    --build-arg BUILD_SOURCE=workspace --build-arg EXPECTED_BUILD_SOURCE=workspace \
    --build-arg OMNI_HOME="$R" --build-arg PROMOTION_CLASS=stability-candidate --build-arg NON_MAIN_LINEAGE=true \
    --build-arg GIT_SHA="$(git rev-parse HEAD)" --build-arg VCS_REF="$(git rev-parse HEAD)" --build-arg RUNTIME_VERSION="$RV" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$W/build.log" 2>&1
  brc=$?; echo "build rc=$brc $(ts)"; tail -4 "$W/build.log"
  if [ "$brc" != 0 ]; then
    grep -qi keychain "$W/build.log" && echo "build failed on the docker credential keychain: set docker_config: isolated for this host"
    exit 1
  fi
  docker images --filter reference="$P*" --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}'
  echo "up start $(ts)"
  uv run -q python -m omnibase_infra.docker.catalog.cli up local --env-file "$W/local.env" > "$W/up.log" 2>&1
  urc=$?; echo "up rc=$urc $(ts)"; tail -6 "$W/up.log"
  [ "$urc" = 0 ] || exit 1
  ;;
probe)
  echo "probe $(ts)"
  # `up` returns as soon as the containers start; the runtimes take minutes to
  # discover contracts and join their groups. Wait until no runtime container is
  # still in its health start period and both ports answer, or PROBE_WAIT_S.
  start=$(date +%s); end=$(( start + ${PROBE_WAIT_S:-1500} ))
  while :; do
    starting=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}} {{.Status}}' | grep -- '-runtime' | grep -c 'health: starting')
    c1=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "localhost:$MAIN/health")
    c2=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "localhost:$EFF/health")
    if [ "$starting" = 0 ] && [ "$c1" != 000 ] && [ "$c2" != 000 ]; then break; fi
    if [ "$(date +%s)" -ge "$end" ]; then echo "health wait timed out"; break; fi
    sleep 15
  done
  echo "health wait $(( $(date +%s) - start ))s ended $(ts) starting=$starting $MAIN=$c1 $EFF=$c2"
  if [ "$KIND" = prepr-slot ]; then slot_ports_readback; fi
  echo "== identity"
  # the label is printed for the reader and never judged: it records intent only
  echo "image-revision-label $(docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$M" 2>&1 | tail -1)"
  echo "build GIT_SHA $(git -C "$R/omnibase_infra" rev-parse HEAD) market $(git -C "$R/omnimarket" rev-parse HEAD)"
  # every repository a PR is under test in has its whole installed package
  # compared with its build tree, in both runtimes, so identity never depends
  # on which files a caller listed in ID_FILES
  SUBJECTS=""
  [ -n "${INFRA_PR:-}${INFRA_GROUP:-}${INFRA_REF_SHA:-}" ] && SUBJECTS="omnibase_infra"
  [ -n "${MARKET_PR:-}${MARKET_GROUP:-}${MARKET_REF_SHA:-}" ] && SUBJECTS="$SUBJECTS omnimarket"
  for repo in $SUBJECTS; do
    echo "identity-subject $repo"
    echo "src-diff $repo files=$(git -C "$R/$repo" diff --name-only origin/dev...HEAD -- src/ 2>/dev/null | grep -c .)"
    for c in $M $E; do
      docker exec "$c" python -c "$IMG_LIST_PY" "$repo" > "/tmp/$TAG-img-$repo-$c.txt" 2> "/tmp/$TAG-img-$repo-$c.err" \
        || echo "pkg-tree-list-error $repo $c $(tail -1 "/tmp/$TAG-img-$repo-$c.err")"
      python3 -c "$PKG_TREE_PY" "$R/$repo" "$repo" "$c" "/tmp/$TAG-img-$repo-$c.txt" 2>&1 | tail -16
    done
  done
  for c in $M $E; do echo "$c omnimarket=$(docker exec $c python -c "import importlib.metadata as m; print(m.version('omnimarket'))" 2>&1 | tail -1) infra=$(docker exec $c python -c "import importlib.metadata as m; print(m.version('omnibase_infra'))" 2>&1 | tail -1)"; done
  for spec in ${ID_FILES:-}; do repo=${spec%%:*}; f=${spec#*:}; pkg=${repo}
    want=$(git -C "$R/$repo" show "HEAD:src/$f" 2>/dev/null | shasum -a 256 | cut -c1-12)
    got=$(docker exec $M python -c "import $pkg, hashlib, os; b=os.path.dirname(os.path.dirname($pkg.__file__)); print(hashlib.sha256(open(b+'/$f','rb').read()).hexdigest()[:12])" 2>&1 | tail -1)
    echo "file $f image=$got build-tree=$want match=$([ "$got" = "$want" ] && echo yes || echo NO)"; done
  echo "== health"
  for p in $MAIN $EFF; do rm -f "/tmp/$TAG-h$p.json"; code=$(curl -s -o "/tmp/$TAG-h$p.json" -w '%{http_code}' --max-time 10 "localhost:$p/health")
    echo "port $p HTTP $code $(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (OSError, ValueError):
    d={}
det=d.get("details") if isinstance(d.get("details"),dict) else {}
print("status",d.get("status"),"healthy",det.get("healthy"),"failed_handlers",det.get("failed_handlers"))' "/tmp/$TAG-h$p.json")"; done
  # a slot has no migration-gate container: prepr_verify_lane.sh ran both
  # migrations against its fresh databases and exits non-zero when one fails
  [ "$KIND" = prepr-slot ] && echo "slot-migration-gate health=$([ -f "$W/migrated" ] && echo healthy || echo unknown)"
  docker ps -aq --filter label=com.docker.compose.project=$P | xargs docker inspect -f '{{.Name}} health={{if .State.Health}}{{.State.Health.Status}}{{end}} restarts={{.RestartCount}} {{.State.Status}} exit={{.State.ExitCode}}'
  if [ -z "${LIVE_GREP:-}" ]; then
    LIVE_GREP=$(derive_live_grep $SUBJECTS)
    LIVE_NA=$(derive_live_not_applicable $SUBJECTS)
    echo "live-grep derived from the diff: ${LIVE_GREP:-none (${LIVE_NA:+every changed node declares no topics and no runtime profile}${LIVE_NA:-the PR changes no node that exists at its head})}"
    if [ -n "$LIVE_NA" ]; then echo "live-grep not-applicable: ${LIVE_NA% } (contract declares no topics and no runtime profile)"; fi
  fi
  echo "== wiring"
  for c in $M $E; do L=$(docker logs $c 2>&1); echo "$c lines=$(echo "$L" | wc -l | tr -d ' ') autowire-fail=$(echo "$L" | grep -c 'Auto-wiring failed for') dup-dispatcher=$(echo "$L" | grep -c 'Cannot register duplicate dispatcher ID') ERROR=$(echo "$L" | grep -c ERROR) Traceback=$(echo "$L" | grep -c Traceback)"
    echo "$L" | grep -E 'ERROR|Traceback|Auto-wiring failed' | sed -E 's/^[0-9T:.,Z -]+//' | cut -c1-160 | sort | uniq -c | sort -rn | head -4
    # the contract names the non-strict wiring pass gave up on, so a run can be
    # compared with a base control at dev (a failure dev already has is not the PR's)
    echo "failed-contracts $c: $(echo "$L" | grep 'Auto-wiring failed for' | sed -E 's/.*enforce\): //' | awk -v RS='; ' -F': ' 'NF>1{print $1}' | sort -u | tr '\n' ' ')"
    echo "$L" | grep 'Auto-wiring failed for' | sed -E 's/.*enforce\): //' | awk -v RS='; ' 'NF{sub(/^ +/, ""); print}' \
      | sed -E "s/^([^:]+): .*failed: (.*)$/  failed-contract-reason $c \\1: \\2/" | cut -c1-260
    # noglob: a derived term carries a bracket class, never a file pattern
    set -f; for n in ${LIVE_GREP:-}; do echo "  $c mentions $n: $(echo "$L" | grep -cE "$n")"; done; set +f; done
  if [ "$KIND" = prepr-slot ] && { [ -n "${SQL:-}" ] || [ -n "${GROUP_GREP:-}" ]; }; then
    echo "== sql and consumer groups: not read on a prepr slot (its databases and groups live on the dev lane's shared servers)"
  elif [ -n "${SQL:-}" ]; then echo "== sql"; PG=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 postgres)
    for db in $(docker exec "$PG" psql -U postgres -Atc "select datname from pg_database where datname not in ('postgres','template0','template1')"); do
      echo "-- db $db"; docker exec "$PG" psql -U postgres -d "$db" -Atc "$SQL" 2>&1 | head -12; done; fi
  if [ "$KIND" != prepr-slot ] && [ -n "${GROUP_GREP:-}" ]; then echo "== consumer groups"; RP=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 'redpanda$')
    docker exec "$RP" rpk group list 2>&1 | grep -E "$GROUP_GREP" | head -10; echo "groups matching: $(docker exec "$RP" rpk group list 2>&1 | grep -cE "$GROUP_GREP") of $(docker exec "$RP" rpk group list 2>&1 | wc -l)"; fi
  uptime
  ;;
post-bringup)
  # OMN-20117: one command the caller names, run on the host once the stack is up
  # and probed, from the test-merge tree. It is how a proof restarts a runtime
  # during a live call on a pre-PR slot. The command gets the member's own names
  # and ports and nothing else of the host's; its exit status is printed on a
  # line of its own for the judge, and this phase itself exits 0 so teardown
  # reads the same either way. The driver's phase budget bounds its run time.
  if [ -z "${POST_BRINGUP_CMD:-}" ]; then
    echo "no-post-bringup: the params name no POST_BRINGUP_CMD"
    exit 0
  fi
  cd "$R/omnibase_infra" || exit 1
  echo "post-bringup start $(ts) project=$P main=$M effects=$E ports=$MAIN,$EFF"
  echo "post-bringup cmd: $POST_BRINGUP_CMD"
  PROJECT=$P MAIN_RUNTIME=$M EFFECTS_RUNTIME=$E MAIN_PORT=$MAIN EFFECTS_PORT=$EFF \
    bash -c "$POST_BRINGUP_CMD" > "$W/post-bringup.out" 2>&1
  prc=$?
  tail -40 "$W/post-bringup.out"
  echo "post-bringup rc=$prc $(ts)"
  exit 0
  ;;
tests)
  if [ -z "${TESTS:-}" ] && [ -z "${SUITE_REPO:-}" ]; then
    echo "no-focused-tests: the PR changes no test files"
    exit 0
  fi
  for spec in ${TESTS:-}; do repo=${spec%%:*}; f=${spec#*:}; echo "$repo $f" >> /tmp/$TAG-tests.lst; done
  # A suite run (OMN-20085) names the repo whose unit suite runs whole, on its own
  # budget (the driver's); it replaces the repo's file list.
  if [ -n "${SUITE_REPO:-}" ]; then echo "$SUITE_REPO tests/unit" >> /tmp/$TAG-tests.lst; fi
  for repo in $(cut -d' ' -f1 /tmp/$TAG-tests.lst 2>/dev/null | sort -u); do
    cd "$T/$repo" || continue
    FILES=$(awk -v r=$repo '$1==r{print $2}' /tmp/$TAG-tests.lst | tr '\n' ' ')
    KIND_WORD=focused; XARGS=""
    if [ "$repo" = "${SUITE_REPO:-}" ]; then KIND_WORD=suite; XARGS="-n ${SUITE_WORKERS:-4}"; fi
    echo "== $repo at $(git rev-parse HEAD) sync $(ts)"; uv sync -q --frozen 2>&1 | tail -2
    uv run --frozen pytest $FILES $XARGS -q -rf -p no:cacheprovider > "/tmp/$TAG-$repo-pytest.out" 2>&1; frc=$?
    tail -6 "/tmp/$TAG-$repo-pytest.out"; echo "$repo $KIND_WORD rc=$frc $(ts)"
    # Dev control: every test that failed here is run again at dev in the same
    # clone and on the same host. A test that fails at dev too (a host git
    # version, say) is dev-inherited, not the PR's; one that passes at dev, or
    # does not exist there, is the PR's. The control is bounded (DEV_CONTROL_MAX,
    # default 25 ids): a failure past the bound has no control line and so stays
    # the PR's, which keeps a broken suite from rerunning itself one id at a time.
    FAILED=$(sed -n 's/^FAILED \([^ ]*\).*/\1/p' "/tmp/$TAG-$repo-pytest.out" | sort -u)
    if [ "$frc" != 0 ] && [ -n "$FAILED" ]; then
      HEADSHA=$(git rev-parse HEAD)
      git fetch -q origin dev && git switch -q --detach FETCH_HEAD && uv sync -q --frozen 2>&1 | tail -1
      n=0
      for id in $FAILED; do
        if [ "$n" -ge "${DEV_CONTROL_MAX:-25}" ]; then echo "dev-control-bound $repo skipped=$(( $(echo $FAILED | wc -w) - n )) bound=${DEV_CONTROL_MAX:-25}"; break; fi
        n=$((n + 1))
        uv run --frozen pytest "$id" -q -p no:cacheprovider > /dev/null 2>&1; drc=$?
        echo "dev-control $repo $id rc=$drc at $(git rev-parse HEAD)"
      done
      git switch -q --detach "$HEADSHA"
    fi
  done
  ;;
exercise)
  # The caller's command, run once against the live stack after the probe, from
  # the build tree. The driver judges it by the rc line below, so this phase
  # itself exits 0 once the command has run.
  : "${POST_UP_COMMAND:?POST_UP_COMMAND is required for the exercise phase}"
  cd "$R/omnibase_infra" || { echo "exercise rc=1 no build tree at $R/omnibase_infra"; exit 0; }
  export MAIN_PORT=$MAIN EFFECTS_PORT=$EFF PROJECT=$P
  echo "exercise start $(ts) command: $POST_UP_COMMAND"
  bash -c "$POST_UP_COMMAND" 2>&1; erc=$?
  echo "exercise rc=$erc $(ts)"
  ;;
teardown)
  if [ "$KIND" = prepr-slot ]; then
    if [ -f "$W/foreign-slot" ]; then
      echo "slot $PREPR_SLOT belongs to another holder: not torn down"; SV=FOREIGN; SI=0
    else
      TD="$R/omnibase_infra/scripts/runtime_build/prepr_teardown_slot.sh"
      if [ ! -f "$TD" ]; then
        git clone -q --depth 1 --branch dev https://github.com/OmniNode-ai/omnibase_infra.git "$W/td"
        TD="$W/td/scripts/runtime_build/prepr_teardown_slot.sh"
      fi
      bash "$TD" --slot "$PREPR_SLOT" --reason "${REASON:-lab pool teardown $TAG}" --report-out "/tmp/$TAG-slot-teardown.json" > "/tmp/$TAG-slot-teardown.log" 2>&1
      echo "prepr_teardown_slot rc=$? $(ts)"; tail -4 "/tmp/$TAG-slot-teardown.log"
      read -r SV SI <<EOF_TD
$(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (OSError, ValueError):
    d={}
r=d.get("residue") or {}
print(d.get("verdict","MISSING"), r.get("images",0))' "/tmp/$TAG-slot-teardown.json")
EOF_TD
      python3 -c 'import json,sys
try:
    print("slot residue", json.dumps(json.load(open(sys.argv[1])).get("residue")))
except (OSError, ValueError):
    print("slot residue unreadable")' "/tmp/$TAG-slot-teardown.json"
    fi
    echo "slot-teardown verdict=$SV"
    drop_source_trees
    cd "$HOME" || exit 1; rm -rf "$W"
    echo "== zero residue $(ts)"
    echo "containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') volumes=$(docker volume ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') networks=$(docker network ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') images=$SI listeners=$(lsn | grep -cE ":($PORTS)\b") workdir=$([ -e "$W" ] && echo present || echo gone)"
    echo "positive control $PC containers $(docker ps -a --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ') running $(docker ps --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ')"
    exit 0
  fi
  IMGS=$(docker images --filter reference="$P*" -q | sort -u)
  C=$(docker ps -a --filter label=com.docker.compose.project=$P -q); echo "containers $(echo $C | wc -w)"; [ -n "$C" ] && docker rm -f $C >/dev/null; echo "rm rc=$?"
  V=$(docker volume ls --filter label=com.docker.compose.project=$P -q); echo "volumes $(echo $V | wc -w)"; [ -n "$V" ] && docker volume rm $V >/dev/null; echo "vol rm rc=$?"
  N=$(docker network ls --filter label=com.docker.compose.project=$P -q); echo "networks $(echo $N | wc -w)"; [ -n "$N" ] && docker network rm $N >/dev/null; echo "net rm rc=$?"
  for i in $IMGS; do docker image rm -f "$i" >/dev/null; echo "image rm $i rc=$?"; done
  # Anonymous volumes an image declares carry no compose label. Remove only the ones
  # that are dangling now AND were absent from this run's pre-snapshot, so a volume
  # that was already on the host is never touched.
  if [ -f "/tmp/$TAG-snap-pre.txt" ]; then
    PRE=$(sed -n '/^## volumes/,/^## networks/p' "/tmp/$TAG-snap-pre.txt" | grep -v '^##')
    for v in $(docker volume ls -qf dangling=true); do
      echo "$PRE" | grep -qxF "$v" || { docker volume rm "$v" >/dev/null; echo "anon volume rm $v rc=$?"; }
    done
  else echo "no pre-snapshot: anonymous volumes left as they are"; fi
  drop_source_trees
  cd "$HOME" || exit 1; rm -rf "$W"
  echo "== zero residue $(ts)"
  echo "containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') volumes=$(docker volume ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') networks=$(docker network ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') images=$(docker images --filter reference="$P*" -q | wc -l | tr -d ' ') listeners=$(lsn | grep -cE ":($PORTS)\b") workdir=$([ -e "$W" ] && echo present || echo gone)"
  echo "positive control $PC containers $(docker ps -a --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ') running $(docker ps --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ')"
  ;;
*) echo "unknown phase $PHASE"; exit 2;;
esac
