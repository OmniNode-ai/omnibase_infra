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
#   phases: snap-pre clone build probe tests teardown snap-post
#
# It only ever touches its own compose project and the work directory named by W
# in the params file. Every other project on the host (a dogfood lane, a runner,
# a declared lane) is read for a positive control and never written.
#
# Two kinds of member (config/prepr_runtime_pool.yaml):
#   isolated    the laptop-bundle project omnibase-infra-local, with its own
#               Postgres, Redpanda and Valkey on the fixed ports.
#   prepr-slot  a numbered .201 pre-PR slot (omnibase-infra-prepr-N), brought up
#               ONLY through prepr_verify_lane.sh and destroyed ONLY through
#               prepr_teardown_slot.sh, both from the test-merge tree. Neither
#               takes a lane argument; the project is derived from the slot.
#
# Params (sourced): TAG W [INFRA_PR INFRA_HEAD | INFRA_GROUP [INFRA_GROUP_BASE INFRA_GROUP_TREE]] [MARKET_PR MARKET_HEAD]
#   MODEL_ENDPOINT [ID_FILES LIVE_GREP GROUP_GREP SQL TESTS CORE_REF SPI_REF COMPAT_REF
#   DOCKER_CONFIG_MODE SLOT_KIND PREPR_SLOT PROJECT MAIN_PORT EFFECTS_PORT SLOT_PORTS
#   POSITIVE_CONTROL REASON]
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

# The facts a pr-head lab-proof receipt binds (OMN-19566): the proven head, the
# dev sha it was proven against, their merge base, and the digest of the PR's own
# diff. The digest is `git diff --no-color --no-ext-diff <merge-base>..<head>`
# hashed with sha256, byte for byte the command lab_pass_receipt.py's
# compute_pr_diff_digest_from_repo runs, so a verifier elsewhere recomputes the
# same value from the same two shas.
pr_subject() { # dir repo pr head base
  local d=$1 repo=$2 pr=$3 h=$4 base=$5 mb dg
  mb=$(git -C "$d" merge-base "$base" "$h" 2>/dev/null) || { echo "pr-subject-unavailable $repo#$pr no merge base with $base"; return 0; }
  dg=$(git -C "$d" diff --no-color --no-ext-diff "$mb..$h" | shasum -a 256 | cut -d' ' -f1)
  echo "pr-subject $repo#$pr head=$h base=$base merge-base=$mb diff-digest=sha256:$dg"
}

fetch_pr() { # dir repo pr expected -> merges PR head into dev (no-ff), falls back to raw head on conflict
  local d=$1 repo=$2 pr=$3 exp=$4
  git clone -q --branch dev "https://github.com/OmniNode-ai/$repo.git" "$d"
  git -C "$d" fetch -q origin "pull/$pr/head"
  local h; h=$(git -C "$d" rev-parse FETCH_HEAD)
  echo "$repo#$pr fetched head $h expected $exp match=$([ "$h" = "$exp" ] && echo yes || echo NO)"
  pr_subject "$d" "$repo" "$pr" "$h" "$(git -C "$d" rev-parse origin/dev)"
  if git -C "$d" -c user.name=prover -c user.email=prover@lab.invalid merge -q --no-ff --no-edit FETCH_HEAD >/dev/null 2>&1; then
    echo "$repo test-merge clean $(git -C "$d" rev-parse HEAD) (dev $(git -C "$d" rev-parse origin/dev))"
  else
    git -C "$d" merge --abort
    git -C "$d" switch -q --detach FETCH_HEAD
    echo "$repo test-merge CONFLICTS with dev; building the raw head $(git -C "$d" rev-parse HEAD)"
  fi
}

# A group of pull requests is proved on the commit the merge queue would land:
# the group base (dev, or the exact dev sha the calling lane planned on) with
# each member squashed on in queue order, one commit per member, the way the
# omnibase_infra dev merge queue squashes (OMN-18893). The author, committer and
# dates are fixed, so the same base and heads give the same commit shas here and
# in prepr_pool_group.py on the lane's machine; the tree hash is what the landed
# commit is later compared with. A member that does not squash cleanly fails the
# clone: members are chosen with disjoint files, so a conflict is a planning
# error, never something to build around.
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
  echo "clone start $(ts)"; mkdir -p "$R" "$T"
  if [ -n "${INFRA_GROUP:-}" ]; then
    fetch_group "$R/omnibase_infra" omnibase_infra "$INFRA_GROUP" "${INFRA_GROUP_BASE:-}" "${INFRA_GROUP_TREE:-}" || { echo "clone failed: group did not build"; exit 1; }
    # the focused tests run against the group commit itself
    git clone -q "https://github.com/OmniNode-ai/omnibase_infra.git" "$T/omnibase_infra"; git -C "$T/omnibase_infra" fetch -q "$R/omnibase_infra" HEAD; git -C "$T/omnibase_infra" switch -q --detach FETCH_HEAD
  elif [ -n "${INFRA_PR:-}" ]; then fetch_pr "$R/omnibase_infra" omnibase_infra "$INFRA_PR" "$INFRA_HEAD"
    git clone -q "https://github.com/OmniNode-ai/omnibase_infra.git" "$T/omnibase_infra"; git -C "$T/omnibase_infra" fetch -q origin "pull/$INFRA_PR/head"; git -C "$T/omnibase_infra" switch -q --detach FETCH_HEAD
  else git clone -q --branch dev https://github.com/OmniNode-ai/omnibase_infra.git "$R/omnibase_infra"; fi
  if [ -n "${MARKET_PR:-}" ]; then fetch_pr "$R/omnimarket" omnimarket "$MARKET_PR" "$MARKET_HEAD"
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
  # the model rung comes from the pool config (config/prepr_runtime_pool.yaml)
  : "${MODEL_ENDPOINT:?MODEL_ENDPOINT is required}"
  sed -i.bak -E "s|endpoint_url: &model_endpoint \"[^\"]*\"|endpoint_url: \\&model_endpoint \"${MODEL_ENDPOINT}\"|" "$W/local.bifrost.yaml"
  grep -n 'model_endpoint "' "$W/local.bifrost.yaml"
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
  echo "== identity"
  # the label is printed for the reader and never judged: it records intent only
  echo "image-revision-label $(docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$M" 2>&1 | tail -1)"
  echo "build GIT_SHA $(git -C "$R/omnibase_infra" rev-parse HEAD) market $(git -C "$R/omnimarket" rev-parse HEAD)"
  # every repository a PR is under test in has its whole installed package
  # compared with its build tree, in both runtimes, so identity never depends
  # on which files a caller listed in ID_FILES
  SUBJECTS=""
  [ -n "${INFRA_PR:-}${INFRA_GROUP:-}" ] && SUBJECTS="omnibase_infra"
  [ -n "${MARKET_PR:-}" ] && SUBJECTS="$SUBJECTS omnimarket"
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
  echo "== wiring"
  for c in $M $E; do L=$(docker logs $c 2>&1); echo "$c lines=$(echo "$L" | wc -l | tr -d ' ') autowire-fail=$(echo "$L" | grep -c 'Auto-wiring failed for') dup-dispatcher=$(echo "$L" | grep -c 'Cannot register duplicate dispatcher ID') ERROR=$(echo "$L" | grep -c ERROR) Traceback=$(echo "$L" | grep -c Traceback)"
    echo "$L" | grep -E 'ERROR|Traceback|Auto-wiring failed' | sed -E 's/^[0-9T:.,Z -]+//' | cut -c1-160 | sort | uniq -c | sort -rn | head -4
    # the contract names the non-strict wiring pass gave up on, so a run can be
    # compared with a base control at dev (a failure dev already has is not the PR's)
    echo "failed-contracts $c: $(echo "$L" | grep 'Auto-wiring failed for' | sed -E 's/.*enforce\): //' | awk -v RS='; ' -F': ' 'NF>1{print $1}' | sort -u | tr '\n' ' ')"
    echo "$L" | grep 'Auto-wiring failed for' | sed -E 's/.*enforce\): //' | awk -v RS='; ' 'NF{sub(/^ +/, ""); print}' \
      | sed -E "s/^([^:]+): .*failed: (.*)$/  failed-contract-reason $c \\1: \\2/" | cut -c1-260
    for n in ${LIVE_GREP:-}; do echo "  $c mentions $n: $(echo "$L" | grep -c "$n")"; done; done
  if [ "$KIND" = prepr-slot ] && { [ -n "${SQL:-}" ] || [ -n "${GROUP_GREP:-}" ]; }; then
    echo "== sql and consumer groups: not read on a prepr slot (its databases and groups live on the dev lane's shared servers)"
  elif [ -n "${SQL:-}" ]; then echo "== sql"; PG=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 postgres)
    for db in $(docker exec "$PG" psql -U postgres -Atc "select datname from pg_database where datname not in ('postgres','template0','template1')"); do
      echo "-- db $db"; docker exec "$PG" psql -U postgres -d "$db" -Atc "$SQL" 2>&1 | head -12; done; fi
  if [ "$KIND" != prepr-slot ] && [ -n "${GROUP_GREP:-}" ]; then echo "== consumer groups"; RP=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 'redpanda$')
    docker exec "$RP" rpk group list 2>&1 | grep -E "$GROUP_GREP" | head -10; echo "groups matching: $(docker exec "$RP" rpk group list 2>&1 | grep -cE "$GROUP_GREP") of $(docker exec "$RP" rpk group list 2>&1 | wc -l)"; fi
  uptime
  ;;
tests)
  for spec in ${TESTS:-}; do repo=${spec%%:*}; f=${spec#*:}; echo "$repo $f" >> /tmp/$TAG-tests.lst; done
  for repo in $(cut -d' ' -f1 /tmp/$TAG-tests.lst 2>/dev/null | sort -u); do
    cd "$T/$repo" || continue
    FILES=$(awk -v r=$repo '$1==r{print $2}' /tmp/$TAG-tests.lst | tr '\n' ' ')
    echo "== $repo at $(git rev-parse HEAD) sync $(ts)"; uv sync -q --frozen 2>&1 | tail -2
    uv run --frozen pytest $FILES -q -rf -p no:cacheprovider > "/tmp/$TAG-$repo-pytest.out" 2>&1; frc=$?
    tail -6 "/tmp/$TAG-$repo-pytest.out"; echo "$repo focused rc=$frc $(ts)"
    # Dev control: every test that failed here is run again at dev in the same
    # clone and on the same host. A test that fails at dev too (a host git
    # version, say) is dev-inherited, not the PR's; one that passes at dev, or
    # does not exist there, is the PR's.
    FAILED=$(sed -n 's/^FAILED \([^ ]*\).*/\1/p' "/tmp/$TAG-$repo-pytest.out" | sort -u)
    if [ "$frc" != 0 ] && [ -n "$FAILED" ]; then
      HEADSHA=$(git rev-parse HEAD)
      git fetch -q origin dev && git switch -q --detach FETCH_HEAD && uv sync -q --frozen 2>&1 | tail -1
      for id in $FAILED; do
        uv run --frozen pytest "$id" -q -p no:cacheprovider > /dev/null 2>&1; drc=$?
        echo "dev-control $repo $id rc=$drc at $(git rev-parse HEAD)"
      done
      git switch -q --detach "$HEADSHA"
    fi
  done
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
  cd "$HOME" || exit 1; rm -rf "$W"
  echo "== zero residue $(ts)"
  echo "containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') volumes=$(docker volume ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') networks=$(docker network ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') images=$(docker images --filter reference="$P*" -q | wc -l | tr -d ' ') listeners=$(lsn | grep -cE ":($PORTS)\b") workdir=$([ -e "$W" ] && echo present || echo gone)"
  echo "positive control $PC containers $(docker ps -a --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ') running $(docker ps --filter label=com.docker.compose.project=$PC -q | wc -l | tr -d ' ')"
  ;;
*) echo "unknown phase $PHASE"; exit 2;;
esac
