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
# It only ever touches the compose project omnibase-infra-local and the work
# directory named by W in the params file. Every other project on the host (a
# dogfood lane, a runner) is read for a positive control and never written.
#
# Params (sourced): TAG W [INFRA_PR INFRA_HEAD] [MARKET_PR MARKET_HEAD]
#   MODEL_ENDPOINT [ID_FILES LIVE_GREP GROUP_GREP SQL TESTS CORE_REF SPI_REF COMPAT_REF
#   DOCKER_CONFIG_MODE]
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
P=omnibase-infra-local; M=$P-omninode-runtime; E=$P-runtime-effects
R=$W/root; T=$W/test
ts() { date -u +%FT%TZ; }
lsn() { if command -v ss >/dev/null 2>&1; then ss -ltn; else lsof -nP -iTCP -sTCP:LISTEN; fi; }

fetch_pr() { # dir repo pr expected -> merges PR head into dev (no-ff), falls back to raw head on conflict
  local d=$1 repo=$2 pr=$3 exp=$4
  git clone -q --branch dev "https://github.com/OmniNode-ai/$repo.git" "$d"
  git -C "$d" fetch -q origin "pull/$pr/head"
  local h; h=$(git -C "$d" rev-parse FETCH_HEAD)
  echo "$repo#$pr fetched head $h expected $exp match=$([ "$h" = "$exp" ] && echo yes || echo NO)"
  if git -C "$d" -c user.name=prover -c user.email=prover@lab.invalid merge -q --no-ff --no-edit FETCH_HEAD >/dev/null 2>&1; then
    echo "$repo test-merge clean $(git -C "$d" rev-parse HEAD) (dev $(git -C "$d" rev-parse origin/dev))"
  else
    git -C "$d" merge --abort
    git -C "$d" switch -q --detach FETCH_HEAD
    echo "$repo test-merge CONFLICTS with dev; building the raw head $(git -C "$d" rev-parse HEAD)"
  fi
}

case "$PHASE" in
snap-pre|snap-post)
  out=/tmp/$TAG-$PHASE.txt
  { ts; uptime
    echo "## ps"; docker ps -a --format '{{.Names}} {{.ID}} {{.Label "com.docker.compose.project"}}' | sort
    echo "## volumes"; docker volume ls -q | sort
    echo "## networks"; docker network ls --format '{{.Name}}' | sort
    echo "## images"; docker images -q | sort -u
    echo "## listeners"; lsn | grep -E ':(5436|19092|16379|8085|8086)\b' || echo none
  } > "$out"
  if [ "$PHASE" = snap-post ] && [ -f "/tmp/$TAG-snap-pre.txt" ]; then
    D=$(diff <(sed '1,2d' "/tmp/$TAG-snap-pre.txt") <(sed '1,2d' "$out") | grep -E '^[<>]')
    echo "snapshot-diff=$(printf '%s' "$D" | grep -c .)"; printf '%s\n' "$D" | head -20
  fi
  echo "$PHASE lines=$(wc -l < "$out") local-containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') listeners=$(grep -cE ':(5436|19092|16379|8085|8086)\b' "$out")"
  ;;
clone)
  echo "clone start $(ts)"; mkdir -p "$R" "$T"
  if [ -n "${INFRA_PR:-}" ]; then fetch_pr "$R/omnibase_infra" omnibase_infra "$INFRA_PR" "$INFRA_HEAD"
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
    c1=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 localhost:8085/health)
    c2=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 localhost:8086/health)
    if [ "$starting" = 0 ] && [ "$c1" != 000 ] && [ "$c2" != 000 ]; then break; fi
    if [ "$(date +%s)" -ge "$end" ]; then echo "health wait timed out"; break; fi
    sleep 15
  done
  echo "health wait $(( $(date +%s) - start ))s ended $(ts) starting=$starting 8085=$c1 8086=$c2"
  echo "== identity"
  docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$M" 2>&1
  echo "build GIT_SHA $(git -C "$R/omnibase_infra" rev-parse HEAD) market $(git -C "$R/omnimarket" rev-parse HEAD)"
  for c in $M $E; do echo "$c omnimarket=$(docker exec $c python -c "import importlib.metadata as m; print(m.version('omnimarket'))" 2>&1 | tail -1) infra=$(docker exec $c python -c "import importlib.metadata as m; print(m.version('omnibase_infra'))" 2>&1 | tail -1)"; done
  for spec in ${ID_FILES:-}; do repo=${spec%%:*}; f=${spec#*:}; pkg=${repo}
    want=$(git -C "$R/$repo" show "HEAD:src/$f" 2>/dev/null | shasum -a 256 | cut -c1-12)
    got=$(docker exec $M python -c "import $pkg, hashlib, os; b=os.path.dirname(os.path.dirname($pkg.__file__)); print(hashlib.sha256(open(b+'/$f','rb').read()).hexdigest()[:12])" 2>&1 | tail -1)
    echo "file $f image=$got build-tree=$want match=$([ "$got" = "$want" ] && echo yes || echo NO)"; done
  echo "== health"
  for p in 8085 8086; do rm -f "/tmp/$TAG-h$p.json"; code=$(curl -s -o "/tmp/$TAG-h$p.json" -w '%{http_code}' --max-time 10 "localhost:$p/health")
    echo "port $p HTTP $code $(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (OSError, ValueError):
    d={}
det=d.get("details") if isinstance(d.get("details"),dict) else {}
print("status",d.get("status"),"healthy",det.get("healthy"),"failed_handlers",det.get("failed_handlers"))' "/tmp/$TAG-h$p.json")"; done
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
  if [ -n "${SQL:-}" ]; then echo "== sql"; PG=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 postgres)
    for db in $(docker exec "$PG" psql -U postgres -Atc "select datname from pg_database where datname not in ('postgres','template0','template1')"); do
      echo "-- db $db"; docker exec "$PG" psql -U postgres -d "$db" -Atc "$SQL" 2>&1 | head -12; done; fi
  if [ -n "${GROUP_GREP:-}" ]; then echo "== consumer groups"; RP=$(docker ps --filter label=com.docker.compose.project=$P --format '{{.Names}}' | grep -m1 'redpanda$')
    docker exec "$RP" rpk group list 2>&1 | grep -E "$GROUP_GREP" | head -10; echo "groups matching: $(docker exec "$RP" rpk group list 2>&1 | grep -cE "$GROUP_GREP") of $(docker exec "$RP" rpk group list 2>&1 | wc -l)"; fi
  uptime
  ;;
tests)
  for spec in ${TESTS:-}; do repo=${spec%%:*}; f=${spec#*:}; echo "$repo $f" >> /tmp/$TAG-tests.lst; done
  for repo in $(cut -d' ' -f1 /tmp/$TAG-tests.lst 2>/dev/null | sort -u); do
    cd "$T/$repo" || continue
    FILES=$(awk -v r=$repo '$1==r{print $2}' /tmp/$TAG-tests.lst | tr '\n' ' ')
    echo "== $repo at $(git rev-parse HEAD) sync $(ts)"; uv sync -q --frozen 2>&1 | tail -2
    uv run --frozen pytest $FILES -q -p no:cacheprovider 2>&1 | tail -6; echo "$repo focused rc=${PIPESTATUS[0]} $(ts)"
  done
  ;;
teardown)
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
  echo "containers=$(docker ps -a --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') volumes=$(docker volume ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') networks=$(docker network ls --filter label=com.docker.compose.project=$P -q | wc -l | tr -d ' ') images=$(docker images --filter reference="$P*" -q | wc -l | tr -d ' ') listeners=$(lsn | grep -cE ':(5436|19092|16379|8085|8086)\b') workdir=$([ -e "$W" ] && echo present || echo gone)"
  echo "positive control dogfood containers $(docker ps -a --filter label=com.docker.compose.project=omnibase-infra-dogfood -q | wc -l | tr -d ' ') running $(docker ps --filter label=com.docker.compose.project=omnibase-infra-dogfood -q | wc -l | tr -d ' ')"
  ;;
*) echo "unknown phase $PHASE"; exit 2;;
esac
