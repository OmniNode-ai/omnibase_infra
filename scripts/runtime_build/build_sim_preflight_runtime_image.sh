#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Build a local image from a private, content-addressed sim context.
set -euo pipefail
# Git hook location variables override -C and could target the invoking repo.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_OBJECT_DIRECTORY
unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE GIT_PREFIX

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT
readonly SOURCE_HOME="${OMNI_HOME:?OMNI_HOME is required}"
readonly CORE="${SOURCE_HOME}/omni_worktrees/OMN-19726/omnibase_core"
readonly MARKET="${SIM_PREFLIGHT_MARKET_SOURCE:-${SOURCE_HOME}/omni_worktrees/OMN-19728/omnimarket}"
readonly COMPAT="${SOURCE_HOME}/omnibase_compat"
readonly SPI="${SOURCE_HOME}/omnibase_spi"
readonly PINS="${ROOT}/docker/sim-preflight-runtime-source-pins.json"

stage_only=false
if [[ $# -eq 1 && "$1" = --stage-only ]]; then
  stage_only=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--stage-only]" >&2; exit 64
fi
if [[ "$stage_only" = true ]]; then
  [[ -n "${SIM_PREFLIGHT_STAGE_REPORT_OUT:-}" ]] || { echo "SIM_PREFLIGHT_STAGE_REPORT_OUT is required" >&2; exit 64; }
  [[ ! -e "$SIM_PREFLIGHT_STAGE_REPORT_OUT" ]] || { echo "refusing to overwrite stage report" >&2; exit 64; }
else
  [[ -n "${SIM_PREFLIGHT_IMAGE_PROVENANCE_OUT:-}" ]] || { echo "SIM_PREFLIGHT_IMAGE_PROVENANCE_OUT is required" >&2; exit 64; }
  [[ ! -e "$SIM_PREFLIGHT_IMAGE_PROVENANCE_OUT" ]] || { echo "refusing to overwrite image provenance" >&2; exit 64; }
fi
for source in "$ROOT" "$CORE" "$MARKET" "$COMPAT" "$SPI"; do
  [[ -e "$source/.git" ]] || { echo "required source checkout absent: $source" >&2; exit 65; }
done

cd "$ROOT"
UV_NO_SYNC=1 uv run python scripts/runtime_build/verify_sim_preflight_runtime_source_pins.py --profile "$PINS" >/dev/null
infra_head="$(git -C "$ROOT" rev-parse HEAD)"
core_head="$(git -C "$CORE" rev-parse HEAD)"
market_head="$(git -C "$MARKET" rev-parse HEAD)"
compat_head="$(git -C "$COMPAT" rev-parse HEAD)"
spi_head="$(git -C "$SPI" rev-parse HEAD)"
jq -e --arg infra "$infra_head" --arg core "$core_head" \
  '.sources.omnibase_infra == $infra and .sources.omnibase_core == $core' "$PINS" >/dev/null || {
    echo "source HEAD differs from declared sim runtime pins" >&2; exit 65;
  }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/sim-preflight-image.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
ctx="$tmp/context"
mkdir -p "$ctx/workspace/sibling-repos"
# Infra is deliberately dirty during integration. Snapshot only runtime-image
# inputs; the generated workspace comes solely from immutable sibling archives.
for item in LICENSE README.md pyproject.toml uv.lock src contracts docker config scripts/runtime_build scripts/seed-keycloak-clients.py; do
  [[ -e "$ROOT/$item" ]] || continue
  parent="$(dirname "$item")"
  [[ "$parent" = . ]] && parent=""
  mkdir -p "$ctx/$parent"
  rsync -a --exclude='.env*' --exclude='*.pem' --exclude='*.key' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
    "$ROOT/$item" "$ctx/$parent/"
done

archive_repo() {
  local name="$1" source="$2" head="$3"
  local target="$ctx/workspace/sibling-repos/$name"
  mkdir -p "$target"
  git -C "$source" archive "$head" | tar -x -C "$target"
  [[ "$(git -C "$source" rev-parse HEAD)" = "$head" ]] || {
    echo "$name HEAD changed during archive" >&2; exit 65;
  }
}
archive_repo omnibase_core "$CORE" "$core_head"
archive_repo omnibase_compat "$COMPAT" "$compat_head"
archive_repo omnimarket "$MARKET" "$market_head"

UV_NO_SYNC=1 uv run python scripts/runtime_build/prepare_sim_preflight_workspace.py \
  --context "$ctx" --source-home "$SOURCE_HOME" --infra "$ROOT" --core "$CORE" --market "$MARKET" \
  --head "omnibase_infra=$infra_head" --head "omnibase_core=$core_head" \
  --head "omnibase_spi=$spi_head" --head "omnibase_compat=$compat_head" \
  --head "omnimarket=$market_head" >"$tmp/digests.json"
UV_NO_SYNC=1 uv run python scripts/runtime_build/prepare_sim_preflight_runtime_dependencies.py \
  --context "$ctx" >"$tmp/dependency-profile.json"
infra_digest="$(jq -er .infra_snapshot_sha256 "$tmp/digests.json")"
core_digest="$(jq -er .core_snapshot_sha256 "$tmp/digests.json")"
market_digest="$(jq -er .market_snapshot_sha256 "$tmp/digests.json")"
compat_digest="$(jq -er .compat_snapshot_sha256 "$tmp/digests.json")"
recipe_digest="$(jq -er .staged_dockerfile_sha256 "$tmp/dependency-profile.json")"
canonical_recipe_digest="$(jq -er .canonical_dockerfile_sha256 "$tmp/dependency-profile.json")"
pyrage_digest="$(jq -er .pyrage_wheel_sha256 "$tmp/dependency-profile.json")"
tag="omnibase-infra-sim-preflight:${infra_digest:0:12}-${core_digest:0:8}-${market_digest:0:8}-${compat_digest:0:8}-${recipe_digest:0:8}"

if [[ "$stage_only" = true ]]; then
  umask 077
  jq -n --slurpfile digests "$tmp/digests.json" \
    --slurpfile dependencies "$tmp/dependency-profile.json" \
    --slurpfile pin_comparison "$ctx/workspace/sibling-pin-comparison.json" \
    --slurpfile vcs "$ctx/workspace/sibling-vcs-provenance.json" \
    --arg infra_head "$infra_head" --arg core_head "$core_head" \
    --arg market_head "$market_head" --arg compat_head "$compat_head" \
    '{profile:"sim-preflight-stage-report-v1", heads:{omnibase_infra:$infra_head,
      omnibase_core:$core_head, omnimarket:$market_head, omnibase_compat:$compat_head},
      digests:$digests[0], dependency_profile:$dependencies[0],
      pin_comparison:$pin_comparison[0], vcs:$vcs[0]}' \
    >"$SIM_PREFLIGHT_STAGE_REPORT_OUT"
  chmod 600 "$SIM_PREFLIGHT_STAGE_REPORT_OUT"
  printf '%s\n' "sim-preflight-stage-validated report=$SIM_PREFLIGHT_STAGE_REPORT_OUT"
  exit 0
fi

cd "$ctx"
docker build --file docker/Dockerfile.runtime --tag "$tag" \
  --label "io.omninode.sim-preflight.infra_head=$infra_head" \
  --label "io.omninode.sim-preflight.infra_snapshot_sha256=$infra_digest" \
  --label "io.omninode.sim-preflight.core_head=$core_head" \
  --label "io.omninode.sim-preflight.core_snapshot_sha256=$core_digest" \
  --label "io.omninode.sim-preflight.market_head=$market_head" \
  --label "io.omninode.sim-preflight.market_snapshot_sha256=$market_digest" \
  --label "io.omninode.sim-preflight.compat_head=$compat_head" \
  --label "io.omninode.sim-preflight.compat_snapshot_sha256=$compat_digest" \
  --label "io.omninode.sim-preflight.canonical_dockerfile_sha256=$canonical_recipe_digest" \
  --label "io.omninode.sim-preflight.staged_dockerfile_sha256=$recipe_digest" \
  --label "io.omninode.sim-preflight.pyrage_wheel_sha256=$pyrage_digest" \
  --label "io.omninode.sim-preflight.dependency_profile=sim-preflight-runtime-active-base-v1" \
  --build-arg BUILD_SOURCE=workspace --build-arg EXPECTED_BUILD_SOURCE=workspace \
  --build-arg OMNI_HOME=/workspace --build-arg GIT_SHA="$infra_head" \
  --build-arg VCS_REF="$infra_head" --build-arg RUNTIME_SOURCE_HASH="$infra_digest" \
  --build-arg PROMOTION_CLASS=stability-candidate --build-arg NON_MAIN_LINEAGE=true \
  --build-arg RUNTIME_VERSION="sim-preflight-${infra_digest:0:12}" .

image_id="$(docker image inspect --format '{{.Id}}' "$tag")"
umask 077
jq -n --arg image_ref "$tag" --arg image_id "$image_id" \
  --arg infra_head "$infra_head" --arg infra_digest "$infra_digest" \
  --arg core_head "$core_head" --arg core_digest "$core_digest" \
  --arg market_head "$market_head" --arg market_digest "$market_digest" \
  --arg compat_head "$compat_head" --arg compat_digest "$compat_digest" \
  --arg canonical_recipe_digest "$canonical_recipe_digest" \
  --arg recipe_digest "$recipe_digest" --arg pyrage_digest "$pyrage_digest" \
  '{profile:"sim-preflight-image-provenance-v2", image_ref:$image_ref, image_id:$image_id,
    infra_head:$infra_head, infra_snapshot_sha256:$infra_digest,
    core_head:$core_head, core_snapshot_sha256:$core_digest,
    market_head:$market_head, market_snapshot_sha256:$market_digest,
    compat_head:$compat_head, compat_snapshot_sha256:$compat_digest,
    canonical_dockerfile_sha256:$canonical_recipe_digest,
    staged_dockerfile_sha256:$recipe_digest,
    pyrage_wheel_sha256:$pyrage_digest,
    dependency_profile:"sim-preflight-runtime-active-base-v1"}' \
  >"$SIM_PREFLIGHT_IMAGE_PROVENANCE_OUT"
chmod 600 "$SIM_PREFLIGHT_IMAGE_PROVENANCE_OUT"
printf '%s\n' "sim-preflight-image-built image_ref=$tag image_id=$image_id"
