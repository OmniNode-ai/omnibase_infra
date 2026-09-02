#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
#
# deploy-runtime.sh -- Stable runtime deployment for omnibase_infra
#
# Rsyncs the current repository to a versioned deployment root
# (~/.omnibase/infra/deployed/{version}/), then runs docker compose
# from that stable location. This eliminates the directory-derived
# compose project name collision that occurs when multiple repo
# copies (omnibase_infra2, omnibase_infra4, etc.) all share the
# same compose project name.
#
# Pattern: real rsync copies (not symlinks), versioned directories,
# dry-run by default.
#
# Usage:
#   ./scripts/deploy-runtime.sh                   # Dry-run preview
#   ./scripts/deploy-runtime.sh --execute         # Deploy + build
#   ./scripts/deploy-runtime.sh --execute --restart  # Deploy + build + restart
#   ./scripts/deploy-runtime.sh --print-compose-cmd  # Show compose commands
#   ./scripts/deploy-runtime.sh --help            # Full usage

set -euo pipefail

# Source contract-rendered runtime policy first, then operator env overrides, so
# all ${VAR} references in docker-compose.infra.yml resolve from exported shell
# environment without making Compose the owner of activation policy.
#
# OMN-14958: the operator env file path is parameterized via
# OMNIBASE_OPERATOR_ENV_FILE (default: ${HOME}/.omnibase/.env) so execution
# contexts whose $HOME does not carry the operator env -- the containerized
# omninode-deploy-runner is the live case -- can point at a provisioned
# mount instead (docker-compose.runners.yml binds the host operator env at
# /run/omnibase-operator.env). A MISSING file is a NAMED, actionable
# precondition failure (exit 64), never bash's bare
# "source: .../.omnibase/.env: No such file or directory" crash that killed
# deploy run 29977968728 before any build/compose action.
SCRIPT_DIR_FOR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_FOR_ENV="$(cd "${SCRIPT_DIR_FOR_ENV}/.." && pwd)"

# OMN-15718: bounded compose-up deadline + stranded-container reconciliation,
# shared with refresh_stability_lane.sh. Sourced early -- its functions are
# only CALLED later, but sourcing here keeps every helper file load in one
# place near the top of the script.
# shellcheck source=./runtime_build/compose_wait_timeout.sh
source "${SCRIPT_DIR_FOR_ENV}/runtime_build/compose_wait_timeout.sh"

OPERATOR_OMNI_HOME="${OMNI_HOME:-}"
OPERATOR_HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-}"
OMNIBASE_OPERATOR_ENV_FILE="${OMNIBASE_OPERATOR_ENV_FILE:-${HOME}/.omnibase/.env}"
if [[ ! -e "${OMNIBASE_OPERATOR_ENV_FILE}" ]]; then
    {
        echo "[deploy-runtime] ERROR: OPERATOR_ENV_MISSING -- operator env file not found:"
        echo "  ${OMNIBASE_OPERATOR_ENV_FILE}"
        echo "  deploy-runtime.sh requires the operator env (POSTGRES_PASSWORD etc.) for"
        echo "  docker compose \${VAR} interpolation. Either provision it at"
        echo "  \${HOME}/.omnibase/.env, or set OMNIBASE_OPERATOR_ENV_FILE to a readable"
        echo "  env file. In the containerized deploy runner this is the read-only bind"
        echo "  mount at /run/omnibase-operator.env wired by DEPLOY_RUNNER_OPERATOR_ENV_FILE"
        echo "  in docker/docker-compose.runners.yml (OMN-14958)."
    } >&2
    exit 64
fi
# OMN-14983: existence does not prove readability. Distinguish "missing"
# from "present but unreadable" explicitly so a permissions problem never
# reaches a raw `source` crash or masquerades as a different failure.
if [[ ! -r "${OMNIBASE_OPERATOR_ENV_FILE}" ]]; then
    {
        echo "[deploy-runtime] ERROR: OPERATOR_ENV_UNREADABLE -- operator env file exists but this process cannot read it:"
        echo "  ${OMNIBASE_OPERATOR_ENV_FILE}"
        echo "  effective uid=$(id -u) ($(id -un 2>/dev/null || echo unknown))"
        echo "  Check the file's ownership/permissions. In the containerized deploy"
        echo "  runner, OMNIBASE_OPERATOR_ENV_FILE should point at the root-phase-init"
        echo "  copy in the runner-owned deploy-runner-creds volume, not directly at the"
        echo "  raw /run/omnibase-operator.env read-only mount (OMN-14983)."
    } >&2
    exit 64
fi
set -a
# shellcheck source=/dev/null
source "${REPO_ROOT_FOR_ENV}/docker/runtime-policy.env"
# shellcheck source=/dev/null
source "${OMNIBASE_OPERATOR_ENV_FILE}"
set +a
if [[ -n "${OPERATOR_OMNI_HOME}" ]]; then
    export OMNI_HOME="${OPERATOR_OMNI_HOME}"
fi
if [[ -n "${OPERATOR_HEALTH_CHECK_URL}" ]]; then
    export HEALTH_CHECK_URL="${OPERATOR_HEALTH_CHECK_URL}"
else
    unset HEALTH_CHECK_URL
fi
unset OPERATOR_OMNI_HOME
unset OPERATOR_HEALTH_CHECK_URL

# =============================================================================
# Constants
# =============================================================================

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
readonly SCRIPT_VERSION="1.0.0"

# Deployment root -- all versioned deployments live under this tree
readonly DEPLOY_ROOT="${HOME}/.omnibase/infra"
readonly REGISTRY_FILE="${DEPLOY_ROOT}/registry.json"
readonly LOCK_DIR="${DEPLOY_ROOT}/.deploy.lock"

# OMN-15218: env-var NAMES the lane-deploy attribution preflight reads. Held as
# names (not values) so the guard's error text and the Python preflight can never
# drift into naming two different knobs.
readonly ONEX_DEPLOY_REASON_VAR="ONEX_DEPLOY_REASON"
readonly ONEX_DEPLOY_GRANT_ACK_VAR="ONEX_DEPLOY_GRANT_ACK"

# Maximum number of deployed versions to retain. Older deployments are pruned
# after each successful deployment. The currently active deployment (tracked in
# registry.json) is never removed regardless of age.
readonly MAX_DEPLOYMENTS="${MAX_DEPLOYMENTS:-5}"

# Runtime services to restart (excludes infrastructure: postgres, redpanda, valkey)
readonly RUNTIME_SERVICES=(
    omninode-runtime
    runtime-effects
    runtime-worker
    projection-api
    agent-actions-consumer
    skill-lifecycle-consumer
    intelligence-api
    omninode-contract-resolver
)

# OMN-17448: services declared ONLY in docker/docker-compose.dev-lane.yml, so
# they exist on the dev lane and on no other. They cannot join RUNTIME_SERVICES
# above: that array is lane-agnostic, and a prod or stability-test
# `up -d --no-deps <service>` naming a service absent from that lane's merged
# compose fails the whole deploy.
#
# These are the standalone projection writers. A handler with the runner shape
# is deliberately NOT dispatched by the shared kernel (OMN-15905/OMN-16874) --
# the kernel still SUBSCRIBES its topics, so without a dedicated process it
# consumes every message, commits every offset, and writes nothing, silently.
# The onex-dev k8s overlay has run these as Deployments since OMN-15905; the
# compose lane ran zero of them until this ticket.
readonly DEV_LANE_ONLY_RUNTIME_SERVICES=(
    projection-tenant-registry-writer
    projection-delegation-writer
)

# OMN-14873: optional scoped-build/restart override. When RUNTIME_BUILD_SERVICES_OVERRIDE
# is set (a space-separated service-name list), build_images() and restart_services()
# operate on ONLY that subset instead of the full RUNTIME_SERVICES fan-out. Unset (the
# default) leaves every existing caller -- prod, dev, cut-lab-ref.sh, the cold bring-up --
# byte-for-byte unchanged (RUNTIME_BUILD_SERVICES == RUNTIME_SERVICES).
#
# scripts/runtime_build/refresh_stability_lane.sh sets this to the 4 known-good core
# services (omninode-runtime runtime-effects runtime-worker projection-api) so a
# workspace-mode build never attempts the 4 release-only services with the still-open
# BUILD_SOURCE selector-mismatch defect (OMN-14262 residual: agent-actions-consumer,
# skill-lifecycle-consumer, intelligence-api, omninode-contract-resolver). This makes the
# scoping a controlled decision, not a side effect of a partial `docker compose build`
# failure leaving only the good images tagged by accident.
if [[ -n "${RUNTIME_BUILD_SERVICES_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word-splitting of an operator-provided
    # space-separated service list (not a glob; no IFS surprises expected here)
    RUNTIME_BUILD_SERVICES=(${RUNTIME_BUILD_SERVICES_OVERRIDE})
else
    RUNTIME_BUILD_SERVICES=("${RUNTIME_SERVICES[@]}")
fi

resolve_lane_runtime_services() {
    # Populate a caller-provided array (by name) with the runtime services to
    # build/restart for one lane: the lane-agnostic set, plus the dev-lane-only
    # additions when (and only when) the target IS the dev lane. Lane identity
    # is read the same way every other function here reads it -- an empty
    # overlay filename means the bare `omnibase-infra` dev project -- so this
    # cannot drift from resolve_compose_file_args().
    local _out_args_name="$1"
    local compose_project="$2"

    eval "${_out_args_name}=()"
    local svc
    for svc in "${RUNTIME_BUILD_SERVICES[@]}"; do
        eval "${_out_args_name}+=( $(printf '%q' "${svc}") )"
    done

    # A scoped-build override is an explicit operator instruction to touch ONLY
    # the named services; silently appending to it would defeat the point.
    if [[ -n "${RUNTIME_BUILD_SERVICES_OVERRIDE:-}" ]]; then
        return 0
    fi

    local overlay_filename
    overlay_filename="$(resolve_lane_overlay_filename "${compose_project}")"
    if [[ -n "${overlay_filename}" ]]; then
        return 0
    fi

    for svc in "${DEV_LANE_ONLY_RUNTIME_SERVICES[@]}"; do
        eval "${_out_args_name}+=( $(printf '%q' "${svc}") )"
    done
}
readonly RUNTIME_BUILD_SERVICES
# Migration services refreshed (one-shot) before the --no-deps runtime restart.
# Order matters: forward-migration applies the omnibase_infra schema, then
# intelligence-migration applies the omniintelligence schema, then migration-gate
# stamps db_metadata.migrations_complete and stays up as a healthcheck keepalive.
#
# OMN-13220: intelligence-migration was MISSING here. The compose file gates
# omninode-runtime on `intelligence-migration: condition: service_completed_successfully`,
# but restart_services() uses `up -d --no-deps`, which bypasses depends_on. On a
# fresh-DB lane that left public.db_metadata for omniintelligence unstamped, so
# the runtime crash-looped. The preflight must run it explicitly.
#
# One-shot services (run-to-completion, exit 0) are listed in
# RUNTIME_MIGRATION_ONESHOTS so the preflight can `docker wait` on them; the
# keepalive migration-gate is deliberately excluded from that wait set.
readonly RUNTIME_MIGRATION_SERVICES=(
    forward-migration
    intelligence-migration
    migration-gate
)
readonly RUNTIME_MIGRATION_ONESHOTS=(
    forward-migration
    intelligence-migration
)
# Broker readiness services brought up (and waited on) before the runtime
# restart. redpanda-partition-cap raises topic_partitions_per_shard so the cold
# 1300+-topic provisioning burst on first boot does not exhaust the default
# single-shard partition ceiling (OMN-11886 / OMN-13220). Because the runtime
# restart is `--no-deps`, the compose depends_on chain (which includes
# redpanda-partition-cap as service_completed_successfully) is bypassed, so the
# preflight must apply the cap explicitly before the kernel provisions topics.
readonly BROKER_READINESS_SERVICE="redpanda"
readonly BROKER_PARTITION_CAP_SERVICE="redpanda-partition-cap"
# Core data-plane infra that the migration preflight + runtime depend on but
# that the `--no-deps` restart path never starts itself (OMN-13594). On a fully
# COLD lane (no prior containers) nothing brings postgres/valkey up before
# run_runtime_migration_preflight runs forward-migration `--no-deps`, so the
# migration's 30x2s Postgres-readiness probe exhausts -> exit 1 -> auto-rollback.
# ensure_core_infra_ready() brings these up + waits BEFORE the preflight; on a
# WARM lane `up -d --wait` on already-healthy services is an idempotent no-op.
# redpanda is intentionally excluded here -- warm_broker_topic_provisioning owns
# broker readiness (and its collision-tolerant reachability probe).
readonly CORE_INFRA_SERVICES=(
    postgres
    valkey
)
# Cold-start consumer-group join budget (OMN-13220). On a fully-cold lane the
# kernel joins a consumer group per subscribed topic; with 1300+ topics on a
# freshly-provisioned broker the default 30s per-consumer KAFKA_TIMEOUT_SECONDS
# blew on the slow group-coordinator tail and the kernel recycled before it
# reached healthy. Raise the per-consumer start budget for the restart-driven
# boot. Operator-overridable; clamped to the config field bound (le=300).
readonly COLD_START_KAFKA_TIMEOUT_SECONDS="${COLD_START_KAFKA_TIMEOUT_SECONDS:-180}"
readonly REQUIRED_PROJECTION_TABLES=(
    delegation_events
    node_service_registry
)

# Minimum Docker Compose version (nested variable expansion support)
readonly MIN_COMPOSE_VERSION="2.20"

# Health check parameters
readonly HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://${INFRA_HOST:?INFRA_HOST required}:8085/health}"
readonly HEALTH_CHECK_RETRIES=15
readonly HEALTH_CHECK_INTERVAL=4

# =============================================================================
# Defaults
# =============================================================================

MODE="dry-run"           # dry-run | execute
FORCE=false
RESTART=false
# OMN-15218: the raw argv this invocation was called with, captured before
# parse_args consumes it, so the attribution record names the exact command that
# mutated the lane instead of a reconstruction.
DEPLOY_INVOCATION_ARGS=()
# OMN-15218: the attribution record emitted by the lane-deploy preflight, folded
# into registry.json by write_registry() so "who/what/why" is readable from the
# same file that already answers "what version is deployed".
LANE_ATTRIBUTION_RECORD_JSON=""
# Set after rsync to enable automatic cleanup of orphaned deployment directories
# on failure. If this is non-empty and the deployment directory is NOT the active
# deployment in registry.json, the trap handler will remove it.
DEPLOY_DIR_TO_CLEANUP=""
# Default is hardcoded and safe; any changes must comply with ^[a-zA-Z0-9_-]+$ (see parse_args).
COMPOSE_PROFILE="runtime"
PRINT_COMPOSE_CMD=false
# When true (--cold), run the cold-lane FULL bring-up path (OMN-13414): build in
# workspace mode from the merged-dev siblings, bring up deps + migration
# one-shots, then bring the WHOLE --profile runtime project up (not just the
# RUNTIME_SERVICES subset the warm --restart path recreates). Two gotchas this
# encodes: the runtime profile is mandatory (a bare `up -d` starts nothing) and
# the build must be workspace-sourced (release packages cannot carry un-released
# merged-dev code, so a release image starts a cold lane on stale code).
COLD_FULL_BRINGUP=false
# When true (--prod, or ONEX_DEPLOY_LANE=prod), the prod promotion-lineage guard
# runs before any build: the source tree must be clean AND HEAD must be an
# ancestor-of/equal-to origin/main. Prevents building the prod image from a
# dirty or dev-only tree (OMN-12626, R1).
PROD_LANE=false
if [[ "${ONEX_DEPLOY_LANE:-}" == "prod" ]]; then
    PROD_LANE=true
fi
# When --force overwrites an existing deployment, the previous directory is
# moved here as a backup. On success the backup is removed; on failure
# cleanup_on_exit() restores it.
FORCE_BACKUP_DIR=""
# OMN-13364: path (relative to the deploy target) of the vendored forward-migration
# tree. The backup-restore path in cleanup_on_exit() reverts the WHOLE deployment
# tree, including freshly-built migrations, which silently regressed the deployed
# migrations to the pre-build snapshot (dropped node_projection_delegation/
# 0015_generation_corpus_acceptance.sql in the 2026-06-19 stability redeploy).
# After a restore, the freshly-synced migration tree is re-applied from this
# snapshot so the deployed migrations always match the build source (origin/dev).
readonly MIGRATION_TREE_REL_PATH="docker/migrations/forward"
# Absolute path to a preserved copy of the freshly-synced vendored migration
# tree, captured after sync_files() (so the restore can re-apply it). Empty until
# the snapshot is taken; the snapshot dir is removed on exit.
MIGRATION_TREE_SNAPSHOT_DIR=""
# Set to true only when ALL deployment phases complete successfully.
# Used by cleanup_on_exit to determine if the --force backup can be safely removed.
DEPLOYMENT_COMPLETE=false
# OMN-15352: path to a file recording each RUNTIME_BUILD_SERVICES image's
# pre-build `:latest` id (or empty if none existed), taken by
# snapshot_latest_image_tags() right before build_images() runs. On a failed
# deploy, cleanup_on_exit() restores every snapshotted tag (or removes an
# unverified tag that had no prior state) so a later `docker compose up -d`
# without --build can never silently resolve an untested image (F3). Empty
# until the snapshot is taken; the snapshot file is removed on exit.
LATEST_TAG_SNAPSHOT_FILE=""
# OMN-15352: compose project name, mirrored into a global right after
# resolve_compose_project() resolves it in main(). cleanup_on_exit() is an
# EXIT-trap handler with no arguments, so it cannot receive compose_project as
# a parameter -- it reads this global to resolve the same image names
# snapshot_latest_image_tags() recorded them under.
DEPLOY_COMPOSE_PROJECT=""

# =============================================================================
# Logging
# =============================================================================

log_info() {
    # Print an informational log message to stdout.
    printf '[deploy] %s\n' "$*"
}

log_warn() {
    # Print a warning message to stderr.
    printf '[deploy] WARNING: %s\n' "$*" >&2
}

log_error() {
    # Print an error message to stderr.
    printf '[deploy] ERROR: %s\n' "$*" >&2
}

log_step() {
    # Print a section header for a deployment phase.
    printf '\n[deploy] === %s ===\n' "$*"
}

log_cmd() {
    # Print a command-echo line showing the command being executed.
    printf '[deploy]   > %s\n' "$*"
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    # Print usage information and exit.
    cat <<EOF
${SCRIPT_NAME} v${SCRIPT_VERSION} -- Stable runtime deployment for omnibase_infra

Rsyncs the current repo to ~/.omnibase/infra/deployed/{version}/,
then runs docker compose from that stable location.

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    (none)              Dry-run mode (default). Preview what would be deployed.
    --execute           Actually deploy: rsync, write registry, build images.
    --force             Required to overwrite an existing version directory.
    --restart           Restart runtime containers after build (requires --execute).
                        WARM path: recreates only the RUNTIME_SERVICES subset
                        with 'up -d --no-deps'. Use on a lane whose deps + broker
                        are already up.
    --cold              COLD-lane FULL bring-up (OMN-13414). For a lane that was
                        GC/idle-reclaimed and torn down to zero containers, and
                        (OMN-16803) for a PARTIALLY cold lane where some services
                        are missing while its deps stay up — the warm --restart
                        path cannot repair that case, because it resolves a
                        running image id per service and an absent container has
                        none. Allowed for dev and stability-test; REFUSED for
                        prod (workspace images are non-main-lineage) and judge
                        (read-only lane). Forces a workspace-mode build from the local
                        merged-dev siblings (BUILD_SOURCE=workspace + OMNI_HOME +
                        sibling REF build-args via stage_workspace.sh), brings up
                        deps + the migration one-shots, then brings the WHOLE
                        '--profile runtime' project up (every consumer/projection
                        service, not just RUNTIME_SERVICES). Requires --execute and
                        OMNI_HOME; incompatible with --prod (workspace images are
                        non-main-lineage and the prod gate refuses them). Two
                        gotchas it solves: the runtime profile is mandatory (a bare
                        'docker compose up -d' starts NOTHING), and the default
                        BUILD_SOURCE=release cannot rebuild a cold lane from
                        un-released merged-dev code. See
                        knowledge-base:runbooks/cold-lane-full-bringup.md.
    --profile <name>    Docker compose profile (default: runtime).
    --print-compose-cmd Print exact compose commands without executing, then exit.
    --prod              Enforce the prod promotion-lineage guard before build:
                        source tree must be clean AND HEAD an ancestor-of/equal-to
                        origin/main. Also honored via ONEX_DEPLOY_LANE=prod.
    --help              Show this help message and exit.

ATTRIBUTION + GRANT INTERLOCK (OMN-15218)
    ONEX_DEPLOY_REASON      REQUIRED for the governed lanes (stability-test, prod,
                            judge). A real justification, ideally naming a ticket;
                            placeholders are rejected. Recorded durably with the
                            actor (user/uid/host/ssh peer/parent command) and the
                            invoking command line.
    ONEX_DEPLOY_TICKET      Optional explicit OMN-#### (otherwise parsed from the
                            reason).
    ONEX_DEPLOY_GRANT_ACK   Comma-separated grant ids. A stability-test deploy is
                            REFUSED while unconsumed, unexpired prod-promotion
                            grants exist at onex_change_control@main; proceeding
                            requires naming EVERY live grant id here (or the token
                            'unreadable-grant-state' when grant state cannot be
                            resolved — which also fails closed). The
                            acknowledgement is written into the record.

DEPLOYMENT ROOT
    ~/.omnibase/infra/
    +-- .deploy.lock/                       mkdir-based concurrency guard
    +-- registry.json                       tracks active deployment
    +-- deploy-log.jsonl                    append-only lane-deploy attribution log
    +-- deploy-attribution/                 per-run attribution records
    +-- deployed/
        +-- {version}/                      build directory
            +-- pyproject.toml
            +-- uv.lock
            +-- src/omnibase_infra/
            +-- contracts/
            +-- docker/
                +-- docker-compose.infra.yml
                +-- Dockerfile.runtime
                +-- entrypoint-runtime.sh
                +-- .env                    preserved across deploys
                +-- .env.local              preserved (user overrides)
                +-- certs/                  preserved (TLS certs)
                +-- migrations/forward/

EXAMPLES
    # Preview what would be deployed
    ${SCRIPT_NAME}

    # Deploy and build images
    ${SCRIPT_NAME} --execute

    # Deploy, build, and restart containers (WARM lane: deps already up)
    ${SCRIPT_NAME} --execute --restart

    # COLD lane full bring-up from merged dev (workspace build + full --profile up)
    OMNI_HOME=/path/to/omni_home ${SCRIPT_NAME} --execute --cold

    # Redeploy same version (overwrite)
    ${SCRIPT_NAME} --execute --force

    # Print compose commands for manual use
    ${SCRIPT_NAME} --print-compose-cmd

    # Check registry
    cat ~/.omnibase/infra/registry.json | jq .

    # Verify image labels match deployed SHA
    docker inspect omninode-runtime \\
        --format='{{index .Config.Labels "org.opencontainers.image.revision"}}'
EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    # Parse command-line arguments and set global mode/flag variables.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --execute)
                MODE="execute"
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --restart)
                RESTART=true
                shift
                ;;
            --cold)
                COLD_FULL_BRINGUP=true
                shift
                ;;
            --profile)
                if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                    log_error "--profile requires a value"
                    exit 1
                fi
                # Validate profile name: only alphanumeric, hyphens, and underscores
                # are allowed to prevent invalid compose project names.
                if [[ ! "$2" =~ ^[a-zA-Z0-9_-]+$ ]]; then
                    log_error "--profile value must contain only alphanumeric characters, hyphens, and underscores."
                    log_error "  Got: '$2'"
                    exit 1
                fi
                COMPOSE_PROFILE="$2"
                shift 2
                ;;
            --print-compose-cmd)
                PRINT_COMPOSE_CMD=true
                shift
                ;;
            --prod)
                PROD_LANE=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                log_error "Run '${SCRIPT_NAME} --help' for usage."
                exit 1
                ;;
        esac
    done

    # Validate flag combinations
    if [[ "${RESTART}" == true && "${MODE}" != "execute" ]]; then
        log_error "--restart requires --execute"
        exit 1
    fi

    # OMN-13414: --cold is the cold-lane FULL bring-up. It forces a workspace
    # build (so a release-pinned BUILD_SOURCE is a contradiction) and produces a
    # non-main-lineage image the prod-promotion gate refuses (so --prod / prod
    # lane is incompatible).
    if [[ "${COLD_FULL_BRINGUP}" == true ]]; then
        if [[ "${BUILD_SOURCE:-}" == "release" ]]; then
            log_error "--cold performs a workspace-mode build from the merged-dev siblings, but BUILD_SOURCE=release was set."
            log_error "  A release image cannot carry un-released merged-dev code; a cold lane would boot on stale code."
            log_error "  Unset BUILD_SOURCE (it defaults to workspace under --cold) or set BUILD_SOURCE=workspace."
            exit 64
        fi
        if [[ "${PROD_LANE}" == true ]]; then
            log_error "--cold is a workspace-mode (non-main-lineage) bring-up and cannot target the prod lane."
            log_error "  The prod-promotion gate refuses workspace / stability-candidate images (OMN-13669)."
            log_error "  Promote a clean-main release to prod via the gated node path instead."
            exit 1
        fi
    fi
}

resolve_compose_project() {
    # Runtime compose files use fixed container names, and the deploy-agent
    # targets the canonical "omnibase-infra" project. Keep deploy-runtime on the
    # same project by default so rebuild/recreate updates the live stack instead
    # of creating a parallel profile-derived project such as
    # "omnibase-infra-runtime".
    local compose_project="${OMNIBASE_INFRA_COMPOSE_PROJECT:-omnibase-infra}"

    if [[ ! "${compose_project}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "OMNIBASE_INFRA_COMPOSE_PROJECT must contain only alphanumeric characters, hyphens, and underscores."
        log_error "  Got: '${compose_project}'"
        exit 1
    fi

    echo "${compose_project}"
}

# Compose project -> lane (overlay) mapping. The dev lane (bare omnibase-infra
# project) runs from docker-compose.infra.yml alone; every non-dev lane LAYERS
# its overlay so the overlay's container_name + project name + lane network win.
#
# OMN-13581: deploy-runtime.sh historically passed ONLY `-f infra.yml` on every
# `docker compose` call, including warm_broker_topic_provisioning's `up redpanda`
# step. The base infra compose hardcodes `container_name: omnibase-infra-redpanda`
# (the DEV name) and the dev network, so running the warmup against a non-dev
# project (e.g. omnibase-infra-stability-test) makes compose try to (re)create
# redpanda as the DEV-named container, which collides with the live dev broker,
# gets a Docker hash prefix, and lands in 'created' -- DESTROYING the lane's own
# correctly-named broker. That left the stability lane broker-less for ~3 days.
# Layering the matching overlay gives redpanda the lane-prefixed container_name +
# lane network, so the lane's broker is targeted and never displaced.
#
# This mirrors the authoritative, tested lane->compose-file mapping in
# scripts/deploy-agent/deploy_agent/executor.py (_LANE_CONFIGS): stability-test
# layers docker-compose.stability-test.yml, prod layers docker-compose.prod.yml,
# judge layers docker-compose.judge.yml. The dev project gets no overlay.
resolve_lane_name() {
    # Echo the LANE name derived from a compose project (OMN-15218).
    #   omnibase-infra                -> dev
    #   omnibase-infra-stability-test -> stability-test
    #   omnibase-infra-prod           -> prod
    #   omnibase-infra-judge          -> judge
    # Single derivation shared by the hot-patch preflight and the lane-deploy
    # attribution guard, so one deploy can never be recorded under two different
    # lane names. Unknown suffixes echo through unchanged; the callers that must
    # fail closed on an unknown lane (resolve_lane_overlay_filename) do their own
    # allowlist check.
    local compose_project="$1"
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"
    if [[ -z "${lane}" ]]; then
        lane="dev"
    fi
    echo "${lane}"
}

resolve_lane_overlay_filename() {
    # Echo the overlay compose FILENAME (relative to docker/) for a compose
    # project, or nothing for the bare dev project. Fails closed: an unknown
    # non-dev project aborts rather than silently running on the dev config (the
    # exact failure mode that displaced the lane broker).
    local compose_project="$1"

    # Lane = compose project suffix after the canonical "omnibase-infra" prefix.
    # omnibase-infra                -> "" (dev, no overlay)
    # omnibase-infra-stability-test -> "stability-test"
    # omnibase-infra-prod           -> "prod"
    # omnibase-infra-judge          -> "judge"
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"

    case "${lane}" in
        "")
            # Dev lane: infra.yml alone (fixed dev container names are correct here).
            return 0
            ;;
        stability-test|prod|judge)
            echo "docker-compose.${lane}.yml"
            return 0
            ;;
        *)
            log_error "Unknown lane '${lane}' derived from compose project '${compose_project}'."
            log_error "  deploy-runtime.sh only knows the dev / stability-test / prod / judge lanes."
            log_error "  Refusing to deploy: running a non-dev lane on the bare infra.yml config"
            log_error "  would recreate the DEV-named redpanda and displace this lane's broker"
            log_error "  (OMN-13581). Add the lane's overlay mapping before deploying it."
            exit 1
            ;;
    esac
}

resolve_compose_file_args() {
    # Populate a caller-provided array (passed by name) with the full
    # `-f <file>` token sequence for a deployment: always
    # docker-compose.infra.yml, plus the lane overlay (docker-compose.<lane>.yml)
    # for any non-dev compose project (OMN-13581).
    #
    # Usage:
    #   local -a compose_args
    #   resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    #   docker compose -p "${compose_project}" "${compose_args[@]}" ...
    local _out_args_name="$1"
    local deploy_target="$2"
    local compose_project="$3"

    local docker_dir="${deploy_target}/docker"
    eval "${_out_args_name}=(-f $(printf '%q' "${docker_dir}/docker-compose.infra.yml"))"

    local overlay_filename
    overlay_filename="$(resolve_lane_overlay_filename "${compose_project}")"
    if [[ -n "${overlay_filename}" ]]; then
        eval "${_out_args_name}+=( -f $(printf '%q' "${docker_dir}/${overlay_filename}") )"
    else
        # Dev/lab lane (bare omnibase-infra project). OMN-15379: layer the
        # dev-lane overlay, whose ONLY content is ONEX_MIGRATION_LANE=dev for
        # forward-migration — the lane indicator that releases the
        # node_projection_registration trio (operator ruling 15, lab lane is the
        # FORCE proving ground). It is a separate file precisely so no non-dev
        # lane can inherit it from the base: see the header of
        # docker/docker-compose.dev-lane.yml. Unset indicator = FULL fence, so
        # omitting this file degrades safely (the lane comes up without
        # node_service_registry) rather than dangerously.
        eval "${_out_args_name}+=( -f $(printf '%q' "${docker_dir}/docker-compose.dev-lane.yml") )"
    fi
}

resolve_lane_runtime_container_name() {
    # Echo the lane-scoped `container_name` of the omninode-runtime main container
    # for a compose project. Each lane overlay prefixes the runtime container_name
    # with its lane (the compose *service* key stays "omninode-runtime" in every
    # overlay -- only container_name changes):
    #   omnibase-infra                -> omninode-runtime               (dev, no prefix)
    #   omnibase-infra-stability-test -> omninode-stability-test-runtime
    #   omnibase-infra-prod           -> omninode-prod-runtime
    #   omnibase-infra-judge          -> omninode-judge-runtime
    #
    # verify_deployment must probe THIS name, not the hardcoded dev name: the dev
    # `omninode-runtime` container commonly runs alongside a non-dev lane on the
    # same host, so an anchored `name=^/omninode-runtime$` filter would resolve the
    # wrong (dev) container and emit a false image-label mismatch (OMN-13826).
    #
    # Fails closed on an unknown non-dev lane, using the same whitelist as
    # resolve_lane_overlay_filename, so a typo'd project can't silently probe dev.
    local compose_project="$1"

    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"

    case "${lane}" in
        "")
            echo "omninode-runtime"
            ;;
        stability-test|prod|judge)
            echo "omninode-${lane}-runtime"
            ;;
        *)
            log_error "Unknown lane '${lane}' derived from compose project '${compose_project}'."
            log_error "  deploy-runtime.sh only knows the dev / stability-test / prod / judge lanes."
            log_error "  Refusing to resolve a runtime container name for an unknown lane (OMN-13826)."
            exit 1
            ;;
    esac
}

# =============================================================================
# Prerequisites
# =============================================================================

check_command() {
    # Validate that a required command exists in PATH.
    local cmd="$1"
    local purpose="$2"
    if ! command -v "${cmd}" &>/dev/null; then
        log_error "'${cmd}' is required (${purpose}) but not found in PATH."
        exit 1
    fi
}

check_compose_version() {
    # Verify Docker Compose meets the minimum version requirement.
    local version_output
    version_output="$(docker compose version --short 2>/dev/null || true)"

    if [[ -z "${version_output}" ]]; then
        log_error "docker compose plugin not found. Install Docker Compose v2.20+."
        exit 1
    fi

    # Strip leading 'v' if present
    version_output="${version_output#v}"

    # Compare major.minor
    local major minor
    major="$(echo "${version_output}" | cut -d. -f1)"
    minor="$(echo "${version_output}" | cut -d. -f2)"
    local req_major req_minor
    req_major="$(echo "${MIN_COMPOSE_VERSION}" | cut -d. -f1)"
    req_minor="$(echo "${MIN_COMPOSE_VERSION}" | cut -d. -f2)"

    # Validate version components are numeric before arithmetic comparison
    local component
    for component in "${major}" "${minor}" "${req_major}" "${req_minor}"; do
        if [[ ! "${component}" =~ ^[0-9]+$ ]]; then
            log_error "Non-numeric version component: '${component}' (from version '${version_output}')."
            log_error "Expected format: MAJOR.MINOR (e.g., 2.20)."
            exit 1
        fi
    done

    if (( major < req_major || (major == req_major && minor < req_minor) )); then
        log_error "Docker Compose ${MIN_COMPOSE_VERSION}+ required (found ${version_output})."
        log_error "Nested variable expansion requires Compose >= ${MIN_COMPOSE_VERSION}."
        exit 1
    fi

    log_info "Docker Compose version: ${version_output}"
}

validate_prerequisites() {
    # Check that all required external commands and Docker Compose version are available.
    log_step "Validate Prerequisites"

    check_command docker  "container runtime"
    check_command git     "version control"

    if [[ "${PRINT_COMPOSE_CMD}" == false ]]; then
        check_command rsync   "file synchronization"
        check_command jq      "JSON processing"
        check_command curl    "deployment verification"
    fi

    check_compose_version
}

# =============================================================================
# Repository Validation
# =============================================================================

resolve_repo_root() {
    # Walk up from script location to find pyproject.toml
    local dir
    dir="$(cd "$(dirname "$0")" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/pyproject.toml" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    log_error "Cannot find repository root (no pyproject.toml found above script)."
    exit 1
}

validate_repo_structure() {
    # Verify that all required files and directories exist in the repository.
    local repo_root="$1"
    local missing=()

    [[ -f "${repo_root}/pyproject.toml" ]]                          || missing+=("pyproject.toml")
    [[ -f "${repo_root}/uv.lock" ]]                                 || missing+=("uv.lock")
    [[ -d "${repo_root}/src/omnibase_infra" ]]                      || missing+=("src/omnibase_infra/")
    [[ -d "${repo_root}/contracts" ]]                                || missing+=("contracts/")
    [[ -d "${repo_root}/docker" ]]                                   || missing+=("docker/")
    [[ -f "${repo_root}/docker/docker-compose.infra.yml" ]]         || missing+=("docker/docker-compose.infra.yml")
    [[ -f "${repo_root}/docker/Dockerfile.runtime" ]]               || missing+=("docker/Dockerfile.runtime")
    [[ -f "${repo_root}/docker/entrypoint-runtime.sh" ]]            || missing+=("docker/entrypoint-runtime.sh")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Repository structure validation failed. Missing:"
        for item in "${missing[@]}"; do
            log_error "  - ${item}"
        done
        exit 1
    fi

    log_info "Repository structure validated."

    # VirtioFS bind-mount conflict detection
    # docker-compose has two mounts:
    #   ../contracts:/app/contracts:ro
    #   ../src/omnibase_infra/nodes:/app/contracts/nodes:ro  (overlays nodes/ subdirectory)
    # When ../contracts/nodes exists but ../src/omnibase_infra/nodes does NOT exist,
    # the overlay source is missing and containers see an empty nodes/ directory.
    local parent_dir
    parent_dir="$(dirname "${repo_root}")"
    local contracts_nodes="${parent_dir}/contracts/nodes"
    local src_nodes="${repo_root}/src/omnibase_infra/nodes"

    if [[ ! -d "${contracts_nodes}" ]]; then
        log_warn "VirtioFS CHECK: ${contracts_nodes} not found (rsync may not have run yet — advisory)"
    elif [[ ! -d "${src_nodes}" ]]; then
        log_error "VirtioFS CHECK: ${contracts_nodes} exists but ${src_nodes} does not"
        log_error "  This will cause an empty bind-mount overlay at /app/contracts/nodes"
        log_error "  Fix: ensure src/omnibase_infra/nodes/ exists in the deploy root"
        exit 1
    else
        log_info "VirtioFS CHECK: both mount sources exist"
    fi
}

# =============================================================================
# Identity -- version + git SHA
# =============================================================================

read_version() {
    # Extract the project version from pyproject.toml [project] section (PEP 621).
    local repo_root="$1"
    local version

    # Extract version from the [project] section of pyproject.toml.
    # A naive grep -m1 '^version' could match a version key in any TOML
    # section (e.g. a dependency table).  This awk approach activates only
    # inside [project] and deactivates when the next section header
    # is reached, ensuring we read the project version.
    version="$(awk '
        /^\[project\]/ { in_section=1; next }
        /^\[/          { in_section=0 }
        in_section && /^version[[:space:]]*=/ {
            gsub(/.*=[[:space:]]*"/, "");
            gsub(/".*/, "");
            print;
            exit
        }
    ' "${repo_root}/pyproject.toml")"

    if [[ -z "${version}" ]]; then
        log_error "Could not read version from pyproject.toml [project] section"
        exit 1
    fi

    echo "${version}"
}

read_git_sha() {
    # Read the 12-character abbreviated git SHA of HEAD for VCS_REF labeling.
    local repo_root="$1"
    local sha

    sha="$(git -C "${repo_root}" rev-parse --short=12 HEAD 2>/dev/null || true)"

    if [[ -z "${sha}" ]]; then
        log_error "Could not determine git SHA. Is this a git repository?"
        exit 1
    fi

    echo "${sha}"
}

read_repo_ref_or_main() {
    # Return a full git SHA for sibling workspace repos when available. Docker
    # build args use the value in install URLs, so "main" is only a fallback.
    local repo_path="$1"
    local sha

    sha="$(git -C "${repo_path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "${sha}" ]]; then
        echo "${sha}"
    else
        echo "main"
    fi
}

resolve_build_source() {
    # Resolve the selected Dockerfile dependency source.
    #
    # OMN-13414: a cold-lane FULL bring-up (--cold) must build from the local
    # workspace siblings at merged-dev SHAs, never from the PyPI release packages
    # — a release image cannot carry un-released merged-dev code, which is exactly
    # what a cold/GC-reclaimed lane has to be rebuilt from. --cold therefore forces
    # workspace mode; validate_build_source_config then requires OMNI_HOME, and
    # parse_args has already rejected a contradictory BUILD_SOURCE=release.
    if [[ "${COLD_FULL_BRINGUP}" == true ]]; then
        echo "workspace"
        return 0
    fi
    echo "${BUILD_SOURCE:-release}"
}

resolve_expected_build_source() {
    # Default the Dockerfile assertion to the selected source. This preserves
    # release-mode behavior while allowing BUILD_SOURCE=workspace without
    # requiring operators to set a second env var by hand.
    local build_source="$1"
    echo "${EXPECTED_BUILD_SOURCE:-${build_source}}"
}

resolve_promotion_class() {
    # OMN-13669: compute PROMOTION_CLASS OCI label from build_source.
    # workspace builds are stability-candidates (non-main-lineage dev images);
    # release/clean-main builds default to clean-main.
    local build_source="$1"
    if [[ "${build_source}" == "workspace" ]]; then
        echo "stability-candidate"
    else
        echo "clean-main"
    fi
}

resolve_non_main_lineage() {
    # OMN-13669: compute NON_MAIN_LINEAGE OCI label from build_source.
    # workspace builds are non-main-lineage; release builds are not.
    local build_source="$1"
    if [[ "${build_source}" == "workspace" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

validate_build_source_config() {
    # Validate build-source selector agreement before staging or Docker build.
    local build_source expected_build_source omni_home
    build_source="$(resolve_build_source)"
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    omni_home="${OMNI_HOME:-}"

    case "${build_source}" in
        workspace|release) ;;
        *)
            log_error "Invalid BUILD_SOURCE='${build_source}'; expected workspace or release."
            exit 64
            ;;
    esac

    case "${expected_build_source}" in
        workspace|release) ;;
        *)
            log_error "Invalid EXPECTED_BUILD_SOURCE='${expected_build_source}'; expected workspace or release."
            exit 64
            ;;
    esac

    if [[ "${build_source}" != "${expected_build_source}" ]]; then
        log_error "BUILD_SOURCE selector mismatch: BUILD_SOURCE='${build_source}' EXPECTED_BUILD_SOURCE='${expected_build_source}'."
        exit 64
    fi

    if [[ "${build_source}" == "workspace" && -z "${omni_home}" ]]; then
        log_error "BUILD_SOURCE=workspace requires OMNI_HOME before staging or build."
        exit 64
    fi
}

stage_workspace_if_needed() {
    # Populate workspace/sibling-repos/ from the operator-selected OMNI_HOME so
    # Dockerfile.runtime can install exact local sibling repo contents.
    local repo_root="$1"
    local build_source omni_home stage_script
    build_source="$(resolve_build_source)"
    if [[ "${build_source}" != "workspace" ]]; then
        return 0
    fi

    omni_home="${OMNI_HOME:-}"
    stage_script="${repo_root}/scripts/runtime_build/stage_workspace.sh"
    if [[ ! -f "${stage_script}" ]]; then
        log_error "Workspace staging script not found: ${stage_script}"
        log_error "Cannot proceed with BUILD_SOURCE=workspace."
        exit 1
    fi

    log_step "Stage Workspace Sibling Repos"
    log_cmd "OMNI_HOME=${omni_home} bash ${stage_script}"
    (cd "${repo_root}" && OMNI_HOME="${omni_home}" bash "${stage_script}")

    check_sibling_lock_pins "${repo_root}" "${omni_home}"
}

check_sibling_lock_pins() {
    # Fail-fast preflight (OMN-12987): every vendored sibling's version/SHA must
    # match the consuming repo's (omnimarket) uv.lock pin. The 2026-06-11
    # stability crash shipped a 13-day-stale infra 0.37.0 because the build
    # ignored the dev lock; this guard refuses to build a stale image.
    local repo_root="$1"
    local omni_home="$2"
    local guard="${repo_root}/scripts/runtime_build/check_sibling_lock_pins.py"
    if [[ ! -f "${guard}" ]]; then
        log_error "Sibling lock-pin preflight not found: ${guard}"
        log_error "Cannot verify vendored siblings match the consuming lock. Aborting."
        exit 1
    fi

    log_step "Sibling Lock-Pin Preflight (OMN-12987)"
    # Write under sibling-repos/ so it rides along with the directory the
    # Dockerfile already COPYs into the build image (no extra COPY needed).
    mkdir -p "${repo_root}/workspace/sibling-repos"
    local provenance_out="${repo_root}/workspace/sibling-repos/.sibling-lock-pins.json"
    local python_bin
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the sibling lock-pin preflight."
        exit 1
    fi

    # The check_sibling_lock_pins.py interface changed under OMN-12977/12987:
    # the original single-output flag was removed in favor of --lock (required,
    # the pin authority), repeatable --repo PACKAGE=PATH (the canonical clones
    # the build vendors), and --output (where to write the comparison JSON).
    # The consuming repo's uv.lock (omnimarket) is the pin authority.
    local lock_path="${omni_home}/omnimarket/uv.lock"
    # --build-source workspace: this preflight only runs on the workspace path
    # (stage_workspace_if_needed short-circuits unless BUILD_SOURCE=workspace), so
    # a registry-sourced sibling whose clone is FORWARD of the lock (the OMN-13929
    # disarm-bump steady state) is non-fatal (OMN-13902). Backward / git-sourced
    # drift stays fatal.
    local guard_args=(
        --lock "${lock_path}"
        --repo "omnibase-infra=${omni_home}/omnibase_infra"
        --repo "omnibase-core=${omni_home}/omnibase_core"
        --repo "omnibase-spi=${omni_home}/omnibase_spi"
        --repo "omnibase-compat=${omni_home}/omnibase_compat"
        --output "${provenance_out}"
        --build-source workspace
    )
    # Operator override (OMN-12977): ALLOW_SIBLING_PIN_DRIFT=1 records drift in
    # the provenance artifact and proceeds instead of aborting. Never the default.
    if [[ "${ALLOW_SIBLING_PIN_DRIFT:-0}" == "1" ]]; then
        guard_args+=(--allow-drift)
        log_warn "ALLOW_SIBLING_PIN_DRIFT=1 -- passing --allow-drift to sibling lock-pin preflight (OMN-12977)"
    fi

    log_cmd "OMNI_HOME=${omni_home} ${guard} ${guard_args[*]}"
    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! OMNI_HOME="${omni_home}" uv run --project "${repo_root}" python "${guard}" \
            "${guard_args[@]}"; then
            log_error "Sibling lock-pin preflight FAILED. Refusing to build a stale image."
            exit 1
        fi
    else
        if ! OMNI_HOME="${omni_home}" "${python_bin}" "${guard}" \
            "${guard_args[@]}"; then
            log_error "Sibling lock-pin preflight FAILED. Refusing to build a stale image."
            exit 1
        fi
    fi
    log_info "Sibling lock-pin preflight passed: all vendored siblings match the lock."
}

check_git_dirty() {
    # Warn if the working tree has uncommitted or untracked changes.
    local repo_root="$1"
    local status_output
    status_output="$(git -C "${repo_root}" status --porcelain 2>/dev/null || true)"
    if [[ -n "${status_output}" ]]; then
        log_warn "Working tree has uncommitted changes."
        log_warn "The deployed SHA will not match the actual file contents."
        # Show untracked files separately for visibility
        local untracked
        untracked="$(echo "${status_output}" | grep '^??' || true)"
        if [[ -n "${untracked}" ]]; then
            local untracked_count
            untracked_count="$(echo "${untracked}" | wc -l | tr -d ' ')"
            log_warn "  Includes ${untracked_count} untracked file(s)."
        fi
    fi
}

guard_prod_promotion_lineage() {
    # Fail-fast when building the prod lane from a dirty or non-promoted tree.
    #
    # Delegates to scripts/check_prod_promotion_lineage.py so the clean-tree +
    # ancestor-of-origin/main lineage rules are enforced by a single, tested
    # source of truth. Only runs when --prod / ONEX_DEPLOY_LANE=prod is set;
    # non-prod lanes keep the advisory check_git_dirty warning (OMN-12626, R1).
    local repo_root="$1"
    if [[ "${PROD_LANE}" != true ]]; then
        return 0
    fi

    log_step "Prod Promotion-Lineage Guard (OMN-12626)"

    local guard="${repo_root}/scripts/check_prod_promotion_lineage.py"
    if [[ ! -f "${guard}" ]]; then
        log_error "Prod promotion-lineage guard not found: ${guard}"
        log_error "Cannot build prod from an unverifiable source tree. Aborting."
        exit 1
    fi

    # Prefer the repo venv, then uv, then system python3 — fail-fast if none run.
    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the prod lineage guard."
        exit 1
    fi

    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! uv run --project "${repo_root}" python "${guard}" --repo "${repo_root}"; then
            log_error "Prod promotion-lineage guard FAILED. Refusing to build prod."
            exit 1
        fi
    else
        if ! "${python_bin}" "${guard}" --repo "${repo_root}"; then
            log_error "Prod promotion-lineage guard FAILED. Refusing to build prod."
            exit 1
        fi
    fi

    log_info "Prod promotion-lineage guard passed: source clean + promoted."
}

guard_lane_deploy_attribution() {
    # Lane-deploy ATTRIBUTION + live-grant INTERLOCK preflight (OMN-15218).
    #
    # Runs BEFORE any mutation (and in dry-run, so an operator sees the refusal
    # during preview rather than after a build starts). Delegates every rule to
    # scripts/preflight_lane_deploy_attribution.py — one tested source of truth
    # for: mandatory ONEX_DEPLOY_REASON on governed lanes (stability-test / prod
    # / judge), durable actor+command+ticket capture, and the refuse-by-default
    # interlock when live, unconsumed prod-promotion grants at
    # onex_change_control@main pin the stability lane's proof.
    #
    # The preflight prints its human summary on stderr and the JSON attribution
    # record on stdout; the record is captured here and folded into registry.json
    # by write_registry().
    local repo_root="$1"
    local compose_project="$2"

    local lane
    lane="$(resolve_lane_name "${compose_project}")"

    log_step "Lane Deploy Attribution + Grant Interlock (OMN-15218)"

    local preflight="${repo_root}/scripts/preflight_lane_deploy_attribution.py"
    if [[ ! -f "${preflight}" ]]; then
        log_error "Lane-deploy attribution preflight not found: ${preflight}"
        log_error "Refusing to deploy lane '${lane}' with no attribution mechanism."
        log_error "  Two unattributed stability rebuilds in two days (2026-07-26, 2026-07-27)"
        log_error "  are exactly what this preflight exists to make impossible (OMN-15218)."
        exit 1
    fi

    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the lane-deploy attribution preflight."
        exit 1
    fi

    local preflight_args=(
        --lane "${lane}"
        --compose-project "${compose_project}"
        --source "deploy-runtime.sh"
        --invoking-command "${SCRIPT_NAME} ${DEPLOY_INVOCATION_ARGS[*]}"
    )
    # Dry-run evaluates and reports but writes no durable record — nothing was
    # deployed, so nothing is attributed; the verdict is still shown.
    if [[ "${MODE}" != "execute" ]]; then
        preflight_args+=(--check-only)
    fi

    log_cmd "${preflight} ${preflight_args[*]}"

    local preflight_exit=0
    if [[ "${python_bin}" == "uv-run" ]]; then
        LANE_ATTRIBUTION_RECORD_JSON="$(uv run --project "${repo_root}" python "${preflight}" \
            "${preflight_args[@]}")" || preflight_exit=$?
    else
        LANE_ATTRIBUTION_RECORD_JSON="$("${python_bin}" "${preflight}" \
            "${preflight_args[@]}")" || preflight_exit=$?
    fi

    if [[ "${preflight_exit}" -ne 0 ]]; then
        log_error "Lane-deploy attribution preflight REFUSED this deploy (exit ${preflight_exit})."
        log_error "  Lane: ${lane} (${compose_project})"
        log_error "  Set ${ONEX_DEPLOY_REASON_VAR} to a real justification, and — when live"
        log_error "  prod-promotion grants pin this lane — acknowledge each grant id via"
        log_error "  ${ONEX_DEPLOY_GRANT_ACK_VAR}. Both are recorded in the attribution record."
        log_error "  Never route around this by calling docker compose directly (OMN-15218)."
        exit 1
    fi

    log_info "Lane-deploy attribution recorded; grant interlock clear."
}

guard_cold_bringup_lane_scope() {
    # Lane-scope guard for the --cold FULL bring-up (OMN-16803).
    #
    # WHY THIS EXISTS. Before OMN-16803 the only --cold lane restriction lived in
    # parse_args, keyed on PROD_LANE — which is set by `--prod` or
    # ONEX_DEPLOY_LANE=prod, NOT by the compose project. parse_args runs before
    # resolve_compose_project, so `--cold` with
    # OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra-prod and no --prod flag
    # sailed straight past it. This guard runs AFTER the lane is resolved, so it
    # sees the real target regardless of how it was selected.
    #
    # LANE POLICY:
    #   prod          REFUSED. A --cold build is workspace-mode (non-main-lineage)
    #                 and the prod-promotion gate refuses those images (OMN-13669).
    #                 Promote a clean-main release via the gated node path.
    #   judge         REFUSED. The lane map declares judge "NOT authorized for
    #                 mutation — read-only" (CLAUDE.md lane table).
    #   stability-test ALLOWED, and this is the OMN-16803 correction. The
    #                 cold-lane runbook previously scoped --cold to dev only,
    #                 lumping stability in with prod under a prod-specific
    #                 rationale ("a workspace image the prod gate refuses"). That
    #                 rationale does not transfer: stability-test is built in
    #                 workspace mode BY DESIGN — its own sanctioned refresh
    #                 (scripts/runtime_build/refresh_stability_lane.sh) sets
    #                 BUILD_SOURCE=workspace DEPLOY_REF=origin/dev. A
    #                 workspace/non-main-lineage image is precisely what the
    #                 candidate-proving lane is supposed to run.
    #
    #                 The gap that scoping left open: a PARTIALLY cold governed
    #                 lane had no recovery path at all. The warm --restart path
    #                 refuses (refresh_stability_lane.sh:418 exit 64 cannot
    #                 resolve a running image id for an absent container), --cold
    #                 was runbook-forbidden, and the only remaining mechanisms
    #                 were the OMN-15243 forbidden raw-compose signatures or a
    #                 preflight bypass. The stability lane sat 6/13 for a month
    #                 behind exactly that dead end (OMN-16803).
    #
    # This guard does NOT weaken anything: the attribution + live-grant interlock
    # (OMN-15218) and the hot-patch ledger preflight (OMN-13014) both still run on
    # this path, unchanged. The --cold-start hot-patch carve-out (OMN-16111) is
    # per-container skip-not-fail, so containers that DO exist on a partially cold
    # lane are still fully tripwire-probed.
    local compose_project="$1"

    if [[ "${COLD_FULL_BRINGUP}" != true ]]; then
        return 0
    fi

    local lane
    lane="$(resolve_lane_name "${compose_project}")"

    case "${lane}" in
        prod)
            log_error "--cold cannot target the prod lane (resolved lane='${lane}', project='${compose_project}')."
            log_error "  A cold bring-up builds a workspace-mode, non-main-lineage image and the"
            log_error "  prod-promotion gate refuses those (OMN-13669)."
            log_error "  Promote a clean-main release to prod via the gated node path instead."
            exit 1
            ;;
        judge)
            log_error "--cold cannot target the judge lane (resolved lane='${lane}', project='${compose_project}')."
            log_error "  The lane map declares judge NOT authorized for mutation — read-only."
            exit 1
            ;;
        stability-test)
            log_info "Cold/partial-cold bring-up targeting the stability-test lane (OMN-16803)."
            log_info "  Sanctioned: this lane is workspace-mode by design (refresh_stability_lane.sh"
            log_info "  sets BUILD_SOURCE=workspace). Attribution + grant interlock and the hot-patch"
            log_info "  ledger preflight both still gate this run."
            log_info "  NOTE: set RUNTIME_BUILD_SERVICES_OVERRIDE to the 4 core services"
            log_info "  (omninode-runtime runtime-effects runtime-worker projection-api) — a"
            log_info "  workspace build of the other 4 is still broken by OMN-14262."
            ;;
        *)
            log_info "Cold bring-up targeting lane '${lane}' (project '${compose_project}')."
            ;;
    esac
}

guard_hotpatch_ledger() {
    # Hot-patch ledger rebuild preflight (OMN-13014, retro B-1).
    #
    # In-container hot-patches (.prepatch sibling discipline) silently revert
    # on any image rebuild / force-recreate. When a hot-patch ledger exists on
    # this host, refuse to build a lane whose recorded patches have source PRs
    # not merged into the build ref, or whose containers carry unledgered
    # .prepatch files. Delegates to scripts/preflight_hotpatch_ledger.py.
    # Sole bypass: HOTPATCH_PREFLIGHT_BYPASS with a Rule-10 user-approval
    # receipt ('# skip-token-allowed: <receipt-id>'), validated by the gate.
    local repo_root="$1"
    local git_sha="$2"
    local compose_project="$3"

    log_step "Hot-Patch Ledger Preflight (OMN-13014)"

    local ledger_path="${HOTPATCH_LEDGER_PATH:-/data/omninode/hotpatch-ledger/ledger.yaml}"
    if [[ ! -f "${ledger_path}" ]]; then
        log_warn "No hot-patch ledger at ${ledger_path} — nothing recorded on this host; gate skipped."
        log_warn "If containers here carry live hot-patches, STOP and write the ledger first."
        return 0
    fi

    local gate="${repo_root}/scripts/preflight_hotpatch_ledger.py"
    if [[ ! -f "${gate}" ]]; then
        log_error "Hot-patch ledger exists at ${ledger_path} but the gate script is missing: ${gate}"
        log_error "Refusing to rebuild over recorded hot-patches without the preflight."
        exit 1
    fi

    # Lane = compose project suffix (omnibase-infra-stability-test -> stability-test);
    # the bare dev project (omnibase-infra) maps to lane 'dev'.
    local lane
    lane="$(resolve_lane_name "${compose_project}")"

    # Workspace builds vendor sibling repos from OMNI_HOME clones; the gate
    # resolves each ledger row's repo build ref (clone HEAD unless overridden
    # via --build-ref) and runs git merge-base --is-ancestor per merge commit.
    local clones_root="${OMNI_HOME:-}"
    if [[ -z "${clones_root}" ]]; then
        log_error "Hot-patch ledger present but OMNI_HOME is unset."
        log_error "Cannot resolve build-input clones for the hot-patch preflight."
        exit 1
    fi

    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the hot-patch ledger preflight."
        exit 1
    fi

    local gate_args=(
        --lane "${lane}"
        --ledger "${ledger_path}"
        --clones-root "${clones_root}"
        --build-ref "omnibase_infra=${git_sha}"
    )
    if [[ "${COLD_FULL_BRINGUP}" == true ]]; then
        # OMN-16111: a from-scratch cold bring-up legitimately has ledgered
        # containers that don't exist yet (this run is what creates them) --
        # tell the gate so it skips-not-fails their tripwire probe instead of
        # treating "container absent" as an unexpected warm-lane vanish.
        gate_args+=(--cold-start)
    fi
    log_cmd "${gate} ${gate_args[*]}"
    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! uv run --project "${repo_root}" python "${gate}" "${gate_args[@]}"; then
            log_error "Hot-patch ledger preflight FAILED. Refusing to rebuild over live hot-patches."
            exit 1
        fi
    else
        if ! "${python_bin}" "${gate}" "${gate_args[@]}"; then
            log_error "Hot-patch ledger preflight FAILED. Refusing to rebuild over live hot-patches."
            exit 1
        fi
    fi

    log_info "Hot-patch ledger preflight passed: all recorded patches merged into the build ref."
}

# =============================================================================
# Concurrency Lock
# =============================================================================

acquire_lock() {
    # Acquire a mkdir-based concurrency lock to prevent parallel deployments.
    mkdir -p "${DEPLOY_ROOT}"

    local pid_file="${LOCK_DIR}/pid"

    # Use mkdir for atomic, cross-platform locking (works on macOS + Linux).
    # mkdir is atomic on all POSIX systems -- it either creates the directory
    # or fails if it already exists, with no race window.
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        # Lock acquired -- write PID immediately to avoid a window where the
        # lock directory exists but has no PID file (Issue: if the script is
        # killed between mkdir and PID write, subsequent runs cannot verify
        # the lock owner and refuse to proceed).
        echo $$ > "${pid_file}"
    else
        # Lock directory exists -- check for stale lock by verifying the
        # owning PID is still alive.
        if [[ -f "${pid_file}" ]]; then
            local lock_pid
            lock_pid="$(cat "${pid_file}" 2>/dev/null || true)"
            # Validate PID is numeric before using it in kill -0.
            # A corrupted or empty PID file is treated as a stale lock.
            if [[ -n "${lock_pid}" ]] && ! [[ "${lock_pid}" =~ ^[0-9]+$ ]]; then
                log_warn "Stale lock detected (PID file contains non-numeric value: '${lock_pid}')."
                log_warn "Treating as corrupted lock and cleaning up..."
                lock_pid=""
            fi
            if [[ -z "${lock_pid}" ]] || ! kill -0 "${lock_pid}" 2>/dev/null; then
                if [[ -n "${lock_pid}" ]]; then
                    log_warn "Stale lock detected (PID ${lock_pid} is no longer running)."
                fi
                log_warn "Cleaning up stale lock and re-acquiring..."
                # Re-read the PID file before removing the lock directory.
                # Between the initial stale check and this point, another
                # process may have legitimately acquired the lock. If the
                # PID file now contains a live process, abort cleanup.
                local recheck_pid
                recheck_pid="$(cat "${pid_file}" 2>/dev/null || true)"
                if [[ -n "${recheck_pid}" ]] && [[ "${recheck_pid}" =~ ^[0-9]+$ ]] \
                        && kill -0 "${recheck_pid}" 2>/dev/null; then
                    log_error "Lock was re-acquired by PID ${recheck_pid} during stale cleanup."
                    log_error "A concurrent deployment is legitimately running. Exiting."
                    exit 2
                fi
                rm -rf "${LOCK_DIR}"
                # Retry mkdir in a short loop to handle the race between rm
                # and mkdir where another process could acquire the lock.
                local lock_acquired=false
                local retry
                for retry in 1 2 3; do
                    if mkdir "${LOCK_DIR}" 2>/dev/null; then
                        # Write PID immediately after acquiring the lock to
                        # eliminate the window where the lock exists without
                        # a PID file.
                        echo $$ > "${pid_file}"
                        lock_acquired=true
                        break
                    fi
                    # Another process grabbed the lock between our rm and mkdir.
                    # Brief sleep before retrying to avoid tight spin.
                    log_warn "Lock contention on retry ${retry}/3, waiting..."
                    sleep 1
                done
                if [[ "${lock_acquired}" != true ]]; then
                    log_error "Another process acquired the lock during stale cleanup."
                    log_error "A concurrent deployment is legitimately running. Exiting."
                    exit 2
                fi
                # Fall through to set up traps and continue
            else
                log_error "Another deployment is in progress (locked by PID ${lock_pid})."
                log_error "If the previous deployment crashed, remove the lock manually:"
                log_error "  rm -rf ${LOCK_DIR}"
                exit 2
            fi
        else
            # Lock directory exists but has no PID file. This happens when the
            # script was killed (e.g., SIGKILL) between mkdir and PID write.
            # Treat as a stale lock and attempt recovery, same as a dead PID.
            log_warn "Lock directory exists but has no PID file (likely interrupted deployment)."
            log_warn "Treating as stale lock and cleaning up..."
            rm -rf "${LOCK_DIR}"
            local lock_acquired=false
            local retry
            for retry in 1 2 3; do
                if mkdir "${LOCK_DIR}" 2>/dev/null; then
                    echo $$ > "${pid_file}"
                    lock_acquired=true
                    break
                fi
                log_warn "Lock contention on retry ${retry}/3, waiting..."
                sleep 1
            done
            if [[ "${lock_acquired}" != true ]]; then
                log_error "Another process acquired the lock during stale cleanup."
                log_error "A concurrent deployment is legitimately running. Exiting."
                exit 2
            fi
        fi
    fi

    # Ensure lock is released on exit (normal, error, or signal).
    # EXIT handles cleanup for normal/error exits.
    # INT/TERM/HUP must explicitly exit after cleanup so the script
    # does not continue executing after receiving a termination signal.
    #
    # ASSUMPTION: acquire_lock() is only called during execute mode (see main()).
    # Dry-run and --print-compose-cmd exit before reaching this code.
    # These traps REPLACE (not chain) any existing EXIT/INT/TERM/HUP traps;
    # this is acceptable because no prior traps are set in this script.
    trap 'cleanup_on_exit' EXIT
    trap 'cleanup_on_exit; exit 1' INT TERM HUP

    log_info "Acquired deployment lock (PID $$)."
}

# =============================================================================
# Cleanup -- partial deployment rollback, --force backup restore, + lock release
# =============================================================================

containers_bound_to_deploy_dir() {
    # OMN-17287: print the names of RUNNING containers that have a bind mount
    # whose source is at or under "$1". Empty output means none.
    #
    # This is the difference between an ORPHANED deployment directory (nothing
    # references it -- safe to remove) and a LIVE one (the lane is serving out
    # of it right now). cleanup_on_exit() must never remove the latter.
    local dir="${1%/}"
    local ids id name src

    [[ -n "${dir}" ]] || return 0
    command -v docker >/dev/null 2>&1 || return 0

    ids="$(docker ps --quiet 2>/dev/null || true)"
    [[ -n "${ids}" ]] || return 0

    while IFS= read -r id; do
        [[ -n "${id}" ]] || continue
        name="$(docker inspect --format '{{.Name}}' "${id}" 2>/dev/null || true)"
        name="${name#/}"
        [[ -n "${name}" ]] || name="${id}"
        while IFS= read -r src; do
            [[ -n "${src}" ]] || continue
            if [[ "${src}" == "${dir}" || "${src}" == "${dir}/"* ]]; then
                printf '%s\n' "${name}"
                break
            fi
        done < <(docker inspect \
            --format '{{range .Mounts}}{{println .Source}}{{end}}' \
            "${id}" 2>/dev/null || true)
    done <<< "${ids}"
}

cleanup_on_exit() {
    # Remove orphaned deployment directory on failure and restore --force backups.
    # If DEPLOY_DIR_TO_CLEANUP is set and registry.json does NOT point to it,
    # the deployment was partial and should be removed. If a --force backup
    # exists (FORCE_BACKUP_DIR), restore it on failure or remove it on success.
    if [[ -n "${DEPLOY_DIR_TO_CLEANUP}" && -d "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
        local active_path=""
        if [[ -f "${REGISTRY_FILE}" ]]; then
            active_path="$(jq -r '.deploy_path // empty' "${REGISTRY_FILE}" 2>/dev/null || true)"
        fi
        if [[ "${active_path}" != "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
            # OMN-17287: a directory the lane's containers are still bind-mounted
            # to is NOT an orphan. DEPLOY_DIR_TO_CLEANUP stays armed through
            # restart_services()/bringup_full_stack() (OMN-15352 made the registry
            # write commit-on-success, so there is no earlier safe disarm point),
            # which means any failure after containers start would otherwise
            # rm -rf the payload out from under a running lane. Docker then
            # re-creates the missing bind sources as empty root-owned directories
            # on the next container restart, and the runtime fail-fasts with
            # "RuntimeHostProcess requires 'service_name'" because /app/contracts
            # is empty. Refuse, and leave the lane recoverable -- a re-run rsyncs
            # over this directory anyway.
            local bound_containers
            bound_containers="$(containers_bound_to_deploy_dir "${DEPLOY_DIR_TO_CLEANUP}")"
            if [[ -n "${bound_containers}" ]]; then
                log_error "================================================================="
                log_error "REFUSING to remove partial deployment: ${DEPLOY_DIR_TO_CLEANUP}"
                log_error "Running containers are still bind-mounted to it:"
                while IFS= read -r _bound_name; do
                    [[ -n "${_bound_name}" ]] || continue
                    log_error "  - ${_bound_name}"
                done <<< "${bound_containers}"
                log_error "Removing it would strand those containers on a deleted"
                log_error "payload (empty /app/contracts on their next restart)."
                log_error "The directory is left in place. Re-run the deploy to"
                log_error "re-sync it, or stop the lane first if you must remove it."
                log_error "================================================================="
            else
                log_warn "Cleaning up partial deployment: ${DEPLOY_DIR_TO_CLEANUP}"
                rm -rf "${DEPLOY_DIR_TO_CLEANUP}" 2>/dev/null || true
            fi
        fi
    fi

    # If a --force backup exists, decide whether to restore it or clean it up
    # based on whether the full deployment completed successfully.
    if [[ -n "${FORCE_BACKUP_DIR}" && -d "${FORCE_BACKUP_DIR}" ]]; then
        # Derive the original deployment directory from the backup path.
        # Backup convention: {deploy_target}.bak -> restore to {deploy_target}
        local original_dir="${FORCE_BACKUP_DIR%.bak}"
        if [[ "${DEPLOYMENT_COMPLETE}" != "true" ]]; then
            # Deployment did not complete -- restore previous working deployment.
            # This covers both pre-registry failures (rsync/sanity) and
            # post-registry failures (build/restart/verify).
            log_warn "Restoring previous deployment from backup: ${FORCE_BACKUP_DIR}"
            rm -rf "${original_dir}" 2>/dev/null || true
            if ! mv "${FORCE_BACKUP_DIR}" "${original_dir}" 2>/dev/null; then
                log_error "================================================================="
                log_error "CRITICAL: Failed to restore previous deployment from backup!"
                log_error "Backup location: ${FORCE_BACKUP_DIR}"
                log_error "Expected restore target: ${original_dir}"
                log_error "Manual recovery required: mv '${FORCE_BACKUP_DIR}' '${original_dir}'"
                log_error "================================================================="
            else
                # OMN-13364: the restored tree carries the PRE-BUILD vendored
                # migration tree. Re-apply the freshly-synced migration tree
                # (snapshot taken after sync_files) so the deployed migrations
                # match the build source instead of silently regressing to the
                # backup's stale snapshot (which dropped a forward migration in
                # the 2026-06-19 stability redeploy).
                restore_migration_tree_after_revert "${original_dir}"
                # OMN-15352: registry.json is no longer stale here. write_registry()
                # is now commit-on-success -- it only runs after every phase that
                # can fail has passed, so on any non-success exit (including this
                # restore branch) it never ran this invocation, and registry.json
                # still holds whatever it held before this deploy started.
                log_info "registry.json is unaffected by this restore (written only on full deploy success, OMN-15352)."
            fi
        else
            # Full deployment succeeded -- backup is stale, clean it up.
            log_info "Cleaning up stale backup: ${FORCE_BACKUP_DIR}"
            rm -rf "${FORCE_BACKUP_DIR}" 2>/dev/null || true
        fi
        FORCE_BACKUP_DIR=""
    fi

    # OMN-15352 F3: restore every RUNTIME_BUILD_SERVICES `:latest` tag to its
    # pre-build state on any non-success exit. This is independent of whether a
    # --force backup exists -- a build/restart/verify/readback failure can leave
    # `:latest` pointed at an unverified image even on a first-ever (non-force)
    # deploy, so it is not covered by the FORCE_BACKUP_DIR branch above.
    if [[ "${DEPLOYMENT_COMPLETE}" != "true" ]]; then
        restore_latest_image_tags
        # OMN-15718: retagging `:latest` does not touch running container
        # state. A restart_services()/bringup_full_stack() call that failed or
        # timed out partway through can leave RUNTIME_BUILD_SERVICES containers
        # stranded in 'Created' (never started). Reconcile every one of them
        # back to running, or explicitly tear it down if it cannot recover --
        # never leave the lane in an ambiguous state that needs manual `docker
        # start`/`docker ps` forensics.
        reconcile_runtime_container_start_state
    fi
    if [[ -n "${LATEST_TAG_SNAPSHOT_FILE}" && -f "${LATEST_TAG_SNAPSHOT_FILE}" ]]; then
        rm -f "${LATEST_TAG_SNAPSHOT_FILE}" 2>/dev/null || true
    fi
    LATEST_TAG_SNAPSHOT_FILE=""

    # OMN-13364: remove the migration-tree snapshot taken after sync_files.
    if [[ -n "${MIGRATION_TREE_SNAPSHOT_DIR}" && -d "${MIGRATION_TREE_SNAPSHOT_DIR}" ]]; then
        rm -rf "${MIGRATION_TREE_SNAPSHOT_DIR}" 2>/dev/null || true
    fi
    MIGRATION_TREE_SNAPSHOT_DIR=""

    # Release concurrency lock
    rm -rf "${LOCK_DIR}" 2>/dev/null || true
}

assert_deployed_migration_tree_synced() {
    # OMN-13415: assert the deployed (bind-mounted) forward-migration tree is
    # byte-identical to the canonical clone @ the target SHA before any migration
    # runs. The stability-promotion footgun (stale 0016, missing 0018/0019) made a
    # lane look "deployed" while applying the wrong migration SQL; this gate makes
    # that drift abort the deploy instead of silently mis-migrating.
    local deploy_target="$1"
    local repo_root="$2"
    local git_sha="$3"
    local deployed_tree="${deploy_target}/${MIGRATION_TREE_REL_PATH}"

    if [[ ! -d "${deployed_tree}" ]]; then
        # No bind-mounted forward-migration tree in this deployment layout; nothing
        # to assert (matches snapshot_migration_tree's own no-tree tolerance).
        log_warn "No deployed migration tree at ${deployed_tree}; skipping sync assertion."
        return 0
    fi

    local check_script="${repo_root}/scripts/check_deployed_migration_tree_sync.py"
    if [[ ! -f "${check_script}" ]]; then
        log_error "Migration-sync gate script missing: ${check_script}"
        exit 1
    fi

    log_info "Asserting deployed migration tree == canonical clone @ ${git_sha} (OMN-13415)..."
    if ! python3 "${check_script}" \
        --deployed-tree "${deployed_tree}" \
        --clone-root "${repo_root}" \
        --ref "${git_sha}" \
        --tree-rel-path "${MIGRATION_TREE_REL_PATH}"; then
        log_error "Deployed migration tree is OUT OF SYNC with the canonical clone @ ${git_sha}."
        log_error "Aborting deploy to avoid applying a stale migration set (OMN-13415)."
        exit 1
    fi
    log_info "Deployed migration tree is in sync with the canonical clone @ ${git_sha}."
}

snapshot_migration_tree() {
    # Preserve a copy of the freshly-synced vendored forward-migration tree so a
    # later backup-restore (cleanup_on_exit) can re-apply it instead of leaving
    # the restored tree on the backup's stale, pre-build migrations (OMN-13364).
    local deploy_target="$1"
    local src_tree="${deploy_target}/${MIGRATION_TREE_REL_PATH}"

    if [[ ! -d "${src_tree}" ]]; then
        # No vendored migration tree to protect (e.g. a deployment layout that
        # does not bind-mount forward migrations). Nothing to snapshot.
        log_warn "No vendored migration tree at ${src_tree}; skipping snapshot."
        return 0
    fi

    local snapshot_dir="${deploy_target}.migrations.snapshot"
    rm -rf "${snapshot_dir}" 2>/dev/null || true
    mkdir -p "${snapshot_dir}"
    # Mirror the tree exactly so re-apply is a faithful copy of the build source.
    rsync -a --delete "${src_tree}/" "${snapshot_dir}/"
    MIGRATION_TREE_SNAPSHOT_DIR="${snapshot_dir}"
    log_info "Snapshotted vendored migration tree for restore safety: ${snapshot_dir}"
}

restore_migration_tree_after_revert() {
    # Re-apply the freshly-synced vendored migration tree onto a restored
    # deployment tree so a backup-restore never silently regresses migrations to
    # the backup's pre-build snapshot (OMN-13364).
    local restored_dir="$1"

    if [[ -z "${MIGRATION_TREE_SNAPSHOT_DIR}" || ! -d "${MIGRATION_TREE_SNAPSHOT_DIR}" ]]; then
        # The failure happened before sync_files snapshotted the tree (e.g. an
        # rsync/sanity failure). In that case nothing newer than the backup was
        # produced, so the backup's migration tree is already the correct one.
        log_warn "No migration-tree snapshot to re-apply; restored tree keeps the backup migrations."
        return 0
    fi

    local dst_tree="${restored_dir}/${MIGRATION_TREE_REL_PATH}"
    log_warn "Re-applying freshly-built vendored migration tree onto restored deployment:"
    log_warn "  ${MIGRATION_TREE_SNAPSHOT_DIR}/ -> ${dst_tree}/"
    mkdir -p "${dst_tree}"
    if rsync -a --delete "${MIGRATION_TREE_SNAPSHOT_DIR}/" "${dst_tree}/"; then
        log_warn "Migration tree re-applied: deployed migrations match the build source, not the backup."
    else
        log_error "================================================================="
        log_error "CRITICAL: Failed to re-apply the vendored migration tree after restore!"
        log_error "The restored deployment may carry STALE migrations (silent loss risk)."
        log_error "Manual recovery: rsync -a --delete '${MIGRATION_TREE_SNAPSHOT_DIR}/' '${dst_tree}/'"
        log_error "================================================================="
    fi
}

# =============================================================================
# Prune -- remove old deployments beyond retention limit
# =============================================================================

prune_old_deployments() {
    # Remove old deployment directories that exceed the retention limit.
    local deployed_root="${DEPLOY_ROOT}/deployed"

    if [[ ! -d "${deployed_root}" ]]; then
        return 0
    fi

    log_step "Prune Old Deployments"

    # Determine active deployment path from registry
    local active_path=""
    if [[ -f "${REGISTRY_FILE}" ]]; then
        active_path="$(jq -r '.deploy_path // empty' "${REGISTRY_FILE}" 2>/dev/null || true)"
    fi

    # Collect all deployment directories sorted by modification time,
    # newest first. Each entry is a full path like
    # ~/.omnibase/infra/deployed/1.2.3/
    local all_deployments=()
    local version_dir
    for version_dir in "${deployed_root}"/*/; do
        [[ -d "${version_dir}" ]] || continue
        # Skip backup directories from failed --force deploys
        [[ "$(basename "${version_dir}")" == *.bak ]] && continue
        all_deployments+=("${version_dir%/}")
    done

    # Sort by modification time (newest first) using stat.
    # macOS stat uses -f '%m' for epoch; GNU stat uses -c '%Y'.
    local sorted_deployments=()
    if stat -f '%m' / >/dev/null 2>&1; then
        # macOS (BSD stat)
        while IFS= read -r line; do
            sorted_deployments+=("${line}")
        done < <(
            for d in "${all_deployments[@]}"; do
                printf '%s %s\n' "$(stat -f '%m' "${d}")" "${d}"
            done | sort -rn | awk '{print $2}'
        )
    else
        # Linux (GNU stat)
        while IFS= read -r line; do
            sorted_deployments+=("${line}")
        done < <(
            for d in "${all_deployments[@]}"; do
                printf '%s %s\n' "$(stat -c '%Y' "${d}")" "${d}"
            done | sort -rn | awk '{print $2}'
        )
    fi

    local total="${#sorted_deployments[@]}"
    if (( total <= MAX_DEPLOYMENTS )); then
        log_info "Deployment count (${total}) within retention limit (${MAX_DEPLOYMENTS}). No pruning needed."
        return 0
    fi

    log_info "Found ${total} deployments, retention limit is ${MAX_DEPLOYMENTS}. Pruning..."

    local kept=0
    local pruned=0
    for deploy_dir in "${sorted_deployments[@]}"; do
        if (( kept < MAX_DEPLOYMENTS )); then
            kept=$((kept + 1))
            continue
        fi

        # Never remove the currently active deployment
        if [[ "${deploy_dir}" == "${active_path}" ]]; then
            log_info "  Skipping active deployment: ${deploy_dir}"
            continue
        fi

        log_info "  Removing old deployment: ${deploy_dir}"
        rm -rf "${deploy_dir}"
        pruned=$((pruned + 1))
    done

    log_info "Pruned ${pruned} old deployment(s). Kept ${kept}."
}

# =============================================================================
# Guard -- refuse to overwrite unless --force
# =============================================================================

guard_existing_deployment() {
    # Refuse to overwrite an existing deployment directory unless --force is set.
    # When --force is active, the existing directory is moved to a .bak backup
    # so it can be restored if the new deployment fails.
    local deploy_target="$1"

    if [[ -d "${deploy_target}" ]]; then
        if [[ "${FORCE}" == true ]]; then
            log_warn "====================================================="
            log_warn "OVERWRITING existing deployment at:"
            log_warn "  ${deploy_target}"
            log_warn "====================================================="

            # Back up the existing deployment so cleanup_on_exit can restore
            # it if the new deployment fails partway through.
            local backup_dir="${deploy_target}.bak"

            # Remove any leftover backup from a previous failed --force deploy
            if [[ -d "${backup_dir}" ]]; then
                log_warn "Removing stale backup: ${backup_dir}"
                rm -rf "${backup_dir}"
            fi

            log_info "Backing up existing deployment to: ${backup_dir}"
            if ! mv "${deploy_target}" "${backup_dir}"; then
                log_error "Failed to back up existing deployment."
                log_error "Cannot proceed with --force: unable to move '${deploy_target}' to '${backup_dir}'"
                exit 1
            fi
            FORCE_BACKUP_DIR="${backup_dir}"
        else
            log_error "Deployment directory already exists:"
            log_error "  ${deploy_target}"
            log_error ""
            log_error "This version has already been deployed."
            log_error "To overwrite, re-run with --force:"
            log_error "  ${SCRIPT_NAME} --execute --force"
            exit 1
        fi
    fi
}

# =============================================================================
# Preview
# =============================================================================

count_files() {
    # Count regular files in a directory (up to 5 levels deep).
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        # -maxdepth 5: prevent runaway traversal in deeply nested trees
        # -type f: matches only regular files (symlinks are excluded by default
        #   since find does not follow them without -L)
        find "${dir}" -maxdepth 5 -type f | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

show_preview() {
    # Display a summary of what would be deployed in dry-run mode.
    local repo_root="$1"
    local version="$2"
    local git_sha="$3"
    local deploy_target="$4"
    local compose_project="$5"

    log_step "Deployment Preview"

    log_info "Source repository:    ${repo_root}"
    log_info "Version:             ${version}"
    log_info "Git SHA:             ${git_sha}"
    log_info "Deploy target:       ${deploy_target}"
    log_info "Compose project:     ${compose_project}"
    log_info "Compose profile:     ${COMPOSE_PROFILE}"
    log_info "Mode:                ${MODE}"
    log_info "Force overwrite:     ${FORCE}"
    log_info "Restart containers:  ${RESTART}"
    log_info ""
    log_info "File counts (source):"
    log_info "  src/omnibase_infra/  $(count_files "${repo_root}/src/omnibase_infra") files"
    log_info "  contracts/           $(count_files "${repo_root}/contracts") files"
    log_info "  docker/              $(count_files "${repo_root}/docker") files"
    log_info "  scripts/runtime_build/ $(count_files "${repo_root}/scripts/runtime_build") files"
    log_info "  workspace/sibling-repos/ $(count_files "${repo_root}/workspace/sibling-repos") files"

    # .env strategy
    if [[ -d "${deploy_target}" && -f "${deploy_target}/docker/.env" ]]; then
        log_info "  .env strategy:       preserve existing"
    elif [[ -f "${repo_root}/docker/.env" ]]; then
        log_info "  .env strategy:       copy from repo docker/.env"
    elif [[ -f "${repo_root}/docker/.env.example" ]]; then
        log_info "  .env strategy:       copy from .env.example (WARNING: edit before use)"
    else
        log_info "  .env strategy:       none available (WARNING: compose will fail)"
    fi
}

# =============================================================================
# Sync -- rsync repository to deployment target
# =============================================================================

resolve_core_contracts_dir() {
    # Resolve the omnibase_core runtime contracts directory (OMN-6698 / OMN-15122).
    #
    # Populates two caller-provided variables, both passed by name (never via
    # command substitution -- a `$(...)` wrapper forks a subshell, and a `local -n`
    # array populated inside that subshell would NOT propagate back to the
    # caller): the first receives every path probed, in probe order; the second
    # receives the resolved directory on success (left empty on failure).
    #
    # Usage:
    #   local -a probed=()
    #   local resolved=""
    #   if resolve_core_contracts_dir probed resolved; then ...
    #
    # Primary: filesystem resolution from the OMNI_HOME sibling clone/checkout
    # -- src/omnibase_core/contracts/runtime_data relative to the omnibase_core
    # repo root. This is the deploy source of truth (the same pinned sibling
    # clone the rest of the workspace build vendors from) and it works even
    # when omnibase_core is not pip-installed on the deploy runner at all --
    # the OMN-15122 failure: `importlib.util.find_spec('omnibase_core')`
    # returned None on the runner's python3, and the previous editable-install
    # fallback assumed a site-packages-shaped layout
    # (`<pkg_dir>/../.. /contracts/runtime_data`) that does not match the real
    # source-tree layout (`<repo>/src/omnibase_core/contracts/runtime_data`).
    #
    # Secondary: python import resolution, kept for hosts where omnibase_core
    # IS installed (e.g. a developer workstation running this script directly
    # against a pip/editable install with no OMNI_HOME sibling clone).
    #
    # Fails closed: leaves the resolved-dir output empty and returns 1 if
    # neither probe resolves.
    # NOTE: internal locals below are deliberately prefixed `_resolve_ccd_*`
    # (never `resolved`/`probed`) -- a `local -n` nameref breaks silently if a
    # plain local variable inside this function shares the caller-chosen target
    # name (e.g. caller passes a variable literally named "resolved"), aliasing
    # the nameref to itself instead of the caller's variable. Verified via a
    # standalone bash harness during development: an earlier draft using
    # `local resolved=""` here produced an empty resolved-dir output AND
    # incorrectly returned success on the fail-closed case once the caller's
    # own variable was also named "resolved".
    local _out_probed_name="$1"
    local _out_resolved_name="$2"
    eval "${_out_resolved_name}=''"
    local _resolve_ccd_dir=""

    if [[ -n "${OMNI_HOME:-}" ]]; then
        local fs_candidate="${OMNI_HOME}/omnibase_core/src/omnibase_core/contracts/runtime_data"
        eval "${_out_probed_name}+=( $(printf '%q' "${fs_candidate}") )"
        if [[ -d "${fs_candidate}" ]]; then
            _resolve_ccd_dir="${fs_candidate}"
        fi
    else
        eval "${_out_probed_name}+=( $(printf '%q' "<OMNI_HOME unset -- cannot probe the sibling clone filesystem path>") )"
    fi

    if [[ -z "${_resolve_ccd_dir}" ]]; then
        local py_candidate
        py_candidate="$(python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('omnibase_core')
if spec and spec.origin:
    pkg_dir = pathlib.Path(spec.origin).parent
    runtime_data = pkg_dir / 'contracts' / 'runtime_data'
    if not runtime_data.is_dir():
        # Fallback: check sibling contracts/runtime_data directory (editable installs)
        runtime_data = pkg_dir.parent.parent / 'contracts' / 'runtime_data'
    if runtime_data.is_dir():
        print(runtime_data)
" 2>/dev/null || true)"
        if [[ -n "${py_candidate}" ]]; then
            eval "${_out_probed_name}+=( $(printf '%q' "${py_candidate} (python find_spec('omnibase_core') resolution)") )"
            _resolve_ccd_dir="${py_candidate}"
        else
            eval "${_out_probed_name}+=( $(printf '%q' "<python find_spec('omnibase_core') returned no importable spec/origin>") )"
        fi
    fi

    if [[ -n "${_resolve_ccd_dir}" ]]; then
        eval "${_out_resolved_name}=$(printf '%q' "${_resolve_ccd_dir}")"
        return 0
    fi
    return 1
}

sync_files() {
    # Rsync repository files to the versioned deployment target directory.
    local repo_root="$1"
    local deploy_target="$2"

    log_step "Sync Files"

    mkdir -p "${deploy_target}/docker"

    # 1. Root files (pyproject.toml, uv.lock, README.md, LICENSE)
    log_info "Syncing root files..."
    log_cmd "rsync pyproject.toml, uv.lock, README.md, LICENSE"
    rsync -a \
        "${repo_root}/pyproject.toml" \
        "${repo_root}/uv.lock" \
        "${deploy_target}/"

    # Copy README.md and LICENSE if they exist (optional files)
    for f in README.md LICENSE; do
        if [[ -f "${repo_root}/${f}" ]]; then
            rsync -a "${repo_root}/${f}" "${deploy_target}/"
        fi
    done

    # 2. Source code
    log_info "Syncing src/ directory..."
    log_cmd "rsync -a --delete src/ -> deployed"
    rsync -a --delete \
        "${repo_root}/src/" "${deploy_target}/src/"

    # 3. Contracts (if directory exists)
    if [[ -d "${repo_root}/contracts/" ]]; then
        log_info "Syncing contracts/..."
        log_cmd "rsync -a --delete contracts/ -> deployed"
        rsync -a --delete \
            "${repo_root}/contracts/" "${deploy_target}/contracts/"
    else
        log_info "No contracts/ directory present, skipping contracts sync."
    fi

    # 3b. Copy omnibase_core runtime contract YAMLs into contracts/runtime/
    # OMN-6698: The bind-mount (../contracts:/app/contracts:ro) in docker-compose
    # overrides the Dockerfile's baked-in contracts. The Dockerfile copies these
    # from the installed omnibase_core package (contracts/runtime_data/), but
    # the bind-mount hides them. We must copy them into the deployed contracts/
    # directory so they survive the bind-mount override.
    #
    # OMN-15122: resolution is delegated to resolve_core_contracts_dir(), which
    # probes the OMNI_HOME sibling clone's real source-tree path FIRST (the
    # deploy runner has no omnibase_core installed at all, so the prior
    # python-only resolution always failed there) and falls back to python
    # import resolution second.
    check_command python3 "locating omnibase_core runtime contracts"
    local -a core_contracts_probed=()
    local core_contracts_dir=""
    resolve_core_contracts_dir core_contracts_probed core_contracts_dir || true

    if [[ -n "${core_contracts_dir}" && -d "${core_contracts_dir}" ]]; then
        log_info "Copying omnibase_core runtime contracts from ${core_contracts_dir}..."
        mkdir -p "${deploy_target}/contracts/runtime"
        local expected_core_runtime_count=5
        local yaml_count=0
        for yaml_file in "${core_contracts_dir}"/*.yaml; do
            if [[ -f "${yaml_file}" ]]; then
                cp -f "${yaml_file}" "${deploy_target}/contracts/runtime/"
                yaml_count=$((yaml_count + 1))
            fi
        done
        if (( yaml_count < expected_core_runtime_count )); then
            log_error "Expected at least ${expected_core_runtime_count} runtime contract YAMLs in ${core_contracts_dir}, found ${yaml_count}."
            log_error "Aborting deployment to avoid runtime startup failure."
            exit 1
        fi
        log_info "Copied ${yaml_count} runtime contract YAMLs from omnibase_core."
    else
        log_error "Could not locate omnibase_core runtime contracts."
        log_error "Probed the following paths:"
        for probed_path in "${core_contracts_probed[@]}"; do
            log_error "  - ${probed_path}"
        done
        log_error "Aborting deployment to avoid runtime startup failure."
        log_error "Ensure OMNI_HOME points at a clone containing omnibase_core (with"
        log_error "src/omnibase_core/contracts/runtime_data), or that omnibase_core is"
        log_error "installed: uv pip install omnibase-core"
        exit 1
    fi

    # 3c. Config (repo-tracked deploy-time config baked into the image)
    # OMN-15696: Dockerfile.runtime COPYs config/runner_fleet.yaml (OMN-15676),
    # but sync_files() never rsynced config/ into the deployed build context, so
    # any --force redeploy or cold bring-up that recreates deployed/<version>/
    # fails the image build with "failed to calculate checksum of ref
    # ...:/config/runner_fleet.yaml: not found" -- the same COPY-without-matching-
    # rsync class OMN-12987 fixed for workspace/. Sync the whole directory (not
    # just runner_fleet.yaml) so a future config/ COPY addition doesn't reopen
    # the same gap.
    if [[ -d "${repo_root}/config/" ]]; then
        log_info "Syncing config/..."
        log_cmd "rsync -a --delete config/ -> deployed"
        rsync -a --delete \
            "${repo_root}/config/" "${deploy_target}/config/"
    else
        log_info "No config/ directory present, skipping config sync."
    fi

    # 4. Docker files -- with preserve allowlist
    #    .env, .env.local, certs/, overrides/ survive --delete
    #    Excludes use a leading '/' to anchor them to the transfer root (docker/),
    #    so only top-level .env and .env.local are excluded; nested .env files in
    #    subdirectories are synced normally.
    log_info "Syncing docker/ (preserving .env, .env.local, certs/, overrides/)..."
    log_cmd "rsync -a --delete --exclude='/.env' --exclude='/.env.local' --exclude='/certs/' --exclude='/overrides/' docker/ -> deployed"
    # Note: .env is excluded from rsync -- env vars come from the shell environment
    # (sourced from ~/.omnibase/.env at script top). No stale .env copy needed.
    rsync -a --delete \
        --exclude='/.env' \
        --exclude='/.env.local' \
        --exclude='/certs/' \
        --exclude='/overrides/' \
        "${repo_root}/docker/" "${deploy_target}/docker/"

    # 5. Runtime build context paths required by docker/Dockerfile.runtime.
    # Release-mode builds still COPY these paths, even when sibling repos are
    # represented only by the committed .gitkeep placeholder.
    stage_workspace_if_needed "${repo_root}"

    log_info "Syncing runtime build context..."
    mkdir -p "${deploy_target}/scripts" "${deploy_target}/workspace"
    log_cmd "rsync -a --delete scripts/runtime_build/ -> deployed"
    rsync -a --delete \
        "${repo_root}/scripts/runtime_build/" "${deploy_target}/scripts/runtime_build/"
    log_cmd "rsync -a --delete workspace/sibling-repos/ -> deployed"
    rsync -a --delete \
        "${repo_root}/workspace/sibling-repos/" "${deploy_target}/workspace/sibling-repos/"
    # The lock-pin preflight result (OMN-12987) lives under sibling-repos/ as
    # .sibling-lock-pins.json, so the rsync above already carries it into the
    # build context for the in-image provenance merge.

    # Carry the root-level workspace/ file that Dockerfile.runtime COPYs
    # (workspace/sibling-pin-comparison.json, line ~278). The sibling-repos/
    # rsync above only covers the subdirectory; without this the deployed build
    # context lacks the comparison file and `docker build` fails with
    # "failed to calculate checksum of ref ...:/workspace/sibling-pin-comparison.json:
    # not found" -- the bug fixed in OMN-12987 (the dev compose build only worked
    # because it runs from the repo root where the committed placeholder exists).
    # Release mode ships the committed placeholder; workspace mode ships the real
    # expected-vs-actual comparison stage_workspace.sh wrote into the repo root
    # (OMN-12977). The regression test
    # tests/scripts/test_deploy_runtime_build_context.py asserts every
    # COPY-from-workspace path the Dockerfile references is staged here, so a
    # future Dockerfile COPY without a matching rsync fails CI.
    log_cmd "rsync -a workspace/sibling-pin-comparison.json -> deployed"
    rsync -a \
        "${repo_root}/workspace/sibling-pin-comparison.json" \
        "${deploy_target}/workspace/sibling-pin-comparison.json"

    # Carry the per-repo VCS provenance file Dockerfile.runtime COPYs
    # (workspace/sibling-vcs-provenance.json, OMN-13030). Same rationale as the
    # pin-comparison file above: the sibling-repos/ rsync only covers the
    # subdirectory, so without this the deployed build context lacks the file and
    # `docker build` fails on the COPY. Release mode ships the committed
    # placeholder; workspace mode ships the real per-repo {vcs_ref, vcs_dirty,
    # vcs_branch} stage_workspace.sh wrote into the repo root.
    log_cmd "rsync -a workspace/sibling-vcs-provenance.json -> deployed"
    rsync -a \
        "${repo_root}/workspace/sibling-vcs-provenance.json" \
        "${deploy_target}/workspace/sibling-vcs-provenance.json"

    # 6. Migration scripts (bind-mounted by docker-compose.infra.yml) plus the
    # Keycloak realm reconciler docker/Dockerfile.runtime COPYs into the image
    # (scripts/seed-keycloak-clients.py, OMN-16026). This include-list is the
    # only rsync that ever touches scripts/ files outside scripts/runtime_build/
    # (which has its own directory-wide sync above), so every file the runtime
    # image or its Job manifests need out of scripts/ must be listed here
    # explicitly -- tests/scripts/test_deploy_runtime_build_context.py derives
    # the required set from docker/Dockerfile.runtime's COPY sources and fails
    # CI if a future COPY isn't matched by an --include here.
    log_info "Syncing migration scripts..."
    mkdir -p "${deploy_target}/scripts"
    rsync -a \
        --include='run-forward-migrations.sh' \
        --include='check_migrations_complete.sh' \
        --include='run-intelligence-migrations.sh' \
        --include='seed-keycloak-clients.py' \
        --exclude='*' \
        "${repo_root}/scripts/" "${deploy_target}/scripts/"

    log_info "Sync complete."
}

# =============================================================================
# Env Setup (REMOVED -- F65 / OMN-6910)
# =============================================================================
# The old setup_env() copied ~/.omnibase/.env into a stale snapshot at
# ${deploy_target}/docker/.env. Docker compose --env-file then read from
# that snapshot instead of the live shell environment. This caused env var
# changes to be silently ignored until the next full redeploy.
#
# Fix: source ~/.omnibase/.env at script top (see line 28) and let docker
# compose resolve ${VAR} references from the shell environment directly.
# No --env-file, no stale copies.

# =============================================================================
# Compose Project Collision Detection
# =============================================================================
#
# Detects whether the target compose project name is currently owned by a
# DIFFERENT deployment directory. This guards against the Feb 15 (OMN-2233)
# class of incident where multiple repo copies share the same compose project
# name, causing containers from the wrong copy to silently continue running.
#
# How it works:
#   Docker labels every container with the working directory of the compose
#   invocation via com.docker.compose.project.working_dir. We compare that
#   label against the resolved deploy target to detect cross-copy ownership.
#
# Scenarios:
#   - No running containers for the project  → no collision, safe to proceed
#   - Running containers from THIS deploy dir → already deployed, safe to proceed
#   - Running containers from a DIFFERENT dir → COLLISION, exit 1
#
# The check runs in BOTH dry-run and execute modes so operators see the
# warning even during a preview.

check_compose_project_collision() {
    local compose_project="$1"
    local deploy_target="$2"

    log_step "Compose Project Collision Check"

    # Query running containers for this compose project name.
    # Use --all (not just running) to catch stopped-but-not-removed containers
    # that still hold the project label, which would cause collisions on `up`.
    local running_dirs
    running_dirs="$(
        docker ps --all \
            --filter "label=com.docker.compose.project=${compose_project}" \
            --format '{{index .Labels "com.docker.compose.project.working_dir"}}' \
            2>/dev/null \
        | sort -u \
        | grep -v '^$' \
        || true
    )"

    if [[ -z "${running_dirs}" ]]; then
        log_info "No running containers for project '${compose_project}'. No collision."
        return 0
    fi

    log_info "Found containers for project '${compose_project}' from: ${running_dirs}"

    # Normalize paths: resolve symlinks so that ~/.omnibase and /home/... compare equal.
    local resolved_deploy_target
    resolved_deploy_target="$(cd "${deploy_target}" 2>/dev/null && pwd -P || echo "${deploy_target}")"

    local collision_detected=false
    local colliding_dirs=()

    while IFS= read -r running_dir; do
        [[ -z "${running_dir}" ]] && continue

        local resolved_running_dir
        resolved_running_dir="$(cd "${running_dir}" 2>/dev/null && pwd -P || echo "${running_dir}")"

        if [[ "${resolved_running_dir}" != "${resolved_deploy_target}" ]]; then
            collision_detected=true
            colliding_dirs+=("${running_dir}")
        fi
    done <<< "${running_dirs}"

    if [[ "${collision_detected}" == true ]]; then
        log_error "============================================================"
        log_error "COMPOSE PROJECT COLLISION DETECTED"
        log_error "============================================================"
        log_error ""
        log_error "Compose project '${compose_project}' is already running"
        log_error "from a DIFFERENT directory:"
        for dir in "${colliding_dirs[@]}"; do
            log_error "  Running from: ${dir}"
        done
        log_error "  You are in:   ${deploy_target}"
        log_error ""
        log_error "Proceeding would deploy from this copy while the other copy's"
        log_error "containers continue to own the compose project. This causes"
        log_error "silent failures where code changes have no effect."
        log_error ""
        log_error "To resolve:"
        log_error "  1. Stop containers from the other copy first:"
        log_error "     docker compose -p ${compose_project} down"
        log_error "  2. Then re-run this script."
        log_error ""
        log_error "Or, if you are certain this is the correct copy:"
        log_error "  Manually stop all containers for project '${compose_project}'"
        log_error "  and remove the stale deployment from: ${colliding_dirs[0]}"
        log_error "============================================================"
        exit 1
    fi

    log_info "Collision check passed: containers are from the expected deployment directory."
}

# =============================================================================
# Sanity Check -- validate compose can resolve all paths
# =============================================================================

sanity_check() {
    # Validate that docker compose config resolves cleanly from the deployed directory.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Post-Sync Sanity Check"

    log_info "Validating compose configuration from deployed directory..."
    log_cmd "docker compose -p ${compose_project} ${compose_args[*]} config --quiet"

    local config_output
    if ! config_output="$(docker compose \
        -p "${compose_project}" \
        "${compose_args[@]}" \
        config --quiet 2>&1)"; then
        log_error "Compose configuration validation failed."
        if [[ -n "${config_output}" ]]; then
            log_error "Compose output:"
            while IFS= read -r line; do
                log_error "  ${line}"
            done <<< "${config_output}"
        fi
        log_error "The deployed directory structure may be incomplete."
        log_error "Check that src/, contracts/, and docker/ are properly synced."
        exit 1
    fi

    log_info "Compose configuration is valid."
}

# =============================================================================
# Registry -- atomic write of deployment metadata
# =============================================================================

write_registry() {
    # Atomically write deployment metadata to registry.json.
    local version="$1"
    local git_sha="$2"
    local deploy_target="$3"
    local repo_root="$4"
    local compose_project="$5"

    log_step "Write Registry"

    local deployed_at
    deployed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    local tmp_file="${REGISTRY_FILE}.tmp"

    # Restrict temp file permissions to 600 (owner-only read/write) to prevent
    # other users from reading deployment metadata while the file is being written.
    local old_umask
    old_umask="$(umask)"
    umask 077

    # OMN-15218: carry the attribution record (actor, reason, ticket, invoking
    # command, grant-interlock verdict + any acknowledgement) into registry.json.
    # Before this, registry.json answered "what is deployed" but nothing on the
    # box answered "who deployed it and why" — the exact gap that made two
    # stability rebuilds unattributable. Defensive: if the record is missing or
    # not valid JSON, write `null` rather than corrupting the registry (the
    # preflight already hard-failed the deploy if it could not produce one).
    local attribution_json="null"
    if [[ -n "${LANE_ATTRIBUTION_RECORD_JSON}" ]] \
        && jq -e . >/dev/null 2>&1 <<<"${LANE_ATTRIBUTION_RECORD_JSON}"; then
        attribution_json="${LANE_ATTRIBUTION_RECORD_JSON}"
    fi

    jq -n \
        --arg active_version "${version}" \
        --arg git_sha "${git_sha}" \
        --arg deploy_path "${deploy_target}" \
        --arg source_repo "${repo_root}" \
        --arg deployed_at "${deployed_at}" \
        --arg compose_project "${compose_project}" \
        --arg profile "${COMPOSE_PROFILE}" \
        --argjson attribution "${attribution_json}" \
        '{
            active_version: $active_version,
            git_sha: $git_sha,
            deploy_path: $deploy_path,
            source_repo: $source_repo,
            deployed_at: $deployed_at,
            compose_project: $compose_project,
            profile: $profile,
            attribution: $attribution
        }' > "${tmp_file}"

    # Restore original umask before continuing
    umask "${old_umask}"

    # Atomic rename
    mv "${tmp_file}" "${REGISTRY_FILE}"

    log_info "Registry written: ${REGISTRY_FILE}"
    log_info "  version:         ${version}"
    log_info "  git_sha:         ${git_sha}"
    log_info "  deployed_at:     ${deployed_at}"
    log_info "  compose_project: ${compose_project}"
    if [[ "${attribution_json}" != "null" ]]; then
        log_info "  actor:           $(jq -r '.actor.identity // "unknown"' <<<"${attribution_json}")"
        log_info "  reason:          $(jq -r '.reason // ""' <<<"${attribution_json}")"
        log_info "  grant verdict:   $(jq -r '.grant_guard.verdict // "unknown"' <<<"${attribution_json}")"
    fi
}

# =============================================================================
# Image tag snapshot -- protect `:latest` from a failed build (OMN-15352 F3)
# =============================================================================

snapshot_latest_image_tags() {
    # Record the pre-build `:latest` image id for every RUNTIME_BUILD_SERVICES
    # member, so a failed deploy can restore each tag to what it resolved to
    # before this invocation (or remove a tag that had no prior state). Runs
    # unconditionally right before build_images(): `docker compose build` only
    # (re)tags `:latest` on a SUCCESSFUL build, so whatever we capture here is
    # always a correct pre-image of the tag regardless of how this run ends.
    local compose_project="$1"

    local snapshot_file
    snapshot_file="$(mktemp "${DEPLOY_ROOT}/.latest-tag-snapshot.XXXXXX" 2>/dev/null || true)"
    if [[ -z "${snapshot_file}" ]]; then
        log_warn "Could not create :latest tag snapshot file; :latest rollback protection is disabled for this run."
        return 0
    fi

    local service image_name prior_id
    for service in "${RUNTIME_BUILD_SERVICES[@]}"; do
        image_name="${compose_project}-${service}"
        prior_id="$(docker image inspect "${image_name}:latest" --format '{{.Id}}' 2>/dev/null || true)"
        printf '%s\t%s\n' "${service}" "${prior_id}" >>"${snapshot_file}"
    done

    LATEST_TAG_SNAPSHOT_FILE="${snapshot_file}"
    log_info "Snapshotted pre-build :latest image ids for ${#RUNTIME_BUILD_SERVICES[@]} service(s) (rollback safety)."
}

restore_latest_image_tags() {
    # Restore every RUNTIME_BUILD_SERVICES `:latest` tag to its pre-build state
    # (OMN-15352 F3). Called only from cleanup_on_exit() on a non-success exit.
    # A service that had no prior `:latest` image (recorded as an empty id by
    # snapshot_latest_image_tags()) has its now-unverified tag removed instead
    # of being left resolvable by a later `docker compose up -d` without
    # --build.
    if [[ -z "${LATEST_TAG_SNAPSHOT_FILE}" || ! -f "${LATEST_TAG_SNAPSHOT_FILE}" ]]; then
        return 0
    fi
    if [[ -z "${DEPLOY_COMPOSE_PROJECT}" ]]; then
        log_warn "DEPLOY_COMPOSE_PROJECT is unset; cannot restore :latest image tags."
        return 0
    fi

    local service prior_id image_name
    while IFS=$'\t' read -r service prior_id; do
        [[ -n "${service}" ]] || continue
        image_name="${DEPLOY_COMPOSE_PROJECT}-${service}"
        if [[ -n "${prior_id}" ]]; then
            if docker tag "${prior_id}" "${image_name}:latest" 2>/dev/null; then
                log_warn "Restored ${image_name}:latest to its pre-build image ${prior_id}."
            else
                log_error "Failed to restore ${image_name}:latest to pre-build image ${prior_id}."
                log_error "Manual recovery: docker tag ${prior_id} ${image_name}:latest"
            fi
        else
            # No prior :latest existed for this service -- remove the tag this
            # failed run may have created rather than leave it pointing at an
            # image that was never proven to deploy.
            docker rmi "${image_name}:latest" 2>/dev/null || true
        fi
    done <"${LATEST_TAG_SNAPSHOT_FILE}"
}

reconcile_runtime_container_start_state() {
    # OMN-15718: companion to restore_latest_image_tags() -- retagging
    # `:latest` only fixes what a FUTURE `docker compose up` would build/pull;
    # it does nothing for containers THIS run already force-recreated (or
    # attempted to) and that are now stranded in 'Created' because a
    # depends_on:condition:service_healthy dependency (e.g. migration-gate)
    # can never become healthy. Reconcile every RUNTIME_BUILD_SERVICES
    # container back to running, or explicitly tear it down.
    #
    # Best-effort: needs DEPLOY_COMPOSE_PROJECT (set once main() resolves the
    # compose project) and DEPLOY_DIR_TO_CLEANUP (set once main() resolves
    # deploy_target; reset to "" only on full success, so it is still the
    # correct deploy_target here on any failure path). If either is unset --
    # a failure early enough in main() that neither had been resolved yet --
    # there is nothing to reconcile because no compose call could have run.
    if [[ -z "${DEPLOY_COMPOSE_PROJECT}" || -z "${DEPLOY_DIR_TO_CLEANUP}" || ! -d "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
        return 0
    fi

    local -a compose_args
    resolve_compose_file_args compose_args "${DEPLOY_DIR_TO_CLEANUP}" "${DEPLOY_COMPOSE_PROJECT}" || return 0

    local service container_id
    for service in "${RUNTIME_BUILD_SERVICES[@]}"; do
        container_id="$(docker compose -p "${DEPLOY_COMPOSE_PROJECT}" "${compose_args[@]}" --profile "${COMPOSE_PROFILE}" ps -q "${service}" 2>/dev/null || true)"
        if [[ -z "${container_id}" ]]; then
            continue
        fi
        # Guarded (`|| true`): this runs inside the EXIT trap, under `set -e`
        # -- reconcile_container_running_state returning 1 (it had to tear a
        # container down) must not abort the rest of cleanup_on_exit (lock
        # release etc).
        reconcile_container_running_state "${container_id}" "${service}" || true
    done
}

# =============================================================================
# Build -- docker compose build with VCS_REF label
# =============================================================================

build_images() {
    # Build Docker images with VCS_REF, BUILD_DATE, and deployment identity args.
    # RUNTIME_SOURCE_HASH and COMPOSE_PROJECT are stamped into the image so the
    # startup banner in entrypoint-runtime.sh can display them on container start.
    # This makes deployment drift visible in logs without git forensics.
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Build Images"

    local -a build_scope
    resolve_lane_runtime_services build_scope "${compose_project}"

    local build_date
    build_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    # OMN-12965: stamp org.opencontainers.image.version from pyproject so the
    # runtime image carries a real version instead of the Dockerfile placeholder
    # (0.1.0). A placeholder version degrades every proof packet.
    local runtime_version
    runtime_version="$(read_version "${deploy_target}")"
    local omni_home="${OMNI_HOME:-}"
    local build_source
    build_source="$(resolve_build_source)"
    local expected_build_source
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    # OMN-13669: stamp OCI provenance labels so the prod-promotion gate and
    # lineage guard can refuse workspace images for prod. Computed from
    # build_source: workspace => stability-candidate/true; release => clean-main/false.
    local promotion_class
    promotion_class="$(resolve_promotion_class "${build_source}")"
    local non_main_lineage
    non_main_lineage="$(resolve_non_main_lineage "${build_source}")"
    local compat_ref="main"
    local omnimarket_ref="dev"
    if [[ -n "${omni_home}" ]]; then
        compat_ref="$(read_repo_ref_or_main "${omni_home}/omnibase_compat")"
        omnimarket_ref="$(read_repo_ref_or_main "${omni_home}/omnimarket")"
    fi

    # Build timeout in seconds (default: 15 minutes). Prevents the known issue
    # where `docker compose build` hangs indefinitely after images are built.
    # Override via DOCKER_BUILD_TIMEOUT_SECONDS env var. (OMN-5462)
    local build_timeout="${DOCKER_BUILD_TIMEOUT_SECONDS:-900}"

    local cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        build
        --progress=plain
        --build-arg "GIT_SHA=${git_sha}"
        --build-arg "VCS_REF=${git_sha}"
        --build-arg "RUNTIME_VERSION=${runtime_version}"
        --build-arg "BUILD_DATE=${build_date}"
        --build-arg "RUNTIME_SOURCE_HASH=${git_sha}"
        --build-arg "COMPOSE_PROJECT=${compose_project}"
        --build-arg "BUILD_SOURCE=${build_source}"
        --build-arg "EXPECTED_BUILD_SOURCE=${expected_build_source}"
        --build-arg "PROMOTION_CLASS=${promotion_class}"
        --build-arg "NON_MAIN_LINEAGE=${non_main_lineage}"
        --build-arg "OMNI_HOME=${omni_home}"
        --build-arg "OMNIBASE_COMPAT_REF=${compat_ref}"
        --build-arg "OMNIMARKET_REF=${omnimarket_ref}"
        # OMN-14873: scope the build to RUNTIME_BUILD_SERVICES (defaults to the full
        # RUNTIME_SERVICES fan-out; see the override comment above its declaration).
        # OMN-17448: plus the dev-lane-only standalone projection writers, which
        # are declared in that lane's overlay and must be built for it.
        "${build_scope[@]}"
    )

    log_info "Building images with VCS_REF=${git_sha} RUNTIME_VERSION=${runtime_version} RUNTIME_SOURCE_HASH=${git_sha} COMPOSE_PROJECT=${compose_project}..."
    log_info "Build scope: ${build_scope[*]}"
    log_info "Build source: BUILD_SOURCE=${build_source} EXPECTED_BUILD_SOURCE=${expected_build_source} PROMOTION_CLASS=${promotion_class} NON_MAIN_LINEAGE=${non_main_lineage} OMNI_HOME=${omni_home}"
    log_info "Plugin refs: OMNIBASE_COMPAT_REF=${compat_ref} OMNIMARKET_REF=${omnimarket_ref}"
    log_info "Build timeout: ${build_timeout}s (set DOCKER_BUILD_TIMEOUT_SECONDS to override)"
    log_cmd "${cmd[*]}"

    # Use timeout to prevent indefinite hangs after build completes (OMN-5462).
    # Exit code 124 = timeout fired; we treat this as success if images exist.
    if timeout "${build_timeout}" "${cmd[@]}"; then
        log_info "Image build complete."
    elif [[ $? -eq 124 ]]; then
        log_warn "Build timed out after ${build_timeout}s — images may still be usable. Continuing."
    else
        log_error "Image build failed."
        return 1
    fi
}

# =============================================================================
# Restart -- bring up runtime services only
# =============================================================================

resolve_broker_container() {
    # Resolve the running broker container id/name for the given compose project.
    #
    # OMN-13364: the broker's fixed container_name (e.g. omnibase-infra-redpanda)
    # is NOT a reliable handle — when it collides with another project's broker,
    # Docker prefixes it with a random hash (3ed1fdb8d50b_omnibase-infra-redpanda).
    # The compose service label (com.docker.compose.service=redpanda) survives
    # the prefix, so resolve by compose project + service label instead of by an
    # exact container-name string match.
    local compose_project="$1"
    docker ps -q \
        --filter "label=com.docker.compose.project=${compose_project}" \
        --filter "label=com.docker.compose.service=${BROKER_READINESS_SERVICE}" \
        2>/dev/null \
        | head -1
}

assert_broker_reachable() {
    # Return 0 when the broker is actually reachable on the lane network.
    #
    # Keys readiness off `rpk cluster health` executed INSIDE the broker
    # container (talking to the broker on TCP/9092 over the lane network), not
    # off an exact container-name match or the compose-wait exit status. This is
    # what lets the warmup tolerate a Docker-prefixed broker name and an
    # already-present healthy broker without false-failing (OMN-13364).
    local compose_project="$1"
    local attempts="${BROKER_REACHABLE_RETRIES:-15}"
    local interval="${BROKER_REACHABLE_INTERVAL:-4}"

    local broker_container
    broker_container="$(resolve_broker_container "${compose_project}")"
    if [[ -z "${broker_container}" ]]; then
        log_error "No running broker container found for project '${compose_project}'"
        log_error "  (label com.docker.compose.service=${BROKER_READINESS_SERVICE})."
        return 1
    fi
    log_info "Resolved broker container: ${broker_container} (probing reachability)"

    local attempt=0
    while (( attempt < attempts )); do
        attempt=$((attempt + 1))
        # rpk talks to the broker on the internal listener (redpanda:9092 / TCP).
        # `cluster health` succeeding means the broker is reachable AND serving;
        # that is the readiness signal the partition-cap rpk calls below need.
        if docker exec "${broker_container}" \
            rpk cluster health -X brokers=redpanda:9092 >/dev/null 2>&1; then
            log_info "Broker reachable: rpk cluster health OK (attempt ${attempt})."
            return 0
        fi
        log_info "  Broker not ready yet (attempt ${attempt}/${attempts}) -- waiting ${interval}s..."
        sleep "${interval}"
    done

    log_error "Broker ${broker_container} did not become reachable after ${attempts} attempts."
    return 1
}

ensure_core_infra_ready() {
    # Bring up + wait for the core data-plane infra (postgres, valkey) BEFORE the
    # migration preflight + runtime restart (OMN-13594).
    #
    # The `--restart` path runs warm_broker_topic_provisioning ->
    # run_runtime_migration_preflight -> restart_services, and every one of those
    # uses `up -d --no-deps`, which bypasses the compose `depends_on` chain. On a
    # WARM lane that is fine (postgres/valkey are already up). On a fully COLD
    # lane (no prior containers) NOTHING starts postgres/valkey first, so
    # forward-migration (`--no-deps`) has no database to connect to: its 30x2s
    # readiness probe exhausts -> exit 1 -> the deploy auto-rolls back. This is
    # the exact cold-start defect OMN-13594 filed against this script.
    #
    # Bring the core infra up explicitly here and BLOCK on its healthchecks via
    # `--wait`. On a warm lane this is an idempotent no-op (up -d on a healthy
    # service does nothing, --wait returns immediately). On a cold lane it
    # creates + warms postgres/valkey so the preflight's forward-migration sees a
    # live database on its first attempt.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Core Infra Readiness (cold-start guard, OMN-13594)"

    local core_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --wait --wait-timeout "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}"
        "${CORE_INFRA_SERVICES[@]}"
    )
    log_info "Ensuring core infra healthy before preflight: ${CORE_INFRA_SERVICES[*]}"
    log_cmd "${core_up_cmd[*]}"
    # OMN-15718: bounded, not just guarded -- a stuck compose-internal wait
    # (e.g. a dependency that can never become healthy) must fail fast with a
    # typed timeout, not hang the whole deploy indefinitely.
    if ! compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${core_up_cmd[@]}"; then
        log_error "Core infra (${CORE_INFRA_SERVICES[*]}) did not become healthy."
        log_error "Migration preflight needs a live Postgres; aborting before it"
        log_error "wastes the 30x2s readiness budget and triggers a rollback (OMN-13594)."
        return 1
    fi
    log_info "Core infra healthy: ${CORE_INFRA_SERVICES[*]}."
}

warm_broker_topic_provisioning() {
    # Bring the broker + partition cap to readiness before the --no-deps runtime
    # restart so the cold-start topic-provisioning burst does not crash-loop the
    # kernel (OMN-13220). The runtime restart bypasses depends_on, so the
    # compose-declared redpanda-partition-cap gate never fires on a restart-only
    # deploy — apply it here, explicitly, before the kernel boots.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Broker Topic-Provisioning Warmup"

    # 1. Ensure the broker itself is up and healthy. `up -d` is a no-op when it
    # is already running; the --wait flag blocks until the healthcheck passes so
    # the partition-cap rpk calls below do not race a still-starting broker.
    #
    # OMN-13364: the compose `up --wait` is best-effort, not the source of truth
    # for broker readiness. When the broker container_name collides with another
    # project's broker, Docker assigns a random prefix (e.g.
    # 3ed1fdb8d50b_omnibase-infra-redpanda) and/or leaves the recreate in
    # 'Created'; `up -d --wait` then errors or never reaches healthy even though
    # a healthy broker is already reachable on the lane network. Do NOT treat
    # that as a deploy failure (it would trigger the backup-restore path, which
    # reverts the freshly-built vendored migration tree). Key broker readiness
    # off ACTUAL reachability (`rpk cluster health` on TCP/9092 inside the lane)
    # via assert_broker_reachable below, not off the compose-wait exit status.
    local broker_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --wait --wait-timeout "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}"
        "${BROKER_READINESS_SERVICE}"
    )
    log_info "Ensuring broker is healthy: ${BROKER_READINESS_SERVICE}"
    log_cmd "${broker_up_cmd[*]}"
    # OMN-15718: bounded as well as guarded -- a name-collision false-fail must
    # stay non-fatal (rpk probe below decides), but a genuinely stuck wait must
    # still fail fast with a typed timeout instead of hanging this guard open.
    if ! compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${broker_up_cmd[@]}"; then
        log_warn "Broker compose up --wait did not report healthy (possible"
        log_warn "name-prefix collision or already-present broker). Falling back"
        log_warn "to a direct broker-reachability probe before deciding."
    fi

    # Source of truth: probe the broker directly. Tolerates a Docker-prefixed
    # container name and an already-present healthy broker (OMN-13364).
    if ! assert_broker_reachable "${compose_project}"; then
        log_error "Broker is not reachable on the lane network after warmup."
        log_error "Cold-start topic provisioning cannot proceed; aborting."
        return 1
    fi

    # 2. Apply the partition cap (run-to-completion). force-recreate re-runs the
    # one-shot even if a prior run left an exited container behind.
    local cap_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --force-recreate
        "${BROKER_PARTITION_CAP_SERVICE}"
    )
    log_info "Applying broker partition cap: ${BROKER_PARTITION_CAP_SERVICE}"
    log_cmd "${cap_up_cmd[*]}"
    # OMN-16110: this `up` is bounded AND guarded -- its exit code is NOT the
    # source of truth for whether the cap was applied. A stale daemon-phantom
    # container record for this service (listed `Dead` by `docker ps -a`, but
    # "No such container" on both `docker inspect` and `docker rm -f`; no
    # backing directory under the daemon's containers dir) makes compose's
    # convergence plan try to start the phantom AFTER it has already
    # recreated and started the real one-shot; that trailing start fails the
    # whole `up` with "No such container" even though the cap container is up
    # and running (observed 2026-08-24 on the .201 dev lane; deterministic
    # while the phantom record persists, and the record can only be cleared
    # by a dockerd restart). Same doctrine as the broker `up --wait` above
    # (OMN-13364): run the up best-effort, then decide success off the actual
    # named container's run-to-completion (`docker wait` == 0) below, which
    # remains fail-closed -- if the up truly created nothing, `docker wait`
    # errors or times out and this step still aborts. The OMN-15718 bounded
    # deadline is preserved.
    if ! compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${cap_up_cmd[@]}"; then
        log_warn "Partition-cap compose up exited non-zero (possible stale/phantom"
        log_warn "container record for ${BROKER_PARTITION_CAP_SERVICE} -- OMN-16110)."
        log_warn "Deciding off the one-shot's own run-to-completion below."
    fi

    local cap_container="${compose_project}-${BROKER_PARTITION_CAP_SERVICE}"
    # OMN-15718: `docker wait` blocks until the container exits, with no
    # deadline of its own -- bound it the same way as the `up` calls above so
    # a stuck one-shot fails typed instead of hanging this step forever.
    local cap_wait_cmd=(timeout --kill-after=15 "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" docker wait "${cap_container}")
    log_cmd "${cap_wait_cmd[*]}"
    local cap_wait_exit_code
    cap_wait_exit_code="$("${cap_wait_cmd[@]}")" || true
    if [[ "${cap_wait_exit_code}" != "0" ]]; then
        if [[ -z "${cap_wait_exit_code}" ]]; then
            log_error "COMPOSE_UP_TIMEOUT: 'docker wait ${cap_container}' did not complete within ${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}s -- killed."
        fi
        log_error "${BROKER_PARTITION_CAP_SERVICE} did not complete successfully."
        log_error "Broker partition cap not applied; cold-start topic provisioning may crash-loop the runtime."
        return 1
    fi
    log_info "Broker partition cap applied."
}

run_runtime_migration_preflight() {
    # Run bounded migration services before --no-deps runtime restarts.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Runtime Migration Preflight"

    for service in "${RUNTIME_MIGRATION_SERVICES[@]}"; do
        local cmd=(
            docker compose
            -p "${compose_project}"
            "${compose_args[@]}"
            --profile "${COMPOSE_PROFILE}"
            up -d --no-deps --force-recreate
            "${service}"
        )
        log_info "Refreshing migration service: ${service}"
        log_cmd "${cmd[*]}"
        # OMN-15718: bounded (not guarded -- a real failure here must still
        # abort under set -e exactly as before; only the hang risk is closed).
        compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${cmd[@]}"
        # One-shot migrations (forward-migration, intelligence-migration) run to
        # completion and must exit 0 before the dependent schema/runtime work
        # proceeds. migration-gate is a long-running healthcheck keepalive, NOT a
        # one-shot, so it is deliberately excluded from the wait set
        # (OMN-13220). Deriving the container name from the compose project keeps
        # the wait pointed at the lane being deployed (OMN-12987): the base
        # compose names it <compose-project>-<service> and each lane overlay
        # follows the same form (e.g. omnibase-infra-intelligence-migration for
        # dev, omnibase-infra-stability-test-intelligence-migration for stability).
        local is_oneshot=false
        local oneshot
        for oneshot in "${RUNTIME_MIGRATION_ONESHOTS[@]}"; do
            if [[ "${service}" == "${oneshot}" ]]; then
                is_oneshot=true
                break
            fi
        done
        if [[ "${is_oneshot}" == true ]]; then
            local migration_container="${compose_project}-${service}"
            # OMN-15718: bound the wait itself, same reasoning as the
            # partition-cap wait above.
            local wait_cmd=(timeout --kill-after=15 "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" docker wait "${migration_container}")
            log_cmd "${wait_cmd[*]}"
            local migration_wait_result
            migration_wait_result="$("${wait_cmd[@]}")" || true
            if [[ "${migration_wait_result}" != "0" ]]; then
                if [[ -z "${migration_wait_result}" ]]; then
                    log_error "COMPOSE_UP_TIMEOUT: 'docker wait ${migration_container}' did not complete within ${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}s -- killed."
                fi
                log_error "${service} did not complete successfully."
                return 1
            fi
        fi
    done

    # Postgres follows the same lane-derivable naming as forward-migration:
    # <compose-project>-postgres (omnibase-infra-postgres for dev,
    # omnibase-infra-stability-test-postgres for stability). Deriving it keeps
    # the projection-table probe pointed at the lane being deployed instead of
    # always hitting the dev-lane postgres (OMN-12987).
    local postgres_container="${compose_project}-postgres"
    for table_name in "${REQUIRED_PROJECTION_TABLES[@]}"; do
        local check_cmd=(
            docker exec "${postgres_container}"
            psql
            -U postgres
            -d omnidash_analytics
            -tAc
            "SELECT to_regclass('public.${table_name}') IS NOT NULL"
        )
        log_info "Checking projection table: omnidash_analytics.${table_name}"
        log_cmd "${check_cmd[*]}"
        if [[ "$("${check_cmd[@]}")" != "t" ]]; then
            log_error "Missing projection table omnidash_analytics.${table_name}; aborting runtime restart."
            return 1
        fi
    done
}

bringup_full_stack() {
    # Cold-lane FULL bring-up: bring the WHOLE --profile runtime project up
    # (OMN-13414).
    #
    # The warm --restart path recreates only the RUNTIME_SERVICES subset with
    # `up -d --no-deps`; on a cold/GC-reclaimed lane every other service in the
    # runtime profile (the projection/consumer fleet, autoheal, etc.) stays down.
    # This brings the entire project up. Two gotchas it encodes:
    #
    #   1. `--profile "${COMPOSE_PROFILE}"` (runtime) is MANDATORY. Runtime
    #      services are gated behind the compose runtime profile; a bare
    #      `docker compose up -d` matches NO profiled service and starts nothing.
    #   2. NO `--no-deps`. Unlike restart_services, the full up honors the compose
    #      depends_on chain (postgres/valkey -> redpanda + partition-cap ->
    #      forward/intelligence migration one-shots -> runtime + consumers), so
    #      the whole stack starts in dependency order. The explicit preflight in
    #      main() (ensure_core_infra_ready / warm_broker_topic_provisioning /
    #      run_runtime_migration_preflight) has already warmed deps + the
    #      one-shots, so this is the idempotent full fan-out over the rest of the
    #      profile.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Cold-Lane Full Bring-Up (OMN-13414)"

    local cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d
    )

    log_info "Bringing the FULL ${COMPOSE_PROFILE}-profile project up: ${compose_project}"
    log_cmd "${cmd[*]}"

    # OMN-15718: bounded (not guarded -- a real failure here must still abort
    # under set -e exactly as before; only the hang risk is closed). This
    # honors the full depends_on chain (no --no-deps), which is precisely the
    # path that can block indefinitely on a dependency that never becomes
    # healthy.
    compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${cmd[@]}"

    log_info "Full project up."
}

restart_services() {
    # Restart runtime containers via docker compose up --force-recreate.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Restart Runtime Services"

    # OMN-17448: the dev lane additionally recreates the standalone projection
    # writers, which are declared only in its own overlay.
    local -a lane_services
    resolve_lane_runtime_services lane_services "${compose_project}"

    local cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --force-recreate
        "${lane_services[@]}"
    )

    log_info "Restarting services: ${lane_services[*]}"
    log_cmd "${cmd[*]}"

    # OMN-15718: bounded (not guarded -- a real failure here must still abort
    # under set -e exactly as before; only the hang risk is closed). --no-deps
    # skips STARTING dependencies but compose still honors a target service's
    # own depends_on:condition:service_healthy before creating/starting it
    # (e.g. migration-gate for runtime-effects/runtime-worker) -- if that
    # dependency can never become healthy this call would otherwise hang
    # indefinitely instead of failing fast (the exact 2026-08-05 defect,
    # OMN-15718).
    compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${cmd[@]}"

    log_info "Services restarted."
}

# =============================================================================
# Verify -- health check + label inspection + log sentinels
# =============================================================================

verify_deployment() {
    # Run health checks and verify image labels match the deployed SHA.
    local git_sha="$1"
    local compose_project="$2"

    log_step "Verify Deployment"

    # Resolve the lane-scoped runtime container name so every probe below targets
    # THIS lane's container, not the hardcoded dev `omninode-runtime` (OMN-13826).
    local runtime_container_name
    runtime_container_name="$(resolve_lane_runtime_container_name "${compose_project}")"

    # 1. Health endpoint
    log_info "Checking health endpoint (${HEALTH_CHECK_URL})..."
    local attempt=0
    local healthy=false

    while (( attempt < HEALTH_CHECK_RETRIES )); do
        attempt=$((attempt + 1))
        if curl -sf --connect-timeout 2 --max-time 5 "${HEALTH_CHECK_URL}" >/dev/null 2>&1; then
            healthy=true
            break
        fi
        log_info "  Attempt ${attempt}/${HEALTH_CHECK_RETRIES} -- waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep "${HEALTH_CHECK_INTERVAL}"
    done

    if [[ "${healthy}" == true ]]; then
        log_info "Health check passed."
    else
        log_error "Health check FAILED after ${HEALTH_CHECK_RETRIES} attempts."
        log_error "Service is not responding at ${HEALTH_CHECK_URL}"
        log_error "Check container logs: docker logs ${runtime_container_name}"
        exit 1
    fi

    # 2. Resolve runtime container ID. Prefer the lane-scoped runtime container
    # name (docker-compose.<lane>.yml prefixes container_name per lane), then fall
    # back to a compose label lookup. The label fallback filters by the lane's
    # compose project AND the compose service key -- which stays "omninode-runtime"
    # in every overlay -- so the project filter is what disambiguates the lane.
    log_info "Checking image labels for VCS_REF..."
    local container_id
    container_id="$(docker ps -q --filter "name=^/${runtime_container_name}$" | head -1)"
    if [[ -z "${container_id}" ]]; then
        container_id="$(
            docker ps -q \
                --filter "label=com.docker.compose.project=${compose_project}" \
                --filter "label=com.docker.compose.service=omninode-runtime" \
                | head -1
        )"
    fi

    if [[ -z "${container_id}" ]]; then
        log_warn "Could not resolve container ID for ${runtime_container_name}; skipping label/log checks."
        return 0
    fi

    # 3. Image label verification
    local label
    label="$(docker inspect "${container_id}" \
        --format='{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"

    if [[ "${label}" == "${git_sha}" ]]; then
        log_info "Image label matches: org.opencontainers.image.revision=${label}"
    elif [[ -n "${label}" ]]; then
        log_warn "Image label mismatch:"
        log_warn "  Expected: ${git_sha}"
        log_warn "  Found:    ${label}"
        log_warn "The running container may be from a previous build."
        log_warn "The fail-closed deploy readback (RT-6) below will reject this deploy."
    else
        log_warn "Could not read image label (container may not exist yet)."
        log_warn "The fail-closed deploy readback (RT-6) below will reject this deploy."
    fi

    # OMN-12965: verify org.opencontainers.image.version is a real version, not
    # the Dockerfile placeholder (0.1.0) or blank. A placeholder/blank identity
    # degrades every proof packet (runtime SHA + image digest are required
    # citations in accepted evidence).
    local version_label
    version_label="$(docker inspect "${container_id}" \
        --format='{{index .Config.Labels "org.opencontainers.image.version"}}' 2>/dev/null || true)"
    if [[ -z "${version_label}" || "${version_label}" == "0.1.0" ]]; then
        log_error "Image version label is blank/placeholder: org.opencontainers.image.version='${version_label}'"
        log_error "Runtime image identity is degraded (OMN-12965). Rebuild with RUNTIME_VERSION from pyproject."
        exit 1
    fi
    log_info "Image version label OK: org.opencontainers.image.version=${version_label}"

    # 4. Log sentinel: entrypoint ran
    log_info "Checking log sentinels..."
    local logs
    logs="$(docker logs "${container_id}" 2>&1 | tail -50 || true)"

    if echo "${logs}" | grep -q "Schema fingerprint stamped"; then
        log_info "Sentinel found: 'Schema fingerprint stamped' (entrypoint ran)."
    else
        log_warn "Sentinel not found: 'Schema fingerprint stamped'"
        log_warn "The entrypoint may not have completed yet."
    fi
}

# =============================================================================
# Deploy readback -- RT-6 (OMN-14469): fail-closed Class-3 mechanism
# =============================================================================

readback_deployed_ref() {
    # TERMINAL deploy readback: read a fact only the freshly-built image could
    # carry off the RUNNING container and assert it equals the intended ref
    # (docs/plans/2026-07-12-mechanical-release-trains.md §4, RT-6).
    #
    # The fact is the org.opencontainers.image.revision label, stamped from
    # VCS_REF at build time (docker/Dockerfile.runtime). verify_deployment()
    # above reads the same label but only *warns* on a mismatch and returns 0 --
    # so a stale / mis-targeted container (deployed code != intended ref) passes
    # today. This step makes that FAIL-CLOSED: it delegates to the previously
    # dead scripts/verify_deployed_versions.py (OMN-5608), which reads the label
    # (and, when release versions are declared, the installed package versions)
    # and exits non-zero on any mismatch. It is NOT an optional flag -- it runs
    # unconditionally after every restart / cold bring-up. Without it,
    # "deployed" / "live-readback" proof classes are unfalsifiable.
    #
    # OMN-15348: verifies EXACTLY the services this run actually rebuilt/
    # recreated -- RUNTIME_BUILD_SERVICES (which already resolves to the
    # RUNTIME_BUILD_SERVICES_OVERRIDE subset when OMN-14873 scoping is in
    # play, else the full RUNTIME_SERVICES set). Prior to this fix the
    # readback was hardcoded to the single omninode-runtime container
    # regardless of scope: a scoped rebuild of e.g. runtime-effects left
    # omninode-runtime's stale label untouched, RT-6 read THAT container,
    # false-FAILed, and auto-triggered restore-previous-deployment on a
    # deploy that never touched omninode-runtime at all. Looping the
    # verified set over RUNTIME_BUILD_SERVICES means an out-of-scope
    # container's stale label is never probed, so it can neither fail the
    # deploy nor trigger the restore.
    local git_sha="$1"
    local version="$2"
    local compose_project="$3"
    local repo_root="$4"
    local deploy_target="$5"

    log_step "Deploy Readback (RT-6, fail-closed) [OMN-14469]"

    # An unresolvable intended SHA means a stale-image readback cannot be proven.
    # Fail closed rather than certify a deploy whose target ref is 'unknown'.
    if [[ "${git_sha}" == "unknown" ]]; then
        log_error "Cannot read back deploy identity: intended git SHA is 'unknown'."
        log_error "A stale-image readback is unprovable without the intended ref. Refusing to certify (RT-6)."
        exit 1
    fi

    # Compose file args to resolve non-runtime service containers below by
    # `docker compose ps -q <service>` -- robust to lanes/services (e.g.
    # runtime-worker on the dev lane) that have no fixed container_name and
    # get a compose-assigned one.
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    # The readback script lives next to this script (sync_files does NOT copy
    # scripts/ into the deploy target), so resolve it from the repo root.
    local readback="${repo_root}/scripts/verify_deployed_versions.py"
    if [[ ! -f "${readback}" ]]; then
        log_error "Deploy readback script not found: ${readback}"
        log_error "Cannot certify the deploy without the readback. Aborting (RT-6)."
        exit 1
    fi

    # verify_deployed_versions.py is pure-stdlib; prefer the repo venv, then a
    # bare python3. (No uv needed -- keep the terminal step fast and dep-free.)
    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the deploy readback."
        exit 1
    fi

    # Always assert each in-scope container's revision == the intended git SHA.
    # Also assert the runtime package version inside the primary
    # omninode-runtime container matches the built version (always true on
    # every lane, and only meaningful for that container's image); operators
    # can declare extra sibling versions to assert via
    # READBACK_EXPECTED_VERSIONS.
    local expected_versions="omnibase-infra=${version}"
    if [[ -n "${READBACK_EXPECTED_VERSIONS:-}" ]]; then
        expected_versions="${expected_versions},${READBACK_EXPECTED_VERSIONS}"
    fi

    # OMN-17448: the readback's in-scope set must match what was actually built
    # and restarted, or a dev-lane writer would be created and never verified.
    local -a readback_scope
    resolve_lane_runtime_services readback_scope "${compose_project}"

    log_info "Verifying in-scope service(s): ${readback_scope[*]}"

    local service
    for service in "${readback_scope[@]}"; do
        local container_name=""
        if [[ "${service}" == "omninode-runtime" ]]; then
            # Keep the pre-existing, individually-tested resolver for the
            # primary runtime container (lane-prefixed container_name).
            container_name="$(resolve_lane_runtime_container_name "${compose_project}")"
        else
            # Every other RUNTIME_SERVICES member either has no fixed
            # container_name (e.g. dev-lane runtime-worker, compose-assigned)
            # or a lane-prefix convention that differs per service
            # (projection-api -> omnimarket-*, intelligence-api ->
            # omnibase-*). Resolve live via the compose service key instead
            # of hardcoding a second name map (OMN-13826-class lesson).
            container_name="$(docker compose -p "${compose_project}" "${compose_args[@]}" ps -q "${service}" 2>/dev/null || true)"
            if [[ -z "${container_name}" ]]; then
                log_error "Deploy readback FAILED (RT-6): could not resolve a running container for in-scope service '${service}'."
                log_error "Refusing to certify this deploy. Rebuild + recreate the lane's runtime and re-run."
                exit 1
            fi
        fi

        local -a readback_args=(
            --container "${container_name}"
            --expected-revision "${git_sha}"
        )
        if [[ "${service}" == "omninode-runtime" ]]; then
            readback_args+=(--versions "${expected_versions}")
        fi

        log_cmd "${python_bin} ${readback} ${readback_args[*]}"
        if ! "${python_bin}" "${readback}" "${readback_args[@]}"; then
            log_error "Deploy readback FAILED (RT-6): service '${service}' (container ${container_name}) is NOT the intended ref ${git_sha}."
            log_error "Deployed code != intended ref (stale / mis-targeted image, or version drift)."
            log_error "Refusing to certify this deploy. Rebuild + recreate the lane's runtime and re-run."
            exit 1
        fi

        log_info "Deploy readback passed: ${service} (${container_name}) revision == ${git_sha} (RT-6)."
    done

    log_info "Deploy readback passed for all ${#readback_scope[@]} in-scope service(s) (RT-6)."
}

# =============================================================================
# Print Compose Commands
# =============================================================================

print_compose_commands() {
    # Print the exact docker compose commands this script would execute.
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    # OMN-13581: print the SAME `-f` token sequence the script executes, including
    # the lane overlay for non-dev projects, so copy-pasted operator commands do
    # not silently run on the bare infra.yml config (which displaces the broker).
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    local compose_f="${compose_args[*]}"
    local omni_home="${OMNI_HOME:-}"
    local build_source
    build_source="$(resolve_build_source)"
    local expected_build_source
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    # OMN-13669: stamp OCI provenance labels so the prod-promotion gate can refuse
    # workspace images for prod. Computed from build_source (workspace =>
    # stability-candidate/true; release => clean-main/false).
    local promotion_class
    promotion_class="$(resolve_promotion_class "${build_source}")"
    local non_main_lineage
    non_main_lineage="$(resolve_non_main_lineage "${build_source}")"
    local compat_ref="main"
    local omnimarket_ref="dev"
    if [[ -n "${omni_home}" ]]; then
        compat_ref="$(read_repo_ref_or_main "${omni_home}/omnibase_compat")"
        omnimarket_ref="$(read_repo_ref_or_main "${omni_home}/omnimarket")"
    fi

    log_step "Compose Commands"

    log_info "These are the exact commands this script would run from the deployed directory."
    log_info "Note: env vars resolve from shell environment (sourced from ~/.omnibase/.env)."
    log_info ""
    log_info "Build:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    build \\"
    log_info "    --build-arg VCS_REF=${git_sha} \\"
    log_info "    --build-arg BUILD_DATE=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\") \\"
    log_info "    --build-arg RUNTIME_SOURCE_HASH=${git_sha} \\"
    log_info "    --build-arg COMPOSE_PROJECT=${compose_project} \\"
    log_info "    --build-arg BUILD_SOURCE=${build_source} \\"
    log_info "    --build-arg EXPECTED_BUILD_SOURCE=${expected_build_source} \\"
    log_info "    --build-arg PROMOTION_CLASS=${promotion_class} \\"
    log_info "    --build-arg NON_MAIN_LINEAGE=${non_main_lineage} \\"
    log_info "    --build-arg OMNI_HOME=${omni_home} \\"
    log_info "    --build-arg OMNIBASE_COMPAT_REF=${compat_ref} \\"
    log_info "    --build-arg OMNIMARKET_REF=${omnimarket_ref}"
    log_info ""
    log_info "Restart runtime services:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d --no-deps --force-recreate \\"
    log_info "    ${RUNTIME_BUILD_SERVICES[*]}"
    log_info ""
    log_info "Full stack up (infra + runtime):"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d"
    log_info ""
    log_info "Stop all:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    down"
    log_info ""
    log_info "Logs:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    logs -f"
    log_info ""
    log_info "Status:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    ps"
}

# =============================================================================
# Summary
# =============================================================================

show_summary() {
    # Display post-deployment summary with next-step commands.
    local deploy_target="$1"
    local version="$2"
    local git_sha="$3"
    local compose_project="$4"
    # OMN-13581: surface the lane-overlay-aware `-f` sequence in operator
    # next-step commands too, so a copy-paste does not run the lane on infra.yml.
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    local compose_f="${compose_args[*]}"

    log_step "Deployment Summary"

    log_info "Deploy path:       ${deploy_target}"
    log_info "Version:           ${version}"
    log_info "Git SHA:           ${git_sha}"
    log_info "Compose project:   ${compose_project}"
    log_info "Profile:           ${COMPOSE_PROFILE}"
    log_info "Registry:          ${REGISTRY_FILE}"
    log_info ""
    log_info "Next steps (source ~/.omnibase/.env before running):"

    if [[ "${RESTART}" == false && "${COLD_FULL_BRINGUP}" == false ]]; then
        log_info "  To start containers, run:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      ${compose_f} \\"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      up -d"
    else
        log_info "  Containers are running. Check status:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      ${compose_f} \\"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      ps"
    fi

    log_info ""
    log_info "  Verify deployment:"
    log_info "    cat ${REGISTRY_FILE} | jq ."
    log_info "    docker inspect omninode-runtime --format='{{index .Config.Labels \"org.opencontainers.image.revision\"}}'"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Orchestrate the full deployment workflow from validation through verification.
    # OMN-15218: capture raw argv before parse_args consumes it so the attribution
    # record carries the literal command that touched the lane.
    DEPLOY_INVOCATION_ARGS=("$@")
    parse_args "$@"

    # Phase 1: Validate prerequisites
    validate_prerequisites

    # Resolve repository root
    local repo_root
    repo_root="$(resolve_repo_root)"
    log_info "Repository root: ${repo_root}"

    # Validate repo structure
    validate_repo_structure "${repo_root}"

    # Phase 2: Identity -- version + git SHA
    log_step "Build Identity"
    local version git_sha
    version="$(read_version "${repo_root}")"
    git_sha="$(read_git_sha "${repo_root}")"

    # Validate version format before using it in path construction.
    # A malformed version could create unexpected directory structures.
    # Policy: only stable release versions (MAJOR.MINOR.PATCH) are allowed for
    # deployment. Pre-release suffixes (e.g., 1.2.3-rc.1, 1.2.3-beta) are
    # intentionally rejected to ensure only tested releases reach production.
    if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: '${version}'"
        log_error "Expected semantic version (e.g., 1.2.3). Check pyproject.toml [project] version."
        exit 1
    fi

    # Validate git SHA format for VCS_REF image labeling.
    # Accept short (7+) or full (40) hex SHAs. read_git_sha uses --short=12
    # but other inputs (e.g., CI injection) may vary.
    # Normalize to lowercase first -- some CI systems produce uppercase hex.
    git_sha=$(echo "${git_sha}" | tr '[:upper:]' '[:lower:]')
    if [[ ! "${git_sha}" =~ ^[0-9a-f]{7,40}$ ]]; then
        log_warn "Could not read valid git SHA (got: '${git_sha}')."
        log_warn "The VCS_REF Docker label may be inaccurate."
        git_sha="unknown"
    fi

    log_info "Version: ${version}"
    log_info "Git SHA: ${git_sha}"
    check_git_dirty "${repo_root}"
    validate_build_source_config

    # Prod lane: hard-fail on dirty/non-promoted source before any build/deploy.
    # Runs in both dry-run and execute modes so operators see the rejection
    # during preview, not after a build starts (OMN-12626, R1).
    guard_prod_promotion_lineage "${repo_root}"

    # Compute paths
    local deploy_target="${DEPLOY_ROOT}/deployed/${version}"
    local compose_project
    compose_project="$(resolve_compose_project)"
    # OMN-15352: mirror into the global cleanup_on_exit() (a no-argument EXIT
    # trap handler) reads to resolve :latest image names on a failed deploy.
    DEPLOY_COMPOSE_PROJECT="${compose_project}"

    # --cold lane-scope guard (OMN-16803). Runs here, not in parse_args, because
    # the target lane is only known once the compose project is resolved — the
    # parse_args PROD_LANE check cannot see a prod lane selected purely via
    # OMNIBASE_INFRA_COMPOSE_PROJECT.
    guard_cold_bringup_lane_scope "${compose_project}"

    # Lane-deploy attribution + live-grant interlock (OMN-15218). FIRST gate that
    # runs once the target lane is known and BEFORE anything is built, recreated,
    # or restarted: an unattributed deploy must not get as far as touching an
    # image, and a stability refresh must not silently erode the stability-proven
    # premise of a live prod-promotion grant.
    guard_lane_deploy_attribution "${repo_root}" "${compose_project}"

    # Hot-patch ledger preflight: refuse to rebuild over live in-container
    # hot-patches whose source PRs are not merged into the build ref.
    # Runs in both dry-run and execute modes (OMN-13014, retro B-1).
    guard_hotpatch_ledger "${repo_root}" "${git_sha}" "${compose_project}"

    # --print-compose-cmd: show commands and exit
    if [[ "${PRINT_COMPOSE_CMD}" == true ]]; then
        print_compose_commands "${deploy_target}" "${compose_project}" "${git_sha}"
        exit 0
    fi

    # Phase 2.5: Compose project collision check
    # Runs in both dry-run and execute modes so operators see collisions during
    # preview. Skipped only when Docker is unavailable (non-fatal in that case).
    if command -v docker &>/dev/null; then
        check_compose_project_collision "${compose_project}" "${deploy_target}"
    else
        log_warn "Docker not available -- skipping compose project collision check."
    fi

    # Phase 3: Preview
    show_preview "${repo_root}" "${version}" "${git_sha}" "${deploy_target}" "${compose_project}"

    # Dry-run mode: stop here
    if [[ "${MODE}" == "dry-run" ]]; then
        log_step "Dry Run Complete"
        log_info "No changes were made. To deploy, re-run with --execute:"
        log_info "  ${SCRIPT_NAME} --execute"
        exit 0
    fi

    # =========================================================================
    # Execute mode from here
    # =========================================================================

    # Phase 4: Lock
    acquire_lock

    # Phase 5: Guard
    guard_existing_deployment "${deploy_target}"

    # Phase 6: Sync
    sync_files "${repo_root}" "${deploy_target}"

    # OMN-13415: assert the freshly-synced deployed (bind-mounted) forward-migration
    # tree is byte-identical to the canonical clone @ the target SHA BEFORE the
    # forward-migration phase. The stability-promotion footgun was a stale
    # bind-mounted tree (old 0016, no 0018/0019) that made the lane look "deployed"
    # while running the wrong migration SQL — caught only by an out-of-band rsync.
    # This gate makes that drift fail the deploy instead of silently mis-migrating.
    assert_deployed_migration_tree_synced "${deploy_target}" "${repo_root}" "${git_sha}"

    # OMN-13364: snapshot the freshly-synced vendored migration tree so a later
    # backup-restore (cleanup_on_exit) re-applies it instead of reverting the
    # deployed migrations to the backup's stale, pre-build snapshot.
    snapshot_migration_tree "${deploy_target}"

    # Mark deployment directory for cleanup on failure. If the build or any
    # later phase fails, cleanup_on_exit() will remove this orphaned directory
    # (unless registry.json already points to it). OMN-15352: stays armed for
    # the whole deploy now that the registry write is commit-on-success -- there
    # is no longer an early point at which disarming it would be safe.
    DEPLOY_DIR_TO_CLEANUP="${deploy_target}"

    # Phase 7: Env setup -- REMOVED (F65 / OMN-6910)
    # Shell environment is sourced at script top; no stale .env copy needed.

    # Phase 8: Sanity check
    sanity_check "${deploy_target}" "${compose_project}"

    # Phase 9: Registry write is DEFERRED to commit-on-success, after Phase 12
    # (OMN-15352). Everything that can actually fail -- build, migration
    # preflight, restart, readback -- runs first; registry.json is written only
    # once none of it failed, so a failed deploy never leaves the registry
    # asserting a version that was never running. See the write_registry() call
    # near the deployment-complete marker below.

    # Snapshot the pre-build `:latest` image id for every service this build
    # will retag, so a failed deploy can restore it (OMN-15352 F3).
    snapshot_latest_image_tags "${compose_project}"

    # Phase 10: Build
    build_images "${deploy_target}" "${compose_project}" "${git_sha}"

    # Phase 11: Bring runtime up (optional).
    #   --cold    -> cold-lane FULL bring-up: deps + migration one-shots + the
    #               WHOLE --profile runtime project (OMN-13414).
    #   --restart -> WARM path: deps + migration one-shots + recreate only the
    #               RUNTIME_SERVICES subset (--no-deps).
    # Both share the cold-start preflight (core infra readiness, broker partition
    # cap, migration one-shots, raised Kafka consumer-start budget); they differ
    # only in the final `up` (full-profile fan-out vs RUNTIME_SERVICES recreate).
    if [[ "${COLD_FULL_BRINGUP}" == true || "${RESTART}" == true ]]; then
        # Raise the per-consumer Kafka consumer-start budget for the restart-driven
        # cold boot (OMN-13220). x-runtime-env reads KAFKA_TIMEOUT_SECONDS from the
        # shell environment (default 30s when unset); exporting it here propagates
        # the raised cold-start value to every runtime container compose recreates.
        # Validate + clamp to ModelKafkaEventBusConfig.timeout_seconds bounds
        # (ge=1, le=300) so an operator override cannot produce a config the
        # kernel rejects at boot.
        local cold_start_timeout="${COLD_START_KAFKA_TIMEOUT_SECONDS}"
        if [[ ! "${cold_start_timeout}" =~ ^[0-9]+$ ]]; then
            log_error "COLD_START_KAFKA_TIMEOUT_SECONDS must be a positive integer (got: '${cold_start_timeout}')."
            return 1
        fi
        if (( cold_start_timeout < 1 )); then
            cold_start_timeout=1
        elif (( cold_start_timeout > 300 )); then
            log_warn "COLD_START_KAFKA_TIMEOUT_SECONDS=${cold_start_timeout} exceeds the config max (300); clamping to 300."
            cold_start_timeout=300
        fi
        export KAFKA_TIMEOUT_SECONDS="${cold_start_timeout}"
        log_info "Cold-start Kafka consumer-start budget: KAFKA_TIMEOUT_SECONDS=${KAFKA_TIMEOUT_SECONDS}s"
        # OMN-13594: bring up + wait for postgres/valkey BEFORE the migration
        # preflight. On a cold lane the preflight's forward-migration runs
        # `--no-deps` and would otherwise hit a non-existent Postgres, exhaust its
        # readiness budget, and trigger an auto-rollback. Idempotent no-op on a
        # warm lane.
        ensure_core_infra_ready "${deploy_target}" "${compose_project}"
        warm_broker_topic_provisioning "${deploy_target}" "${compose_project}"
        run_runtime_migration_preflight "${deploy_target}" "${compose_project}"
        if [[ "${COLD_FULL_BRINGUP}" == true ]]; then
            # Cold lane: fan out across the WHOLE runtime profile (OMN-13414).
            bringup_full_stack "${deploy_target}" "${compose_project}"
        else
            # Warm lane: recreate only the RUNTIME_SERVICES subset (--no-deps).
            restart_services "${deploy_target}" "${compose_project}"
        fi
    fi

    # Phase 12: Verify (after a cold bring-up or a warm --restart)
    if [[ "${COLD_FULL_BRINGUP}" == true || "${RESTART}" == true ]]; then
        verify_deployment "${git_sha}" "${compose_project}"
        # Phase 12b: TERMINAL fail-closed deploy readback (RT-6, OMN-14469). Runs
        # only when this invocation actually started containers (there is nothing
        # to read back otherwise). A stale / mis-targeted running container is
        # rejected here instead of passing with only verify_deployment's warning.
        readback_deployed_ref "${git_sha}" "${version}" "${compose_project}" "${repo_root}" "${deploy_target}"
    fi

    # Phase 9 (commit-on-success, OMN-15352): every phase that can fail --
    # build, migration preflight, restart, readback -- has now passed. Write
    # the registry only now, closing the write-ahead window that let a failed
    # deploy leave registry.json asserting a version that was never actually
    # running.
    write_registry "${version}" "${git_sha}" "${deploy_target}" "${repo_root}" "${compose_project}"

    # Registry now points to this deployment -- disable partial cleanup.
    DEPLOY_DIR_TO_CLEANUP=""

    # All phases completed successfully. Mark deployment as complete so that
    # cleanup_on_exit knows the backup can be safely removed rather than restored.
    DEPLOYMENT_COMPLETE=true

    # Remove the --force backup (if any) since the new deployment is fully
    # built and running. cleanup_on_exit would also handle this (since
    # DEPLOYMENT_COMPLETE=true), but explicit cleanup here keeps the success
    # path self-documenting.
    if [[ -n "${FORCE_BACKUP_DIR}" && -d "${FORCE_BACKUP_DIR}" ]]; then
        log_info "Removing previous deployment backup: ${FORCE_BACKUP_DIR}"
        rm -rf "${FORCE_BACKUP_DIR}"
        FORCE_BACKUP_DIR=""
    fi

    # Phase 13: Summary
    show_summary "${deploy_target}" "${version}" "${git_sha}" "${compose_project}"

    # Phase 14: Prune old deployments (non-fatal -- must not trigger rollback)
    prune_old_deployments || log_warn "Pruning old deployments failed (non-fatal)"
}

main "$@"
