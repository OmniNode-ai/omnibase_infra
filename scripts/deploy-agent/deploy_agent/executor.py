# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executor with phase timeouts for deploy operations."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, ConfigDict

from deploy_agent.build_budget import (
    ModelBuildBudget,
    derive_image_build_budget,
)
from deploy_agent.compose_budget import (
    ModelPhaseBudget,
    derive_runtime_phase_budget,
)
from deploy_agent.events import (
    DEV_LANE_ONLY_BUILDABLE_SERVICES,
    BuildSource,
    EnumRuntimeLane,
    EnumSelfUpdateBoundary,
    ModelContainerResidue,
    ModelHealthCheck,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)
from deploy_agent.ref_fence import (
    ModelRefLineageFacts,
    assert_ref_not_stale_branch,
)
from deploy_agent.tracking_ref import (
    load_tracking_ref_from_env,
    load_tracking_remote_ref_from_env,
)

# Maps deploy scope to catalog bundle names used by compose_gen.
# Scope.FULL regenerates both core and runtime bundles.
SCOPE_BUNDLES: dict[Scope, list[str]] = {
    Scope.CORE: ["core"],
    Scope.RUNTIME: ["core", "runtime"],
    Scope.FULL: ["core", "runtime"],
}

logger = logging.getLogger(__name__)

REPO_DIR = os.environ.get(
    "DEPLOY_AGENT_REPO_DIR", "/data/omninode/omni_home/omnibase_infra"
)
DEPLOY_AGENT_DIR = os.environ.get(
    "DEPLOY_AGENT_DIR", "/data/omninode/omnibase_infra/scripts/deploy-agent"
)
# The TRACKED compose base every deploy path layers its lane overlay on:
# this agent, scripts/deploy-runtime.sh, and
# scripts/runtime_build/refresh_stability_lane.sh alike. Its only writer is git.
COMPOSE_FILE = f"{REPO_DIR}/docker/docker-compose.infra.yml"
# OMN-17291: where the catalog render lands. A BUILD ARTIFACT, gitignored, never
# the tracked file above -- see compose_gen() for why that distinction is the
# whole point.
COMPOSE_GEN_OUTPUT_FILE = f"{REPO_DIR}/docker/docker-compose.generated.yml"
COMPOSE_PROJECT = "omnibase-infra"
RUNTIME_POLICY_ENV_FILE = Path(REPO_DIR) / "docker" / "runtime-policy.env"

PHASE_TIMEOUTS = {
    Phase.PREFLIGHT: 30,
    Phase.GIT: 60,
    Phase.COMPOSE_GEN: 120,
    Phase.CORE: 300,
    # OMN-18057: this entry no longer bounds the runtime compose-up. It bounds
    # the runtime IMAGE operations (build, pinned-digest pull) and the migration
    # preflight, which are not gated on any healthcheck. The compose-up ceiling
    # is derived from the compose model -- see runtime_compose_up_budget below,
    # and deploy_agent.compose_budget for why a constant cannot express it.
    Phase.RUNTIME: 300,
    Phase.VERIFICATION: 120,
}

# OMN-18057: the two windows _compose_up gives the lane to settle after the
# compose command returns -- named rather than inline so the recovery path reads
# as a policy and can be exercised without a three-minute unit test.
CONTAINER_VERIFY_TIMEOUT_SECONDS = 120
CONTAINER_RECOVERY_VERIFY_TIMEOUT_SECONDS = 60

# OMN-18057: added to the largest gating start_period to form the runtime
# compose-up ceiling. It must cover the unhealthy-detection tail the compose
# file states for the runtime family -- interval 30s x retries 5 = 150s beyond
# start_period -- plus the stop/create/start of the rest of the selected set.
RUNTIME_COMPOSE_UP_MARGIN_SECONDS = 300

# OMN-18057: the floor the derived ceiling can never fall below. MEASURED
# 2026-09-08 (ledger :5076): the minimum viable budget for the ten-service dev
# force-recreate was 336s (bootstrap 249.6s, :8085 bound at t+321s). The floor
# sits above it with room, and applies when the compose model declares no
# health-gated start_period at all.
RUNTIME_COMPOSE_UP_FLOOR_SECONDS = 600

# OMN-18072: the runtime IMAGE-BUILD ceiling's two derived terms. `docker
# compose --profile runtime build` runs ONE BuildKit solve over the Dockerfile
# every buildable service in the profile shares, then exports one image per
# service -- so the cost scales with the Dockerfile's work-step count and with
# the service count, and both are read from the model in build_budget.
#
# MEASURED on the dev lane, 2026-09-09, from this agent's own job history:
#   6c323639  build start 01:44:30.717Z -> first core container Created
#             01:48:24.652Z  =>  <= 233.9s over 60 steps / 9 images, SUCCEEDED
#             (~3.9s per step)
#   79171e79  images exported at t+209.3s, killed at t+300.0s
#   2788af33  killed at t+300.3s with a WARM BuildKit cache
# The per-step budget is ~4x the measured per-step cost and the per-image
# budget ~7x the measured export cost, because this ceiling exists to catch a
# HUNG build, not to be tight around a healthy one: two of the three
# observations above are right-censored (killed, not measured), and a genuinely
# cold cache after a prune is longer than any of them.
RUNTIME_IMAGE_BUILD_PER_STEP_SECONDS = 15
RUNTIME_IMAGE_BUILD_PER_IMAGE_SECONDS = 20

# OMN-18072: twice the flat constant that killed two consecutive live rebuilds.
# A model that read as unexpectedly small must never re-derive a ceiling at or
# below the one already proven insufficient.
RUNTIME_IMAGE_BUILD_FLOOR_SECONDS = 600

PhaseCallback = Callable[[Phase, PhaseStatus], None]

RUNTIME_HEALTH_TARGETS: tuple[tuple[str, int], ...] = (
    ("omninode-runtime", 8085),
    ("runtime-effects", 8086),
)
RUNTIME_MIGRATION_SERVICES: tuple[str, ...] = (
    "forward-migration",
    "migration-gate",
)
REQUIRED_PROJECTION_TABLES: tuple[str, ...] = (
    "delegation_events",
    "node_service_registry",
)

_BUILD_SOURCE_ALLOWED = ", ".join(source.value for source in BuildSource)
_BUILD_PROVENANCE_BY_SOURCE: dict[BuildSource, tuple[str, str]] = {
    BuildSource.WORKSPACE: ("stability-candidate", "true"),
    BuildSource.RELEASE: ("clean-main", "false"),
}

# Promotion-lineage guard (OMN-12626, R1). Loaded from scripts/ by file path
# because scripts/ is not an importable package. The guard refuses to build a
# prod-bound (release-mode) image from a dirty or non-promoted source tree.
_PROMOTION_GUARD_PATH = Path(REPO_DIR) / "scripts" / "check_prod_promotion_lineage.py"


def _load_promotion_guard() -> ModuleType:
    """Load the prod promotion-lineage guard module from scripts/ by path.

    Raises RuntimeError (fail-fast) when the guard is missing so a release
    build can never silently skip the clean-tree + promoted-lineage check.
    """
    mod_name = "check_prod_promotion_lineage"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if not _PROMOTION_GUARD_PATH.is_file():
        raise RuntimeError(
            "prod promotion-lineage guard not found at "
            f"{_PROMOTION_GUARD_PATH}; cannot verify clean+promoted build source."
        )
    spec = importlib.util.spec_from_file_location(mod_name, _PROMOTION_GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load prod promotion-lineage guard from {_PROMOTION_GUARD_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def assert_release_build_promoted(
    build_source: BuildSource,
    *,
    repo_dir: str = REPO_DIR,
    runtime_lane: EnumRuntimeLane = EnumRuntimeLane.PROD,
) -> None:
    """Enforce clean + promoted source for prod-bound (release-mode) builds.

    Release-mode builds produce the digest that is later pinned and promoted to
    prod. They MUST come from a clean working tree whose HEAD is an
    ancestor-of/equal-to origin/main. Workspace builds (local dev iteration) are
    exempt by design — they never reach prod.

    OMN-16442: the assertion is also scoped to the lanes whose artifacts can
    reach prod. It used to be applied lane-blind, which made it unsatisfiable
    on the dev lane rather than merely strict: a dev head is by construction
    NOT an ancestor of a release-synced ``origin/main``, so the ancestry half
    refused every dev release-mode build on every day, and the clean-tree half
    refused on any stray file in the deploy-source clone. Measured live
    2026-09-08 on command c73cc38a: DIRTY_TREE first, then NOT_PROMOTED for
    HEAD e42519c5b against origin/main 276d69383.

    The exemption is the DEV lane and only the DEV lane, named rather than
    derived from a negation, because the stability lane is where the
    ``stability-proven`` digest of a prod promotion grant comes from
    (CLAUDE.md rule 12, OMN-15243) and must keep the same lineage requirement
    prod has. The default is ``PROD`` so an undeclared lane fails CLOSED —
    the gate applies unless a caller says which exempt lane it is on.

    Raises the guard's ``ProdLineageError`` when the source is dirty or
    not promoted. Fails the build CLOSED before any docker build side effects.
    """
    if build_source != BuildSource.RELEASE:
        return
    if runtime_lane == EnumRuntimeLane.DEV:
        logger.info(
            "assert_release_build_promoted: lane %s is exempt from the prod "
            "promotion-lineage assertion (a dev head is never an ancestor of "
            "the release-synced origin/main); build source %s not checked",
            runtime_lane.value,
            repo_dir,
        )
        return
    guard = _load_promotion_guard()
    sha = guard.assert_prod_build_promoted(Path(repo_dir))
    logger.info(
        "assert_release_build_promoted: release build source %s is clean and "
        "promoted (HEAD %s is ancestor-of/equal-to origin/main)",
        repo_dir,
        sha[:12],
    )


class DigestMismatchError(RuntimeError):
    """Raised when the running container image digest != the requested digest.

    Fails the deploy closed before any health check runs — a lane must never be
    marked healthy while serving an artifact that does not match the pinned
    (stability-proven) digest.
    """


class ProdStabilityDigestMissingError(RuntimeError):
    """Raised when a prod deploy request lacks a matching stability READY digest.

    This is a boundary-level guard: it must fire before any deploy effect runs,
    not just before health checks.
    """


class ModelLaneConfig(BaseModel):
    """Per-lane compose file(s), compose project, and health targets.

    The base ``docker-compose.infra.yml`` is always the first compose file;
    non-dev lanes layer their overlay (``docker-compose.<lane>.yml``) on top so
    the overlay's container names, project, and host port bindings win.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: EnumRuntimeLane
    compose_files: tuple[str, ...]
    compose_project: str
    postgres_container: str
    runtime_health_targets: tuple[tuple[str, int], ...]


_STABILITY_OVERLAY = f"{REPO_DIR}/docker/docker-compose.stability-test.yml"
_PROD_OVERLAY = f"{REPO_DIR}/docker/docker-compose.prod.yml"
# OMN-15379: the dev/lab lane's own overlay. Its only content is
# ``ONEX_MIGRATION_LANE=dev`` on forward-migration -- the lane indicator that
# releases the node_projection_registration trio (0000/0001/0002, CREATE +
# heartbeat + ENABLE/FORCE ROW LEVEL SECURITY) from the operator fence, per
# operator ruling 15 which makes the lab the FORCE proving ground. It is a
# SEPARATE file, not a line in the base compose, so that no non-dev lane can
# inherit it: every lane overlay merges the base, and stability-test's
# forward-migration override inherits the base ``environment:`` block wholesale.
# Unset indicator = FULL fence, so this list is fail-closed on omission.
# Must stay matched with ``resolve_compose_file_args`` in
# ``scripts/deploy-runtime.sh``.
_DEV_LANE_OVERLAY = f"{REPO_DIR}/docker/docker-compose.dev-lane.yml"

# OMN-15181 round 3 (Finding 9): maps each prod runtime service to the compose
# env var that repoints its `image:` field (docker-compose.prod.yml). This is
# the single source mapping consumed by ``_resolve_prod_image_env`` -- no
# forked second copy of the service->env-var relationship.
PROD_IMAGE_ENV_VAR_FOR_SERVICE: dict[str, str] = {
    "omninode-runtime": "PROD_OMNINODE_RUNTIME_IMAGE",
    "runtime-effects": "PROD_RUNTIME_EFFECTS_IMAGE",
}

_LANE_CONFIGS: dict[EnumRuntimeLane, ModelLaneConfig] = {
    EnumRuntimeLane.DEV: ModelLaneConfig(
        lane=EnumRuntimeLane.DEV,
        compose_files=(COMPOSE_FILE, _DEV_LANE_OVERLAY),
        compose_project=COMPOSE_PROJECT,
        postgres_container="omnibase-infra-postgres",
        runtime_health_targets=RUNTIME_HEALTH_TARGETS,
    ),
    EnumRuntimeLane.STABILITY_TEST: ModelLaneConfig(
        lane=EnumRuntimeLane.STABILITY_TEST,
        compose_files=(COMPOSE_FILE, _STABILITY_OVERLAY),
        compose_project="omnibase-infra-stability-test",
        postgres_container="omnibase-infra-stability-test-postgres",
        # OMN-15181: must match the container_name: override in
        # docker-compose.stability-test.yml, never the dev-lane bare name —
        # docker inspect on the bare name returns "no such object" for this lane.
        runtime_health_targets=(
            ("omninode-stability-test-runtime", 18085),
            ("omninode-stability-test-runtime-effects", 18086),
        ),
    ),
    EnumRuntimeLane.PROD: ModelLaneConfig(
        lane=EnumRuntimeLane.PROD,
        compose_files=(COMPOSE_FILE, _PROD_OVERLAY),
        compose_project="omnibase-infra-prod",
        postgres_container="omnibase-infra-prod-postgres",
        # OMN-15181: must match the container_name: override in
        # docker-compose.prod.yml, never the dev-lane bare name — the live
        # PREFLIGHT-STOP defect was verify_running_image_digest inspecting
        # the bare "omninode-runtime"/"runtime-effects" names, which exist on
        # no real lane, so every prod deploy raised DigestMismatchError
        # unconditionally regardless of actual outcome.
        runtime_health_targets=(
            ("omninode-prod-runtime", 28085),
            ("omninode-prod-runtime-effects", 28086),
        ),
    ),
}


def lane_config_for(lane: EnumRuntimeLane) -> ModelLaneConfig:
    """Return the compose/project/health configuration for a runtime lane."""
    return _LANE_CONFIGS[lane]


# OMN-15181 round 4 (Finding 11): single shared source mapping a canonical
# requested service name to its position in every lane's ``runtime_health_targets``
# tuple (see ``_LANE_CONFIGS`` above -- each lane orders its targets
# (runtime, runtime-effects)). Consumed by both ``resolve_stability_ready_digest``
# and ``verify_running_image_digest`` so per-service digest resolution never
# forks a second, independently-drifting copy of "which index is which
# service" -- the same discipline ``PROD_IMAGE_ENV_VAR_FOR_SERVICE`` already
# established for the compose image-env override.
SERVICE_HEALTH_TARGET_INDEX: dict[str, int] = {
    "omninode-runtime": 0,
    "runtime-effects": 1,
}


def _health_target_container(lane: EnumRuntimeLane, service: str) -> str:
    """Return the lane-qualified container name for a canonical service name.

    Fails loud (``RuntimeError``) on an unknown service rather than silently
    defaulting to index 0 -- the exact live defect this fix closes: every
    prod stability-digest check was comparing against the RUNTIME container
    regardless of which service the request actually targeted, wrongly
    rejecting a runtime-effects command carrying its own (genuinely
    stability-proven) effects digest.
    """
    index = SERVICE_HEALTH_TARGET_INDEX.get(service)
    if index is None:
        raise RuntimeError(
            f"no runtime_health_targets mapping for service {service!r}; known "
            f"services: {sorted(SERVICE_HEALTH_TARGET_INDEX)}"
        )
    container_name, _ = lane_config_for(lane).runtime_health_targets[index]
    return container_name


def resolve_prod_target_service(cmd: ModelRebuildRequested) -> str:
    """Return the single image-bearing service a prod request targets.

    Mirrors ``_resolve_prod_image_env``'s target-service resolution so the
    stability-digest guard and post-deploy verification check the SAME
    service the deploy actually touches. Falls back to the pre-round-4
    default (``"omninode-runtime"``) when the request does not name exactly
    one image-bearing service (e.g. an empty-``services`` full-scope
    deploy) -- disambiguating a single shared digest across multiple
    services is out of scope for this fix.
    """
    candidates = [s for s in cmd.services if s in PROD_IMAGE_ENV_VAR_FOR_SERVICE]
    if len(candidates) == 1:
        return candidates[0]
    return "omninode-runtime"


def _compose_file_args(lane: EnumRuntimeLane) -> list[str]:
    """Return the ``-f <file>`` token sequence for a lane's compose invocation."""
    args: list[str] = []
    for compose_file in lane_config_for(lane).compose_files:
        args.extend(["-f", compose_file])
    return args


def assert_prod_request_has_stability_digest(
    cmd: ModelRebuildRequested, *, stability_ready_digest: str | None
) -> None:
    """Reject a prod request lacking a matching stability READY digest.

    Boundary-level guard: this is invoked before any deploy effect so a prod
    deploy can never start without a stability-proven artifact. Non-prod lanes
    are unaffected.
    """
    if cmd.runtime_lane != EnumRuntimeLane.PROD:
        return
    if stability_ready_digest is None:
        raise ProdStabilityDigestMissingError(
            "prod deploy rejected: no stability-test READY digest is available; "
            "production may only deploy a digest already proven in stability-test"
        )
    if cmd.image_digest != stability_ready_digest:
        raise ProdStabilityDigestMissingError(
            "prod deploy rejected: requested image_digest "
            f"{cmd.image_digest!r} does not equal the stability-test READY digest "
            f"{stability_ready_digest!r}"
        )


def _requested_services_for_up(
    scope: Scope, services: list[str], *, lane: EnumRuntimeLane | None = None
) -> list[str]:
    """Return the explicit service list compose should recreate for this scope.

    Runtime scope must always target the runtime service list directly so a
    runtime-only rebuild cannot recreate core infra dependencies via compose's
    dependency graph. Core and full scopes retain the historical behavior
    (compose chooses services from the active profile) by returning an empty
    list when no explicit service selection was provided.

    OMN-9455: a runtime rebuild on 2026-04-22 invoked ``docker compose
    --profile runtime up -d --force-recreate --pull always`` without a service
    list or ``--no-deps``. Compose recreated dependency graph services and
    collided with the live ``omnibase-infra-infisical`` container, breaking
    Redpanda/Postgres/Valkey/Phoenix. Forcing an explicit runtime service list
    combined with ``--no-deps`` in ``_compose_up`` prevents that regression.

    OMN-18108: ``lane`` is threaded through so a DEV runtime deploy also
    recreates the dev-lane-only services. It defaults to ``None`` (the
    lane-agnostic base list) so a caller that names no lane cannot silently
    acquire them, and prod/stability behaviour is unchanged.
    """
    if services:
        return services
    if scope == Scope.RUNTIME:
        return services_for_scope(scope, lane=lane)
    return []


def _coerce_build_source(value: BuildSource | str, *, layer: str) -> BuildSource:
    try:
        return BuildSource(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {layer} BUILD_SOURCE={value!r}; expected one of: {_BUILD_SOURCE_ALLOWED}"
        ) from exc


def _build_source_build_args(
    build_source: BuildSource | str,
    *,
    expected_build_source: BuildSource | str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return validated immutable build-source args for docker compose build."""
    selected = _coerce_build_source(build_source, layer="deploy-agent")
    expected = _coerce_build_source(
        selected if expected_build_source is None else expected_build_source,
        layer="expected",
    )
    if selected != expected:
        raise RuntimeError(
            "BUILD_SOURCE selector disagreement: "
            f"deploy-agent selected {selected.value!r}, "
            f"Dockerfile expected {expected.value!r}"
        )

    source_env = os.environ if env is None else env
    omni_home = source_env.get("OMNI_HOME", "").strip()
    if selected == BuildSource.WORKSPACE and not omni_home:
        raise RuntimeError("BUILD_SOURCE=workspace requires OMNI_HOME before build")

    promotion_class, non_main_lineage = _BUILD_PROVENANCE_BY_SOURCE[selected]

    return [
        "--build-arg",
        f"BUILD_SOURCE={selected.value}",
        "--build-arg",
        f"EXPECTED_BUILD_SOURCE={expected.value}",
        "--build-arg",
        f"PROMOTION_CLASS={promotion_class}",
        "--build-arg",
        f"NON_MAIN_LINEAGE={non_main_lineage}",
        "--build-arg",
        f"OMNI_HOME={omni_home}",
    ]


def _run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


def _load_runtime_policy_env(path: Path | None = None) -> dict[str, str]:
    """Load contract-rendered runtime policy env values."""
    env_path = RUNTIME_POLICY_ENV_FILE if path is None else path
    return _load_dotenv_file(env_path)


def _load_dotenv_file(env_path: Path) -> dict[str, str]:
    """Parse one committed dotenv file with shell-compatible quoting."""
    if not env_path.exists():
        return {}

    policy_env: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        policy_env[key.strip()] = _parse_runtime_policy_env_value(value)
    return policy_env


def _parse_runtime_policy_env_value(value: str) -> str:
    """Parse one runtime-policy dotenv value with shell-compatible quoting."""
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        tokens = shlex.split(f"VALUE={stripped}", posix=True)
    except ValueError:
        return stripped
    if len(tokens) != 1 or not tokens[0].startswith("VALUE="):
        return stripped
    return tokens[0].split("=", 1)[1]


class UndecodedAnsiCQuotingError(RuntimeError):
    """Raised when an inherited env value is still in un-decoded ANSI-C form.

    See :func:`_undecoded_ansi_c_quoted_names` for why this is fatal rather
    than repairable.
    """


def _undecoded_ansi_c_quoted_names(env: Mapping[str, str]) -> list[str]:
    """Return the NAMES of values still wrapped in bash ANSI-C ``$'...'`` quoting.

    OMN-18073. This agent inherits its environment from systemd, whose unit
    files declare ``EnvironmentFile=<the operator env store>``. **systemd's
    env-file parser does not implement bash ANSI-C ``$'...'`` quoting.** Given
    a line written in that form it keeps the literal ``$'`` and ``'`` wrapper
    and drops every backslash escape, so each ``\\n`` collapses to the bare
    letter ``n``. ``deploy-runtime.sh`` bash-``source``s the very same file and
    decodes it correctly, which is why only the agent path is affected.

    Measured on the lab host 2026-09-09 with a synthetic, non-secret value of
    the same shape: bash ``source`` yielded a 60-byte value with 4 real
    newlines; systemd's ``EnvironmentFile`` yielded the same bytes still
    wrapped in ``$'``/``'`` with those 4 newlines rendered as the letter ``n``.

    **The damage is irreversible in transit, so this guard refuses rather than
    repairs.** Once the backslashes are gone, nothing downstream can tell a
    newline's ``n`` from an ``n`` that belongs to the payload -- a base64 PEM
    body legitimately contains the letter. Any "normalizer" that guesses is
    reconstructing a different value and calling it the original.

    Refusing is the whole point. :func:`_compose_env` hands this mapping
    straight to ``docker compose``, whose ``${VAR:-}`` interpolation writes it
    into every container the deploy creates. Passing a provably-mangled
    credential through silently is what produced a 100% OCC-mint outage that
    ran for eight hours before anyone noticed: the runtime's own secret
    resolver logged only a ``no_mapping`` warning, the call site's
    ``env_var_fallback`` handed the mangled bytes to pyjwt, and the resulting
    ``InvalidKeyError`` surfaced nowhere near the transport that caused it.

    Names only -- a value is never returned, logged, or included in the error.
    """
    return sorted(
        name
        for name, value in env.items()
        if value.startswith("$'") and value.endswith("'") and len(value) >= 3
    )


def _assert_no_undecoded_ansi_c_quoting(env: Mapping[str, str]) -> None:
    """Fail loud if any inherited value is still ANSI-C quoted (OMN-18073)."""
    offenders = _undecoded_ansi_c_quoted_names(env)
    if not offenders:
        return
    raise UndecodedAnsiCQuotingError(
        "Refusing to hand docker compose an environment carrying un-decoded "
        f"bash ANSI-C quoting. Variable name(s): {', '.join(offenders)}. "
        "These values reached this process through systemd's "
        "EnvironmentFile=, which does not implement $'...' quoting: it keeps "
        "the literal wrapper and drops every backslash escape, so each \\n "
        "became the bare letter n. The original bytes cannot be recovered "
        "from what is left, so this is refused rather than repaired. Repair "
        "the operator env store: write the value as a real multi-line "
        'double-quoted entry (VAR="<line>\\n<line>\\n") -- the one shape both '
        "bash `source` and systemd's EnvironmentFile decode identically -- "
        "then restart this unit so it re-reads the file. No value is printed "
        "by this guard."
    )


def _compose_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in _load_runtime_policy_env().items():
        env.setdefault(key, value)
    # OMN-18073: refuse before compose interpolation writes a provably-mangled
    # value into every container this deploy creates.
    _assert_no_undecoded_ansi_c_quoting(env)
    postgres_host = env.get("POSTGRES_HOST", "127.0.0.1")
    postgres_port = env.get("POSTGRES_PORT", "5436")
    postgres_dsn = env.get("OMNIDASH_ANALYTICS_DB_URL") or (
        "postgresql://postgres:"
        f"{env.get('POSTGRES_PASSWORD', 'postgres')}@{postgres_host}:{postgres_port}/omnidash_analytics"
    )
    # OMN-17291: four sentinel placeholders used to be injected here --
    # CI_CALLBACK_TOKEN, LINEAR_WEBHOOK_SECRET, WAITLIST_NOTIFIER_SLACK_BOT_TOKEN
    # and WAITLIST_NOTIFIER_SLACK_CHANNEL_ID, all set to a literal
    # "parse-only" string. They existed for one reason: compose_gen had
    # overwritten the tracked compose file with the catalog render, and the
    # render's ci-relay / linear-relay / waitlist-signup-notifier services carry
    # those names as `${VAR:?}`, so `config` could not parse without them.
    #
    # They are gone because their reason is gone. Measured 2026-09-08 against
    # this repo at 00a821b1: those four names appear in ZERO committed compose
    # files -- not the base, not any lane overlay -- so no compose command this
    # agent issues after the single-writer fix references them. They were also
    # never sufficient: on the lab host, and in this agent's own process
    # environment, the render still fails interpolation on ONEX_TENANT_DB_URL
    # with all four supplied. Keeping a sentinel that neither fixes anything nor
    # is needed by anything is the silent default this ticket exists to remove.
    #
    # This is a real derived DSN, not a sentinel, so it stays.
    env.setdefault("OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN", postgres_dsn)
    # OMN-15181 round 3: prod-lane image repoint overrides (PROD_*_IMAGE) win
    # over any ambient os.environ value -- this is the one call site allowed
    # to override rather than setdefault, since it carries the caller's
    # explicit per-dispatch pinned-image resolution.
    if extra_env:
        env.update(extra_env)
    return env


def _env_with_repo_pythonpath(env: Mapping[str, str]) -> dict[str, str]:
    """Return env with this deploy repo's src path first on PYTHONPATH."""
    repo_src = f"{REPO_DIR}/src"
    current = env.get("PYTHONPATH", "")
    return {
        **env,
        "PYTHONPATH": f"{repo_src}:{current}" if current else repo_src,
    }


def _runtime_version_from_pyproject(repo_dir: str = REPO_DIR) -> str:
    """Return the runtime package version stamped into rebuilt images."""
    pyproject = Path(repo_dir) / "pyproject.toml"
    if not pyproject.is_file():
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data["project"]["version"]).strip()
    except (FileNotFoundError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"Could not resolve RUNTIME_VERSION from {pyproject}"
        ) from exc
    if not version:
        raise RuntimeError(f"Empty RUNTIME_VERSION in {pyproject}")
    return version


def _runtime_health_passed(result: subprocess.CompletedProcess) -> bool:
    """Return whether a runtime /health response proves deploy readiness."""
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    details = payload.get("details")
    if not isinstance(details, dict):
        return False
    return (
        payload.get("status") == "healthy"
        and details.get("is_running") is True
        and details.get("config_prefetch_status") in {"ok", "skipped"}
    )


def _compose_service_states(
    lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
) -> dict[str, tuple[str, int | None]]:
    result = subprocess.run(
        [
            "docker",
            "compose",
            *_compose_file_args(lane),
            "-p",
            lane_config_for(lane).compose_project,
            "ps",
            "-a",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_compose_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:200])
    states: dict[str, tuple[str, int | None]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        service = str(payload.get("Service") or "")
        if not service:
            continue
        exit_code = payload.get("ExitCode")
        states[service] = (
            str(payload.get("State") or ""),
            exit_code if isinstance(exit_code, int) else None,
        )
    return states


def _container_image_id(container_name: str) -> str | None:
    """Return the image id (``sha256:...``) a running container was created from.

    Uses ``docker inspect --format {{.Image}} <container>`` — a
    CONTAINER-inspect field. ``.RepoDigests`` exists only on IMAGE-inspect
    objects (and is empty for locally-built images that were never pushed to
    a registry, which is the normal case for the stability-test artifact) —
    running a ``.RepoDigests``-based format against a container always fails
    with "map has no entry for key RepoDigests" (OMN-15181 round-2 defect,
    live-reproduced on omninode-pc). Both digest-verification call sites
    (``resolve_stability_ready_digest``, ``verify_running_image_digest``)
    share this one helper — no forked second implementation.

    Returns ``None`` (fail-closed via the caller) when the container cannot
    be inspected.
    """
    result = _run(
        ["docker", "inspect", "--format", "{{.Image}}", container_name],
        timeout=PHASE_TIMEOUTS[Phase.VERIFICATION],
    )
    if result.returncode != 0:
        logger.warning(
            "_container_image_id: could not inspect %s: %s",
            container_name,
            result.stderr.strip() or result.stdout.strip(),
        )
        return None
    observed = result.stdout.strip()
    if not observed:
        logger.warning(
            "_container_image_id: empty image id for %s",
            container_name,
        )
        return None
    return observed


def _service_satisfied(state: str, exit_code: int | None) -> bool:
    if state == "running":
        return True
    return state == "exited" and exit_code == 0


def verify_containers_up(
    expected_containers: list[str],
    timeout_s: int = 120,
    *,
    lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
) -> tuple[bool, list[str]]:
    """Poll compose until services are running or completed successfully."""
    deadline = time.monotonic() + timeout_s
    last_states: dict[str, tuple[str, int | None]] = {}
    while time.monotonic() < deadline:
        try:
            last_states = _compose_service_states(lane)
        except RuntimeError as exc:
            logger.warning("verify_containers_up: docker compose ps failed: %s", exc)
            time.sleep(2)
            continue
        missing = [
            service
            for service in expected_containers
            if not _service_satisfied(*last_states.get(service, ("missing", None)))
        ]
        if not missing:
            return True, []
        logger.info(
            "verify_containers_up: waiting for %d service(s): %s",
            len(missing),
            missing,
        )
        time.sleep(2)
    try:
        last_states = _compose_service_states(lane)
    except RuntimeError:
        last_states = {}
    missing = [
        service
        for service in expected_containers
        if not _service_satisfied(*last_states.get(service, ("missing", None)))
    ]
    return False, missing


def runtime_compose_up_budget(
    lane: EnumRuntimeLane, expected_services: list[str]
) -> ModelPhaseBudget:
    """Derive the runtime compose-up ceiling from the lane's own compose files.

    Reads the SAME files the deploy is about to invoke (``lane_config_for``), so
    a compose change that lengthens a gating healthcheck moves the ceiling with
    it. See ``deploy_agent.compose_budget`` for the derivation and for why the
    previous bare ``300`` could not express this.
    """
    return derive_runtime_phase_budget(
        lane_config_for(lane).compose_files,
        expected_services,
        margin_seconds=RUNTIME_COMPOSE_UP_MARGIN_SECONDS,
        floor_seconds=RUNTIME_COMPOSE_UP_FLOOR_SECONDS,
    )


def runtime_image_build_budget(
    profile: str, compose_files: tuple[str, ...] = (COMPOSE_FILE,)
) -> ModelBuildBudget:
    """Derive the ``docker compose --profile <profile> build`` ceiling (OMN-18072).

    Reads the SAME compose files the build is about to invoke and the
    Dockerfiles they name, so a new runtime service or a new Dockerfile step
    moves the ceiling with it. See ``deploy_agent.build_budget`` for the
    measurements and for why a percentile over the recorded history is not
    derivable from a history whose two longest entries were killed at the flat
    constant rather than measured.

    ``compose_files`` defaults to the tracked base alone, which is what the
    lane-agnostic build passes. OMN-18108's DEV-lane addendum build passes the
    base plus the lane overlay: the ceiling it derives is therefore an
    over-estimate (it counts the base images too, which that command does not
    rebuild). Over-estimating a ceiling only delays a kill; under-estimating it
    kills a healthy build, which is the failure OMN-18072 existed to remove.
    """
    return derive_image_build_budget(
        compose_files,
        profile,
        per_step_seconds=RUNTIME_IMAGE_BUILD_PER_STEP_SECONDS,
        per_image_seconds=RUNTIME_IMAGE_BUILD_PER_IMAGE_SECONDS,
        floor_seconds=RUNTIME_IMAGE_BUILD_FLOOR_SECONDS,
    )


class DeployExecutor:
    def __init__(self) -> None:
        # OMN-18057: services a phase left in a non-running state, and whether
        # per-container recovery then got them up. Read by the agent when it
        # builds the terminal event so residue is a recorded fact rather than
        # something an operator has to go and find on the host.
        self.container_residue: list[ModelContainerResidue] = []
        # OMN-17135: repo -> the commit SHA RT-1 actually resolved and vendored
        # for that sibling. The requested ref pins omnibase_infra only, so
        # without this the terminal event named one repository's commit and left
        # the other three to be inferred from the image.
        self.sibling_source_refs: dict[str, str] = {}

    def reset_deploy_observations(self) -> None:
        """Clear per-job observations at the start of a rebuild."""
        self.container_residue = []
        self.sibling_source_refs = {}

    def _record_container_residue(
        self, stuck: list[str], *, lane: EnumRuntimeLane
    ) -> None:
        """Record the live state of every service that did not reach running."""
        try:
            states = _compose_service_states(lane)
        except RuntimeError as exc:
            logger.warning(
                "could not read compose state for residue recording: %s", exc
            )
            states = {}
        known = {item.service for item in self.container_residue}
        for service in stuck:
            if service in known:
                continue
            state, exit_code = states.get(service, ("unknown", None))
            self.container_residue.append(
                ModelContainerResidue(service=service, state=state, exit_code=exit_code)
            )

    def _mark_residue_recovered(self, recovered: set[str]) -> None:
        """Flip residue entries whose service came up under recovery."""
        self.container_residue = [
            item.model_copy(update={"recovered": True})
            if item.service in recovered
            else item
            for item in self.container_residue
        ]

    def validate_llm_endpoint_env_contract(self) -> None:
        """Fail runtime deploys when configured LLM endpoints drift from contract."""
        script = f"{REPO_DIR}/scripts/check_llm_endpoint_env_contract.py"
        venv_python = f"{REPO_DIR}/.venv/bin/python"
        base_cmd = (
            [venv_python, script]
            if Path(venv_python).is_file()
            else ["uv", "run", "--project", REPO_DIR, "python", script]
        )

        env_files: list[Path] = []
        explicit_env_file = os.environ.get("OMNIBASE_ENV_FILE")
        fallback_candidates = [
            Path.home() / ".omnibase" / ".env",
            Path(REPO_DIR) / ".env",
        ]
        candidates = (
            [Path(explicit_env_file), *fallback_candidates]
            if explicit_env_file
            else fallback_candidates
        )
        for candidate in candidates:
            if candidate.is_file() and candidate not in env_files:
                env_files.append(candidate)
        if explicit_env_file and not any(
            f == Path(explicit_env_file) for f in env_files
        ):
            raise RuntimeError(
                f"LLM endpoint env contract check failed: OMNIBASE_ENV_FILE does not exist: {explicit_env_file}"
            )

        commands = [base_cmd]
        commands.extend([*base_cmd, "--env-file", str(path)] for path in env_files)
        for cmd in commands:
            result = _run(cmd, timeout=PHASE_TIMEOUTS[Phase.PREFLIGHT])
            if result.returncode != 0:
                source = "process environment"
                if "--env-file" in cmd:
                    source = cmd[cmd.index("--env-file") + 1]
                raise RuntimeError(
                    "LLM endpoint env contract check failed for "
                    f"{source}: {result.stderr.strip() or result.stdout.strip()}"
                )

    def self_update(
        self,
        *,
        boundary: EnumSelfUpdateBoundary,
        skip: bool = False,
        on_before_reexec: Callable[[], None] | None = None,
    ) -> None:
        """Pull and re-exec deploy-agent itself if behind its tracking ref.

        Called ONLY at a job boundary, never between the phases of a deploy
        (OMN-16442). ``boundary`` is required and has no default: every call
        site names where it fired, the journal line carries that name, and a
        future mid-deploy caller cannot quietly omit it.

        Why the boundary is the whole contract. This method replaces the
        process image. Until 2026-09-08 it was invoked as the first statement
        of ``rebuild_scope`` -- that is, after preflight, git, compose_gen and
        seed had already run for an accepted command. Command
        ``8d0c861a-f91e-4ca2-954e-a073759dd39d`` on the .201 dev lane is the
        live proof of what that costs: those four phases succeeded, this method
        logged ``behind origin/dev ... pulling and re-execing``, and the
        replacement process logged ``Recovered 1 crashed job(s)`` and published
        the command as ``status=failed``. A process that re-execs mid-deploy
        cannot finish the deploy it is executing, so a deploy that starts on
        version X must be allowed to complete on version X.

        The two legal boundaries are declared in
        ``EnumSelfUpdateBoundary``: ``PRE_ACCEPT`` (before a polled command is
        marked started) and ``POST_TERMINAL`` (after a job's terminal status is
        published and the single-flight lock is released).

        ``on_before_reexec`` is invoked once the decision to update has been
        made and immediately before the pull, and only then. The ``PRE_ACCEPT``
        caller passes a callback that rewinds its committed consumer offset to
        the un-accepted command, so the replacement process re-reads that
        command instead of skipping it -- update-then-process, not
        process-then-die. Callers with nothing to hand off pass nothing.

        The branch is DECLARED, never hardcoded: ``DEPLOY_AGENT_TRACKING_REF``
        is required and has no default (OMN-16442, see
        ``deploy_agent.tracking_ref``). This method previously compared against
        a literal ``origin/main``; the .201 dev agent's clone is on ``dev``,
        hundreds of commits ahead of a release-synced ``main``, so it never
        self-updated and could not pick up its own fixes.

        Safety rails:
        - Skipped entirely when DEPLOY_AGENT_NO_SELF_UPDATE=1 is set.
        - Skipped when the working tree carries TRACKED modifications (a pull
          would discard uncommitted work). UNTRACKED files do not block
          (OMN-16442): ``git status --porcelain`` reports the whole repository
          regardless of the ``-C`` subdirectory, and the deploy path drops
          untracked build byproducts (historically
          ``workspace/deploy-source-refs.json``) into whatever repo root it runs
          from. One such file made the unfiltered check read dirty forever, so
          the agent skipped every self-update and could not pick up its own
          fixes -- the exact failure this method exists to prevent. A pull can
          only lose work that git is tracking; ``--untracked-files=no`` is the
          narrowest gate that still protects it.
        - skip=True (--skip-self-update CLI flag) bypasses the check.
        - Container mode (DEPLOY_AGENT_MODE=container) exits with code 42
          instead of os.execv so the supervisor can respawn from the new binary.
        - Host mode re-execs ``DEPLOY_AGENT_LAUNCHER`` when the launcher
          exported it, and the bare interpreter otherwise (OMN-18073). os.execv
          inherits the caller's environment, so an interpreter re-exec can only
          ever carry forward the environment this process started with -- which
          is how a mangled credential survived four self-updates on 2026-09-09
          while the code advanced normally. The launcher re-``source``s the
          operator env store, so code-on-disk and env-on-disk both become
          code-and-env-in-process.

        Raises:
            RuntimeError: when ``DEPLOY_AGENT_TRACKING_REF`` is unset. The
                kill-switch and ``skip=True`` are checked first, so a
                deliberately disabled self-update never needs the variable.
        """
        if skip or os.environ.get("DEPLOY_AGENT_NO_SELF_UPDATE") == "1":
            logger.info(
                "self_update[boundary=%s]: skipped (kill-switch active)", boundary.value
            )
            return

        branch = load_tracking_ref_from_env()
        remote_ref = f"origin/{branch}"
        agent_dir = os.environ.get("DEPLOY_AGENT_DIR", DEPLOY_AGENT_DIR)
        timeout = 60

        # Abort only on TRACKED modifications — a pull cannot lose an untracked
        # file, and untracked deploy byproducts in the clone are exactly what
        # made this gate never open (OMN-16442).
        status_result = _run(
            ["git", "-C", agent_dir, "status", "--porcelain", "--untracked-files=no"],
            timeout=timeout,
        )
        if status_result.returncode != 0:
            logger.warning(
                "self_update[boundary=%s]: git status failed (exit=%d), skipping update",
                boundary.value,
                status_result.returncode,
            )
            return
        tracked_changes = status_result.stdout.strip()
        if tracked_changes:
            logger.warning(
                "self_update[boundary=%s]: working tree has tracked modifications, "
                "skipping update to avoid data loss: %s",
                boundary.value,
                tracked_changes.replace("\n", "; "),
            )
            return

        # Fetch the declared tracking ref.
        fetch_result = _run(
            ["git", "-C", agent_dir, "fetch", "origin", branch],
            timeout=timeout,
        )
        if fetch_result.returncode != 0:
            logger.warning(
                "self_update[boundary=%s]: git fetch failed (exit=%d), skipping update: %s",
                boundary.value,
                fetch_result.returncode,
                fetch_result.stderr[:200],
            )
            return

        head_result = _run(
            ["git", "-C", agent_dir, "rev-parse", "HEAD"],
            timeout=timeout,
        )
        remote_result = _run(
            ["git", "-C", agent_dir, "rev-parse", remote_ref],
            timeout=timeout,
        )
        if head_result.returncode != 0 or remote_result.returncode != 0:
            logger.warning(
                "self_update[boundary=%s]: rev-parse failed, skipping update",
                boundary.value,
            )
            return

        local_sha = head_result.stdout.strip()
        remote_sha = remote_result.stdout.strip()

        if local_sha == remote_sha:
            logger.info(
                "self_update[boundary=%s]: already at %s (%s), nothing to do",
                boundary.value,
                remote_ref,
                local_sha[:12],
            )
            return

        logger.info(
            "self_update[boundary=%s]: behind %s (local=%s remote=%s), pulling and "
            "re-execing",
            boundary.value,
            remote_ref,
            local_sha[:12],
            remote_sha[:12],
        )

        # Hand off before the process image is replaced. The PRE_ACCEPT caller
        # rewinds its committed consumer offset here so the replacement process
        # re-reads the command that triggered this update rather than skipping
        # past it.
        if on_before_reexec is not None:
            on_before_reexec()

        pull_result = _run(
            ["git", "-C", agent_dir, "pull", "--ff-only", "origin", branch],
            timeout=timeout,
        )
        if pull_result.returncode != 0:
            logger.warning(
                "self_update[boundary=%s]: git pull failed (exit=%d), skipping re-exec: %s",
                boundary.value,
                pull_result.returncode,
                pull_result.stderr[:200],
            )
            return

        # Sync deps so new imports are available after re-exec.
        uv_result = _run(
            ["uv", "sync", "--project", agent_dir],
            timeout=120,
        )
        if uv_result.returncode != 0:
            logger.warning(
                "self_update[boundary=%s]: uv sync failed (exit=%d), proceeding with "
                "re-exec anyway: %s",
                boundary.value,
                uv_result.returncode,
                uv_result.stderr[:200],
            )

        mode = os.environ.get("DEPLOY_AGENT_MODE", "host")
        if mode == "container":
            # Let systemd/compose restart us from the freshly-pulled source.
            logger.info(
                "self_update[boundary=%s]: container mode — exiting with code 42 for "
                "supervisor respawn",
                boundary.value,
            )
            sys.exit(42)
        else:
            launcher = os.environ.get("DEPLOY_AGENT_LAUNCHER")
            if launcher:
                # OMN-18073: re-exec THROUGH the launcher, not the bare
                # interpreter. os.execv replaces the process image but inherits
                # the caller's environment verbatim, so an interpreter re-exec
                # carries the environment this process started with forward
                # forever. On 2026-09-09 that is exactly what happened: the
                # agent re-execed four times across the day, advancing its code
                # from b0b46c18 to c65d8a8b normally, while the mangled
                # ONEXBOT_OCC_PRIVATE_KEY it had inherited from systemd's
                # EnvironmentFile= at 01:38:54Z survived every one of them.
                # Only a systemd restart re-read the file. Going through
                # deploy/deploy-agent-launch.sh re-`source`s the operator env
                # store, so a repaired or rotated value is picked up at the next
                # job boundary instead of needing an operator restart.
                logger.info(
                    "self_update[boundary=%s]: host mode — re-execing through "
                    "launcher %s (re-reads the operator env store)",
                    boundary.value,
                    launcher,
                )
                os.execv(launcher, [launcher, *sys.argv[1:]])  # noqa: S606
            else:
                logger.info(
                    "self_update[boundary=%s]: host mode — re-execing process image",
                    boundary.value,
                )
                os.execv(sys.executable, [sys.executable] + sys.argv)  # noqa: S606

    def preflight(self, on_phase_update: PhaseCallback) -> None:
        on_phase_update(Phase.PREFLIGHT, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.PREFLIGHT]

        # Check git remote is reachable
        result = _run(
            ["git", "-C", REPO_DIR, "ls-remote", "--exit-code", "origin"],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git remote unreachable: {result.stderr}")

        # Check docker is available
        result = _run(["docker", "info"], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Docker unavailable: {result.stderr}")

        on_phase_update(Phase.PREFLIGHT, PhaseStatus.SUCCESS)

    def git_pull(self, git_ref: str, on_phase_update: PhaseCallback) -> str:
        on_phase_update(Phase.GIT, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.GIT]

        # Fetch with 1 retry
        result = _run(
            ["git", "-C", REPO_DIR, "fetch", "--all", "--prune"],
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("Git fetch failed, retrying in 5s...")
            time.sleep(5)
            result = _run(
                ["git", "-C", REPO_DIR, "fetch", "--all", "--prune"],
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git fetch failed: {result.stderr}")

        # OMN-18122: refuse a branch alias that would move the clone BACKWARDS
        # before anything touches the tree. The fetch above is what makes the
        # comparison meaningful -- both refs are now current -- and the reset
        # below is the mutation being fenced.
        assert_ref_not_stale_branch(
            self._ref_lineage_facts(git_ref, timeout=timeout),
        )

        # Reset to ref
        result = _run(
            ["git", "-C", REPO_DIR, "reset", "--hard", git_ref],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git reset --hard {git_ref} failed: {result.stderr}")

        # Get SHA
        result = _run(
            ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
            timeout=timeout,
        )
        sha = result.stdout.strip()

        on_phase_update(Phase.GIT, PhaseStatus.SUCCESS)
        return sha

    def _ref_lineage_facts(self, git_ref: str, *, timeout: int) -> ModelRefLineageFacts:
        """Ask git where ``git_ref`` sits relative to this lane's tracking branch.

        Gathers the facts :func:`assert_ref_not_stale_branch` decides on. Kept
        separate from that decision so the rule itself is a pure function with
        no repository on disk (OMN-18122).

        ``git rev-parse --symbolic-full-name`` is what distinguishes a branch
        alias from a commit SHA: it prints a full ref name for a branch and
        nothing at all for a SHA. That is a git-native answer rather than a
        regex over the string, so a tag, a ``HEAD``, and an abbreviated SHA are
        all classified by what they actually resolve to.
        """
        tracking_ref = load_tracking_remote_ref_from_env()

        symbolic = _run(
            ["git", "-C", REPO_DIR, "rev-parse", "--symbolic-full-name", git_ref],
            timeout=timeout,
        )
        is_branch_reference = bool(symbolic.stdout.strip())

        behind = 0
        ahead = 0
        if is_branch_reference and git_ref != tracking_ref:
            counts = _run(
                [
                    "git",
                    "-C",
                    REPO_DIR,
                    "rev-list",
                    "--left-right",
                    "--count",
                    f"{tracking_ref}...{git_ref}",
                ],
                timeout=timeout,
            )
            # Left column: commits on the tracking ref only (the requested ref
            # is BEHIND by these). Right column: commits on the requested ref
            # only (it is AHEAD by these).
            fields = counts.stdout.split()
            if counts.returncode == 0 and len(fields) == 2:
                behind, ahead = int(fields[0]), int(fields[1])
            else:
                # A comparison that could not be made is not evidence that the
                # ref is fine. Refusing here would block every deploy on a
                # transient git failure, so the fence is skipped and the reason
                # is logged loudly rather than swallowed.
                logger.warning(
                    "OMN-18122 ref fence could not compare %s against %s "
                    "(rc=%s, stdout=%r); the fence did NOT run for this deploy",
                    git_ref,
                    tracking_ref,
                    counts.returncode,
                    counts.stdout.strip(),
                )
                is_branch_reference = False

        return ModelRefLineageFacts(
            requested_ref=git_ref,
            tracking_ref=tracking_ref,
            is_branch_reference=is_branch_reference,
            commits_ahead_of_tracking=ahead,
            commits_behind_tracking=behind,
        )

    def _preflight_required_compose_env(
        self,
        *,
        lane: EnumRuntimeLane,
        compose_files: list[str],
        timeout: int,
    ) -> None:
        """Fail with EVERY unset required compose variable named, not just the first.

        OMN-17530. ``docker compose config`` stops at the first unset
        ``${VAR:?}``, so a host missing N of them produces N failed deploys that
        each name one variable. This runs the repo's stdlib-only preflight over
        the same compose files and the same environment the validation is about
        to use, and raises once with the complete list.

        There is no soft-fail branch and no "continue if the script is
        missing": a preflight that can be skipped is the failure mode this
        closes, so a missing or unrunnable script surfaces as a non-zero exit
        and raises like any other refusal. The script is stdlib-only and runs
        under the interpreter running this agent, so a missing project venv
        cannot be the reason the list goes unseen.
        """
        script = f"{REPO_DIR}/scripts/preflight_required_compose_env.py"
        cmd = [
            sys.executable,
            script,
            "--lane",
            lane.value,
            "--runtime-policy-env",
            f"{REPO_DIR}/docker/runtime-policy.env",
        ]
        for compose_file in compose_files:
            cmd.extend(["--compose-file", compose_file])
        result = _run(cmd, timeout=timeout, cwd=REPO_DIR, env=_compose_env())
        if result.returncode != 0:
            raise RuntimeError(
                "REQUIRED_COMPOSE_ENV_MISSING for lane "
                f"{lane.value} -- compose validation was not attempted. "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("required-compose-env preflight: %s", result.stdout.strip())

    def compose_gen(
        self,
        bundles: list[str],
        on_phase_update: PhaseCallback,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> None:
        """Render the catalog to the build artifact and validate both composes.

        Runs ``uv run python -m omnibase_infra.docker.catalog.cli generate
        <bundles> --output <COMPOSE_GEN_OUTPUT_FILE>``, then validates the render
        and the lane's real compose stack.

        OMN-17291 -- WHY THE OUTPUT IS NOT THE TRACKED FILE. Until this ticket
        the ``--output`` here was ``COMPOSE_FILE``, so every deploy overwrote a
        TRACKED file in place (OMN-8430). Two consequences, both measured on the
        lab deploy-source clone:

        1. The clone was permanently dirty. A ``git reset --hard`` in git_pull()
           was undone six seconds later by this method, every deploy, with a
           ~2900-line uncommitted delta that read like a lane edit and was not.
        2. The two files are not the same stack. The render carries 12 services
           the tracked file does not -- the ``runtime`` bundle pulls in
           runtime-integrations (docker/catalog/bundles.yaml) -- and 31 required
           ``${VAR:?}`` names against the tracked file's 50, nine of which
           ``scripts/deploy-runtime.sh`` and
           ``scripts/runtime_build/refresh_stability_lane.sh`` cannot supply. So
           once the render had replaced the tracked file, the sanctioned deploy
           scripts failed compose validation and auto-restored -- the dev and
           stability lanes, the proof surface for beta work, were re-broken by
           every run of this agent.

        The catalog CLI's own declared default output is
        ``docker/docker-compose.generated.yml`` and .gitignore already ignores
        it: the ``--output`` override was the deviation, not the design. The
        tracked file now has exactly one writer (git), and this agent's ``up``
        path uses the same base + lane overlay as every other deploy path.

        OMN-8430's purpose is kept, not dropped: the render is still produced on
        every deploy and is still validated, so a catalog change that cannot
        render fails the deploy here. What it no longer does is silently swap
        the running stack for one no other deploy path can reproduce.

        Generation is non-fatal if the catalog CLI is unavailable — a missing
        virtualenv must not block a deploy. A render that PARSES INVALID is
        fatal, as the lane validation already was (OMN-12865).
        """
        on_phase_update(Phase.COMPOSE_GEN, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.COMPOSE_GEN]

        selected = bundles if bundles else ["core", "runtime"]
        cmd = [
            "uv",
            "run",
            "--project",
            REPO_DIR,
            "python",
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            *selected,
            "--output",
            COMPOSE_GEN_OUTPUT_FILE,
        ]

        result = _run(
            cmd,
            timeout=timeout,
            cwd=REPO_DIR,
            env=_env_with_repo_pythonpath(_compose_env()),
        )
        if result.returncode != 0:
            logger.warning(
                "compose_gen returned non-zero (exit=%d) — continuing; the tracked "
                "compose base is unaffected either way. stderr: %s",
                result.returncode,
                result.stderr[:500],
            )
        else:
            logger.info("compose_gen complete: %s", result.stdout.strip())
            profile = "runtime" if "runtime" in selected else "core"

            # Validate the render itself. This is what OMN-8430 bought and what
            # OMN-17291 keeps: a catalog change that cannot render stops the
            # deploy here, without that render ever touching the tracked file.
            #
            # --no-interpolate is load-bearing, not a weakening. The render
            # requires nine ${VAR:?} names that resolve from no committed
            # source, and five of them are absent from this agent's own process
            # environment (measured 2026-09-08 on the lab host, /proc/<pid>/environ
            # of the live agent: ONEX_TENANT_DB_URL among them). An interpolating
            # check here would therefore fail every deploy on a value nobody can
            # supply. Skipping interpolation still validates the render's schema
            # and structure -- proven by control: the same flags reject a file
            # carrying an unknown service key with exit 1. The render's env
            # contract is covered separately and exactly, by
            # docker/generated-compose-required-env.manifest.txt and its parity
            # test, which is where a change to that contract now surfaces.
            render_cmd = [
                "docker",
                "compose",
                "-f",
                COMPOSE_GEN_OUTPUT_FILE,
                "--profile",
                profile,
                "config",
                "--quiet",
                "--no-interpolate",
            ]
            render_result = _run(
                render_cmd,
                timeout=timeout,
                cwd=REPO_DIR,
                env=_compose_env(),
            )
            if render_result.returncode != 0:
                raise RuntimeError(
                    "compose_gen produced an invalid catalog render at "
                    f"{COMPOSE_GEN_OUTPUT_FILE}: "
                    f"{render_result.stderr.strip() or render_result.stdout.strip()}"
                )

            # Validate the stack this deploy will actually bring up: the tracked
            # base plus the lane overlay (OMN-12865).
            config = lane_config_for(lane)

            # OMN-17530 -- report the WHOLE missing-variable set first.
            #
            # The validation below is `docker compose config`, which reports the
            # FIRST unset ${VAR:?} it reaches and stops. On 2026-09-08 two deploy
            # commands died here ~14 minutes apart on two DIFFERENT variables
            # added by the same PR (ONEX_API_IMAGE, then ONEX_CLOUD_MIGRATE_IMAGE);
            # a mechanical enumeration afterwards put the real count at ten. Each
            # one cost a whole command and a whole lane window to learn one name.
            #
            # This preflight parses every ${VAR:?} out of the same compose files
            # this validation will load, checks them against the same env this
            # validation will run under, and fails with all of them named at
            # once. It reads names only, never values.
            self._preflight_required_compose_env(
                lane=lane,
                compose_files=list(config.compose_files),
                timeout=timeout,
            )

            validate_cmd = [
                "docker",
                "compose",
                *_compose_file_args(lane),
                "-p",
                config.compose_project,
                "--profile",
                profile,
                "config",
                "--quiet",
            ]
            validate_result = _run(
                validate_cmd,
                timeout=timeout,
                cwd=REPO_DIR,
                env=_compose_env(),
            )
            if validate_result.returncode != 0:
                raise RuntimeError(
                    "compose_gen produced invalid lane compose for "
                    f"{lane.value}: "
                    f"{validate_result.stderr.strip() or validate_result.stdout.strip()}"
                )

        on_phase_update(Phase.COMPOSE_GEN, PhaseStatus.SUCCESS)

    def seed_infisical(self, on_phase_update: PhaseCallback) -> None:
        """Seed Infisical with required secrets before runtime containers start.

        Non-fatal: if Infisical is unreachable or seed fails, logs a warning and
        continues — the runtime containers will fall back to env-var resolution.
        This prevents a broken Infisical from blocking deploys entirely.
        """
        from deploy_agent.events import Phase, PhaseStatus

        on_phase_update(Phase.SEED, PhaseStatus.IN_PROGRESS)
        timeout = 120  # 2 minutes max for seed

        seed_script = f"{REPO_DIR}/scripts/seed-infisical.py"
        venv_python = f"{REPO_DIR}/.venv/bin/python"

        import shutil

        # Resolve Python: prefer venv, fall back to uv, then system python3
        python_bin = (
            venv_python
            if Path(venv_python).is_file()
            else ((shutil.which("uv") and "uv run python") or "python3")
        )
        if python_bin == venv_python:
            cmd = [
                python_bin,
                seed_script,
                "--contracts-dir",
                f"{REPO_DIR}/src/omnibase_infra/nodes",
                "--create-missing-keys",
                "--execute",
            ]
        else:
            cmd = [
                "uv",
                "run",
                "--project",
                REPO_DIR,
                "python",
                seed_script,
                "--contracts-dir",
                f"{REPO_DIR}/src/omnibase_infra/nodes",
                "--create-missing-keys",
                "--execute",
            ]

        result = _run(
            cmd,
            timeout=timeout,
            env={**_compose_env(), "PYTHONPATH": f"{REPO_DIR}/src"},
        )
        if result.returncode != 0:
            logger.warning(
                "Infisical seed returned non-zero (exit=%d). Runtime will fall back to env-var resolution. stderr: %s",
                result.returncode,
                result.stderr[:500],
            )
        else:
            logger.info("Infisical seed complete. stdout: %s", result.stdout[-500:])

        # Mark success regardless — seed failure is non-fatal
        on_phase_update(Phase.SEED, PhaseStatus.SUCCESS)

    def rebuild_scope(
        self,
        scope: Scope,
        services: list[str],
        on_phase_update: PhaseCallback,
        *,
        git_sha: str = "",
        git_ref: str = "",
        build_source: BuildSource | str = BuildSource.RELEASE,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        image_digest: str | None = None,
    ) -> list[str]:
        # OMN-16442: no self-update here. This method runs inside an accepted
        # job, after preflight/git/compose_gen/seed; a re-exec from this point
        # aborts the deploy in flight and the replacement process publishes the
        # command as failed after recovering it as a crashed job (live: command
        # 8d0c861a-f91e-4ca2-954e-a073759dd39d, 2026-09-08T16:01Z). Self-update
        # is a job-boundary concern and lives at the two boundaries declared in
        # EnumSelfUpdateBoundary. The parameter is removed rather than defaulted
        # off so this path cannot reach self_update at all.
        phase = Phase.CORE if scope == Scope.CORE else Phase.RUNTIME
        # OMN-18057: residue is per-rebuild, not per-process.
        self.reset_deploy_observations()

        # prod deploys the stability-proven digest — it pulls the pinned image
        # and never rebuilds from a ref (the digest is the authority).
        if lane == EnumRuntimeLane.PROD:
            if not image_digest:
                raise ProdStabilityDigestMissingError(
                    "prod rebuild_scope requires a pinned image_digest"
                )
            self._pull_pinned_image(image_digest, lane)
            if scope == Scope.FULL:
                self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update, lane=lane)
                runtime_services = services_for_scope(Scope.RUNTIME)
                self._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    [],
                    on_phase_update,
                    lane=lane,
                    extra_env=self._resolve_prod_image_env(
                        image_digest, runtime_services
                    ),
                )
                return services_for_scope(Scope.FULL)
            target_services = services if services else services_for_scope(scope)
            prod_env = (
                {}
                if scope == Scope.CORE
                else self._resolve_prod_image_env(image_digest, target_services)
            )
            self._compose_up(
                phase, scope, services, on_phase_update, lane=lane, extra_env=prod_env
            )
            return target_services

        if scope == Scope.FULL:
            # Build images first (both scopes), then bring them up.
            # _compose_build passes --build-arg GIT_SHA so Docker invalidates
            # the COPY src/ layer even when the file-system mtime is cached.
            self._compose_build(
                Scope.CORE,
                git_sha,
                on_phase_update,
                build_source=build_source,
                runtime_lane=lane,
                git_ref=git_ref,
            )
            self._compose_build(
                Scope.RUNTIME,
                git_sha,
                on_phase_update,
                build_source=build_source,
                runtime_lane=lane,
                git_ref=git_ref,
            )
            self._build_dev_lane_only_services(
                git_sha,
                on_phase_update,
                build_source=build_source,
                lane=lane,
                git_ref=git_ref,
            )
            self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update, lane=lane)
            self._compose_up(
                Phase.RUNTIME, Scope.RUNTIME, [], on_phase_update, lane=lane
            )
            return services_for_scope(Scope.FULL, lane=lane)

        self._compose_build(
            scope,
            git_sha,
            on_phase_update,
            build_source=build_source,
            runtime_lane=lane,
            git_ref=git_ref,
        )
        if scope == Scope.RUNTIME:
            self._build_dev_lane_only_services(
                git_sha,
                on_phase_update,
                build_source=build_source,
                lane=lane,
                git_ref=git_ref,
            )
        self._compose_up(phase, scope, services, on_phase_update, lane=lane)
        return services if services else services_for_scope(scope, lane=lane)

    def _build_dev_lane_only_services(
        self,
        git_sha: str,
        on_phase_update: PhaseCallback,
        *,
        build_source: BuildSource | str,
        lane: EnumRuntimeLane,
        git_ref: str,
    ) -> None:
        """Build the DEV lane's own runtime services (OMN-18108). No-op elsewhere.

        These are declared only in ``docker/docker-compose.dev-lane.yml``, which
        the lane-agnostic build above never passes, so without this command the
        up phase recreates them from whatever image they already carry and
        reports success. That is exactly what left eight services ~35 commits
        behind on the .201 dev lane while the deploy agent's own terminal event
        said the deploy succeeded.

        Additive and explicit: a SECOND build naming the services, rather than
        widening the first one's compose files, so prod and stability-test issue
        a byte-identical command to before. The tag-referenced services are
        excluded by ``DEV_LANE_ONLY_BUILDABLE_SERVICES`` -- ``docker compose
        build`` has nothing to do for a service that carries only an ``image:``.
        """
        if lane != EnumRuntimeLane.DEV or not DEV_LANE_ONLY_BUILDABLE_SERVICES:
            return
        logger.info(
            "_build_dev_lane_only_services: building %d dev-lane-only services "
            "the base compose file does not declare: %s",
            len(DEV_LANE_ONLY_BUILDABLE_SERVICES),
            ", ".join(DEV_LANE_ONLY_BUILDABLE_SERVICES),
        )
        self._compose_build(
            Scope.RUNTIME,
            git_sha,
            on_phase_update,
            build_source=build_source,
            runtime_lane=lane,
            git_ref=git_ref,
            compose_files=lane_config_for(lane).compose_files,
            services=DEV_LANE_ONLY_BUILDABLE_SERVICES,
            stage_workspace=False,
        )

    def _pull_pinned_image(self, image_digest: str, lane: EnumRuntimeLane) -> None:
        """Resolve the exact stability-proven image digest for a prod deploy.

        Local-presence-first (OMN-15181 round 2): the granted ``image_digest``
        is a bare image ID (``sha256:...``), not a registry reference, and the
        stability-proven artifact is normally a ``docker compose build``
        output that was never pushed to any registry — a literal
        ``docker pull <sha256:...>`` is an invalid reference on its own and
        always fails (live-reproduced 2026-07-26 on omninode-pc: "pull access
        denied for sha256, repository does not exist").

        Resolution order:

        1. Local presence — ``docker image inspect <image_digest>``. If the
           image already exists on this host, use it as-is; no pull, no
           registry required. This is the normal case.
        2. Registry fallback — only attempted when
           ``DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF`` names a real
           ``repo:tag``/``repo@sha256:...`` reference; pulls that reference,
           then re-inspects locally to confirm the pulled image actually
           matches ``image_digest``.
        3. Fail loud — neither locally present nor a registry reference
           configured: raise ``RuntimeError`` naming the gap. Never silently
           proceeds with a possibly-wrong image.
        """
        inspect_result = _run(
            ["docker", "image", "inspect", image_digest],
            timeout=PHASE_TIMEOUTS[Phase.RUNTIME],
        )
        if inspect_result.returncode == 0:
            logger.info(
                "_pull_pinned_image: digest %s already present locally for lane "
                "%s (compose-built artifact, no registry pull needed)",
                image_digest,
                lane.value,
            )
            return

        registry_ref = os.environ.get(
            "DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF", ""
        ).strip()
        if not registry_ref:
            raise RuntimeError(
                f"pinned image digest {image_digest} is not present locally for "
                f"lane {lane.value}, and no registry reference is configured "
                "(DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF) to pull it from; a "
                "docker-compose-build-only artifact must already exist locally "
                f"on this host: {inspect_result.stderr.strip() or inspect_result.stdout.strip()}"
            )

        pull_result = _run(
            ["docker", "pull", registry_ref],
            timeout=PHASE_TIMEOUTS[Phase.RUNTIME],
        )
        if pull_result.returncode != 0:
            raise RuntimeError(
                f"docker pull {registry_ref} failed for lane {lane.value}: "
                f"{pull_result.stderr.strip() or pull_result.stdout.strip()}"
            )

        verify_result = _run(
            ["docker", "image", "inspect", image_digest],
            timeout=PHASE_TIMEOUTS[Phase.RUNTIME],
        )
        if verify_result.returncode != 0:
            raise RuntimeError(
                f"pulled {registry_ref} for lane {lane.value} but the resulting "
                f"local image does not match pinned digest {image_digest}: "
                f"{verify_result.stderr.strip() or verify_result.stdout.strip()}"
            )
        logger.info(
            "_pull_pinned_image: pulled %s and verified digest %s for lane %s",
            registry_ref,
            image_digest,
            lane.value,
        )

    def _resolve_local_image_reference(self, image_digest: str) -> str:
        """Return a locally-present ``repository:tag`` reference for a pinned digest.

        OMN-15181 round 3 (Finding 9): the granted ``image_digest`` is a bare
        image id (``sha256:...``), which is never a valid value for a compose
        ``image:`` field or a ``docker pull``/``docker run`` reference on its
        own. ``_pull_pinned_image`` only proves the id is present somewhere in
        the local docker store (normally under the stability-test tag, since
        that is where the artifact was built); this method resolves an actual
        ``RepoTags`` entry for that id so the caller can repoint the prod
        compose ``image:`` field at it declaratively — no ``docker tag``/retag
        command, no mutation of the local docker store.

        Raises ``RuntimeError`` (fail loud) when the id cannot be inspected or
        carries no usable repository:tag reference (e.g. a dangling image)
        rather than silently proceeding with an unresolvable pin.
        """
        result = _run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                '{{join .RepoTags ","}}',
                image_digest,
            ],
            timeout=PHASE_TIMEOUTS[Phase.RUNTIME],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"could not resolve a local repository:tag reference for pinned "
                f"digest {image_digest}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        tags = [
            tag
            for tag in result.stdout.strip().split(",")
            if tag and tag != "<none>:<none>"
        ]
        if not tags:
            raise RuntimeError(
                f"pinned digest {image_digest} is present locally but carries no "
                "usable repository:tag reference (RepoTags empty/dangling) — "
                "cannot repoint the prod compose image field at it"
            )
        return str(tags[0])

    def _resolve_prod_image_env(
        self, image_digest: str, services: list[str]
    ) -> dict[str, str]:
        """Return the prod compose env-var overrides for the given services.

        Only services present in ``PROD_IMAGE_ENV_VAR_FOR_SERVICE`` (the prod
        runtime services with a parameterized ``image:`` field) get an
        override; other services (e.g. core infra) are untouched. Returns an
        empty dict — no pinned-image resolution attempted — when no target
        service needs one, so a core-only scope never requires the digest to
        carry a usable local tag.
        """
        targets = [s for s in services if s in PROD_IMAGE_ENV_VAR_FOR_SERVICE]
        if not targets:
            return {}
        reference = self._resolve_local_image_reference(image_digest)
        return {
            PROD_IMAGE_ENV_VAR_FOR_SERVICE[service]: reference for service in targets
        }

    def resolve_stability_ready_digest(
        self, service: str = "omninode-runtime"
    ) -> str | None:
        """Return the image id currently serving ``service`` in stability-test.

        This is the boundary-level source of truth for "stability-proven"
        consumed by ``assert_prod_request_has_stability_digest``: a digest
        counts as READY exactly when it is the image id currently running in
        stability-test for the REQUESTED service — resolved per-service
        (OMN-15181 round 4, Finding 11) via the single shared
        ``_health_target_container`` mapping, no second hardcoded map.
        Defaults to ``"omninode-runtime"`` (the pre-round-4 behavior) when no
        service is given. Returns ``None`` (fail-closed via the caller) when
        the stability-test container for that service cannot be inspected.
        """
        container = _health_target_container(EnumRuntimeLane.STABILITY_TEST, service)
        return _container_image_id(container)

    def verify_running_image_digest(
        self,
        *,
        lane: EnumRuntimeLane,
        expected_digest: str,
        service: str = "omninode-runtime",
    ) -> None:
        """Verify the running ``service`` container's image id == requested.

        FAILS CLOSED (raises ``DigestMismatchError``) on any mismatch. Must run
        before health checks so a lane is never marked healthy while serving an
        artifact that does not match the pinned digest. Resolved per-service
        (OMN-15181 round 4, Finding 11) so a targeted runtime-effects deploy is
        verified against its own container, not the runtime container by
        default (``service`` defaults to ``"omninode-runtime"``, the
        pre-round-4 behavior, when the caller does not name one).
        """
        container = _health_target_container(lane, service)
        observed = _container_image_id(container)
        if observed is None:
            raise DigestMismatchError(
                f"could not inspect running image id for {container} "
                f"(lane {lane.value}, service {service!r}); failing closed"
            )
        if observed != expected_digest:
            raise DigestMismatchError(
                f"running container {container} image id {observed!r} "
                f"does not match requested digest {expected_digest!r} "
                f"(lane {lane.value}, service {service!r}); failing closed"
            )
        logger.info(
            "verify_running_image_digest: %s matches requested digest %s "
            "(lane %s, service %s)",
            container,
            expected_digest,
            lane.value,
            service,
        )

    def deploy_and_verify(
        self,
        *,
        lane: EnumRuntimeLane,
        expected_digest: str,
        on_phase_update: PhaseCallback,
        service: str = "omninode-runtime",
    ) -> list[ModelHealthCheck]:
        """Verify the running digest, then run health checks.

        Digest verification runs first and fails closed: a mismatch aborts
        before any health check, so a lane serving the wrong artifact can never
        be reported healthy. ``service`` (OMN-15181 round 4, Finding 11)
        selects which container is checked — a targeted runtime-effects
        deploy must verify the effects container, not the runtime container
        by default.

        OMN-15181: a digest-mismatch abort must mark ``Phase.VERIFICATION``
        FAILED (not leave it untouched/absent from ``phase_results``) so the
        published completion event carries a truthful status — a swallowed
        verification phase with all other phases SUCCESS would otherwise let
        ``ModelRebuildCompleted.status`` compute "success" for a deploy that
        actually failed verification (job.errors also carries the real error,
        but phase_results must not contradict it). No rollback/recreate is
        attempted here or anywhere in this module on a verify failure — the
        job is reported failed, never silently retried or half-recreated.
        """
        on_phase_update(Phase.VERIFICATION, PhaseStatus.IN_PROGRESS)
        try:
            self.verify_running_image_digest(
                lane=lane, expected_digest=expected_digest, service=service
            )
        except DigestMismatchError:
            on_phase_update(Phase.VERIFICATION, PhaseStatus.FAILED)
            raise
        return self.verify(on_phase_update=on_phase_update, lane=lane)

    @staticmethod
    def _resolve_plugin_ref(repo_dir: str, *, fallback: str) -> str:
        """Return the HEAD SHA of a plugin repo for uv cache busting (OMN-10728).

        BuildKit's uv cache mount is keyed on the install URL, not the resolved
        git HEAD. Passing a bare branch always hits the stale cache entry.
        Passing the full SHA forces a cache miss and a fresh fetch every time
        the branch advances.

        ``fallback`` is the branch name to use when the sibling clone is absent
        or ``git rev-parse`` fails, so manual docker builds without omni_home
        still work. It is supplied by the caller from the declared tracking ref
        (OMN-16442) rather than hardcoded here: the old literal ``"main"``
        resolved a release-synced branch on repos whose integration branch is
        ``dev``, which is precisely the class of stale-ref defect the operator
        ruling names.
        """
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning(
            "_resolve_plugin_ref: git rev-parse failed for %s (exit=%d): %s — falling back to %s",
            repo_dir,
            result.returncode,
            result.stderr[:200],
            fallback,
        )
        return fallback

    @staticmethod
    def _stage_workspace(
        repo_dir: str,
        omni_home: str,
        deploy_ref: str = "",
        sibling_fallback_ref: str = "",
    ) -> dict[str, str]:
        """Stage sibling repos into the Docker build context for workspace mode.

        Runs docker/runtime_build/stage_workspace.sh from the repo root so that
        workspace/sibling-repos/ is populated before `docker compose build`.
        Raises RuntimeError on failure.

        OMN-16442: ``deploy_ref`` is the accepted command's own ``git_ref`` and
        is exported as ``DEPLOY_REF`` for the staging script. The script's
        OMN-17291 guard refuses to stage the AMBIENT host tree when
        ``DEPLOY_REF`` is unset, and it was refusing correctly — the caller was
        wrong. The command envelope carries the pin (``git_ref=origin/dev``,
        resolved from ``DEPLOY_AGENT_TRACKING_REF``) and ``self_update`` had
        already used it one step earlier, but it was dropped between the
        consumer and the staging step, so the sibling build was asserted
        against nothing. Measured live 2026-09-08 on command a5635af0:
        ``Workspace staging failed (exit=5): ERROR: DEPLOY_REF unset``.

        The guard STAYS. This passes the pin the operator supplied; it does not
        weaken, skip, or opt out of the assertion. An empty ``deploy_ref`` is
        deliberately NOT substituted with a fallback ref — nothing is exported
        and the script refuses in its own words, which is the correct outcome
        for a caller that has no pin to offer.

        OMN-17135: ``deploy_ref`` pins ONE repository. The CI path
        (``runtime-rebuild-trigger.yml`` → ``trigger_rebuild_on_merge.py``)
        constrains the published ``git_ref`` to a lowercase hex commit SHA of
        **omnibase_infra**, and that commit exists in no sibling — so RT-1
        aborted on the first one it tried (``ERROR: omnibase_core: cannot
        resolve ref '<infra sha>'``, exit 4) and every CI-triggered agent
        rebuild failed by construction, 13 seconds after acceptance (job
        ``a5b200d5``, 2026-09-09T21:02Z). ``sibling_fallback_ref`` is the ref
        each sibling resolves for itself instead: the declared tracking head.
        Manual requests carrying ``git_ref=origin/dev`` are unaffected, because
        the fallback engages only where the primary names no commit.

        Returns repo → the commit SHA RT-1 actually resolved for that sibling,
        read back from the expected-refs manifest this call pins the location of.
        The path is named explicitly, under the same ``~/.omnibase/state``
        convention the script's own default uses but with a distinct
        agent-and-pid basename, so two concurrent callers cannot read each
        other's manifest and neither can collide with the script's default.
        """
        script = Path(repo_dir) / "scripts" / "runtime_build" / "stage_workspace.sh"
        if not script.exists():
            raise RuntimeError(
                f"workspace staging script not found: {script}. "
                "Cannot proceed with BUILD_SOURCE=workspace."
            )
        staging_env = {**os.environ, "OMNI_HOME": omni_home}
        refs_out = (
            Path.home()
            / ".omnibase"
            / "state"
            / "deploy_source_refs"
            / f"deploy-agent-{os.getpid()}.json"
        )
        refs_out.parent.mkdir(parents=True, exist_ok=True)
        refs_out.unlink(missing_ok=True)
        staging_env["DEPLOY_SOURCE_REFS_OUT"] = str(refs_out)
        if deploy_ref:
            staging_env["DEPLOY_REF"] = deploy_ref
            logger.info(
                "_stage_workspace: staging siblings against DEPLOY_REF=%s "
                "(the accepted command's git_ref)",
                deploy_ref,
            )
        if sibling_fallback_ref:
            staging_env["DEPLOY_SIBLING_FALLBACK_REF"] = sibling_fallback_ref
            logger.info(
                "_stage_workspace: siblings fall back to %s where the requested "
                "ref names no commit in them (OMN-17135)",
                sibling_fallback_ref,
            )
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_dir,
            env=staging_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Workspace staging failed (exit={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("_stage_workspace: %s", result.stdout.strip())
        return DeployExecutor._read_resolved_sibling_refs(refs_out)

    @staticmethod
    def _read_resolved_sibling_refs(refs_out: Path) -> dict[str, str]:
        """Read repo → resolved SHA out of the RT-1 expected-refs manifest.

        Evidence, not a gate: the build already succeeded and its own assertion
        already compared every vendored SHA against this same manifest. An
        unreadable manifest is logged and yields an empty map rather than
        failing a deploy that passed — the fail-closed decision belongs to RT-1,
        which has already made it (OMN-14438).
        """
        try:
            manifest = json.loads(refs_out.read_text(encoding="utf-8"))
            repos = manifest["repos"]
            return {
                str(repo): str(row["expected_sha"])
                for repo, row in repos.items()
                if row.get("expected_sha")
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                "_stage_workspace: could not read resolved sibling refs from %s: %s",
                refs_out,
                exc,
            )
            return {}

    def _compose_build(
        self,
        scope: Scope,
        git_sha: str,
        on_phase_update: PhaseCallback,
        *,
        build_source: BuildSource | str = BuildSource.RELEASE,
        expected_build_source: BuildSource | str | None = None,
        runtime_lane: EnumRuntimeLane = EnumRuntimeLane.PROD,
        git_ref: str = "",
        compose_files: tuple[str, ...] = (COMPOSE_FILE,),
        services: tuple[str, ...] = (),
        stage_workspace: bool = True,
    ) -> None:
        """Build images with --build-arg GIT_SHA to bust the COPY src/ layer cache.

        Without this arg, Docker serves a cached layer even after git pull, so
        the running container silently ships pre-pull code (root cause: PR #1231).

        Also passes OMNIBASE_COMPAT_REF and OMNIMARKET_REF as full commit SHAs
        so the uv cache mount (keyed on URL) misses and fetches fresh code every
        time main advances (OMN-10728 / OMN-11542). ONEX_CHANGE_CONTROL_REF was
        a third such ref until OMN-16296 removed onex_change_control from the
        runtime image; the Dockerfile ARG it fed no longer exists.

        For BUILD_SOURCE=workspace, stages sibling repos into the build context
        via stage_workspace.sh before invoking docker compose build (OMN-9470).

        OMN-18108: ``compose_files`` / ``services`` / ``stage_workspace`` exist
        so a DEV deploy can issue a SECOND, additive build for the dev-lane-only
        services, which are declared in the lane overlay this command otherwise
        never passes. The defaults reproduce the previous command exactly, so
        the prod and stability-test build is byte-unchanged. ``stage_workspace``
        is False on that second call: staging is the expensive, tree-mutating
        half and it has already run for the same refs in the same deploy --
        re-running it would re-vendor identical siblings and reset
        ``sibling_source_refs`` the terminal event reports.
        """
        profile = "core" if scope == Scope.CORE else "runtime"
        # OMN-18072: derived from the build model, never a bare constant. The
        # flat PHASE_TIMEOUTS entry that used to bound this killed two
        # consecutive sanctioned dev rebuilds at exactly 300s, the second with a
        # warm BuildKit cache. It still bounds the pinned-digest pull and the
        # migration preflight, which are not builds and are not affected.
        budget = runtime_image_build_budget(profile, compose_files)
        timeout = budget.timeout_seconds
        logger.info("Phase %s image-build ceiling %s", profile, budget.describe())

        # Validate build-source selector agreement before any side effects.
        # This surfaces selector mismatch and missing OMNI_HOME before staging.
        validated_args = _build_source_build_args(
            build_source,
            expected_build_source=expected_build_source,
        )

        selected_source = _coerce_build_source(build_source, layer="deploy-agent")
        omni_home = os.environ.get("OMNI_HOME", "").strip()

        # OMN-12626 (R1): release-mode builds produce the digest that is later
        # pinned/promoted to prod. Refuse to build one from a dirty or
        # non-promoted (dev-only) source tree before any docker side effects.
        # OMN-16442: scoped to the lanes whose artifacts can reach prod. The
        # default is PROD, so an undeclared lane still runs the gate.
        assert_release_build_promoted(selected_source, runtime_lane=runtime_lane)

        # OMN-16442: the sibling-repo fallback branch is the declared tracking
        # ref, not a literal. It used to be "dev" for omnimarket and "main" for
        # omnibase_compat — an asymmetry with no stated reason, on two repos
        # that both integrate on `dev`.
        sibling_fallback = load_tracking_ref_from_env()

        if selected_source == BuildSource.WORKSPACE and stage_workspace:
            if not omni_home:
                raise RuntimeError(
                    "BUILD_SOURCE=workspace requires OMNI_HOME before build"
                )
            # OMN-16442/OMN-17291: the accepted command's git_ref is the pin the
            # staging script asserts the INFRA clone against.
            # OMN-17135: it is a pin on that ONE repository. The CI path
            # publishes an omnibase_infra commit SHA, which names no commit in
            # omnibase_core / omnibase_compat / omnimarket, so passing it as
            # every sibling's ref made RT-1 abort on the first one and took
            # every CI-triggered rebuild red. Each sibling resolves the declared
            # tracking head instead, and the SHA it lands on is recorded.
            self.sibling_source_refs = self._stage_workspace(
                REPO_DIR,
                omni_home,
                git_ref,
                load_tracking_remote_ref_from_env(),
            )
            if self.sibling_source_refs:
                logger.info(
                    "_compose_build: sibling source refs %s",
                    {repo: sha[:12] for repo, sha in self.sibling_source_refs.items()},
                )
        omnimarket_ref = (
            self._resolve_plugin_ref(
                f"{omni_home}/omnimarket", fallback=sibling_fallback
            )
            if omni_home
            else sibling_fallback
        )
        compat_ref = (
            self._resolve_plugin_ref(
                f"{omni_home}/omnibase_compat", fallback=sibling_fallback
            )
            if omni_home
            else sibling_fallback
        )
        logger.info(
            "_compose_build: BUILD_SOURCE=%s OMNIBASE_COMPAT_REF=%s OMNIMARKET_REF=%s",
            selected_source.value,
            compat_ref[:12],
            omnimarket_ref[:12],
        )

        import datetime

        build_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        runtime_version = _runtime_version_from_pyproject()

        compose_file_args: list[str] = []
        for compose_file in compose_files:
            compose_file_args.extend(["-f", compose_file])
        cmd = [
            "docker",
            "compose",
            *compose_file_args,
            "-p",
            COMPOSE_PROJECT,
            "--profile",
            profile,
            "build",
            "--build-arg",
            f"GIT_SHA={git_sha}",
            "--build-arg",
            f"VCS_REF={git_sha}",
            "--build-arg",
            f"BUILD_DATE={build_date}",
            "--build-arg",
            f"RUNTIME_VERSION={runtime_version}",
            "--build-arg",
            f"OMNIBASE_COMPAT_REF={compat_ref}",
            "--build-arg",
            f"OMNIMARKET_REF={omnimarket_ref}",
        ]
        cmd.extend(validated_args)
        # Service names are positional and go last (`docker compose build
        # [OPTIONS] [SERVICE...]`). Empty by default, which is the lane-agnostic
        # "every buildable service in the profile" command.
        cmd.extend(services)
        # OMN-18072: a blown build ceiling is a build OUTCOME with a name, not
        # a raw TimeoutExpired carrying a forty-token argv dump into the
        # terminal record. Nothing has been recreated at this point -- the build
        # precedes every compose up -- so unlike the OMN-18057 compose-up path
        # there is no residue to recover; the verdict states that explicitly so
        # the terminal event settles instead of leaving an operator to go and
        # check whether the lane is half-recreated.
        try:
            result = _run(cmd, timeout=timeout, env=_compose_env())
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"runtime image build for profile {profile!r} exceeded its "
                f"{timeout}s ceiling and was killed. Ceiling derivation: "
                f"{budget.describe()}. The lane was NOT mutated: the build runs "
                f"before any compose up, so no container was stopped, created "
                f"or recreated by this command."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"Docker compose build failed: {result.stderr}")

    def _compose_up(
        self,
        phase: Phase,
        scope: Scope,
        services: list[str],
        on_phase_update: PhaseCallback,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        on_phase_update(phase, PhaseStatus.IN_PROGRESS)

        config = lane_config_for(lane)
        profile = "core" if scope == Scope.CORE else "runtime"
        requested_services = _requested_services_for_up(scope, services, lane=lane)
        # For runtime scope, verification MUST be bounded by the requested
        # runtime service list so the runtime-only rebuild never implicitly
        # waits on core infra containers (OMN-9455). The same list bounds the
        # ceiling derivation, so an unrelated long healthcheck elsewhere in the
        # compose file cannot inflate it.
        expected = (
            requested_services
            if requested_services
            else services_for_scope(scope, lane=lane)
        )

        if scope == Scope.RUNTIME:
            # OMN-18057: derived from the compose model, never a bare constant.
            budget = runtime_compose_up_budget(lane, expected)
            timeout = budget.timeout_seconds
            logger.info(
                "Phase %s compose-up ceiling %s", phase.value, budget.describe()
            )
            # The migration one-shots are not gated on any healthcheck, so they
            # keep the flat phase bound.
            self._ensure_runtime_migrations_ready(
                lane=lane, timeout=PHASE_TIMEOUTS[Phase.RUNTIME]
            )
        else:
            timeout = PHASE_TIMEOUTS.get(phase, 300)
        cmd = [
            "docker",
            "compose",
            *_compose_file_args(lane),
            "-p",
            config.compose_project,
            "--profile",
            profile,
            "up",
            "-d",
            "--force-recreate",
            "--pull",
            "never" if scope == Scope.RUNTIME else "always",
        ]
        # OMN-9455: runtime scope must pass --no-deps so compose cannot recreate
        # the core infra services (postgres/redpanda/valkey/infisical) declared
        # as depends_on targets of the runtime services.
        if scope == Scope.RUNTIME:
            cmd.append("--no-deps")
        if requested_services:
            cmd.extend(requested_services)

        # OMN-18057: a blown ceiling is a compose-up OUTCOME, not a reason to
        # skip the outcome check. Before this, TimeoutExpired propagated out of
        # here and the verify + per-container recovery below -- the recovery a
        # non-zero exit already gets -- never ran at all, which is how command
        # 23edaf62 left three services in Created with :8086 down.
        try:
            result = _run(cmd, timeout=timeout, env=_compose_env(extra_env))
        except subprocess.TimeoutExpired:
            compose_up_error = (
                f"docker compose up exceeded its {timeout}s ceiling for phase "
                f"{phase.value} and was killed with the lane mid-recreate"
            )
            logger.warning(
                "%s; verifying live service state before deciding the phase verdict",
                compose_up_error,
            )
        else:
            compose_up_error = result.stderr.strip() if result.returncode != 0 else ""
            if compose_up_error:
                logger.warning(
                    "Docker compose up returned non-zero; verifying live service state before failing: %s",
                    compose_up_error[:500],
                )

        # Verify containers actually reached running state — docker compose up exits 0
        # even when containers land in Created state (hit twice in production, 01:33 + 04:48).
        logger.info(
            "Verifying %d container(s) reached running state: %s",
            len(expected),
            expected,
        )
        ok, stuck = verify_containers_up(
            expected, timeout_s=CONTAINER_VERIFY_TIMEOUT_SECONDS, lane=lane
        )
        if not ok:
            self._record_container_residue(stuck, lane=lane)
            logger.warning(
                "Containers stuck after compose up — attempting docker start recovery: %s",
                stuck,
            )
            for name in stuck:
                start_result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        *_compose_file_args(lane),
                        "-p",
                        config.compose_project,
                        "up",
                        "-d",
                        "--no-deps",
                        name,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=_compose_env(extra_env),
                )
                if start_result.returncode == 0:
                    logger.info("docker start %s: ok", name)
                else:
                    logger.warning(
                        "docker start %s failed: %s", name, start_result.stderr[:200]
                    )
            ok, still_stuck = verify_containers_up(
                expected,
                timeout_s=CONTAINER_RECOVERY_VERIFY_TIMEOUT_SECONDS,
                lane=lane,
            )
            self._mark_residue_recovered(set(stuck) - set(still_stuck))
            if not ok:
                detail = f"Containers still not running after docker start recovery: {still_stuck}"
                if compose_up_error:
                    detail = f"Docker compose up failed: {compose_up_error}; {detail}"
                raise RuntimeError(detail)
            logger.info("Recovery succeeded — all containers now running")
        elif compose_up_error:
            # The command was killed or exited non-zero, yet every expected
            # service is running. Stated rather than swallowed: the phase
            # passes on live state, and VERIFICATION still has to prove health.
            logger.warning(
                "compose up did not exit cleanly (%s) but every expected "
                "service is running; phase %s passes on live state",
                compose_up_error,
                phase.value,
            )

        on_phase_update(phase, PhaseStatus.SUCCESS)

    def _ensure_runtime_migrations_ready(
        self,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        timeout: int = 300,
    ) -> None:
        """Run bounded migration services before a runtime-only restart.

        Runtime deploys intentionally use ``--no-deps`` so compose cannot walk
        into core infra and recreate Postgres/Redpanda/Valkey. The migration
        one-shots are not core infra; they are the boot-order contract that
        applies pending projection DDL and exposes the migration health gate.
        """
        config = lane_config_for(lane)
        base_cmd = [
            "docker",
            "compose",
            *_compose_file_args(lane),
            "-p",
            config.compose_project,
            "--profile",
            "runtime",
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
        ]
        for service in RUNTIME_MIGRATION_SERVICES:
            cmd = [*base_cmd, service]
            result = _run(cmd, timeout=timeout, env=_compose_env())
            if result.returncode != 0:
                raise RuntimeError(
                    f"Runtime migration preflight failed for {service}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            ok, stuck = verify_containers_up([service], timeout_s=120, lane=lane)
            if not ok:
                raise RuntimeError(
                    f"Runtime migration preflight did not satisfy {service}: {stuck}"
                )
        for table_name in REQUIRED_PROJECTION_TABLES:
            result = _run(
                [
                    "docker",
                    "exec",
                    config.postgres_container,
                    "psql",
                    "-U",
                    "postgres",
                    "-d",
                    "omnidash_analytics",
                    "-tAc",
                    f"SELECT to_regclass('public.{table_name}') IS NOT NULL",
                ],
                timeout=timeout,
            )
            if result.stdout.strip() != "t":
                raise RuntimeError(
                    "Runtime migration preflight failed: missing "
                    f"omnidash_analytics.{table_name}"
                )

    def verify(
        self,
        on_phase_update: PhaseCallback,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> list[ModelHealthCheck]:
        on_phase_update(Phase.VERIFICATION, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.VERIFICATION]
        checks: list[ModelHealthCheck] = []

        # Check for unhealthy containers
        result = _run(
            ["docker", "ps", "--filter", "health=unhealthy", "--format", "{{.Names}}"],
            timeout=timeout,
        )
        unhealthy = result.stdout.strip()
        if unhealthy:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter health=unhealthy",
                    status="fail",
                    latency_ms=0,
                )
            )
        else:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter health=unhealthy",
                    status="pass",
                    latency_ms=0,
                )
            )

        # Check for restarting containers
        result = _run(
            ["docker", "ps", "--filter", "status=restarting", "--format", "{{.Names}}"],
            timeout=timeout,
        )
        restarting = result.stdout.strip()
        if restarting:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter status=restarting",
                    status="fail",
                    latency_ms=0,
                )
            )
        else:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter status=restarting",
                    status="pass",
                    latency_ms=0,
                )
            )

        # Check projection tables in the database used by runtime DB injection.
        for table_name in REQUIRED_PROJECTION_TABLES:
            result = _run(
                [
                    "docker",
                    "exec",
                    lane_config_for(lane).postgres_container,
                    "psql",
                    "-U",
                    "postgres",
                    "-d",
                    "omnidash_analytics",
                    "-tAc",
                    f"SELECT to_regclass('public.{table_name}') IS NOT NULL",
                ],
                timeout=timeout,
            )
            checks.append(
                ModelHealthCheck(
                    service="postgres",
                    endpoint=f"omnidash_analytics.{table_name} exists",
                    status="pass" if result.stdout.strip() == "t" else "fail",
                    latency_ms=0,
                )
            )

        # Runtime health endpoint checks.
        #
        # OMN-9728: deployment readiness is owned by the runtime health servers.
        # Ports 8000/8001/8002 are LLM endpoints and cannot prove that the
        # runtime or runtime-effects processes are healthy. The host ports vary
        # per lane (dev 8085/8086, stability-test 18085/18086, prod 28085/28086).
        for service, port in lane_config_for(lane).runtime_health_targets:
            start = time.monotonic()
            result = _run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    "10",
                    f"http://localhost:{port}/health",  # url-authority-ok: pre-existing host-loopback health check, deploy-agent curls its own host's lane-scoped port already resolved from lane_config_for; out of scope for OMN-15181 round-2 digest fix
                ],
                timeout=10,
            )
            latency = int((time.monotonic() - start) * 1000)
            checks.append(
                ModelHealthCheck(
                    service=service,
                    endpoint=f"http://localhost:{port}/health",  # url-authority-ok: see loopback rationale above
                    status="pass" if _runtime_health_passed(result) else "fail",
                    latency_ms=latency,
                )
            )

        on_phase_update(Phase.VERIFICATION, PhaseStatus.SUCCESS)
        return checks
