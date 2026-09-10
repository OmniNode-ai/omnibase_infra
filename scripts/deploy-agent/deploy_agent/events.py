# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event schemas for deploy agent. Strongly typed, frozen, standalone."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from deploy_agent.tracking_ref import load_tracking_remote_ref_from_env

TOPIC_REBUILD_REQUESTED = "onex.cmd.deploy.rebuild-requested.v1"
TOPIC_REBUILD_COMPLETED = "onex.evt.deploy.rebuild-completed.v1"
TOPIC_REBUILD_REJECTED = "onex.evt.deploy.rebuild-rejected.v1"
# Dead-letter target for a command record the agent cannot decode or validate
# (OMN-16442). Shape follows the org convention onex.dlq.<producer>.<category>.<version>.
# A record that lands here is one the agent has committed past: it can never be
# decoded by redelivery, and withholding the offset stalls every command behind
# it -- see deploy_agent.consumer for the full argument.
TOPIC_DEPLOY_COMMAND_DLQ = "onex.dlq.omnibase-infra.deploy-command.v1"


class DeployInProgressError(RuntimeError):
    """Raised when a second deploy arrives while one is already running."""


class Scope(StrEnum):
    FULL = "full"
    RUNTIME = "runtime"
    CORE = "core"


class EnumRuntimeLane(StrEnum):
    """Runtime deployment lane.

    Each lane maps to its own compose overlay, compose project, and runtime
    health ports (see ``deploy_agent.executor.lane_config_for``). ``prod``
    deploys a stability-proven image digest rather than rebuilding from a ref.
    """

    DEV = "dev"
    STABILITY_TEST = "stability-test"
    PROD = "prod"


class EnumSelfUpdateBoundary(StrEnum):
    """The job boundary a self-update is allowed to fire at (OMN-16442).

    Self-update pulls the agent's own clone and replaces the process image, so
    it may only run where no job is in flight. Between deploy phases is not
    such a place: on 2026-09-08 command
    ``8d0c861a-f91e-4ca2-954e-a073759dd39d`` re-execed after the seed phase and
    the replacement process published that same command as ``failed`` after
    logging ``Recovered 1 crashed job(s)``.

    ``PRE_ACCEPT``
        In the consumer, after a command has passed the signature, payload,
        lane-fence, busy and dedup checks and BEFORE ``job_store.accept``
        marks it started. Nothing is in flight, and the command's offset is
        rewound rather than committed, so the replacement process re-reads it
        and processes it once.

    ``POST_TERMINAL``
        In the agent, after the single-flight lock is released and the job's
        terminal status has been published. Deferring to here is what lets a
        deploy that starts on version X complete on version X.
    """

    PRE_ACCEPT = "pre_accept"
    POST_TERMINAL = "post_terminal"


class BuildSource(StrEnum):
    WORKSPACE = "workspace"
    RELEASE = "release"


class Phase(StrEnum):
    PREFLIGHT = "preflight"
    GIT = "git"
    COMPOSE_GEN = "compose_gen"
    SEED = "seed"
    CORE = "core"
    RUNTIME = "runtime"
    VERIFICATION = "verification"
    PUBLISH = "publish"


class PhaseStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


# The phases a deploy actually executes, in pipeline order. PUBLISH is
# deliberately absent: it is the act of emitting the terminal event, so an event
# can never carry a settled verdict for it (OMN-18057). Used by the terminal
# reconciliation to decide which phases a raised deploy never reached.
DEPLOY_PHASE_ORDER: tuple[Phase, ...] = (
    Phase.PREFLIGHT,
    Phase.GIT,
    Phase.COMPOSE_GEN,
    Phase.SEED,
    Phase.CORE,
    Phase.RUNTIME,
    Phase.VERIFICATION,
)


SCOPE_SERVICES: dict[Scope, list[str]] = {
    Scope.CORE: ["postgres", "redpanda", "valkey"],
    Scope.RUNTIME: [
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "context-audit-consumer",
        "intelligence-migration",
        "intelligence-api",
        "omninode-contract-resolver",
        "autoheal",
    ],
    Scope.FULL: [],  # resolved as union of core + runtime
}


# OMN-18108: the runtime services the DEV lane declares and no other lane does.
#
# These live only in ``docker/docker-compose.dev-lane.yml``. Membership in
# ``SCOPE_SERVICES[Scope.RUNTIME]`` above would be FATAL, not merely wrong: the
# service name does not exist in the prod, stability-test or judge merged
# compose, so every deploy to those lanes would abort on `no such service`.
# They are a DEV-lane addendum, resolved by ``services_for_scope`` only when the
# caller names that lane.
#
# WHY THIS IS DECLARED HERE AND NOT PARSED FROM THE SHELL SCRIPT
# --------------------------------------------------------------
# ``scripts/deploy-runtime.sh`` carries the same eight names in its
# ``DEV_LANE_ONLY_RUNTIME_SERVICES`` array, and three existing tests parse that
# hand-written literal out of the script by regex. Reshaping the array into a
# file both sides read would break those tests and edit the sanctioned deploy
# path for a refactor's sake. The two declarations are instead bound
# MECHANICALLY and bidirectionally by
# ``tests/unit/test_dev_lane_only_scope_omn18108.py``, which parses the array
# and asserts set equality both ways -- an edit to either side alone is a red
# test, which is the property "single source of truth" was wanted for.
#
# The defect this closes, measured on the .201 dev lane 2026-09-10T00:45Z: the
# runtime family carried the deploy agent's own build
# ``4598a4358bd9f59528875b8b320b6cde54383fb1`` while all eight of these carried
# ``3461e4b0aeae`` from the previous day, ~35 infra commits behind. Not an
# intermittent miss -- the agent's scope could not reach them at all, and
# ``restart: unless-stopped`` keeps a stale image running and healthy, so
# nothing reported it.
DEV_LANE_ONLY_RUNTIME_SERVICES: tuple[str, ...] = (
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "infra-routing-decisions-consumer",
    "onex-api",
    # OMN-18114: the TENANT-domain projection carrier, for a reason that is the
    # OPPOSITE of the writers' reason above and is stated separately so the two
    # do not merge. That service IS declared in docker-compose.infra.yml, so
    # every lane resolves the name -- which is exactly why it cannot be
    # lane-agnostic: a prod or judge `up -d --no-deps tenant-projection-writer`
    # would SUCCEED and start the carrier on a lane that never opted into it,
    # defeating the compose profile that keeps it inert there. Membership here
    # scopes it to the lane whose overlay puts it in the `runtime` profile.
    #
    # It is the only process that owns the eight omnimarket contracts declaring
    # `runtime_profiles: [tenant-projection]`, so an agent scope that could not
    # reach it would leave those eight running an image the rest of the lane had
    # moved past -- the exact defect measured above, on the one service where
    # nothing else would ever notice.
    "tenant-projection-writer",
)

# OMN-18108: the members of the array above that carry an ``image:`` and no
# ``build:`` -- tag-referenced, not lane-built.
#
# ``onex-api`` resolves ``${ONEX_API_IMAGE}`` from the operator env file on the
# host; the image is built out of a different repository. So a governed deploy
# can RECREATE it, which is what it actually needs (a container that is never
# recreated never reads a new environment, and this one carries the lane's
# broker credentials and eleven fail-closed variables), but it CANNOT advance
# the tag. Stating the boundary here, and asserting it in the test, rather than
# leaving an operator to discover that a "successful" deploy left the tag where
# it was. Advancing it needs an image build plus an env repoint, and no
# sanctioned script does either.
DEV_LANE_ONLY_TAG_REFERENCED_SERVICES: frozenset[str] = frozenset({"onex-api"})

# The subset ``docker compose build`` can be handed. Derived, never a second
# hand-written list.
DEV_LANE_ONLY_BUILDABLE_SERVICES: tuple[str, ...] = tuple(
    service
    for service in DEV_LANE_ONLY_RUNTIME_SERVICES
    if service not in DEV_LANE_ONLY_TAG_REFERENCED_SERVICES
)


# OMN-18134: the DEV lane's gateway compose project, and the services in it.
#
# THIS LIST IS NOT A SECOND DEV_LANE_ONLY_RUNTIME_SERVICES. The two are kept
# apart because their compose semantics are OPPOSITE, and merging them would be
# fatal in the same way membership in the base runtime list would be fatal for
# the eight above.
#
# The eight above live in ``docker/docker-compose.dev-lane.yml``, which IS one
# of the DEV lane's ``compose_files``: they are handed to
# ``docker compose -p omnibase-infra`` and that project resolves them. These
# live in ``docker/docker-compose.gateway.yml`` under the compose project
# ``omninode-gateway``, which appears in NO lane's ``compose_files``. Handing
# ``gateway-forwarder`` to the omnibase-infra project aborts the runtime phase
# on `no such service`.
#
# So membership here means exactly one thing: a DEV deploy is RESPONSIBLE for
# this service. Which command deploys it is a separate question, answered by
# ``DeployExecutor._deploy_gateway_lane`` -- it calls the sanctioned
# ``scripts/deploy-gateway.sh``, which owns the gateway project's build, digest
# pin, host-file sync, rollback record and systemd reload. Every call site that
# builds a compose ARGUMENT list subtracts these via
# ``without_gateway_services``.
#
# Only the forwarder is listed. ``gateway-dns-bastion`` is a sidecar of the
# same project that ``deploy-gateway.sh`` builds and that compose starts via
# the forwarder's ``depends_on``; it is not independently addressable as a
# deploy target, and listing it would let a command name it alone and get the
# whole gateway lane deployed anyway.
GATEWAY_COMPOSE_PROJECT = "omninode-gateway"
DEV_LANE_GATEWAY_SERVICES: tuple[str, ...] = ("gateway-forwarder",)


def without_gateway_services(services: Iterable[str]) -> list[str]:
    """Return ``services`` minus anything the gateway compose project owns.

    Used at every site that builds an argument list for the ``omnibase-infra``
    compose project, so a service that is legitimately in DEV *scope* can never
    become a compose *argument* for a project that does not declare it.
    """
    return [s for s in services if s not in DEV_LANE_GATEWAY_SERVICES]


def gateway_services_in(services: Iterable[str]) -> list[str]:
    """Return the subset of ``services`` the gateway compose project owns."""
    return [s for s in services if s in DEV_LANE_GATEWAY_SERVICES]


def services_for_scope(
    scope: Scope, *, lane: EnumRuntimeLane | None = None
) -> list[str]:
    """Return the services a deploy of ``scope`` targets on ``lane``.

    ``lane`` defaults to ``None``, which resolves the lane-agnostic base list
    exactly as before. A caller that does not name a lane therefore never
    silently acquires dev-lane services, and prod/stability-test scope is
    byte-unchanged whether the lane is passed or not (OMN-18108 AC3, OMN-18134
    AC2).

    This is the list of what a deploy is RESPONSIBLE for, not the list of what
    is handed to any one compose project -- see ``DEV_LANE_GATEWAY_SERVICES``.
    """
    if scope == Scope.FULL:
        base = SCOPE_SERVICES[Scope.CORE] + SCOPE_SERVICES[Scope.RUNTIME]
    else:
        base = list(SCOPE_SERVICES[scope])
    if lane == EnumRuntimeLane.DEV and scope in (Scope.RUNTIME, Scope.FULL):
        return (
            base
            + list(DEV_LANE_ONLY_RUNTIME_SERVICES)
            + list(DEV_LANE_GATEWAY_SERVICES)
        )
    return base


class ModelHealthCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    endpoint: str
    status: Literal["pass", "fail"]
    latency_ms: int = 0


class ModelContainerResidue(BaseModel):
    """One service left in a non-running state by a deploy phase (OMN-18057).

    The 2026-09-08 runtime-phase kill left ``runtime-effects``,
    ``runtime-worker`` and ``omninode-contract-resolver`` in ``Created`` with
    :8086 down, and the terminal event said nothing about any of them. Residue
    is recorded whether or not per-container recovery then succeeded, because
    "recovered after the ceiling blew" and "came up first time" are different
    facts about the lane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    state: str
    exit_code: int | None = None
    recovered: bool = False


class ModelRebuildRequested(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    requested_by: str
    scope: Scope
    runtime_lane: EnumRuntimeLane
    build_source: BuildSource = BuildSource.RELEASE
    services: list[str] = Field(default_factory=list)
    # OMN-16442: a command that omits the ref deploys the branch this agent
    # DECLARES it tracks (DEPLOY_AGENT_TRACKING_REF), not a literal. The old
    # default was "origin/main"; on the .201 dev lane that asked the agent to
    # `git reset --hard` its deploy-source clone onto a release-synced branch
    # hundreds of commits behind the code the lane exists to run. There is no
    # default for the variable itself — an undeclared tracking ref raises
    # rather than guessing (rule 8).
    git_ref: str = Field(default_factory=load_tracking_remote_ref_from_env)
    # Carry both ref and digest; the digest is the authority. dev/stability-test
    # may build from a ref and leave the digest unresolved up front; prod must
    # pin the stability-proven digest (enforced below).
    image_ref: str | None = None
    image_digest: str | None = None

    @model_validator(mode="after")
    def validate_services_subset(self) -> ModelRebuildRequested:
        if self.services:
            # OMN-18108: lane-aware, so a dev command may name one of the
            # dev-lane-only services and a prod/stability command may not.
            allowed = services_for_scope(self.scope, lane=self.runtime_lane)
            invalid = [s for s in self.services if s not in allowed]
            if invalid:
                msg = f"Services {invalid} not in scope '{self.scope}'. Allowed: {allowed}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_prod_requires_digest(self) -> ModelRebuildRequested:
        if self.runtime_lane == EnumRuntimeLane.PROD and not self.image_digest:
            msg = (
                "prod runtime_lane requires image_digest: production deploys the "
                "exact stability-proven digest and never rebuilds from a ref"
            )
            raise ValueError(msg)
        return self


class ModelRebuildCompleted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    requested_git_ref: str
    git_sha: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    scope: Scope
    runtime_lane: EnumRuntimeLane
    image_ref: str | None = None
    image_digest: str | None = None
    services_restarted: list[str] = Field(default_factory=list)
    # OMN-17135: repo -> the commit SHA RT-1 resolved and vendored for that
    # sibling in a workspace-mode build. ``requested_git_ref`` above pins ONE
    # repository (omnibase_infra), so on its own it said nothing about which
    # omnibase_core / omnibase_compat / omnimarket commit the image carries.
    # This is EVIDENCE beside the infra pin, never a key: the rule-24 lab-pass
    # receipt is keyed by the infra sha and stays that way. Empty for a
    # release-mode or prod digest deploy, which vendors no sibling trees.
    sibling_refs: dict[str, str] = Field(default_factory=dict)
    phase_results: dict[Phase, PhaseStatus]
    errors: list[str] = Field(default_factory=list)
    health_checks: list[ModelHealthCheck] = Field(default_factory=list)
    container_residue: list[ModelContainerResidue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase_results_are_settled(self) -> ModelRebuildCompleted:
        """A terminal event may only carry settled phase verdicts (OMN-18057).

        Two shapes are refused here rather than merely discouraged upstream:

        * ``PhaseStatus.IN_PROGRESS`` -- the live defect. Command 23edaf62's
          terminal event carried ``runtime: in_progress`` alongside a
          ``completed_at`` and a duration, so the event asserted the deploy was
          over and simultaneously refused to say how it ended. An unreached
          phase is SKIPPED and a phase that raised is FAILED.
        * ``Phase.PUBLISH`` -- this event IS the publish. Its outcome is not
          knowable at the moment the payload is built, and reporting it as
          ``in_progress`` made ``status`` derive "failed" for every deploy the
          agent ever completed, successful ones included.
        """
        if Phase.PUBLISH in self.phase_results:
            raise ValueError(
                "phase_results must not carry Phase.PUBLISH: the completion "
                "event is the publish and cannot report its own outcome"
            )
        unsettled = sorted(
            phase.value
            for phase, status in self.phase_results.items()
            if status in (PhaseStatus.IN_PROGRESS, PhaseStatus.PENDING)
        )
        if unsettled:
            raise ValueError(
                f"phase_results carries unsettled verdicts for {unsettled}: a "
                "terminal event must report failed/skipped for a phase that "
                "raised or was never reached"
            )
        return self

    @computed_field
    @property
    def status(self) -> Literal["success", "failed"]:
        non_skipped = {
            k: v for k, v in self.phase_results.items() if v != PhaseStatus.SKIPPED
        }
        if all(v == PhaseStatus.SUCCESS for v in non_skipped.values()):
            return "success"
        return "failed"
