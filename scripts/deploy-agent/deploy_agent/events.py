# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event schemas for deploy agent. Strongly typed, frozen, standalone."""

from __future__ import annotations

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


def services_for_scope(scope: Scope) -> list[str]:
    if scope == Scope.FULL:
        return SCOPE_SERVICES[Scope.CORE] + SCOPE_SERVICES[Scope.RUNTIME]
    return SCOPE_SERVICES[scope]


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
            allowed = services_for_scope(self.scope)
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
