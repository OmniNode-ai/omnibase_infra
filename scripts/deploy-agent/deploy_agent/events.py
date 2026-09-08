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

    @computed_field
    @property
    def status(self) -> Literal["success", "failed"]:
        non_skipped = {
            k: v for k, v in self.phase_results.items() if v != PhaseStatus.SKIPPED
        }
        if all(v == PhaseStatus.SUCCESS for v in non_skipped.values()):
            return "success"
        return "failed"
