# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event schemas for deploy agent. Strongly typed, frozen, standalone."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

TOPIC_REBUILD_REQUESTED = "onex.cmd.deploy.rebuild-requested.v1"
TOPIC_REBUILD_COMPLETED = "onex.evt.deploy.rebuild-completed.v1"
TOPIC_REBUILD_REJECTED = "onex.evt.deploy.rebuild-rejected.v1"


class DeployInProgressError(RuntimeError):
    """Raised when a second deploy arrives while one is already running."""


class Scope(StrEnum):
    FULL = "full"
    RUNTIME = "runtime"
    CORE = "core"


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
    services: list[str] = Field(default_factory=list)
    git_ref: str = "origin/main"

    @model_validator(mode="after")
    def validate_services_subset(self) -> ModelRebuildRequested:
        if self.services:
            allowed = services_for_scope(self.scope)
            invalid = [s for s in self.services if s not in allowed]
            if invalid:
                msg = f"Services {invalid} not in scope '{self.scope}'. Allowed: {allowed}"
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
