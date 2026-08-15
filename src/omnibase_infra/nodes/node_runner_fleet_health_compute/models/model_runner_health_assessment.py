# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-runner health assessment (OMN-13942) -- the COMPUTE node's classification output.

OMN-14228 Slice A adds the precondition data a remediation gate needs to fail
CLOSED on indeterminate health: per-runner source determinacy plus the typed
re-arm signals that today survive only as free text in ``detail``. This slice
does not add any executor or gate logic -- it only stops dropping data a
future gate would need.

OMN-15255 adds the composite readiness verdict (friction F-04): ``state`` is a
precedence pick, ``readiness`` is a conjunction over ``signals``. The two can
legitimately disagree -- a GitHub-online runner with a fresh heartbeat, an
unhealthy container and two listeners is ``state=HEALTHY`` and
``readiness=NOT_READY``, because container health and listener topology are
not inputs to the precedence chain at all.

OMN-15234 carries GitHub/Docker corroboration facts on the assessment so a
LISTENER_ZOMBIE restart recommendation cannot be derived from stale heartbeat
text alone.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_readiness_state import (
    EnumRunnerReadinessState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_readiness_signal import (
    ModelRunnerReadinessSignal,
)


class ModelRunnerHealthAssessment(BaseModel):
    """Classified health state for a single runner."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ..., description="Runner name (matches ModelRunnerFleetRunnerFact.name)."
    )
    state: EnumRunnerFleetHealthState = Field(
        ..., description="Classified health state."
    )
    detail: str = Field(
        default="", description="Explanation of why this state was chosen."
    )
    is_determinate: bool = Field(
        default=True,
        description=(
            "False when the upstream GitHub or Docker source used to classify "
            "this runner failed (see ModelRunnerFleetSnapshot.github_source_ok/"
            "docker_source_ok). A downstream remediation gate MUST treat this "
            "state as unreliable -- never as a verified HEALTHY -- when False."
        ),
    )
    docker_restart_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Typed re-arm signal for CRASH_LOOPING (ModelRunnerFleetRunnerFact."
            "docker_restart_count at classification time). Carried as a typed "
            "field, not parsed out of `detail`, so a future idempotency key can "
            "key on the actual observed edge."
        ),
    )
    diag_heartbeat_age_seconds: float | None = Field(
        default=None,
        description=(
            "Typed re-arm signal for LISTENER_ZOMBIE (ModelRunnerFleetRunnerFact."
            "diag_heartbeat_age_seconds at classification time). None if the "
            "probe could not determine an age."
        ),
    )
    github_status: str = Field(
        ...,
        description=(
            "GitHub registry status at classification time ('online'/'offline'/"
            "'not_registered'). OMN-15234: this is the registry cross-check the "
            "LISTENER_ZOMBIE restart recommendation is corroborated against -- "
            "'if the registry says online, the stale-heartbeat flag is the bug' "
            "(OMN-15233 runbook rule). Typed, not parsed out of `detail`."
        ),
    )
    github_busy: bool = Field(
        ...,
        description=(
            "Whether GitHub reported a job in flight at classification time. "
            "OMN-15234: a busy runner is never recommended for restart on "
            "heartbeat staleness -- restarting it would kill a live job."
        ),
    )
    docker_status: str = Field(
        default="",
        description=(
            "Docker container state at classification time (running/restarting/"
            "not_found/...). Empty when the docker probe reported none. "
            "OMN-15234: corroborating evidence for the LISTENER_ZOMBIE restart "
            "recommendation."
        ),
    )
    readiness: EnumRunnerReadinessState = Field(
        default=EnumRunnerReadinessState.UNKNOWN,
        description=(
            "Composite readiness (OMN-15255): READY only when every signal in "
            "`signals` PASSes. NOT_READY when any FAILs. UNKNOWN when none "
            "FAIL but at least one is undetermined. Consumers routing work "
            "MUST require READY -- `state == HEALTHY` is NOT equivalent and "
            "never was (the precedence chain does not evaluate container "
            "health or listener topology at all)."
        ),
    )
    signals: tuple[ModelRunnerReadinessSignal, ...] = Field(
        default_factory=tuple,
        description=(
            "Every readiness signal evaluated for this runner, PASSing ones "
            "included, so a reader can tell 'checked and fine' from 'never "
            "checked'."
        ),
    )
    quarantined: bool = Field(
        default=False,
        description=(
            "True iff readiness is NOT_READY -- i.e. a signal was probed and "
            "FAILed. Deliberately NOT set for UNKNOWN: a probe outage is not "
            "evidence of runner failure, and quarantining on it would take "
            "the fleet down on the first blip. UNKNOWN runners are excluded "
            "from routing capacity by `readiness != READY`, which is the "
            "fail-closed half of this pair."
        ),
    )
    quarantine_reason: str = Field(
        default="",
        description="Failing signal names + observed values. Empty when not quarantined.",
    )
    bounce_eligible: bool = Field(
        default=False,
        description=(
            "True iff a force-recreate is a defensible remedy for THIS "
            "runner: quarantined, sources determinate, not executing a job, "
            "and at least one failing signal is actually fixable by a bounce. "
            "Strictly narrower than `quarantined` -- a full host disk or a "
            "GitHub status-lag with healthy local evidence (OMN-14057) "
            "quarantines without ever recommending a restart."
        ),
    )


__all__ = ["ModelRunnerHealthAssessment"]
