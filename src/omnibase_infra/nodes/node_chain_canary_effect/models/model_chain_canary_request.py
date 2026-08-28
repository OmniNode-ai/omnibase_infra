# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for one event-chain canary run (OMN-16773)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelChainCanaryRequest(BaseModel):
    """Configuration for one live delegation-chain probe.

    Note what is NOT here: the probe's correlation id. It is minted fresh
    inside the handler on every run and is deliberately not settable, so no
    caller can pin it and no two runs can share one (OMN-16773 AC1). The
    ``correlation_id`` below is the RUN's id — the sweep's own identity,
    the same convention every scheduled sweep node in this repo uses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Canary run correlation ID.")
    probe_url: str = Field(
        ...,
        description=(
            "Base URL of the runtime ingress to probe, e.g. "
            "'http://host.docker.internal:8085'. REQUIRED with no default: "
            "a silent default would let a misconfigured run probe the wrong "
            "lane and report a green that means nothing (CLAUDE.md Rule 8). "
            "The caller names the lane; this node never guesses it."
        ),
    )
    runtime_command: str = Field(
        default="node_delegate_skill_orchestrator",
        description=(
            "Runtime command dispatched through POST {probe_url}/skill. The "
            "default is the delegation entry point from the recorded "
            "2026-07-30 matrix recipe."
        ),
    )
    task_type: str = Field(
        default="test",
        description=(
            "Delegation task class. One of the ids in omnidash "
            "shared/contracts/delegation-task-types.json. 'test' is class 1 "
            "of the 13-class matrix and the one the OMN-16767 incident was "
            "reproduced on."
        ),
    )
    prompt: str = Field(
        default=(
            "Reply with the single word: alive. This is an automated "
            "liveness probe of the delegation chain."
        ),
        description=(
            "Probe prompt. Deliberately trivial: the canary asserts that the "
            "CHAIN carries a request to a terminal, not that the model is "
            "any good. Inference is local to the lane host, so a run costs "
            "no external spend."
        ),
    )
    max_tokens: int = Field(
        default=32,
        ge=1,
        le=2048,
        description="Cap on generated tokens — keeps a probe run cheap and fast.",
    )
    budget_ms: int = Field(
        default=120_000,
        ge=1_000,
        le=600_000,
        description=(
            "Runtime ingress budget. A terminal that does not land inside "
            "this window is RED. 120s is 2/3 of the 180s budget the live "
            "OMN-16767 reproduction exhausted, so a chain that is merely "
            "slow still reads as failing rather than as intermittently fine."
        ),
    )
    quarantine_bootstrap_servers: str = Field(
        default="",
        description=(
            "Kafka/Redpanda bootstrap servers for the correlation-scoped "
            "quarantine check. EMPTY (the default) means the leg does not "
            "run and the result reports SKIPPED_NOT_CONFIGURED — never a "
            "clean verdict for a check that never happened."
        ),
    )
    quarantine_topic: str = Field(
        default="onex.dlq.omnibase-infra.quarantine.v1",
        description=(
            "Platform quarantine sink. Handlers with no DLQ topic declared "
            "in their own contract fall through to this topic, which is how "
            "the OMN-16767 delegation failures became invisible."
        ),
    )
    quarantine_scan_records: int = Field(
        default=500,
        ge=1,
        le=20_000,
        description=(
            "How many records back from the high-water mark to scan. This "
            "is a TAIL scan by design: the sink held ~8.9M records when "
            "OMN-16767 was diagnosed, so reading it whole is not an option. "
            "The canary only needs the window its own request just landed "
            "in. Sink-wide depth monitoring is OMN-16769, not this node."
        ),
    )
    quarantine_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Wall-clock cap on the quarantine tail scan.",
    )
    settle_seconds: int = Field(
        default=2,
        ge=0,
        le=60,
        description=(
            "Pause between the ingress answering and the quarantine scan. "
            "The DLQ write happens after the handler raises, so scanning "
            "instantly can miss a record that is about to appear and turn a "
            "QUARANTINED run into a vaguer TERMINAL_MISSING one."
        ),
    )

    @field_validator("probe_url")
    @classmethod
    def _validate_probe_url(cls, value: str) -> str:
        stripped = value.strip().rstrip("/")
        if not stripped:
            raise ValueError("probe_url must not be empty")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError(
                f"probe_url must be an absolute http(s) URL, got: {value!r}"
            )
        return stripped


__all__ = ["ModelChainCanaryRequest"]
