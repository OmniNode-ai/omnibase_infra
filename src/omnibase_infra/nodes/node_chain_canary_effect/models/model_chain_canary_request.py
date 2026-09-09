# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for one event-chain canary run (OMN-16773)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    host_aliases_in,
    looks_like_a_dsn,
)

# Read off the generated topic enum, which is itself generated from the
# contract.yaml files — never typed as raw literals here (CLAUDE.md
# "contract-first topic definitions"; the arch-invariants raw-topic-literal
# gate, OMN-3343, fails CI on a hand-typed `onex.evt.*` string in src/).
_DEFAULT_TERMINAL_SUCCESS_TOPICS: tuple[str, ...] = (
    EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value,
)
_DEFAULT_TERMINAL_FAILURE_TOPICS: tuple[str, ...] = (
    EnumOmnimarketTopic.EVT_DELEGATE_SKILL_FAILED_V1.value,
)


def _coerce_topics(value: object) -> object:
    """Accept a comma-separated string as well as a sequence of topics.

    The CLI passes one string per flag (``skill_mapping.yaml`` arg types are
    scalar), so the wire form has to be splittable; the model form stays a
    tuple.
    """
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return value


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
    terminal_bootstrap_servers: str = Field(
        default="",
        description=(
            "Kafka/Redpanda bootstrap servers for the correlation-scoped "
            "TERMINAL readback — the leg that discharges OMN-16025 link 4 "
            "('emission OUTBOX-CONFIRMED via broker readback, not "
            "publish-return'). EMPTY means the run has no evidence about "
            "the terminal and reports TERMINAL_READBACK_NOT_CONFIGURED. It "
            "deliberately does NOT fall back to the ingress response: that "
            "fallback is the OMN-16931 defect this field exists to remove."
        ),
    )
    projection_dsn_env: str = Field(
        default="",
        description=(
            "NAME of the environment variable carrying the Postgres DSN for "
            "the correlation-scoped PROJECTION readback — the leg that "
            "discharges OMN-16025 link 2 ('routing decision PUBLISHED and "
            "PROJECTED, readback from projection, not logs'). A NAME, never a "
            "DSN: this field is serialised into the node payload, onto the "
            "bus and into the event log, and it is built from a CLI flag that "
            "lands in argv, so a DSN here would be durably persisted and "
            "readable from /proc by every process on the host. The NAME is "
            "resolved from the lane's declared `projection_readback.dsn_env` "
            "(omnimarket config/ci_bus_lanes.yaml); the value is injected "
            "into the job environment under that name and read there. EMPTY "
            "means the run has no evidence about the projection and reports "
            "NOT_CONFIGURED. It deliberately does NOT fall back to the bus "
            "terminal: OMN-14843 measured 26 of 38 correlations stranded "
            "mid-FSM while the topic layer was healthy at that same moment, "
            "so a green terminal is not evidence about this layer."
        ),
    )
    ledger_source: str = Field(
        default="",
        description=(
            "Source for the LEDGER CHAIN assembly and replay — the leg that "
            "discharges OMN-16025 link 5 ('complete ledger chain + replay "
            "green through an HONEST tier-2 verifier, SKIP != PASS'). EMPTY "
            "means the run has no evidence about the chain and reports "
            "NOT_CONFIGURED. It deliberately does NOT fall back to the bus "
            "terminal or the projection: a link with no instrument pointed at "
            "it makes no claim."
        ),
    )
    expected_ledger_hops: tuple[str, ...] = Field(
        default=(),
        description=(
            "The hops a COMPLETE chain must carry, end to end. Any assembled "
            "chain missing one of these is CHAIN_INCOMPLETE — the scope's own "
            "wording is 'no gaps tolerated silently', so completeness is "
            "checked against a declared set rather than inferred from "
            "whatever the ledger happened to return."
        ),
    )
    terminal_success_topics: tuple[str, ...] = Field(
        default=_DEFAULT_TERMINAL_SUCCESS_TOPICS,
        description=(
            "Terminal topics that mean the delegation COMPLETED. Defaults to "
            "node_delegate_skill_orchestrator's contract-declared "
            "runtime_dispatch.terminal_events.success, read off the "
            "generated topic enum. A terminal here discharges link 4 and "
            "supports link 3."
        ),
    )
    terminal_failure_topics: tuple[str, ...] = Field(
        default=_DEFAULT_TERMINAL_FAILURE_TOPICS,
        description=(
            "Terminal topics that mean the delegation FAILED. Read back on "
            "the same pass as the success topics: a failure terminal still "
            "discharges link 4 (the emission landed on the bus) while "
            "failing link 3 (execution did not complete). Distinguishing "
            "those two is the diagnostic OMN-16931 adds."
        ),
    )
    terminal_scan_records: int = Field(
        default=500,
        ge=1,
        le=20_000,
        description=(
            "Backlog depth to seek back per terminal topic before waiting "
            "for new records. The terminal may already be on the bus by the "
            "time the ingress answers — run 33251822642 published it 3s "
            "before the ingress replied — so the readback must look "
            "backwards as well as forwards."
        ),
    )
    terminal_readback_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=600,
        description=(
            "FLOOR on the terminal readback window, in seconds. The actual "
            "window is the larger of this and the remainder of budget_ms "
            "after the ingress answered — OMN-16025 link 4 says 'inside the "
            "budget', and giving up at 4,369 ms of a 120,000 ms budget "
            "because the ingress returned early is precisely the OMN-16931 "
            "bug. The floor covers the case where the ingress itself burned "
            "the whole budget."
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

    @field_validator(
        "terminal_success_topics", "terminal_failure_topics", mode="before"
    )
    @classmethod
    def _split_topics(cls, value: object) -> object:
        return _coerce_topics(value)

    @field_validator("terminal_success_topics")
    @classmethod
    def _require_a_success_topic(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A readback with nothing to read is a check that cannot fail."""
        if not value:
            raise ValueError(
                "terminal_success_topics must name at least one topic — a "
                "readback with no topics would report NOT_FOUND for every "
                "run and teach people to ignore this canary"
            )
        return value

    @field_validator("quarantine_bootstrap_servers", "terminal_bootstrap_servers")
    @classmethod
    def _refuse_a_host_alias_broker(cls, value: str) -> str:
        """A broker address may not be a Docker-Desktop host alias (OMN-17926).

        ``host.docker.internal:19092`` was this canary's hardcoded default for
        both broker legs, and it took every run from 2026-09-07T17:44Z onward
        red with ``quarantine_probe_failed`` -- 25 consecutive runs, each of
        which had already fired its delegation successfully. The alias resolves
        only because ``docker-compose.runners.yml`` hands the deploy runner an
        ``extra_hosts: host-gateway`` mapping, and, being an address literal,
        it carries no transport at all -- so the client opened plaintext
        against the SASL/SCRAM listener OMN-18012 Phase B had enabled hours
        earlier and reported the whole bus unreadable.

        The refusal is deliberately at the model boundary rather than in the
        workflow: a workflow default is one edit away from coming back, and
        the next reader of a green canary cannot tell which address it used.
        Pass the lane's declared broker (omnimarket
        ``config/ci_bus_lanes.yaml``, resolved by
        ``lane_transport.load_lane_transport``) instead.

        The ingress ``probe_url`` is deliberately NOT covered: it speaks HTTP
        to a host-published port, that leg has worked on every one of those 25
        runs, and widening the refusal to it would break a working probe to
        make a point.
        """
        aliases = host_aliases_in(value)
        if aliases:
            raise ValueError(
                f"broker address {value!r} uses the Docker-Desktop host alias "
                f"{aliases[0]!r}. That alias resolves on one runner's compose "
                "block and nowhere else, and it declares no transport, so a "
                "client cannot know whether the listener speaks SASL. Resolve "
                "the broker from the lane declaration (omnimarket "
                "config/ci_bus_lanes.yaml) instead of naming a host literal."
            )
        return value

    @field_validator("projection_dsn_env")
    @classmethod
    def _refuse_a_dsn_where_a_name_belongs(cls, value: str) -> str:
        """This field takes a variable NAME. A DSN here is refused (OMN-18060).

        The refusal is at the model boundary because that is the boundary the
        credential would cross. ``onex skill`` builds the node payload from
        CLI flags, so a value passed here is simultaneously in this process's
        argv (world-readable through ``/proc/<pid>/cmdline``), echoed by the
        dispatch step into the run log, and serialised into the event log as
        part of the request — three durable copies of a credential, from one
        flag.

        The message deliberately does not echo the offending value.
        """
        if looks_like_a_dsn(value):
            raise ValueError(
                "projection_dsn_env takes the NAME of the environment "
                "variable carrying the DSN, and the value supplied parses as "
                "a connection string. A DSN passed here would land in argv, "
                "in this run's log and in the event log. Pass the name "
                "declared by the lane's projection_readback.dsn_env "
                "(omnimarket config/ci_bus_lanes.yaml) and inject the value "
                "into the environment under that name."
            )
        return value

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
