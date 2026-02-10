# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Bus audit diagnostic models.

Frozen dataclasses for point-in-time Kafka bus health diagnostics.
These are diagnostic output models, NOT event schemas — hence dataclasses over Pydantic.

Models:
    TopicHealth: Per-topic health snapshot (offsets, sampling, naming, schema).
    EnvelopeStats: JSON parse and envelope field statistics for sampled messages.
    DomainValidationResult: Schema validation outcome for a single topic.
    AuditReport: Aggregate report across all audited topics.

Thread Safety:
    All models are frozen (immutable) and safe to share across threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.diagnostics.enum_naming_compliance import EnumNamingCompliance
from omnibase_infra.diagnostics.enum_topic_status import EnumTopicStatus
from omnibase_infra.diagnostics.enum_verdict import EnumVerdict

# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class EnvelopeStats:
    """JSON parse and envelope field statistics for sampled messages.

    Attributes:
        total_sampled: Total messages sampled from the topic.
        tombstone_count: Messages with null/empty values (excluded from parse rate).
        json_parse_ok: Messages successfully parsed as JSON.
        json_parse_fail: Messages that failed JSON parsing (excluding tombstones).
        has_envelope_id: Messages with a top-level ``envelope_id`` field.
        has_payload: Messages with a top-level ``payload`` field.
        has_timestamp: Messages with a top-level ``timestamp`` field.
    """

    total_sampled: int = 0
    tombstone_count: int = 0
    json_parse_ok: int = 0
    json_parse_fail: int = 0
    has_envelope_id: int = 0
    has_payload: int = 0
    has_timestamp: int = 0

    @property
    def parseable_count(self) -> int:
        """Messages eligible for JSON parsing (total minus tombstones)."""
        return self.total_sampled - self.tombstone_count

    @property
    def json_failure_rate(self) -> float:
        """JSON parse failure rate (0.0-1.0), excluding tombstones.

        Returns 0.0 if no parseable messages exist.
        """
        if self.parseable_count == 0:
            return 0.0
        return self.json_parse_fail / self.parseable_count


@dataclass(frozen=True)
class DomainValidationResult:
    """Schema validation outcome for a single topic.

    Only populated when ``topic_schema_map`` includes an entry for the topic.

    Attributes:
        model_name: Fully-qualified name of the Pydantic model used for validation.
        valid_count: Messages that passed ``model_validate()``.
        invalid_count: Messages that failed ``model_validate()``.
        errors: First N error messages, each capped at 300 characters.
    """

    model_name: str = ""
    valid_count: int = 0
    invalid_count: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TopicHealth:
    """Per-topic health snapshot from the bus audit.

    Attributes:
        topic: Topic name as discovered or expected.
        status: Discovery state on the broker.
        partition_count: Number of partitions (0 if not found).
        total_messages: Sum of (high - low) across all partitions.
        naming: ONEX naming compliance classification.
        envelope_stats: Sampling and envelope field statistics.
        domain_validation: Schema validation result (None if no schema mapped).
        verdict: Computed health verdict.
        issues: Human-readable issue descriptions.
    """

    topic: str
    status: EnumTopicStatus
    partition_count: int = 0
    total_messages: int = 0
    naming: EnumNamingCompliance = EnumNamingCompliance.COMPLIANT
    envelope_stats: EnvelopeStats | None = None
    domain_validation: DomainValidationResult | None = None
    verdict: EnumVerdict = EnumVerdict.PASS
    issues: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AuditReport:
    """Aggregate bus health audit report.

    Attributes:
        broker: Bootstrap servers string used for the audit.
        topics: Per-topic health results.
        overall_verdict: Worst verdict across all topics.
        unexpected_topics: Topics found on broker but not in expected list.
    """

    broker: str
    topics: tuple[TopicHealth, ...] = field(default_factory=tuple)
    overall_verdict: EnumVerdict = EnumVerdict.PASS
    unexpected_topics: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize report to a plain dict for JSON output.

        Returns:
            Nested dict suitable for ``json.dumps()``.
        """
        return {
            "broker": self.broker,
            "overall_verdict": str(self.overall_verdict),
            "topics": [_topic_to_dict(t) for t in self.topics],
            "unexpected_topics": list(self.unexpected_topics),
        }

    def to_human_readable(self) -> str:
        """Render report as a human-readable multi-line string.

        Returns:
            Formatted report for terminal or log output.
        """
        lines: list[str] = []
        lines.append(f"Bus Audit Report -- {self.broker}")
        lines.append(f"Overall Verdict: {self.overall_verdict.value.upper()}")
        lines.append("")

        for th in self.topics:
            icon = {"pass": "+", "warn": "~", "fail": "!"}[th.verdict.value]
            lines.append(f"[{icon}] {th.topic}")
            lines.append(
                f"    Status: {th.status.value}  |  Partitions: {th.partition_count}  |  Messages: {th.total_messages}"
            )
            lines.append(
                f"    Naming: {th.naming.value}  |  Verdict: {th.verdict.value.upper()}"
            )

            if th.envelope_stats is not None:
                es = th.envelope_stats
                lines.append(
                    f"    Envelope: sampled={es.total_sampled}  tombstones={es.tombstone_count}  "
                    f"json_ok={es.json_parse_ok}  json_fail={es.json_parse_fail}"
                )
                lines.append(
                    f"    Fields: envelope_id={es.has_envelope_id}  "
                    f"payload={es.has_payload}  timestamp={es.has_timestamp}"
                )

            if th.domain_validation is not None:
                dv = th.domain_validation
                lines.append(
                    f"    Schema [{dv.model_name}]: valid={dv.valid_count}  invalid={dv.invalid_count}"
                )
                for err in dv.errors:
                    lines.append(f"      - {err}")

            if th.issues:
                for issue in th.issues:
                    lines.append(f"    ** {issue}")
            lines.append("")

        if self.unexpected_topics:
            lines.append("Unexpected topics on broker:")
            for ut in self.unexpected_topics:
                lines.append(f"  - {ut}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# HELPERS
# =============================================================================


def _topic_to_dict(th: TopicHealth) -> dict[str, object]:
    """Serialize a TopicHealth to dict."""
    d: dict[str, object] = {
        "topic": th.topic,
        "status": str(th.status),
        "partition_count": th.partition_count,
        "total_messages": th.total_messages,
        "naming": str(th.naming),
        "verdict": str(th.verdict),
        "issues": list(th.issues),
    }
    if th.envelope_stats is not None:
        es = th.envelope_stats
        d["envelope_stats"] = {
            "total_sampled": es.total_sampled,
            "tombstone_count": es.tombstone_count,
            "json_parse_ok": es.json_parse_ok,
            "json_parse_fail": es.json_parse_fail,
            "has_envelope_id": es.has_envelope_id,
            "has_payload": es.has_payload,
            "has_timestamp": es.has_timestamp,
            "json_failure_rate": es.json_failure_rate,
        }
    if th.domain_validation is not None:
        dv = th.domain_validation
        d["domain_validation"] = {
            "model_name": dv.model_name,
            "valid_count": dv.valid_count,
            "invalid_count": dv.invalid_count,
            "errors": list(dv.errors),
        }
    return d


__all__: list[str] = [
    "AuditReport",
    "DomainValidationResult",
    "EnvelopeStats",
    "TopicHealth",
]
