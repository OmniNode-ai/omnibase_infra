# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Kafka bus health audit engine.

Point-in-time diagnostic tool — run once, see what's on the bus, fix problems.
Uses confluent-kafka (sync) AdminClient and Consumer for broker introspection.

Usage:
    from omnibase_infra.diagnostics import AuditConfig, run_audit

    config = AuditConfig(
        broker="192.168.86.200:29092",
        expected_topics=["onex.evt.platform.node-registration.v1"],
    )
    report = run_audit(config)
    print(report.to_human_readable())

Design Decisions:
    - confluent-kafka (not aiokafka): Sync CLI tool, no event loop needed.
    - Frozen dataclasses for results: Diagnostic output, not event schemas.
    - Consumer group: ``bus-audit.{timestamp}`` — never commits, never interferes.
    - Tombstone tracking: null/empty values excluded from JSON failure rate.

Thread Safety:
    ``run_audit()`` is NOT thread-safe. Each invocation creates and closes its own
    AdminClient and Consumer connections. Multiple concurrent calls are fine as
    long as they use separate AuditConfig instances.

See Also:
    models.py: Result dataclasses and enums.
    EnumConsumerGroupPurpose.AUDIT: Consumer group purpose classification.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, ClusterMetadata
from pydantic import BaseModel

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.diagnostics.enum_naming_compliance import EnumNamingCompliance
from omnibase_infra.diagnostics.enum_topic_status import EnumTopicStatus
from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.diagnostics.models import (
    AuditReport,
    DomainValidationResult,
    EnvelopeStats,
    TopicHealth,
)

logger = logging.getLogger(__name__)

# Maximum error string length stored per schema validation failure.
_MAX_ERROR_LEN = 300

# Maximum number of schema validation errors stored per topic.
_MAX_SCHEMA_ERRORS = 20

# Default number of messages to sample per partition.
_DEFAULT_SAMPLE_PER_PARTITION = 50

# Default consumer poll timeout in seconds.
_DEFAULT_POLL_TIMEOUT = 2.0

# JSON parse failure rate threshold above which verdict is FAIL.
_JSON_FAILURE_THRESHOLD = 0.10


# =============================================================================
# CONFIG
# =============================================================================


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for a bus health audit run.

    Attributes:
        broker: Kafka bootstrap servers string (e.g., ``192.168.86.200:29092``).
        expected_topics: Topic names that MUST exist on the broker.
        legacy_topics: Known non-ONEX topic names (classified LEGACY, not NON_COMPLIANT).
        topic_schema_map: Mapping of topic name to Pydantic model class for schema validation.
        sample_per_partition: Max messages to sample per partition for envelope stats.
        poll_timeout: Consumer poll timeout in seconds.
        envelope_exempt_topics: Topics exempt from envelope field warnings.
    """

    broker: str
    expected_topics: tuple[str, ...] = field(default_factory=tuple)
    legacy_topics: frozenset[str] = field(default_factory=frozenset)
    topic_schema_map: dict[str, type[BaseModel]] = field(default_factory=dict)
    sample_per_partition: int = _DEFAULT_SAMPLE_PER_PARTITION
    poll_timeout: float = _DEFAULT_POLL_TIMEOUT
    envelope_exempt_topics: frozenset[str] = field(default_factory=frozenset)


# =============================================================================
# PUBLIC API
# =============================================================================


def run_audit(config: AuditConfig) -> AuditReport:
    """Execute a point-in-time bus health audit.

    Connects to the broker, discovers topics, samples messages, validates
    naming and schemas, and computes per-topic and overall verdicts.

    Args:
        config: Audit configuration specifying broker, expected topics,
            and optional schema mappings.

    Returns:
        Frozen AuditReport with per-topic health and overall verdict.

    Raises:
        Exception: Propagates confluent-kafka connection errors.
    """
    # Phase 1: Topic discovery
    admin = AdminClient({"bootstrap.servers": config.broker})
    cluster_metadata = admin.list_topics(timeout=10)
    broker_topics: set[str] = set(cluster_metadata.topics.keys())

    # Collect all topics to audit (expected + any discovered)
    all_topics = set(config.expected_topics) | broker_topics

    # Phase 2: Watermark offsets for discovered topics
    offsets_map = _collect_watermarks(config, broker_topics, cluster_metadata)

    # Phase 3: Sample messages for topics that have data
    samples_map = _sample_messages(config, offsets_map)

    # Phase 4: Build per-topic health
    topic_results: list[TopicHealth] = []
    for topic in sorted(all_topics):
        th = _build_topic_health(config, topic, broker_topics, offsets_map, samples_map)
        topic_results.append(th)

    # Phase 5: Unexpected topics
    expected_set = set(config.expected_topics)
    unexpected = tuple(sorted(broker_topics - expected_set))

    # Phase 6: Overall verdict
    overall = EnumVerdict.PASS
    for th in topic_results:
        if th.verdict == EnumVerdict.FAIL:
            overall = EnumVerdict.FAIL
            break
        if th.verdict == EnumVerdict.WARN:
            overall = EnumVerdict.WARN

    return AuditReport(
        broker=config.broker,
        topics=tuple(topic_results),
        overall_verdict=overall,
        unexpected_topics=unexpected,
    )


# =============================================================================
# INTERNALS
# =============================================================================


def _collect_watermarks(
    config: AuditConfig,
    broker_topics: set[str],
    cluster_metadata: ClusterMetadata,
) -> dict[str, list[tuple[int, int, int]]]:
    """Collect (partition, low, high) watermark offsets for discovered topics.

    Returns:
        Dict mapping topic name to list of (partition_id, low, high) tuples.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": config.broker,
            "group.id": f"bus-audit.{int(time.time())}",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": "earliest",
        }
    )

    offsets: dict[str, list[tuple[int, int, int]]] = {}
    try:
        for topic in broker_topics:
            topic_meta = cluster_metadata.topics.get(topic)
            if topic_meta is None:
                continue
            partitions = topic_meta.partitions
            topic_offsets: list[tuple[int, int, int]] = []
            for pid in partitions:
                try:
                    low, high = consumer.get_watermark_offsets(
                        TopicPartition(topic, pid), timeout=5
                    )
                    topic_offsets.append((pid, low, high))
                except Exception:
                    logger.debug("Failed to get watermarks for %s[%d]", topic, pid)
            offsets[topic] = topic_offsets
    finally:
        consumer.close()

    return offsets


def _sample_messages(
    config: AuditConfig,
    offsets_map: dict[str, list[tuple[int, int, int]]],
) -> dict[str, list[bytes | None]]:
    """Sample messages from topics that have data.

    Returns:
        Dict mapping topic name to list of raw message values (bytes or None).
    """
    # Determine which topics have data and need sampling
    topics_to_sample: dict[str, list[TopicPartition]] = {}
    for topic, part_offsets in offsets_map.items():
        assignments: list[TopicPartition] = []
        for pid, low, high in part_offsets:
            if high > low:
                seek_pos = max(low, high - config.sample_per_partition)
                tp = TopicPartition(topic, pid, seek_pos)
                assignments.append(tp)
        if assignments:
            topics_to_sample[topic] = assignments

    if not topics_to_sample:
        return {}

    samples: dict[str, list[bytes | None]] = {}

    consumer = Consumer(
        {
            "bootstrap.servers": config.broker,
            "group.id": f"bus-audit.{int(time.time())}",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": "earliest",
        }
    )
    try:
        for topic, assignments in topics_to_sample.items():
            consumer.assign(assignments)
            topic_samples: list[bytes | None] = []
            # Calculate total messages we expect to sample
            total_expected = sum(config.sample_per_partition for _ in assignments)
            empty_polls = 0
            while len(topic_samples) < total_expected and empty_polls < 3:
                msg = consumer.poll(timeout=config.poll_timeout)
                if msg is None:
                    empty_polls += 1
                    continue
                if msg.error():
                    empty_polls += 1
                    continue
                empty_polls = 0
                topic_samples.append(msg.value())
            samples[topic] = topic_samples
            consumer.unassign()
    finally:
        consumer.close()

    return samples


def _build_topic_health(
    config: AuditConfig,
    topic: str,
    broker_topics: set[str],
    offsets_map: dict[str, list[tuple[int, int, int]]],
    samples_map: dict[str, list[bytes | None]],
) -> TopicHealth:
    """Build a TopicHealth for a single topic."""
    issues: list[str] = []

    # --- Status ---
    if topic not in broker_topics:
        status = EnumTopicStatus.NOT_FOUND
        naming = _classify_naming(topic, config.legacy_topics)
        verdict = (
            EnumVerdict.FAIL if topic in config.expected_topics else EnumVerdict.WARN
        )
        if topic in config.expected_topics:
            issues.append(f"Expected topic not found on broker: {topic}")
        return TopicHealth(
            topic=topic,
            status=status,
            naming=naming,
            verdict=verdict,
            issues=tuple(issues),
        )

    # Topic exists on broker
    part_offsets = offsets_map.get(topic, [])
    partition_count = len(part_offsets)
    total_messages = sum(high - low for _, low, high in part_offsets)

    if total_messages == 0:
        status = EnumTopicStatus.FOUND_EMPTY
    else:
        status = EnumTopicStatus.FOUND_ACTIVE

    # --- Naming ---
    naming = _classify_naming(topic, config.legacy_topics)
    if naming == EnumNamingCompliance.NON_COMPLIANT:
        issues.append(f"Topic name does not comply with ONEX naming: {topic}")

    # --- Envelope stats ---
    envelope_stats: EnvelopeStats | None = None
    raw_samples = samples_map.get(topic, [])
    parsed_messages: list[dict[str, object]] = []
    if raw_samples:
        envelope_stats, parsed_messages = _compute_envelope_stats(raw_samples)

    # --- Domain schema validation ---
    domain_validation: DomainValidationResult | None = None
    schema_model = config.topic_schema_map.get(topic)
    if schema_model is not None and parsed_messages:
        domain_validation = _validate_domain_schema(schema_model, parsed_messages)

    # --- EnumVerdict ---
    verdict = _compute_verdict(
        topic=topic,
        status=status,
        naming=naming,
        envelope_stats=envelope_stats,
        domain_validation=domain_validation,
        config=config,
    )

    return TopicHealth(
        topic=topic,
        status=status,
        partition_count=partition_count,
        total_messages=total_messages,
        naming=naming,
        envelope_stats=envelope_stats,
        domain_validation=domain_validation,
        verdict=verdict,
        issues=tuple(issues),
    )


def _classify_naming(topic: str, legacy_topics: frozenset[str]) -> EnumNamingCompliance:
    """Classify a topic name against ONEX naming convention."""
    result = validate_topic_suffix(topic)
    if result.is_valid:
        return EnumNamingCompliance.COMPLIANT
    if topic in legacy_topics:
        return EnumNamingCompliance.LEGACY
    return EnumNamingCompliance.NON_COMPLIANT


def _compute_envelope_stats(
    raw_samples: list[bytes | None],
) -> tuple[EnvelopeStats, list[dict[str, object]]]:
    """Compute envelope statistics from raw message samples.

    Returns:
        Tuple of (EnvelopeStats, list of successfully parsed JSON dicts).
    """
    total = len(raw_samples)
    tombstone_count = 0
    json_ok = 0
    json_fail = 0
    has_envelope_id = 0
    has_payload = 0
    has_timestamp = 0
    parsed: list[dict[str, object]] = []

    for raw in raw_samples:
        if raw is None or len(raw) == 0:
            tombstone_count += 1
            continue
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            json_fail += 1
            continue

        if not isinstance(obj, dict):
            json_fail += 1
            continue

        json_ok += 1
        parsed.append(obj)

        if "envelope_id" in obj:
            has_envelope_id += 1
        if "payload" in obj:
            has_payload += 1
        if "timestamp" in obj:
            has_timestamp += 1

    stats = EnvelopeStats(
        total_sampled=total,
        tombstone_count=tombstone_count,
        json_parse_ok=json_ok,
        json_parse_fail=json_fail,
        has_envelope_id=has_envelope_id,
        has_payload=has_payload,
        has_timestamp=has_timestamp,
    )
    return stats, parsed


def _validate_domain_schema(
    model: type[BaseModel],
    parsed_messages: list[dict[str, object]],
) -> DomainValidationResult:
    """Validate parsed messages against a Pydantic model.

    Args:
        model: Pydantic model class to validate against.
        parsed_messages: List of JSON-decoded dicts.

    Returns:
        DomainValidationResult with counts and capped error messages.
    """
    valid_count = 0
    invalid_count = 0
    errors: list[str] = []

    for msg in parsed_messages:
        try:
            model.model_validate(msg)
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            if len(errors) < _MAX_SCHEMA_ERRORS:
                err_str = str(e)
                if len(err_str) > _MAX_ERROR_LEN:
                    err_str = err_str[:_MAX_ERROR_LEN] + "..."
                errors.append(err_str)

    return DomainValidationResult(
        model_name=f"{model.__module__}.{model.__qualname__}",
        valid_count=valid_count,
        invalid_count=invalid_count,
        errors=tuple(errors),
    )


def _compute_verdict(
    *,
    topic: str,
    status: EnumTopicStatus,
    naming: EnumNamingCompliance,
    envelope_stats: EnvelopeStats | None,
    domain_validation: DomainValidationResult | None,
    config: AuditConfig,
) -> EnumVerdict:
    """Compute the health verdict for a single topic.

    EnumVerdict rules (evaluated in order, first FAIL/WARN wins):
        FAIL:
            - Expected topic NOT_FOUND
            - Domain schema validation has errors
            - JSON parse failure rate > 10%
        WARN:
            - Expected topic FOUND_EMPTY
            - ONEX naming violation (non-legacy)
            - Envelope missing required fields (non-exempt)
    """
    is_expected = topic in config.expected_topics

    # FAIL conditions
    if status == EnumTopicStatus.NOT_FOUND and is_expected:
        return EnumVerdict.FAIL

    if domain_validation is not None and domain_validation.invalid_count > 0:
        return EnumVerdict.FAIL

    if (
        envelope_stats is not None
        and envelope_stats.json_failure_rate > _JSON_FAILURE_THRESHOLD
    ):
        return EnumVerdict.FAIL

    # WARN conditions
    verdict = EnumVerdict.PASS

    if status == EnumTopicStatus.FOUND_EMPTY and is_expected:
        verdict = EnumVerdict.WARN

    if naming == EnumNamingCompliance.NON_COMPLIANT:
        verdict = EnumVerdict.WARN

    if (
        envelope_stats is not None
        and topic not in config.envelope_exempt_topics
        and envelope_stats.json_parse_ok > 0
    ):
        # Warn if envelope fields are mostly missing
        ok = envelope_stats.json_parse_ok
        if envelope_stats.has_envelope_id < ok or envelope_stats.has_payload < ok:
            verdict = EnumVerdict.WARN

    return verdict


__all__: list[str] = [
    "AuditConfig",
    "run_audit",
]
