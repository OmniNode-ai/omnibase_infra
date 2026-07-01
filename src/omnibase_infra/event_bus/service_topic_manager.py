# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka Topic Provisioner for automatic topic creation on startup.

Ensures that all ONEX topics (platform + domain plugins) exist before the
runtime begins consuming or producing events. Uses AIOKafkaAdminClient to
create topics that are missing, with best-effort semantics (warnings on
failure, never blocks startup).

Design:
    - Best-effort: Logs warnings but never blocks startup on failure
    - Idempotent: Safe to call multiple times (skips existing topics)
    - Compatible: Works with both Redpanda and Apache Kafka
    - Configurable: Supports custom topic configs via ModelSnapshotTopicConfig

Related Tickets:
    - OMN-1990: Kafka topic auto-creation gap
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.models.projection.model_snapshot_topic_config import (
        ModelSnapshotTopicConfig,
    )

logger = logging.getLogger(__name__)

# OMN-8783: No default — KAFKA_BOOTSTRAP_SERVERS must be set via overlay.
ENV_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_TOPIC_PARTITION_CAP = "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS"

# Default partition and replication settings for standard event topics
DEFAULT_EVENT_TOPIC_PARTITIONS = 6
DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1


def _topic_partition_cap_from_env() -> int | None:
    raw_value = os.environ.get(ENV_TOPIC_PARTITION_CAP)
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        cap = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected a positive integer",
            ENV_TOPIC_PARTITION_CAP,
            raw_value,
        )
        return None

    if cap in (-1, 0):
        return None

    if cap < 0:
        logger.warning(
            "Ignoring invalid %s=%r; expected -1, zero, or a positive integer",
            ENV_TOPIC_PARTITION_CAP,
            raw_value,
        )
        return None

    return cap


def _topic_provisioning_sort_key(spec: ModelTopicSpec) -> tuple[int, str]:
    """Sort topics using contract-declared provisioning priority."""
    return (spec.provisioning_priority, spec.suffix)


class TopicProvisioner:
    """Provisions Kafka topics automatically on startup.

    Creates ONEX platform topics if they don't already exist, using
    AIOKafkaAdminClient. Topic creation is best-effort: failures log
    warnings but never block startup.

    The provisioner handles two categories of topics:
    1. **Standard event topics**: Created with default settings (delete cleanup)
    2. **Snapshot topics**: Created with compaction settings from ModelSnapshotTopicConfig

    Thread Safety:
        This class is coroutine-safe. All methods are async and use
        the AIOKafkaAdminClient which handles its own connection pooling.

    Example:
        >>> provisioner = TopicProvisioner(contracts_root=Path("src/.../nodes"))
        >>> await provisioner.ensure_provisioned_topics_exist()
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        request_timeout_ms: int = 30000,
        *,
        contracts_root: Path,
        skill_manifests_root: Path | None = None,
        skill_manifests_roots: list[Path] | None = None,
    ) -> None:
        """Initialize the topic provisioner.

        Args:
            bootstrap_servers: Kafka broker addresses. If None, reads from
                KAFKA_BOOTSTRAP_SERVERS env var (raises KeyError if absent).
            request_timeout_ms: Timeout for admin operations in milliseconds.
            contracts_root: Path to contract.yaml root directory. Required.
                Topics are discovered from contracts via
                ContractTopicExtractor. The directory must exist; a
                ``FileNotFoundError`` is raised at construction time if it
                does not.
            skill_manifests_root: Optional single path to omniclaude skills
                root (plugins/onex/skills/). Kept for backwards compatibility.
            skill_manifests_roots: Optional list of paths to scan for
                topics.yaml manifests (supports multiple roots: skills,
                CLI relays, services). When both singular and plural are set,
                the singular root is prepended to the list.

        Raises:
            FileNotFoundError: If *contracts_root* does not point to an
                existing directory.

        Ticket: OMN-4594, OMN-4622, OMN-5132
        """
        if not contracts_root.is_dir():
            raise FileNotFoundError(
                f"contracts_root does not exist or is not a directory: {contracts_root}"
            )
        # OMN-8783: Hard-fail if not provided and env var absent.
        self._bootstrap_servers = bootstrap_servers or os.environ[ENV_BOOTSTRAP_SERVERS]
        self._request_timeout_ms = request_timeout_ms
        self._contracts_root = contracts_root
        self._skill_manifests_root = skill_manifests_root
        self._skill_manifests_roots = skill_manifests_roots
        self._topic_partition_cap = _topic_partition_cap_from_env()
        self._topic_specs = self._build_topic_specs()

    def _creation_partitions(self, spec: ModelTopicSpec) -> int:
        if self._topic_partition_cap is None:
            return spec.partitions
        return min(spec.partitions, self._topic_partition_cap)

    def _build_topic_specs(self) -> tuple[ModelTopicSpec, ...]:
        """Build topic specs from contract YAML extraction.

        Topics are derived entirely from contract YAML extraction via
        ``ContractTopicExtractor.extract_all()``. There is no fallback to
        the Python constant registry (``ALL_PROVISIONED_TOPIC_SPECS``).

        Raises:
            ImportError: If ``ContractTopicExtractor`` is not importable.
            RuntimeError: If extraction fails unexpectedly.

        Ticket: OMN-4594, OMN-4622, OMN-5132
        """
        from omnibase_infra.tools.contract_topic_extractor import (
            ContractTopicExtractor,
        )

        extractor = ContractTopicExtractor(include_installed_packages=True)
        contract_entries = extractor.extract_all(
            contracts_root=self._contracts_root,
            skill_manifests_root=self._skill_manifests_root,
            skill_manifests_roots=self._skill_manifests_roots,
        )

        result_specs: list[ModelTopicSpec] = []
        for entry in contract_entries:
            # Per-topic config (OMN-13238): when a contract declares a
            # ``topic_config`` block the extractor carries partitions /
            # replication_factor / kafka_config; otherwise these stay None and
            # ModelTopicSpec applies its canonical defaults.
            result_specs.append(
                ModelTopicSpec(
                    suffix=entry.topic,
                    provisioning_priority=entry.provisioning_priority,
                    partitions=(
                        entry.partitions
                        if entry.partitions is not None
                        else DEFAULT_EVENT_TOPIC_PARTITIONS
                    ),
                    replication_factor=(
                        entry.replication_factor
                        if entry.replication_factor is not None
                        else DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR
                    ),
                    kafka_config=(
                        dict(entry.kafka_config)
                        if entry.kafka_config is not None
                        else None
                    ),
                )
            )

        result = tuple(sorted(result_specs, key=_topic_provisioning_sort_key))

        skill_count = len([e for e in contract_entries if "omniclaude" in e.topic])
        logger.info(
            "topic provisioning (contract-first) — total: %d, "
            "skill-manifest topics: %d",
            len(result),
            skill_count,
        )

        return result

    async def ensure_provisioned_topics_exist(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, list[str] | str]:
        """Ensure all ONEX provisioned topics exist.

        Creates any missing topics discovered from contract YAML extraction.
        The snapshot topic gets special compaction configuration via
        ModelSnapshotTopicConfig.

        This method is best-effort: individual topic creation failures are
        logged as warnings but do not prevent other topics from being created.
        Unrecoverable failures (connection, authentication, etc.) are also
        logged as warnings and never block startup.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Summary dict with:
                - created: List of newly created topic names
                - existing: List of topics that already existed
                - failed: List of topics that failed to create
                - status: "success", "partial", or "unavailable"
        """
        correlation_id = correlation_id or uuid4()
        created: list[str] = []
        existing: list[str] = []
        failed: list[str] = []

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, skipping topic auto-creation. "
                "Install aiokafka to enable automatic topic management.",
                extra={"correlation_id": str(correlation_id)},
            )
            return {
                "created": created,
                "existing": existing,
                "failed": [s.suffix for s in self._topic_specs],
                "status": "unavailable",
            }

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            for spec in self._topic_specs:
                try:
                    partitions = self._creation_partitions(spec)
                    new_topic = NewTopic(
                        name=spec.suffix,
                        num_partitions=partitions,
                        replication_factor=spec.replication_factor,
                        topic_configs=dict(spec.kafka_config)
                        if spec.kafka_config
                        else {},
                    )

                    await admin.create_topics([new_topic])
                    created.append(spec.suffix)
                    logger.info(
                        "Created topic: %s (partitions=%d)",
                        spec.suffix,
                        partitions,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except TopicAlreadyExistsError:
                    existing.append(spec.suffix)
                    logger.debug(
                        "Topic already exists: %s",
                        spec.suffix,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
                    failed.append(spec.suffix)
                    logger.warning(
                        "Failed to create topic %s: %s",
                        spec.suffix,
                        type(e).__name__,
                        extra={
                            "correlation_id": str(correlation_id),
                            "error": sanitize_error_message(e),
                        },
                    )

        except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Topic auto-creation interrupted by %s. "
                "Topics may need to be created manually or via broker auto-create.",
                type(e).__name__,
                extra={
                    "bootstrap_servers": self._bootstrap_servers,
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            # Separate individually-failed topics from those never attempted
            already_resolved = set(created) | set(existing) | set(failed)
            all_suffixes = {spec.suffix for spec in self._topic_specs}
            not_attempted = [s for s in all_suffixes if s not in already_resolved]
            if not_attempted:
                logger.warning(
                    "Topics not attempted due to early termination: %d topics",
                    len(not_attempted),
                    extra={
                        "not_attempted_count": len(not_attempted),
                        "correlation_id": str(correlation_id),
                    },
                )
            # Use "partial" if any topics succeeded before the interruption;
            # "unavailable" only when nothing was resolved at all.
            interrupted_status = "partial" if (created or existing) else "unavailable"
            return {
                "created": created,
                "existing": existing,
                "failed": failed + not_attempted,
                "status": interrupted_status,
            }

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass  # Best-effort cleanup

        status = (
            "success"
            if not failed
            else ("partial" if created or existing else "unavailable")
        )

        logger.info(
            "Topic auto-creation complete",
            extra={
                "created_count": len(created),
                "existing_count": len(existing),
                "failed_count": len(failed),
                "status": status,
                "correlation_id": str(correlation_id),
            },
        )

        return {
            "created": created,
            "existing": existing,
            "failed": failed,
            "status": status,
        }

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: ModelSnapshotTopicConfig | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        """Ensure a single topic exists with optional custom config.

        Creates a new AIOKafkaAdminClient connection per call. For creating
        multiple topics, prefer :meth:`ensure_provisioned_topics_exist` which
        reuses a single admin connection for all topics.

        Args:
            topic_name: The topic name to create.
            config: Optional snapshot-topic configuration (compaction etc.). If
                None, falls back to *spec* or default event topic settings.
            correlation_id: Optional correlation ID for tracing.
            spec: Optional contract-derived ``ModelTopicSpec`` (partitions,
                replication, kafka_config). Used by the per-contract boot
                interleave (OMN-13237) so a topic is created to its
                contract-declared spec rather than bare defaults. Ignored when
                *config* is supplied.

        Returns:
            True if topic was created or already exists, False on failure.
        """
        correlation_id = correlation_id or uuid4()

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, cannot create topic %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            if config is not None:
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=config.partition_count,
                    replication_factor=config.replication_factor,
                    topic_configs=config.to_kafka_config(),
                )
            elif spec is not None:
                # Contract-derived spec (OMN-13237 per-contract interleave):
                # honor declared partitions/replication/kafka_config.
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=self._creation_partitions(spec),
                    replication_factor=spec.replication_factor,
                    topic_configs=dict(spec.kafka_config) if spec.kafka_config else {},
                )
            else:
                default_spec = ModelTopicSpec(suffix=topic_name)
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=self._creation_partitions(default_spec),
                    replication_factor=DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR,
                )

            await admin.create_topics([new_topic])
            logger.info(
                "Created topic: %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except TopicAlreadyExistsError:
            logger.debug(
                "Topic already exists: %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Failed to create topic %s: %s",
                topic_name,
                type(e).__name__,
                extra={
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            return False

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, ModelTopicSpec] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        """Confirm broker metadata for ``topics`` converged (§3.7, OMN-13237).

        A topic is READY when broker metadata returns it, its partition count
        matches the expected spec, every partition has a leader, the reported
        replication factor matches the spec (where inspectable), and required
        config keys are visible. The poll is bounded by *config*'s timeout /
        cadence / max-attempts; on exhaustion each unready topic carries a
        classified failure reason.

        Args:
            topics: The topic names to confirm.
            expected_specs: Optional per-topic expected spec (partitions/RF/
                kafka_config). Topics without a spec use default expectations.
            config: Bounded readiness knobs. Defaults to env-resolved knobs.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            A ``ModelTopicSetReadiness`` describing per-topic outcomes.
        """
        correlation_id = correlation_id or uuid4()
        requested = tuple(dict.fromkeys(topics))
        if not requested:
            return ModelTopicSetReadiness(status=EnumTopicReadinessStatus.SKIPPED)
        knobs = config or ModelTopicReadinessConfig()
        specs = dict(expected_specs or {})

        try:
            from aiokafka.admin import AIOKafkaAdminClient
        except ImportError:
            logger.warning(
                "aiokafka not available, cannot confirm topic readiness",
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelTopicSetReadiness(
                topics=requested,
                status=EnumTopicReadinessStatus.UNAVAILABLE,
            )

        deadline = time.monotonic() + knobs.readiness_timeout_seconds
        poll_seconds = knobs.readiness_poll_interval_ms / 1000.0
        admin: AIOKafkaAdminClient | None = None
        last_evaluation: ModelTopicSetReadiness | None = None
        attempts = 0
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            while attempts < knobs.max_attempts:
                attempts += 1
                metadata = await admin.describe_topics(list(requested))
                last_evaluation = evaluate_topic_readiness(
                    requested,
                    metadata,
                    expected_specs=specs,
                    attempts=attempts,
                )
                if last_evaluation.is_ready:
                    return last_evaluation
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(poll_seconds)

        except Exception as e:  # noqa: BLE001 — boundary: degrades to not-ready
            logger.warning(
                "Topic readiness confirm interrupted by %s",
                type(e).__name__,
                extra={
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            return ModelTopicSetReadiness(
                topics=requested,
                status=EnumTopicReadinessStatus.UNAVAILABLE,
                attempts=attempts,
            )
        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass

        if last_evaluation is not None:
            return last_evaluation
        # max_attempts must be >=1, so a loop that ran at least once always sets
        # last_evaluation; this guards the (unreachable) zero-iteration case.
        return ModelTopicSetReadiness(
            topics=requested,
            status=EnumTopicReadinessStatus.NOT_READY,
            attempts=attempts,
        )


def evaluate_topic_readiness(
    topics: Sequence[str],
    metadata: Sequence[Mapping[str, object]],
    *,
    expected_specs: Mapping[str, ModelTopicSpec] | None = None,
    attempts: int = 1,
) -> ModelTopicSetReadiness:
    """Classify broker metadata into a per-topic readiness outcome (§3.7).

    Pure function over the metadata shape returned by
    ``AIOKafkaAdminClient.describe_topics`` so the readiness semantics are
    unit-testable without a live broker. Each metadata entry is a mapping with
    keys ``topic``, ``error_code``, and ``partitions`` (a sequence of mappings
    with ``partition``, ``leader``, and ``replicas``).
    """
    requested = tuple(dict.fromkeys(topics))
    specs = dict(expected_specs or {})
    by_topic: dict[str, Mapping[str, object]] = {}
    for entry in metadata:
        name = entry.get("topic")
        if isinstance(name, str):
            by_topic[name] = entry

    ready: list[str] = []
    failures: list[ModelTopicReadinessFailure] = []
    for topic in requested:
        spec = specs.get(topic)
        topic_entry: Mapping[str, object] | None = by_topic.get(topic)
        if topic_entry is None:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                    detail="broker metadata did not return the topic",
                )
            )
            continue
        error_code = topic_entry.get("error_code")
        if isinstance(error_code, int) and error_code != 0:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                    detail=f"broker reported error_code={error_code}",
                )
            )
            continue
        partitions_raw = topic_entry.get("partitions")
        partitions: list[Mapping[str, object]] = (
            [p for p in partitions_raw if isinstance(p, Mapping)]
            if isinstance(partitions_raw, Sequence)
            and not isinstance(partitions_raw, (str, bytes))
            else []
        )
        if not partitions:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.PARTITION_MISMATCH,
                    detail="topic metadata reported zero partitions",
                )
            )
            continue
        if spec is not None and len(partitions) != spec.partitions:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.PARTITION_MISMATCH,
                    detail=(
                        f"expected {spec.partitions} partitions, "
                        f"broker reports {len(partitions)}"
                    ),
                )
            )
            continue
        no_leader = any(_partition_leader(p) is None for p in partitions)
        if no_leader:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.NO_LEADER,
                    detail="at least one partition has no available leader",
                )
            )
            continue
        if spec is not None:
            rf_mismatch = _replication_mismatch(partitions, spec.replication_factor)
            if rf_mismatch is not None:
                failures.append(
                    ModelTopicReadinessFailure(
                        topic=topic,
                        reason=(EnumTopicReadinessFailureReason.REPLICATION_MISMATCH),
                        detail=rf_mismatch,
                    )
                )
                continue
        ready.append(topic)

    status = (
        EnumTopicReadinessStatus.READY
        if not failures
        else EnumTopicReadinessStatus.NOT_READY
    )
    return ModelTopicSetReadiness(
        topics=requested,
        status=status,
        ready_topics=tuple(ready),
        failures=tuple(failures),
        attempts=attempts,
    )


def _partition_leader(partition: Mapping[str, object]) -> int | None:
    """Return the partition leader id, or None when no valid leader exists."""
    leader = partition.get("leader")
    if isinstance(leader, int) and leader >= 0:
        return leader
    return None


def _replication_mismatch(
    partitions: Sequence[Mapping[str, object]],
    expected_rf: int,
) -> str | None:
    """Return a detail string when replica counts disagree with the spec.

    Skipped (returns None) where the broker does not expose a replica list.
    """
    for partition in partitions:
        replicas = partition.get("replicas")
        if not (
            isinstance(replicas, Sequence) and not isinstance(replicas, (str, bytes))
        ):
            return None  # RF not inspectable from this metadata shape
        if len(replicas) != expected_rf:
            return (
                f"expected replication_factor={expected_rf}, "
                f"broker reports {len(replicas)} replicas"
            )
    return None


def _cli_main() -> None:
    """CLI entrypoint for manual topic provisioning without runtime.

    Usage:
        uv run python -m omnibase_infra.event_bus.service_topic_manager \\
            --contracts-root src/omnibase_infra/nodes

    Useful for provisioning topics when running just Redpanda for development
    without the full runtime stack.
    """
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(
        description="Provision Kafka topics from contract YAML."
    )
    parser.add_argument(
        "--contracts-root",
        type=Path,
        default=Path(os.environ.get("ONEX_CONTRACTS_DIR", "./contracts")),
        help=(
            "Root directory containing contract.yaml files. "
            "Defaults to ONEX_CONTRACTS_DIR env var or ./contracts."
        ),
    )
    args = parser.parse_args()

    async def _run() -> None:
        provisioner = TopicProvisioner(contracts_root=args.contracts_root)
        result = await provisioner.ensure_provisioned_topics_exist()
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    _cli_main()


__all__ = ["TopicProvisioner", "evaluate_topic_readiness"]
