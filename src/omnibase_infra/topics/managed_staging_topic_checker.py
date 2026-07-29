# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Managed-staging (mstg1) topic/group checker against a live broker (OMN-15283).

This is the IAM-capable broker admin surface's counterpart to
``managed_staging_canary_catalog.verify_zero_collision``: instead of proving
the canary namespace is disjoint from *pre-existing* names, this module proves
the canary namespace is (or is not yet) fully *provisioned* on a live broker.

Design
------
1. The catalog is generated the same way every other Phase-1 canary consumer
   generates it -- :func:`~omnibase_infra.topics.managed_staging_canary_catalog.build_canary_catalog_from_candidate`.
   This module never re-implements prefixing or candidate topic extraction.
2. The broker connection uses the exact same admin-client construction as
   ``TopicProvisioner`` (``AIOKafkaAdminClient`` +
   ``build_aiokafka_auth_kwargs_from_env``) so MSK IAM auth behaves identically
   to the real provisioning path.
3. :func:`build_topic_diff` is pure and transport-free (mirrors
   ``verify_zero_collision``'s shape) -- it is testable without a live broker.
4. ``--create-missing`` is opt-in and, even when enabled, only ever creates
   catalog-listed, prefix-scoped topics for names found missing by the diff --
   never a universe sweep, never an out-of-catalog name.

Scope boundary (HARD): this module checks/diffs/creates only the
managed-staging canary namespace on whatever broker it is pointed at. Live
execution against the actual MSK cluster is the AWS lane's step, out of CI
scope here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.topics.managed_staging_canary_catalog import (
    build_canary_catalog_from_candidate,
    load_canary_namespace,
)
from omnibase_infra.topics.model_canary_catalog import ModelCanaryCatalog
from omnibase_infra.topics.model_managed_staging_topic_diff import (
    ModelManagedStagingTopicDiff,
)
from omnibase_infra.topics.model_topic_provisioning_diff import (
    build_provisioning_diff,
)
from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
)

if TYPE_CHECKING:
    from aiokafka.admin import AIOKafkaAdminClient

logger = logging.getLogger(__name__)


def build_topic_diff(
    catalog: ModelCanaryCatalog,
    *,
    existing_topics: Iterable[str],
    existing_groups: Iterable[str],
) -> ModelManagedStagingTopicDiff:
    """Diff a generated canary catalog against a broker snapshot.

    Transport-free and pure -- mirrors
    ``managed_staging_canary_catalog.verify_zero_collision``'s shape so it is
    unit-testable without a live broker.

    Args:
        catalog: The generated canary catalog (the desired state).
        existing_topics: All topic names currently on the broker.
        existing_groups: All consumer group names currently on the broker.

    Returns:
        A :class:`ModelManagedStagingTopicDiff` classifying every catalog name
        as missing/present, plus any out-of-catalog prefix-matching stray name.
    """
    existing_topic_set = set(existing_topics)
    existing_group_set = set(existing_groups)
    catalog_topic_set = set(catalog.topic_names)
    catalog_group_set = set(catalog.groups)

    # OMN-15395: the missing/present split is the SHARED provisioning diff every
    # creation path runs — one diff engine, not a canary-only copy. This module
    # keeps only what is genuinely canary-specific (prefix scoping + consumer
    # groups).
    topic_diff = build_provisioning_diff(sorted(catalog_topic_set), existing_topic_set)
    missing_topics = topic_diff.missing_topics
    present_topics = topic_diff.present_topics
    out_of_catalog_topics = tuple(
        sorted(
            name
            for name in existing_topic_set
            if name.startswith(catalog.topic_prefix) and name not in catalog_topic_set
        )
    )

    missing_groups = tuple(sorted(catalog_group_set - existing_group_set))
    present_groups = tuple(sorted(catalog_group_set & existing_group_set))
    out_of_catalog_groups = tuple(
        sorted(
            name
            for name in existing_group_set
            if name.startswith(catalog.group_prefix) and name not in catalog_group_set
        )
    )

    return ModelManagedStagingTopicDiff(
        topic_prefix=catalog.topic_prefix,
        group_prefix=catalog.group_prefix,
        missing_topics=missing_topics,
        present_topics=present_topics,
        out_of_catalog_topics=out_of_catalog_topics,
        missing_groups=missing_groups,
        present_groups=present_groups,
        out_of_catalog_groups=out_of_catalog_groups,
    )


async def open_admin_client(
    *,
    bootstrap_servers: str,
    request_timeout_ms: int = 30000,
) -> AIOKafkaAdminClient:
    """Open + start an ``AIOKafkaAdminClient`` using the shared MSK IAM auth path.

    Mirrors ``TopicProvisioner``'s admin-client construction exactly (same
    ``build_aiokafka_auth_kwargs_from_env`` call) so this checker authenticates
    identically to the real provisioning path. Caller owns ``close()``.
    """
    from aiokafka.admin import AIOKafkaAdminClient

    auth_kwargs = build_aiokafka_auth_kwargs_from_env()
    admin = AIOKafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=request_timeout_ms,
        **auth_kwargs,
    )
    await admin.start()
    return admin


async def fetch_live_topics_and_groups(
    admin: AIOKafkaAdminClient,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """List live topics + consumer group names from an open admin client.

    Args:
        admin: An already-started ``AIOKafkaAdminClient``.

    Returns:
        ``(topic_names, group_names)``.
    """
    topics = await admin.list_topics()
    groups_raw = await admin.list_consumer_groups()
    # aiokafka returns a list of tuples; the first element is the group id.
    groups = tuple(sorted({entry[0] for entry in groups_raw}))
    return tuple(sorted(topics)), groups


async def create_missing_catalog_topics(
    admin: AIOKafkaAdminClient,
    catalog: ModelCanaryCatalog,
    diff: ModelManagedStagingTopicDiff,
    *,
    policy: ModelTopicProvisioningPolicy | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create only catalog-listed, currently-missing topics (per-contract scoped).

    Never touches an out-of-catalog name and never sweeps the broker's full
    topic universe -- only the names present in ``diff.missing_topics``, which
    are guaranteed (by construction of :func:`build_topic_diff`) to be a
    subset of the catalog's own topic set.

    Every spec is resolved through the environment replication policy BEFORE the
    first ``CreateTopics`` (OMN-15395), so this path cannot mint an RF1 topic on
    the managed cluster even if a catalog entry regressed to one.

    Args:
        admin: An already-started ``AIOKafkaAdminClient``.
        catalog: The generated canary catalog (source of namespace defaults --
            partitions/replication -- per topic).
        diff: The diff previously computed by :func:`build_topic_diff`.
        policy: Replication policy. Defaults to the policy derived from the live
            Kafka client configuration.

    Returns:
        ``(created, failed)`` topic name tuples.

    Raises:
        TopicReplicationPolicyError: A catalog spec violates the policy. Raised
            before any ``CreateTopics``.
    """
    from aiokafka.admin import NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    resolved_policy = policy or ModelTopicProvisioningPolicy.from_env()
    specs_by_name = {spec.suffix: spec for spec in catalog.topics}
    created: list[str] = []
    failed: list[str] = []

    # Fail closed ahead of the create loop: a durability violation anywhere in
    # the batch means nothing is created.
    resolved_by_name = {
        name: resolved_policy.resolve_spec(spec)
        for name in diff.missing_topics
        if (spec := specs_by_name.get(name)) is not None
    }

    for name in diff.missing_topics:
        spec = resolved_by_name.get(name)
        if spec is None:
            # Defensive: build_topic_diff guarantees missing_topics subset of
            # catalog.topic_names, so this branch should be unreachable.
            failed.append(name)
            continue
        new_topic = NewTopic(
            name=spec.suffix,
            num_partitions=spec.partitions,
            replication_factor=spec.replication_factor,
            topic_configs=dict(spec.kafka_config) if spec.kafka_config else {},
        )
        try:
            await admin.create_topics([new_topic])
            created.append(name)
        except TopicAlreadyExistsError:
            created.append(name)
        except Exception:  # noqa: BLE001 — boundary: report, do not raise
            logger.warning("Failed to create catalog topic %s", name)
            failed.append(name)

    return tuple(created), tuple(failed)


def _render_report(
    diff: ModelManagedStagingTopicDiff,
    *,
    created: Sequence[str] = (),
    failed: Sequence[str] = (),
) -> str:
    payload: dict[str, object] = {
        "topic_prefix": diff.topic_prefix,
        "group_prefix": diff.group_prefix,
        "missing_topics": list(diff.missing_topics),
        "present_topics": list(diff.present_topics),
        "out_of_catalog_topics": list(diff.out_of_catalog_topics),
        "missing_groups": list(diff.missing_groups),
        "present_groups": list(diff.present_groups),
        "out_of_catalog_groups": list(diff.out_of_catalog_groups),
        "is_fully_present": diff.is_fully_present,
        "created_topics": list(created),
        "failed_topics": list(failed),
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the managed-staging (mstg1) canary topic/group catalog "
            "against a live broker. Check-only by default; --create-missing "
            "opts into creating catalog-listed missing topics."
        )
    )
    parser.add_argument(
        "--namespace-path",
        type=Path,
        default=None,
        help=(
            "Path to the canary namespace YAML. Defaults to "
            "managed_staging_canary_catalog.DEFAULT_CANARY_CATALOG_PATH."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory candidate_contract_roots resolve against. "
            "Defaults to the repo root."
        ),
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=None,
        help=(
            "Kafka bootstrap servers. Defaults to the same resolution "
            "ModelKafkaEventBusConfig.default() uses (KAFKA_BOOTSTRAP_SERVERS "
            "env var)."
        ),
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        default=False,
        help=(
            "Create catalog-listed missing topics using the namespace's "
            "declared partition/replication defaults. Default is CHECK-ONLY "
            "(zero mutations)."
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    namespace = load_canary_namespace(args.namespace_path)
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=args.base_dir)

    # Resolve bootstrap servers through the same boundary EventBusKafka/
    # TopicProvisioner use (ModelKafkaEventBusConfig.default() reads
    # KAFKA_BOOTSTRAP_SERVERS via its own, already-declared env resolution) --
    # this module never reads os.environ directly.
    bootstrap_servers = (
        args.bootstrap_servers or ModelKafkaEventBusConfig.default().bootstrap_servers
    )
    admin = await open_admin_client(bootstrap_servers=bootstrap_servers)
    try:
        existing_topics, existing_groups = await fetch_live_topics_and_groups(admin)
        diff = build_topic_diff(
            catalog,
            existing_topics=existing_topics,
            existing_groups=existing_groups,
        )

        created: tuple[str, ...] = ()
        failed: tuple[str, ...] = ()
        if args.create_missing and diff.missing_topics:
            created, failed = await create_missing_catalog_topics(admin, catalog, diff)
    finally:
        await admin.close()

    print(_render_report(diff, created=created, failed=failed))  # noqa: T201

    if failed:
        return 1
    if args.create_missing:
        # After an opt-in create pass, success means every previously-missing
        # topic is now accounted for (created or already existing).
        return 0
    # Fail-closed gate semantics: required *topics* missing is a hard failure.
    # Consumer groups are lazily created by their consumers and are not
    # required to pre-exist, so they never fail the gate on their own.
    return 0 if not diff.missing_topics else 1


def _cli_main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint.

    Usage:
        uv run python -m omnibase_infra.topics.managed_staging_topic_checker \\
            [--create-missing]

    Exits nonzero when required catalog topics/groups are missing (check-only
    mode) or when a create-missing pass fails to create a catalog topic --
    fail-closed for use as a CI/ops gate.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    _cli_main()


__all__: list[str] = [
    "build_topic_diff",
    "create_missing_catalog_topics",
    "fetch_live_topics_and_groups",
    "open_admin_client",
]
