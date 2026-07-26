# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Managed-staging canary topic/group catalog generation + zero-collision readback.

OMN-14727 (managed-staging Phase 1, bucket B7).

This module turns the immutable candidate contract into a concrete canary
topic/group catalog on the shared ``omninode-dev-msk`` cluster, and provides the
zero-collision readback that proves the canary namespace is disjoint from every
pre-existing topic and consumer group.

Design
------
The canary shares the physical MSK cluster with 1089 inert topics and every
existing consumer group. To avoid collision **and** stay authorized by the A1
MSK IAM resource patterns (``onex.*`` / ``omninode.*``), the catalog mints every
canary topic and group under a single common prefix that

1. starts with ``onex.`` (or ``omninode.``) so it is authorized by the IAM
   patterns -- **a prefix outside them fails AUTH (AccessDenied), not a
   collision** -- and
2. carries a unique epoch segment (``mstg1``) so the whole namespace is provably
   disjoint from every pre-existing name.

The candidate's application topic **suffixes**
(``onex.<kind>.<producer>.<event>.v<n>``) are contract-owned and are extracted
from the candidate's node contracts via
:class:`~omnibase_infra.topics.contract_topic_extractor.ContractTopicExtractor`.
The prefix is prepended -- exactly mirroring the
:class:`~omnibase_infra.topics.model_bus_descriptor.ModelBusDescriptor`
``namespace_prefix`` mechanism used by the runtime ``TopicResolver`` -- so a
resolved canary topic looks like
``onex.mstg1.onex.evt.platform.node-registration.v1``.

Zero-collision readback
-----------------------
:func:`verify_zero_collision` proves two independent facts against a snapshot of
the cluster's existing topics + consumer groups:

* **exact-name collision** -- no generated canary name equals an existing name;
* **prefix conflict** -- no existing name already lives under the canary prefix.

The catalog + namespace live entirely in-repo, so the readback runs offline.
The **live** readback against the actual 1089 topics + live consumer groups
needs a broker connection from inside the VPC and is therefore deferred to
Phase 3 (apply-to-cluster). Phase 3 fetches the live topic/group lists and calls
:func:`verify_zero_collision` unchanged.

Scope boundary (HARD): this module generates the catalog and runs the readback.
It never connects to or mutates the MSK cluster -- apply-to-cluster is Phase 3.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import yaml

from omnibase_infra.topics.contract_topic_extractor import ContractTopicExtractor
from omnibase_infra.topics.model_canary_catalog import ModelCanaryCatalog
from omnibase_infra.topics.model_canary_namespace import (
    ModelCanaryNamespace,
    iam_pattern_authorizes,
)
from omnibase_infra.topics.model_collision_report import ModelCollisionReport
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from omnibase_infra.utils.util_consumer_group import normalize_kafka_identifier

# Repo root resolved from this module's location:
# src/omnibase_infra/topics/managed_staging_canary_catalog.py -> parents[3].
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Default declarative namespace input consumed by :func:`load_canary_namespace`.
DEFAULT_CANARY_CATALOG_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "managed_staging_canary_catalog_namespace.yaml"
)


def load_canary_namespace(path: Path | None = None) -> ModelCanaryNamespace:
    """Load and validate the declarative canary namespace from YAML.

    Args:
        path: Path to the namespace YAML. Defaults to
            :data:`DEFAULT_CANARY_CATALOG_PATH`.

    Returns:
        A validated :class:`ModelCanaryNamespace` (raises if a prefix is not
        IAM-authorized).
    """
    source = path if path is not None else DEFAULT_CANARY_CATALOG_PATH
    with open(source, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Canary namespace file {source} did not parse to a mapping.")
    return ModelCanaryNamespace.model_validate(raw)


def _resolve_roots(contract_roots: Iterable[str], base_dir: Path | None) -> list[Path]:
    """Resolve repo-relative contract roots against ``base_dir`` (default repo root)."""
    base = base_dir if base_dir is not None else _REPO_ROOT
    return [base / root for root in contract_roots]


def extract_candidate_topic_suffixes(
    contract_roots: Iterable[str],
    *,
    base_dir: Path | None = None,
) -> tuple[str, ...]:
    """Extract the candidate's contract-owned topic suffixes.

    Scans each contract root with
    :class:`~omnibase_infra.topics.contract_topic_extractor.ContractTopicExtractor`
    and returns the sorted union of every declared (subscribe + publish) topic
    suffix.

    Args:
        contract_roots: Repo-relative directories containing ``contract.yaml``
            files (the candidate's node contracts).
        base_dir: Base directory the roots are resolved against. Defaults to the
            repo root inferred from this module's location.

    Returns:
        Sorted, deduplicated tuple of ONEX topic suffixes.
    """
    extractor = ContractTopicExtractor()
    suffixes: set[str] = set()
    for root in _resolve_roots(contract_roots, base_dir):
        if not root.is_dir():
            continue
        manifest = extractor.scan(root)
        suffixes.update(manifest.all_unique_topics)
    return tuple(sorted(suffixes))


def derive_candidate_group_bases(
    contract_roots: Iterable[str],
    *,
    base_dir: Path | None = None,
) -> tuple[str, ...]:
    """Derive canary consumer-group base names from subscribing candidate nodes.

    One ``<node>.consume.v1`` base is emitted per node contract that declares at
    least one subscribe topic (a consumer). Node names are normalized for Kafka
    consumer-group safety. The canary ``group_prefix`` is applied later by
    :func:`generate_canary_catalog`.

    The authoritative group set is pinned when the candidate tuple is frozen
    (B1 / T24); this derivation is the in-repo source used to generate + prove
    the catalog for Phase 1.

    Args:
        contract_roots: Repo-relative directories containing ``contract.yaml``
            files.
        base_dir: Base directory the roots are resolved against.

    Returns:
        Sorted, deduplicated tuple of group base names (prefix not yet applied).
    """
    extractor = ContractTopicExtractor()
    bases: set[str] = set()
    for root in _resolve_roots(contract_roots, base_dir):
        if not root.is_dir():
            continue
        manifest = extractor.scan(root)
        for node in manifest.nodes.values():
            if node.subscribe_topics:
                normalized = normalize_kafka_identifier(node.node_name)
                bases.add(f"{normalized}.consume.v1")
    return tuple(sorted(bases))


def generate_canary_catalog(
    namespace: ModelCanaryNamespace,
    *,
    topic_suffixes: Sequence[str],
    group_bases: Sequence[str],
) -> ModelCanaryCatalog:
    """Generate the concrete canary catalog by prefixing suffixes + group bases.

    Args:
        namespace: The validated canary namespace (carries prefixes + sizing).
        topic_suffixes: Candidate contract-owned topic suffixes.
        group_bases: Canary consumer-group base names (prefix not yet applied).

    Returns:
        A :class:`ModelCanaryCatalog` with deterministic, sorted, deduplicated
        topics (full names) and groups (full names).
    """
    unique_topics = sorted(set(topic_suffixes))
    topics = tuple(
        ModelTopicSpec(
            suffix=f"{namespace.topic_prefix}{suffix}",
            partitions=namespace.default_partitions,
            replication_factor=namespace.default_replication_factor,
        )
        for suffix in unique_topics
    )
    groups = tuple(
        f"{namespace.group_prefix}{base}" for base in sorted(set(group_bases))
    )
    return ModelCanaryCatalog(
        epoch=namespace.epoch,
        topic_prefix=namespace.topic_prefix,
        group_prefix=namespace.group_prefix,
        topics=topics,
        groups=groups,
    )


def build_canary_catalog_from_candidate(
    namespace: ModelCanaryNamespace | None = None,
    *,
    base_dir: Path | None = None,
) -> ModelCanaryCatalog:
    """One-call generation of the canary catalog from the candidate contract.

    Loads the default namespace (unless supplied), extracts the candidate's
    contract-owned topic suffixes + subscribing-node group bases, and generates
    the catalog.

    Args:
        namespace: Optional pre-loaded namespace. Loaded from the default path
            when omitted.
        base_dir: Base directory the candidate contract roots resolve against.

    Returns:
        The generated :class:`ModelCanaryCatalog`.
    """
    ns = namespace if namespace is not None else load_canary_namespace()
    topic_suffixes = extract_candidate_topic_suffixes(
        ns.candidate_contract_roots, base_dir=base_dir
    )
    group_bases = derive_candidate_group_bases(
        ns.candidate_contract_roots, base_dir=base_dir
    )
    return generate_canary_catalog(
        ns, topic_suffixes=topic_suffixes, group_bases=group_bases
    )


def verify_zero_collision(
    catalog: ModelCanaryCatalog,
    *,
    existing_topics: Iterable[str],
    existing_groups: Iterable[str],
) -> ModelCollisionReport:
    """Prove the canary catalog is disjoint from existing topics + groups.

    Checks two independent facts:

    * **exact-name collision** -- no generated canary name equals an existing
      name;
    * **prefix conflict** -- no existing name already lives under the canary
      prefix (which would mean the namespace was not actually fresh).

    This is transport-free: pass a snapshot of the cluster's existing topics and
    consumer groups. Phase 3 supplies the live lists (from an in-VPC broker
    connection) and calls this function unchanged.

    Args:
        catalog: The generated canary catalog to check.
        existing_topics: All topic names currently on the cluster.
        existing_groups: All consumer group names currently on the cluster.

    Returns:
        A :class:`ModelCollisionReport`; ``report.is_clean`` is ``True`` iff the
        canary namespace is fully disjoint.
    """
    existing_topic_set = set(existing_topics)
    existing_group_set = set(existing_groups)
    canary_topics = catalog.topic_names
    canary_groups = catalog.groups

    colliding_topics = tuple(
        sorted(name for name in canary_topics if name in existing_topic_set)
    )
    colliding_groups = tuple(
        sorted(name for name in canary_groups if name in existing_group_set)
    )
    prefix_conflicting_topics = tuple(
        sorted(t for t in existing_topic_set if t.startswith(catalog.topic_prefix))
    )
    prefix_conflicting_groups = tuple(
        sorted(g for g in existing_group_set if g.startswith(catalog.group_prefix))
    )

    return ModelCollisionReport(
        topic_prefix=catalog.topic_prefix,
        group_prefix=catalog.group_prefix,
        checked_topic_count=len(canary_topics),
        checked_group_count=len(canary_groups),
        existing_topic_count=len(existing_topic_set),
        existing_group_count=len(existing_group_set),
        colliding_topics=colliding_topics,
        colliding_groups=colliding_groups,
        prefix_conflicting_topics=prefix_conflicting_topics,
        prefix_conflicting_groups=prefix_conflicting_groups,
    )


__all__: list[str] = [
    "DEFAULT_CANARY_CATALOG_PATH",
    "ModelCanaryCatalog",
    "ModelCanaryNamespace",
    "ModelCollisionReport",
    "build_canary_catalog_from_candidate",
    "derive_candidate_group_bases",
    "extract_candidate_topic_suffixes",
    "generate_canary_catalog",
    "iam_pattern_authorizes",
    "load_canary_namespace",
    "verify_zero_collision",
]
