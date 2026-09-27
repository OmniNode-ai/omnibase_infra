# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure admission checks for a redacted, sim-only archive selection manifest."""

from __future__ import annotations

from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)
from omnibase_infra.runtime.models.model_sim_archive_freshness_observation import (
    ModelSimArchiveFreshnessObservation,
)
from omnibase_infra.runtime.models.model_sim_archive_selected_record import (
    ModelSimArchiveSelectedRecord,
)
from omnibase_infra.runtime.models.model_sim_archive_selection_manifest import (
    ModelSimArchiveSelectionManifest,
)


def validate_sim_archive_selection(
    manifest: ModelSimArchiveSelectionManifest,
    *,
    topology: PinnedExecutionGraphTopology,
    current: ModelSimArchiveFreshnessObservation,
) -> None:
    """Structurally reject stale or malformed selections; this does not authenticate them.

    The manifest and freshness observation are caller-constructible redacted
    claims.  They are deliberately insufficient enqueue authority.  An
    unwired, trusted source-receipt adapter must compare the selected raw Kafka
    key, value, *ordered raw header sequence*, and broker timestamp to its
    receipt before this structural check and any outbox enqueue.

    ``event_ledger`` is only a secondary witness for source coordinate and
    value digest.  Its normalized/enriched headers must never be compared to
    the broker ordered-header digest.
    """

    if type(manifest) is not ModelSimArchiveSelectionManifest:
        raise TypeError("sim archive selection requires a typed manifest")
    if type(topology) is not PinnedExecutionGraphTopology:
        raise TypeError("sim archive selection requires pinned topology")
    if type(current) is not ModelSimArchiveFreshnessObservation:
        raise TypeError("sim archive selection requires typed freshness observation")
    if manifest.topology_sha256 != topology.version.topology_sha256:
        raise ValueError("selection topology does not match pinned topology")
    if (
        current.exact_owner_rows != 1
        or current.head_count != 1
        or current.foreign_explicit_tenant_count != 0
        or current.conflicting_envelope_count != 0
    ):
        raise ValueError("source ownership scan is not admissible")
    if (
        manifest.owner_binding_sha256 != current.owner_binding_sha256
        or manifest.source_scan_sha256 != current.source_scan_sha256
        or manifest.head_envelope_sha256 != current.head_envelope_sha256
    ):
        raise ValueError("selection is expired against the current source scan")

    declared = topology.declared_chain
    expected_topics = {hop.topic for hop in declared}
    selected_topics = {record.source_topic for record in manifest.selected_records}
    if selected_topics != expected_topics or len(manifest.selected_records) != len(
        declared
    ):
        raise ValueError("selection does not contain exactly the declared chain topics")
    head_topic = topology.read_set.head_topic
    head = next(
        record
        for record in manifest.selected_records
        if record.envelope_sha256 == manifest.head_envelope_sha256
    )
    if head.source_topic != head_topic:
        raise ValueError("selection head is not on the declared head topic")

    by_topic = {record.source_topic: record for record in manifest.selected_records}
    for hop in declared:
        record = by_topic[hop.topic]
        if record.broker_value_sha256 != record.ledger_value_sha256:
            raise ValueError("broker capture and ledger payload digest differ")
        if hop.parent is None:
            continue
        parent = by_topic.get(hop.parent)
        if parent is None or record.parent_envelope_sha256 != parent.envelope_sha256:
            raise ValueError("selection parent edge conflicts with declared topology")


__all__ = [
    "ModelSimArchiveFreshnessObservation",
    "ModelSimArchiveSelectedRecord",
    "ModelSimArchiveSelectionManifest",
    "validate_sim_archive_selection",
]
