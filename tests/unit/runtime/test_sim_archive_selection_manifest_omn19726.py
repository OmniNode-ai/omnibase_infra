# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure redacted manifest admission tests for OMN-19726."""

from __future__ import annotations

from copy import deepcopy

import pytest

from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.sim_archive_selection_manifest import (
    ModelSimArchiveFreshnessObservation,
    ModelSimArchiveSelectedRecord,
    ModelSimArchiveSelectionManifest,
    validate_sim_archive_selection,
)

_TOPOLOGY_SHA = "0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3"
_DIGESTS = tuple(f"{index:064x}" for index in range(1, 16))


def _topology():
    return PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256=_TOPOLOGY_SHA,
        )
    )


def _manifest() -> ModelSimArchiveSelectionManifest:
    topology = _topology()
    head, request, routing, decision, terminal = _DIGESTS[:5]
    records = (
        ModelSimArchiveSelectedRecord(
            source_topic=topology.declared_chain[0].topic,
            source_partition=0,
            source_offset=1,
            envelope_sha256=head,
            broker_key_sha256=None,
            broker_value_sha256=_DIGESTS[5],
            broker_headers_sha256=_DIGESTS[6],
            broker_timestamp_ms=1_001,
            ledger_value_sha256=_DIGESTS[5],
            ledger_headers_normalized=True,
        ),
        ModelSimArchiveSelectedRecord(
            source_topic=topology.declared_chain[1].topic,
            source_partition=0,
            source_offset=2,
            envelope_sha256=request,
            parent_envelope_sha256=head,
            broker_key_sha256=_DIGESTS[6],
            broker_value_sha256=_DIGESTS[7],
            broker_headers_sha256=_DIGESTS[8],
            broker_timestamp_ms=1_002,
            ledger_value_sha256=_DIGESTS[7],
            ledger_headers_normalized=True,
        ),
        ModelSimArchiveSelectedRecord(
            source_topic=topology.declared_chain[2].topic,
            source_partition=0,
            source_offset=3,
            envelope_sha256=routing,
            parent_envelope_sha256=request,
            broker_key_sha256=_DIGESTS[8],
            broker_value_sha256=_DIGESTS[9],
            broker_headers_sha256=_DIGESTS[10],
            broker_timestamp_ms=1_003,
            ledger_value_sha256=_DIGESTS[9],
            ledger_headers_normalized=True,
        ),
        ModelSimArchiveSelectedRecord(
            source_topic=topology.declared_chain[3].topic,
            source_partition=0,
            source_offset=4,
            envelope_sha256=decision,
            parent_envelope_sha256=routing,
            broker_key_sha256=_DIGESTS[10],
            broker_value_sha256=_DIGESTS[11],
            broker_headers_sha256=_DIGESTS[12],
            broker_timestamp_ms=1_004,
            ledger_value_sha256=_DIGESTS[11],
            ledger_headers_normalized=True,
        ),
        ModelSimArchiveSelectedRecord(
            source_topic=topology.declared_chain[4].topic,
            source_partition=0,
            source_offset=5,
            envelope_sha256=terminal,
            parent_envelope_sha256=head,
            broker_key_sha256=_DIGESTS[12],
            broker_value_sha256=_DIGESTS[13],
            broker_headers_sha256=_DIGESTS[14],
            broker_timestamp_ms=1_005,
            ledger_value_sha256=_DIGESTS[13],
            ledger_headers_normalized=True,
        ),
    )
    return ModelSimArchiveSelectionManifest(
        runtime_lane="sim-202",
        correlation_sha256="a" * 64,
        owner_binding_sha256="b" * 64,
        source_scan_sha256="c" * 64,
        head_envelope_sha256=head,
        topology_sha256=_TOPOLOGY_SHA,
        selected_records=records,
        withheld_envelope_sha256=("d" * 64, "e" * 64),
    )


def _current() -> ModelSimArchiveFreshnessObservation:
    manifest = _manifest()
    return ModelSimArchiveFreshnessObservation(
        owner_binding_sha256=manifest.owner_binding_sha256,
        source_scan_sha256=manifest.source_scan_sha256,
        head_envelope_sha256=manifest.head_envelope_sha256,
        exact_owner_rows=1,
        head_count=1,
        foreign_explicit_tenant_count=0,
        conflicting_envelope_count=0,
    )


@pytest.mark.unit
def test_selection_admits_the_declared_five_hop_tree_and_withholds_unlinked_rows() -> (
    None
):
    manifest = _manifest()

    validate_sim_archive_selection(manifest, topology=_topology(), current=_current())

    assert len(manifest.selected_records) == 5
    assert len(manifest.withheld_envelope_sha256) == 2
    assert all(record.ledger_headers_normalized for record in manifest.selected_records)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exact_owner_rows", 0, "ownership"),
        ("head_count", 2, "ownership"),
        ("foreign_explicit_tenant_count", 1, "ownership"),
        ("conflicting_envelope_count", 1, "ownership"),
        ("source_scan_sha256", "f" * 64, "expired"),
    ],
)
def test_selection_refuses_noncurrent_or_ambiguous_source_observation(
    field: str, value: int | str, message: str
) -> None:
    current = _current().model_copy(update={field: value})

    with pytest.raises(ValueError, match=message):
        validate_sim_archive_selection(
            _manifest(), topology=_topology(), current=current
        )


@pytest.mark.unit
def test_selection_refuses_bad_parent_edge_and_payload_digest() -> None:
    manifest = _manifest()
    raw = manifest.model_dump()
    records = deepcopy(raw["selected_records"])
    records[3]["parent_envelope_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent outside"):
        ModelSimArchiveSelectionManifest(**(raw | {"selected_records": records}))

    records = deepcopy(raw["selected_records"])
    records[2]["ledger_value_sha256"] = "f" * 64
    changed = ModelSimArchiveSelectionManifest(**(raw | {"selected_records": records}))
    with pytest.raises(ValueError, match="payload digest"):
        validate_sim_archive_selection(
            changed, topology=_topology(), current=_current()
        )

    records = deepcopy(raw["selected_records"])
    records[0]["ledger_value_sha256"] = "f" * 64
    changed_head = ModelSimArchiveSelectionManifest(
        **(raw | {"selected_records": records})
    )
    with pytest.raises(ValueError, match="payload digest"):
        validate_sim_archive_selection(
            changed_head, topology=_topology(), current=_current()
        )


@pytest.mark.unit
def test_selection_refuses_an_unredacted_or_malformed_identifier() -> None:
    raw = _manifest().model_dump()
    raw["withheld_envelope_sha256"] = ("not-a-digest",)

    with pytest.raises(ValueError, match="withheld_envelope_sha256"):
        ModelSimArchiveSelectionManifest(**raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("broker_key_sha256", "not-a-digest", "broker_key_sha256"),
        ("broker_timestamp_ms", -1, "broker_timestamp_ms"),
    ],
)
def test_selected_record_requires_explicit_key_identity_and_timestamp(
    field: str, value: int | str, message: str
) -> None:
    raw = _manifest().selected_records[0].model_dump()
    with pytest.raises(ValueError, match=message):
        ModelSimArchiveSelectedRecord(**(raw | {field: value}))

    raw.pop("ledger_headers_normalized")
    with pytest.raises(ValueError, match="ledger_headers_normalized"):
        ModelSimArchiveSelectedRecord(**raw)
