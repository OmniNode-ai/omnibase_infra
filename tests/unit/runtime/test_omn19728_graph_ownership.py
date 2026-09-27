# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Full-current envelope ownership is independent of replay cursor bounds."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphCurrentEvidence,
    ExecutionGraphLedgerRecord,
    PinnedExecutionGraphReadSet,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipRefusalError,
    admit_current_ownership,
    admit_verdict_candidates,
)

_HEAD_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_VERDICT_TOPIC = "onex.evt.omnimarket.dod-verify-completed.v1"
_READ_SET = object.__new__(PinnedExecutionGraphReadSet)
# Pure admission fixtures bypass minting; production accepts only registry output.
object.__setattr__(_READ_SET, "topology_version", "test-topology-v1")
object.__setattr__(
    _READ_SET,
    "topics",
    frozenset(
        {
            _HEAD_TOPIC,
            _VERDICT_TOPIC,
            "onex.cmd.omnibase-infra.delegation-request.v1",
            "onex.evt.test.unlinked.v1",
        }
    ),
)
object.__setattr__(_READ_SET, "head_topic", _HEAD_TOPIC)
object.__setattr__(_READ_SET, "verdict_topic", _VERDICT_TOPIC)


def _evidence(
    tenant_id: UUID,
    correlation_id: UUID,
    rows: tuple[ExecutionGraphLedgerRecord, ...],
) -> ExecutionGraphCurrentEvidence:
    # This pure-fold fixture bypasses construction solely to supply a DB proof;
    # integration tests establish that production proof minting is sealed.
    owner = object.__new__(DelegationOwnerProof)
    object.__setattr__(owner, "tenant_id", tenant_id)
    object.__setattr__(owner, "correlation_id", correlation_id)
    return ExecutionGraphCurrentEvidence(owner=owner, ledger_rows=rows)


def _row(
    correlation_id: UUID,
    *,
    topic: str,
    offset: int,
    envelope_id: UUID | None = None,
    parent_id: UUID | None = None,
    tenant_id: UUID | None = None,
    delegation_correlation_id: UUID | None = None,
) -> ExecutionGraphLedgerRecord:
    value: dict[str, object] = {
        "tenant_id": str(tenant_id) if tenant_id else None,
        "payload": {},
    }
    if delegation_correlation_id:
        value["payload"] = {"delegation_correlation_id": str(delegation_correlation_id)}
    return ExecutionGraphLedgerRecord(
        ledger_entry_id=uuid4(),
        topic=topic,
        partition=0,
        kafka_offset=offset,
        event_key=None,
        event_value=json.dumps(value).encode(),
        onex_headers=json.dumps(
            {"parent_message_id": str(parent_id)} if parent_id else {}
        ),
        envelope_id=envelope_id or uuid4(),
        correlation_id=correlation_id,
        event_type=None,
        source=None,
        event_timestamp=None,
        ledger_written_at=datetime.now(UTC),
    )


@pytest.mark.unit
def test_tenantless_descendant_is_owned_but_unreachable_is_withheld() -> None:
    tenant = uuid4()
    correlation = uuid4()
    head = _row(correlation, topic=_HEAD_TOPIC, offset=1, tenant_id=tenant)
    child = _row(
        correlation,
        topic="onex.cmd.omnibase-infra.delegation-request.v1",
        offset=2,
        parent_id=head.envelope_id,
    )
    stray = _row(
        correlation,
        topic="onex.evt.test.unlinked.v1",
        offset=3,
    )

    admitted = admit_current_ownership(
        _evidence(tenant, correlation, (head, child, stray)), _READ_SET
    )

    assert {row.envelope_id for row in admitted.owned_rows} == {
        head.envelope_id,
        child.envelope_id,
    }
    assert admitted.withheld_count == 1
    assert admitted.withheld_envelope_ids == (stray.envelope_id,)
    assert set(admitted.owned_envelope_ids) == {
        head.envelope_id,
        child.envelope_id,
    }
    assert not hasattr(admitted, "evidence")


@pytest.mark.unit
def test_real_five_hop_tenantless_chain_is_owned_by_head_reachability() -> None:
    """The recorded legacy chain has no tenant on any event, including its head."""
    fixture_path = (
        Path(__file__).parents[2] / "fixtures" / "omn19726" / "real_branch_legacy.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    correlation = UUID(fixture["correlation_id"])
    tenant = UUID(fixture["owner_tenant_id"])
    records = fixture["records"]
    read_set = object.__new__(PinnedExecutionGraphReadSet)
    object.__setattr__(read_set, "topology_version", "real-branch-legacy-v1")
    object.__setattr__(
        read_set,
        "topics",
        frozenset({record["topic"] for record in records} | {_VERDICT_TOPIC}),
    )
    object.__setattr__(read_set, "head_topic", _HEAD_TOPIC)
    object.__setattr__(read_set, "verdict_topic", _VERDICT_TOPIC)
    rows = tuple(
        _row(
            correlation,
            topic=record["topic"],
            offset=record["kafka_offset"],
            envelope_id=UUID(record["envelope_id"]),
            parent_id=(
                UUID(record["parent_envelope_id"])
                if record["parent_envelope_id"]
                else None
            ),
        )
        for record in records
    )

    admitted = admit_current_ownership(_evidence(tenant, correlation, rows), read_set)

    expected_hops = {
        UUID(record["envelope_id"])
        for record in records
        if record["stored_hop_index"] is not None
    }
    expected_withheld = {
        UUID(record["envelope_id"])
        for record in records
        if record["stored_hop_index"] is None
    }
    assert admitted.head_envelope_id == UUID(records[0]["envelope_id"])
    assert set(admitted.owned_envelope_ids) == expected_hops
    assert set(admitted.withheld_envelope_ids) == expected_withheld
    assert admitted.withheld_count == 2

    foreign = replace(
        rows[3], event_value=json.dumps({"tenant_id": str(uuid4())}).encode()
    )
    with pytest.raises(ExecutionGraphOwnershipRefusalError, match="foreign"):
        admit_current_ownership(
            _evidence(tenant, correlation, (*rows[:3], foreign, *rows[4:])),
            read_set,
        )

    # A requested bound below the second head is irrelevant to full-current
    # authorization; the owner gate sees both heads before replay selection.
    second_head = _row(correlation, topic=_HEAD_TOPIC, offset=999)
    with pytest.raises(ExecutionGraphOwnershipRefusalError, match="exactly one"):
        admit_current_ownership(
            _evidence(tenant, correlation, (*rows, second_head)), read_set
        )


@pytest.mark.unit
def test_read_set_must_include_head_and_verdict_and_reject_outside_rows() -> None:
    with pytest.raises(TypeError, match="verified topology"):
        PinnedExecutionGraphReadSet(
            topology_version="test-topology-v1",
            topics=frozenset({_HEAD_TOPIC, _VERDICT_TOPIC}),
            head_topic=_HEAD_TOPIC,
            verdict_topic=_VERDICT_TOPIC,
        )

    tenant = uuid4()
    correlation = uuid4()
    head = _row(correlation, topic=_HEAD_TOPIC, offset=1, tenant_id=tenant)
    outside = _row(correlation, topic="onex.evt.test.outside-v1", offset=2)
    with pytest.raises(ExecutionGraphOwnershipRefusalError, match="pinned"):
        admit_current_ownership(
            _evidence(tenant, correlation, (head, outside)), _READ_SET
        )


@pytest.mark.unit
def test_foreign_head_outside_any_replay_bound_refuses_whole_correlation() -> None:
    tenant = uuid4()
    correlation = uuid4()
    head = _row(correlation, topic=_HEAD_TOPIC, offset=1, tenant_id=tenant)
    foreign_head = _row(correlation, topic=_HEAD_TOPIC, offset=999, tenant_id=uuid4())

    with pytest.raises(ExecutionGraphOwnershipRefusalError):
        admit_current_ownership(
            _evidence(tenant, correlation, (head, foreign_head)), _READ_SET
        )


@pytest.mark.unit
def test_conflicting_same_envelope_id_is_not_collapsed_as_redelivery() -> None:
    tenant = uuid4()
    correlation = uuid4()
    head_id = uuid4()
    first = _row(
        correlation,
        topic=_HEAD_TOPIC,
        offset=1,
        envelope_id=head_id,
        tenant_id=tenant,
    )
    conflicting = _row(
        correlation,
        topic=_HEAD_TOPIC,
        offset=2,
        envelope_id=head_id,
        tenant_id=tenant,
        parent_id=uuid4(),
    )

    with pytest.raises(ExecutionGraphOwnershipRefusalError, match="collision"):
        admit_current_ownership(
            _evidence(tenant, correlation, (first, conflicting)), _READ_SET
        )


@pytest.mark.unit
def test_exact_redelivery_does_not_create_second_head() -> None:
    tenant = uuid4()
    correlation = uuid4()
    head = _row(correlation, topic=_HEAD_TOPIC, offset=1, tenant_id=tenant)
    redelivery = replace(head, ledger_entry_id=uuid4(), kafka_offset=2)

    admitted = admit_current_ownership(
        _evidence(tenant, correlation, (head, redelivery)), _READ_SET
    )

    assert len(admitted.owned_rows) == 1


@pytest.mark.unit
def test_verdict_must_name_delegation_and_match_recorded_tenant() -> None:
    tenant = uuid4()
    correlation = uuid4()
    head = _row(correlation, topic=_HEAD_TOPIC, offset=1, tenant_id=tenant)
    admitted = admit_current_ownership(
        _evidence(tenant, correlation, (head,)), _READ_SET
    )
    verdict_correlation = uuid4()
    matching = _row(
        verdict_correlation,
        topic=_VERDICT_TOPIC,
        offset=7,
        tenant_id=tenant,
        delegation_correlation_id=correlation,
    )
    assert admit_verdict_candidates(
        admitted, (matching,), _READ_SET
    ).admitted_verdict_rows == (matching,)

    wrong_delegation = _row(
        verdict_correlation,
        topic=_VERDICT_TOPIC,
        offset=8,
        tenant_id=tenant,
        delegation_correlation_id=uuid4(),
    )
    foreign = _row(
        verdict_correlation,
        topic=_VERDICT_TOPIC,
        offset=9,
        tenant_id=uuid4(),
        delegation_correlation_id=correlation,
    )
    with pytest.raises(ExecutionGraphOwnershipRefusalError):
        admit_verdict_candidates(admitted, (wrong_delegation,), _READ_SET)
    with pytest.raises(ExecutionGraphOwnershipRefusalError):
        admit_verdict_candidates(admitted, (foreign,), _READ_SET)
