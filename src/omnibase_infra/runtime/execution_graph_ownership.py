# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure, full-current ownership admission before any bounded graph replay."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import NoReturn
from uuid import UUID

from omnibase_core.models.execution_graph_replay.model_enum_execution_graph_replay import (
    EnumExecutionGraphRefusalReason,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_source_cursor import (
    ModelExecutionGraphSourceCursor,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphCurrentEvidence,
    ExecutionGraphLedgerRecord,
    PinnedExecutionGraphReadSet,
)


class ExecutionGraphOwnershipRefusalError(ValueError):
    """Internal diagnostic; the serving boundary must return uniform not-found."""

    def __init__(self, reason: EnumExecutionGraphRefusalReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionGraphOwnershipAdmission:
    """Only evidence whose current ownership was established."""

    owner: DelegationOwnerProof
    head_envelope_id: UUID
    owned_rows: tuple[ExecutionGraphLedgerRecord, ...]
    owned_envelope_ids: tuple[UUID, ...]
    withheld_envelope_ids: tuple[UUID, ...]
    withheld_count: int
    admitted_verdict_rows: tuple[ExecutionGraphLedgerRecord, ...] = ()
    withheld_rows: tuple[ExecutionGraphLedgerRecord, ...] = ()


def _refuse(reason: EnumExecutionGraphRefusalReason, detail: str) -> NoReturn:
    raise ExecutionGraphOwnershipRefusalError(reason, detail)


def _canonical_uuid(value: object, *, field: str) -> UUID:
    if not isinstance(value, str) or value != value.strip():
        _refuse(EnumExecutionGraphRefusalReason.INVALID_EVIDENCE, f"invalid {field}")
    try:
        parsed = UUID(value)
    except ValueError:
        _refuse(EnumExecutionGraphRefusalReason.INVALID_EVIDENCE, f"invalid {field}")
    if parsed.int == 0 or str(parsed) != value:
        _refuse(EnumExecutionGraphRefusalReason.INVALID_EVIDENCE, f"invalid {field}")
    return parsed


def _event_body(row: ExecutionGraphLedgerRecord) -> dict[str, object] | None:
    try:
        value = json.loads(row.event_value)
    except (ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _recorded_tenant(row: ExecutionGraphLedgerRecord) -> UUID | None:
    body = _event_body(row)
    if body is None:
        return None
    raw = body.get("tenant_id")
    if raw is None or raw == "":
        return None
    return _canonical_uuid(raw, field="recorded tenant")


def _parent_id(row: ExecutionGraphLedgerRecord) -> UUID | None:
    try:
        headers = json.loads(row.onex_headers)
    except ValueError:
        _refuse(EnumExecutionGraphRefusalReason.INVALID_EVIDENCE, "invalid headers")
    if not isinstance(headers, dict):
        _refuse(EnumExecutionGraphRefusalReason.INVALID_EVIDENCE, "invalid headers")
    raw = headers.get("parent_message_id")
    if raw is None or raw == "":
        return None
    return _canonical_uuid(raw, field="recorded parent")


def _same_envelope(
    left: ExecutionGraphLedgerRecord, right: ExecutionGraphLedgerRecord
) -> bool:
    """An exact redelivery may change delivery coordinates, not event identity."""
    return (
        left.topic == right.topic
        and left.event_key == right.event_key
        and left.event_value == right.event_value
        and left.onex_headers == right.onex_headers
        and left.event_type == right.event_type
        and left.source == right.source
        and left.event_timestamp == right.event_timestamp
        and left.correlation_id == right.correlation_id
    )


def admit_current_ownership(
    evidence: ExecutionGraphCurrentEvidence,
    read_set: PinnedExecutionGraphReadSet,
) -> ExecutionGraphOwnershipAdmission:
    """Classify every current row before a replay cursor can hide any of them."""
    if type(read_set) is not PinnedExecutionGraphReadSet:
        raise TypeError("Ownership admission requires a pinned read set")
    owner = evidence.owner
    distinct: dict[UUID, ExecutionGraphLedgerRecord] = {}
    unknown_id_count = 0
    opaque_ids: set[UUID] = set()
    for row in evidence.ledger_rows:
        if row.topic not in read_set.topics:
            _refuse(
                EnumExecutionGraphRefusalReason.INVALID_EVIDENCE,
                "row is outside the pinned read set",
            )
        if row.correlation_id != owner.correlation_id:
            _refuse(
                EnumExecutionGraphRefusalReason.CORRELATION_AMBIGUOUS,
                "ledger correlation conflicts with owner",
            )
        recorded_tenant = _recorded_tenant(row)
        if recorded_tenant is not None and recorded_tenant != owner.tenant_id:
            _refuse(
                EnumExecutionGraphRefusalReason.CORRELATION_AMBIGUOUS,
                "foreign recorded tenant on correlation",
            )
        if row.envelope_id is None:
            # No envelope identity means no recorded edge can establish its
            # ownership. Withhold it; never invent an id for the renderer.
            unknown_id_count += 1
            continue
        if _event_body(row) is None:
            opaque_ids.add(row.envelope_id)
        prior = distinct.get(row.envelope_id)
        if prior is not None:
            if not _same_envelope(prior, row):
                _refuse(
                    EnumExecutionGraphRefusalReason.ENVELOPE_ID_COLLISION,
                    "envelope ID collision with conflicting recorded event",
                )
            continue
        distinct[row.envelope_id] = row

    heads = [row for row in distinct.values() if row.topic == read_set.head_topic]
    if len(heads) != 1:
        _refuse(
            EnumExecutionGraphRefusalReason.MULTIPLE_CHAIN_HEADS,
            "correlation does not have exactly one chain head",
        )
    head = heads[0]
    assert head.envelope_id is not None
    if _parent_id(head) is not None:
        _refuse(
            EnumExecutionGraphRefusalReason.INVALID_EVIDENCE,
            "chain head records a parent",
        )
    if _event_body(head) is None:
        _refuse(
            EnumExecutionGraphRefusalReason.INVALID_EVIDENCE,
            "chain head body cannot establish tenant ownership",
        )

    owned_ids = {head.envelope_id}
    for row in distinct.values():
        if (
            row.envelope_id not in opaque_ids
            and _recorded_tenant(row) == owner.tenant_id
            and row.envelope_id is not None
        ):
            owned_ids.add(row.envelope_id)
    changed = True
    while changed:
        changed = False
        for row in distinct.values():
            if row.envelope_id in owned_ids or row.envelope_id in opaque_ids:
                continue
            parent_id = _parent_id(row)
            if parent_id in owned_ids:
                assert row.envelope_id is not None
                owned_ids.add(row.envelope_id)
                changed = True

    # Ownership is current, but delivery selection belongs to the bounded fold.
    # Dropping an older delivery here would let a late lower Kafka offset replace
    # evidence that was present at an earlier ingest watermark.
    owned_rows = tuple(
        row for row in evidence.ledger_rows if row.envelope_id in owned_ids
    )
    withheld_envelope_ids = tuple(
        envelope_id for envelope_id in distinct if envelope_id not in owned_ids
    )
    return ExecutionGraphOwnershipAdmission(
        owner=owner,
        head_envelope_id=head.envelope_id,
        owned_rows=owned_rows,
        owned_envelope_ids=tuple(
            envelope_id for envelope_id in distinct if envelope_id in owned_ids
        ),
        withheld_envelope_ids=withheld_envelope_ids,
        withheld_count=unknown_id_count + len(withheld_envelope_ids),
        withheld_rows=tuple(
            row for row in evidence.ledger_rows if row.envelope_id not in owned_ids
        ),
    )


def bound_replay_ownership(
    admission: ExecutionGraphOwnershipAdmission,
    cursors: tuple[ModelExecutionGraphSourceCursor, ...],
) -> ExecutionGraphOwnershipAdmission:
    """Restrict admitted rows and ownership evidence to an immutable replay window.

    Full-current authorization must already have passed. A later parent can
    establish current reachability, but cannot expose a formerly withheld node
    at an earlier bound. Withheld counts obey the same boundary as drawn nodes.
    """
    bounds = {
        (cursor.topic, cursor.partition): cursor.max_ingest_watermark
        for cursor in cursors
    }
    selected: list[ExecutionGraphLedgerRecord] = []
    for row in (*admission.owned_rows, *admission.withheld_rows):
        bound = bounds.get((row.topic, row.partition))
        if bound is None:
            continue
        if row.ingest_watermark is None:
            raise ValueError("replay ownership evidence lacks an ingest watermark")
        if row.ingest_watermark <= bound:
            selected.append(row)

    current_owned = set(admission.owned_envelope_ids)
    reachable = {
        row.envelope_id
        for row in selected
        if row.envelope_id == admission.head_envelope_id
    }
    changed = True
    while changed:
        changed = False
        for row in selected:
            if row.envelope_id in reachable or row.envelope_id not in current_owned:
                continue
            if _parent_id(row) in reachable:
                reachable.add(row.envelope_id)
                changed = True
    owned_ids = reachable.union(
        row.envelope_id
        for row in selected
        if row.envelope_id in current_owned
        and _recorded_tenant(row) == admission.owner.tenant_id
    )
    owned_rows = tuple(row for row in selected if row.envelope_id in owned_ids)
    withheld_rows = tuple(row for row in selected if row.envelope_id not in owned_ids)
    withheld_ids = tuple(
        dict.fromkeys(row.envelope_id for row in withheld_rows if row.envelope_id)
    )
    return replace(
        admission,
        owned_rows=owned_rows,
        owned_envelope_ids=tuple(
            dict.fromkeys(row.envelope_id for row in owned_rows if row.envelope_id)
        ),
        withheld_rows=withheld_rows,
        withheld_envelope_ids=withheld_ids,
        withheld_count=len(withheld_ids)
        + sum(row.envelope_id is None for row in withheld_rows),
    )


def admit_verdict_candidates(
    admission: ExecutionGraphOwnershipAdmission,
    candidates: tuple[ExecutionGraphLedgerRecord, ...],
    read_set: PinnedExecutionGraphReadSet,
) -> ExecutionGraphOwnershipAdmission:
    """Admit only raw verdicts naming this authorized delegation/tenant."""
    if type(read_set) is not PinnedExecutionGraphReadSet:
        raise TypeError("Verdict admission requires a pinned read set")
    delegation_correlation = admission.owner.correlation_id
    tenant_id = admission.owner.tenant_id
    for row in candidates:
        if row.topic != read_set.verdict_topic:
            _refuse(
                EnumExecutionGraphRefusalReason.INVALID_EVIDENCE,
                "candidate is not a verdict envelope",
            )
        body = _event_body(row)
        if body is None or not isinstance(body.get("payload"), dict):
            _refuse(
                EnumExecutionGraphRefusalReason.INVALID_EVIDENCE,
                "verdict body cannot establish delegation correlation",
            )
        payload = body["payload"]
        assert isinstance(payload, dict)
        raw_delegation = payload.get("delegation_correlation_id")
        if (
            _canonical_uuid(raw_delegation, field="verdict delegation correlation")
            != delegation_correlation
        ):
            _refuse(
                EnumExecutionGraphRefusalReason.CORRELATION_AMBIGUOUS,
                "verdict names another delegation",
            )
        recorded_tenant = _recorded_tenant(row)
        if recorded_tenant is not None and recorded_tenant != tenant_id:
            _refuse(
                EnumExecutionGraphRefusalReason.CORRELATION_AMBIGUOUS,
                "verdict records another tenant",
            )
    return replace(admission, admitted_verdict_rows=candidates)


__all__ = [
    "ExecutionGraphOwnershipAdmission",
    "ExecutionGraphOwnershipRefusalError",
    "admit_current_ownership",
    "bound_replay_ownership",
    "admit_verdict_candidates",
]
