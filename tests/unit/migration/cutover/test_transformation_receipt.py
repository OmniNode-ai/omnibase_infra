# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-proven pure receipt and contract invariants for OMN-15420."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from pydantic import ValidationError

from omnibase_infra.migration.cutover.enums import (
    EnumCutoverEventKind,
    EnumCutoverFamilyKind,
    EnumPostCheckpointMode,
    EnumReceiptDimension,
    EnumReceiptStatus,
    EnumReverseDeltaOperation,
)
from omnibase_infra.migration.cutover.models import (
    ModelControlPlaneDeltaEvidence,
    ModelCutoverContinuityEvidence,
    ModelCutoverFamilyContract,
    ModelCutoverJournalRequest,
    ModelPostgresEvidenceQuerySet,
    ModelProjectionReplayEvidence,
    ModelReverseDeltaEntry,
    ModelReverseDeltaProof,
    ModelTransformationEvidence,
    ModelTransformationReceipt,
)
from omnibase_infra.migration.cutover.transformation_receipt_builder import (
    TransformationReceiptBuilder,
)

pytestmark = pytest.mark.unit


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _family_id(family_key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"omninode-cutover:{family_key}")


def _contract(
    *,
    family_key: str = "tenant.usage",
    kind: EnumCutoverFamilyKind = EnumCutoverFamilyKind.PROJECTION,
    mode: EnumPostCheckpointMode = EnumPostCheckpointMode.REVERSE_DELTA,
    source_evidence_contract_hash: str | None = None,
    target_evidence_contract_hash: str | None = None,
) -> ModelCutoverFamilyContract:
    return ModelCutoverFamilyContract(
        family_id=_family_id(family_key),
        family_key=family_key,
        family_kind=kind,
        source_binding_ref="application.legacy",
        target_binding_ref="application.target",
        source_evidence_contract_hash=(
            source_evidence_contract_hash or _hash("query-contract:source")
        ),
        target_evidence_contract_hash=(
            target_evidence_contract_hash or _hash("query-contract:target")
        ),
        post_checkpoint_mode=mode,
        reverse_delta_contract_ref=(
            "contracts/reverse-delta/tenant-usage.yaml"
            if mode is EnumPostCheckpointMode.REVERSE_DELTA
            else ""
        ),
        forward_fix_runbook_ref=(
            "runbooks/control-plane-forward-fix.md"
            if mode is EnumPostCheckpointMode.FORWARD_FIX_ONLY
            else ""
        ),
        dual_write_max_seconds=60,
        observation_window_seconds=300,
    )


def _evidence(label: str = "source") -> ModelTransformationEvidence:
    return ModelTransformationEvidence(
        label=label,
        evidence_contract_hash=_hash(f"query-contract:{label}"),
        keys=("00000000-0000-0000-0000-000000000001", "row-b"),
        row_count=2,
        transformed_row_hashes=tuple(sorted((_hash("row-a"), _hash("row-b")))),
        foreign_keys=("tenant_id->tenant.tenants.id:on_delete=restrict",),
        sequences=("usage_id:2",),
        owners=("tenant.usage:owner_onex_tenant",),
        grants=("tenant_projection_writer:insert,select,update",),
        policies=("tenant_isolation:force:using=tenant_uuid",),
        views_functions=("tenant.usage_view:security_invoker",),
        dependencies=("tenant.usage_view",),
        collision_keys=(),
    )


def _projection_continuity(
    *, target_offset: int = 42
) -> ModelCutoverContinuityEvidence:
    return ModelCutoverContinuityEvidence(
        projection_replays=(
            ModelProjectionReplayEvidence(
                projection_id=uuid5(NAMESPACE_URL, "projection:usage_projection"),
                projection_label="usage_projection",
                projection_version="v7",
                topic="onex.evt.usage-recorded.v1",
                partition=0,
                source_offset=42,
                target_offset=target_offset,
            ),
        )
    )


def _control_continuity(
    *, target_delta_hash: str | None = None
) -> ModelCutoverContinuityEvidence:
    digest = _hash("control-plane-final-delta")
    return ModelCutoverContinuityEvidence(
        control_plane_delta=ModelControlPlaneDeltaEvidence(
            snapshot_id=uuid4(),
            source_snapshot_hash=_hash("control-plane-snapshot"),
            target_snapshot_hash=_hash("control-plane-snapshot"),
            final_delta_id=uuid4(),
            source_final_delta_hash=digest,
            target_final_delta_hash=target_delta_hash or digest,
            source_watermark="0000000000000042",
            target_watermark="0000000000000042",
        )
    )


def test_projection_receipt_is_complete_and_passes() -> None:
    receipt = TransformationReceiptBuilder().build(
        _contract(),
        _evidence("source"),
        _evidence("target"),
        _projection_continuity(),
    )

    assert receipt.status is EnumReceiptStatus.PASS
    assert [check.dimension for check in receipt.checks] == list(EnumReceiptDimension)
    assert all(check.passed for check in receipt.checks)
    assert len(receipt.receipt_hash) == 64

    tampered = receipt.model_dump()
    tampered["receipt_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="complete receipt"):
        ModelTransformationReceipt(**tampered)


def test_receipt_refuses_unregistered_evidence_query_contract() -> None:
    target = _evidence("target").model_copy(
        update={"evidence_contract_hash": _hash("unregistered-query-contract")}
    )

    receipt = TransformationReceiptBuilder().build(
        _contract(),
        _evidence("source"),
        target,
        _projection_continuity(),
    )

    assert receipt.status is EnumReceiptStatus.FAIL
    failed = {check.dimension for check in receipt.checks if not check.passed}
    assert failed == {EnumReceiptDimension.EVIDENCE_CONTRACTS}


def test_control_plane_receipt_requires_snapshot_and_final_delta_parity() -> None:
    contract = _contract(
        family_key="tenant.control-plane",
        kind=EnumCutoverFamilyKind.CONTROL_PLANE,
        mode=EnumPostCheckpointMode.FORWARD_FIX_ONLY,
    )
    service = TransformationReceiptBuilder()

    green = service.build(
        contract,
        _evidence("source"),
        _evidence("target"),
        _control_continuity(),
    )
    red = service.build(
        contract,
        _evidence("source"),
        _evidence("target"),
        _control_continuity(target_delta_hash=_hash("wrong-delta")),
    )

    assert green.status is EnumReceiptStatus.PASS
    assert red.status is EnumReceiptStatus.FAIL
    failed = {check.dimension for check in red.checks if not check.passed}
    assert failed == {EnumReceiptDimension.CONTROL_PLANE_DELTA}


@pytest.mark.parametrize(
    ("dimension", "target", "continuity"),
    [
        (
            EnumReceiptDimension.KEY_SET,
            _evidence("target").model_copy(
                update={
                    "keys": (
                        "00000000-0000-0000-0000-000000000001",
                        "row-c",
                    )
                }
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.ROW_COUNT,
            ModelTransformationEvidence(
                **{
                    **_evidence("target").model_dump(),
                    "keys": ("00000000-0000-0000-0000-000000000001",),
                    "row_count": 1,
                    "transformed_row_hashes": (_hash("row-a"),),
                }
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.TRANSFORMATION_HASH,
            _evidence("target").model_copy(
                update={
                    "transformed_row_hashes": tuple(
                        sorted((_hash("row-a"), _hash("changed-row")))
                    )
                }
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.FOREIGN_KEYS,
            _evidence("target").model_copy(update={"foreign_keys": ()}),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.SEQUENCES,
            _evidence("target").model_copy(update={"sequences": ("usage_id:1",)}),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.OWNERS,
            _evidence("target").model_copy(
                update={"owners": ("tenant.usage:postgres",)}
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.GRANTS,
            _evidence("target").model_copy(update={"grants": ("PUBLIC:select",)}),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.POLICIES,
            _evidence("target").model_copy(
                update={"policies": ("tenant_isolation:enable-only",)}
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.VIEWS_FUNCTIONS,
            _evidence("target").model_copy(
                update={"views_functions": ("tenant.usage_view:definer",)}
            ),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.EVENT_OFFSETS,
            _evidence("target"),
            _projection_continuity(target_offset=41),
        ),
        (
            EnumReceiptDimension.COLLISIONS,
            _evidence("target").model_copy(update={"collision_keys": ("row-b",)}),
            _projection_continuity(),
        ),
        (
            EnumReceiptDimension.DEPENDENCIES,
            _evidence("target").model_copy(update={"dependencies": ()}),
            _projection_continuity(),
        ),
    ],
)
def test_each_seeded_red_dimension_fails_closed(
    dimension: EnumReceiptDimension,
    target: ModelTransformationEvidence,
    continuity: ModelCutoverContinuityEvidence,
) -> None:
    receipt = TransformationReceiptBuilder().build(
        _contract(),
        _evidence("source"),
        target,
        continuity,
    )

    assert receipt.status is EnumReceiptStatus.FAIL
    check = next(check for check in receipt.checks if check.dimension is dimension)
    assert not check.passed
    assert check.detail.startswith("MISMATCH:")


def test_query_contract_rejects_mutation_and_missing_dimensions() -> None:
    valid = {
        "label": "source",
        "keys_sql": "SELECT key::text FROM source_rows",
        "rows_sql": "SELECT row_to_json(source_rows)::text FROM source_rows",
        "foreign_keys_sql": "SELECT signature FROM source_foreign_keys",
        "sequences_sql": "SELECT signature FROM source_sequences",
        "owners_sql": "SELECT signature FROM source_owners",
        "grants_sql": "SELECT signature FROM source_grants",
        "policies_sql": "SELECT signature FROM source_policies",
        "views_functions_sql": "SELECT signature FROM source_dependencies",
        "dependencies_sql": "SELECT signature FROM source_dependencies",
        "collisions_sql": "SELECT key FROM source_collisions",
    }
    ModelPostgresEvidenceQuerySet(**valid)

    with pytest.raises(ValidationError, match="SELECT or WITH"):
        ModelPostgresEvidenceQuerySet(**{**valid, "rows_sql": "DELETE FROM source"})
    incomplete = dict(valid)
    incomplete.pop("owners_sql")
    with pytest.raises(ValidationError, match="owners_sql"):
        ModelPostgresEvidenceQuerySet(**incomplete)


def test_family_contract_refuses_secret_like_bindings_and_ambiguous_mode() -> None:
    with pytest.raises(ValidationError, match="not DSNs"):
        _contract().model_copy(
            update={"source_binding_ref": "postgresql://user:secret@database/app"}
        ).__class__(
            **{
                **_contract().model_dump(),
                "source_binding_ref": "postgresql://user:secret@database/app",
            }
        )

    values = _contract().model_dump()
    values["forward_fix_runbook_ref"] = "runbook.md"
    with pytest.raises(ValidationError, match="cannot declare"):
        ModelCutoverFamilyContract(**values)


def test_journal_requests_refuse_unbounded_dual_write_and_weak_write_proof() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="finite expiry"):
        ModelCutoverJournalRequest(
            kind=EnumCutoverEventKind.DUAL_WRITE_STARTED,
            occurred_at=now,
            evidence_ref="proof/dual-write",
        )
    with pytest.raises(ValidationError, match="database, principal, and schema"):
        ModelCutoverJournalRequest(
            kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
            occurred_at=now,
            evidence_ref="proof/app-write",
            target_sequence=1,
        )

    bounded = ModelCutoverJournalRequest(
        kind=EnumCutoverEventKind.DUAL_WRITE_STARTED,
        occurred_at=now,
        evidence_ref="proof/dual-write",
        dual_write_expires_at=now + timedelta(seconds=30),
    )
    assert bounded.dual_write_expires_at is not None


def test_reverse_delta_proof_refuses_sequence_gaps() -> None:
    family_id = _family_id("tenant.usage")
    entries = (
        ModelReverseDeltaEntry(
            entry_id=uuid4(),
            family_id=family_id,
            target_sequence=7,
            relation="tenant.usage",
            operation=EnumReverseDeltaOperation.INSERT,
            primary_key_hash=_hash("pk-7"),
            before_image_hash=_hash("missing"),
            after_image_hash=_hash("after-7"),
            inverse_artifact_ref="journal/reverse/7",
        ),
        ModelReverseDeltaEntry(
            entry_id=uuid4(),
            family_id=family_id,
            target_sequence=9,
            relation="tenant.usage",
            operation=EnumReverseDeltaOperation.UPDATE,
            primary_key_hash=_hash("pk-9"),
            before_image_hash=_hash("before-9"),
            after_image_hash=_hash("after-9"),
            inverse_artifact_ref="journal/reverse/9",
        ),
    )
    with pytest.raises(ValidationError, match="cover every sequence"):
        ModelReverseDeltaProof(
            proof_id=uuid4(),
            family_id=family_id,
            start_sequence=7,
            end_sequence=9,
            entries=entries,
            quiescence_event_id=uuid4(),
            reconciliation_receipt_id=uuid4(),
            behavioral_readback_ref="proof/readback/tenant-usage",
            proven_at=datetime.now(UTC),
        )
