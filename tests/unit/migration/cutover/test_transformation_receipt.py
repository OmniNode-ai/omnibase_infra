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
    ModelApplicationPathWriteProof,
    ModelConnectionIdentity,
    ModelControlPlaneDeltaEvidence,
    ModelCutoverContinuityEvidence,
    ModelCutoverFamilyContract,
    ModelCutoverJournalRequest,
    ModelPostgresEvidenceQuerySet,
    ModelProjectionReplayEvidence,
    ModelReconciliationInput,
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


def _connection_identity(*, backend_pid: int = 4242) -> ModelConnectionIdentity:
    """Fixed, deterministic identity so two calls with the same ``backend_pid``
    compare equal -- mirroring ``collect_pair``, which reads the server clock
    exactly once and stamps the identical identity onto both evidence sides.
    A fresh ``datetime.now(UTC)`` per call would make source/target evidence
    diverge by construction even when nothing in the test intends a mismatch.
    """
    return ModelConnectionIdentity(
        database="application",
        backend_pid=backend_pid,
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _evidence(label: str = "source") -> ModelTransformationEvidence:
    return ModelTransformationEvidence(
        label=label,
        evidence_contract_hash=_hash(f"query-contract:{label}"),
        binding_ref=f"application.{label}",
        connection_identity=_connection_identity(),
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


def _reconciliation_input(
    contract: ModelCutoverFamilyContract,
    source: ModelTransformationEvidence,
    target: ModelTransformationEvidence,
    continuity: ModelCutoverContinuityEvidence,
    idempotency_key: str,
) -> ModelReconciliationInput:
    return ModelReconciliationInput(
        contract=contract,
        source=source,
        target=target,
        continuity=continuity,
        idempotency_key=idempotency_key,
    )


def test_projection_receipt_is_complete_and_passes() -> None:
    receipt = TransformationReceiptBuilder().build(
        _reconciliation_input(
            _contract(),
            _evidence("source"),
            _evidence("target"),
            _projection_continuity(),
            "idempotency-key:projection-receipt",
        )
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
        _reconciliation_input(
            _contract(),
            _evidence("source"),
            target,
            _projection_continuity(),
            "idempotency-key:unregistered-contract",
        )
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
        _reconciliation_input(
            contract,
            _evidence("source"),
            _evidence("target"),
            _control_continuity(),
            "idempotency-key:control-plane-green",
        )
    )
    red = service.build(
        _reconciliation_input(
            contract,
            _evidence("source"),
            _evidence("target"),
            _control_continuity(target_delta_hash=_hash("wrong-delta")),
            "idempotency-key:control-plane-red",
        )
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
        _reconciliation_input(
            _contract(),
            _evidence("source"),
            target,
            continuity,
            f"idempotency-key:red-{dimension.value}",
        )
    )

    assert receipt.status is EnumReceiptStatus.FAIL
    check = next(check for check in receipt.checks if check.dimension is dimension)
    assert not check.passed
    assert check.detail.startswith("MISMATCH:")


def test_receipt_builder_refuses_evidence_with_mismatched_connection_identity() -> None:
    """GAP 3 forgery-refusal: source/target evidence not collected atomically.

    ``PostgresTransformationEvidenceCollector.collect_pair`` always stamps
    both sides with one server-verified identity read inside a single
    repeatable-read transaction. Two evidence objects with divergent
    ``connection_identity`` values could not have come from a genuine
    ``collect_pair`` call -- they must be refused before any dimension
    comparison runs, not silently reconciled.
    """
    source = _evidence("source")
    target = _evidence("target").model_copy(
        update={"connection_identity": _connection_identity(backend_pid=9999)}
    )
    with pytest.raises(ValueError, match="connection_identity mismatch"):
        TransformationReceiptBuilder().build(
            _reconciliation_input(
                _contract(),
                source,
                target,
                _projection_continuity(),
                "idempotency-key:mismatched-connection-identity",
            )
        )


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
            idempotency_key="dual-write:unbounded",
        )
    with pytest.raises(
        ValidationError, match="independently verified application_path_write_proof"
    ):
        ModelCutoverJournalRequest(
            kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
            occurred_at=now,
            evidence_ref="proof/app-write",
            idempotency_key="app-write:missing-proof",
        )
    with pytest.raises(ValidationError, match="idempotency_key"):
        ModelCutoverJournalRequest(
            kind=EnumCutoverEventKind.BACKFILL_STARTED,
            occurred_at=now,
            evidence_ref="proof/backfill",
            idempotency_key="",
        )

    bounded = ModelCutoverJournalRequest(
        kind=EnumCutoverEventKind.DUAL_WRITE_STARTED,
        occurred_at=now,
        evidence_ref="proof/dual-write",
        idempotency_key="dual-write:bounded",
        dual_write_expires_at=now + timedelta(seconds=30),
    )
    assert bounded.dual_write_expires_at is not None

    proof = ModelApplicationPathWriteProof(
        family_id=uuid4(),
        database_ref="application",
        principal="onex_api",
        schema_ref="tenant",
        target_sequence=1,
        verification_query_hash=_hash("select 1"),
        write_result_hash=_hash("row-1"),
        connection_identity=_connection_identity(),
    )
    write_proven = ModelCutoverJournalRequest(
        kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
        occurred_at=now,
        evidence_ref="proof/app-write",
        idempotency_key="app-write:proven",
        application_path_write_proof=proof,
    )
    assert write_proven.application_path_write_proof is not None
    assert write_proven.application_path_write_proof.principal == "onex_api"


def test_connection_identity_requires_timezone_aware_readback() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ModelConnectionIdentity(
            database="application",
            backend_pid=4242,
            collected_at=datetime.now(),
        )


def test_evidence_requires_binding_ref_and_connection_identity() -> None:
    with pytest.raises(ValidationError, match="binding_ref"):
        ModelTransformationEvidence(
            **{
                **_evidence("source").model_dump(),
                "binding_ref": "",
            }
        )
    with pytest.raises(ValidationError, match="connection_identity"):
        ModelTransformationEvidence(
            **{
                k: v
                for k, v in _evidence("source").model_dump().items()
                if k != "connection_identity"
            }
        )


def test_application_path_write_proof_requires_every_verified_field() -> None:
    with pytest.raises(ValidationError, match="database_ref"):
        ModelApplicationPathWriteProof(
            family_id=uuid4(),
            database_ref="",
            principal="onex_api",
            schema_ref="tenant",
            target_sequence=1,
            verification_query_hash=_hash("select 1"),
            write_result_hash=_hash("row-1"),
            connection_identity=_connection_identity(),
        )
    with pytest.raises(ValidationError):
        ModelApplicationPathWriteProof(
            family_id=uuid4(),
            database_ref="application",
            principal="onex_api",
            schema_ref="tenant",
            target_sequence=1,
            verification_query_hash="not-a-sha256",
            write_result_hash=_hash("row-1"),
            connection_identity=_connection_identity(),
        )


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
