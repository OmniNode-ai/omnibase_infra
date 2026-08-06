# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure builder for complete transformation-aware database receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from omnibase_infra.migration.cutover.enums import (
    EnumCutoverFamilyKind,
    EnumReceiptDimension,
    EnumReceiptStatus,
)
from omnibase_infra.migration.cutover.models import (
    ModelCutoverContinuityEvidence,
    ModelCutoverFamilyContract,
    ModelReceiptCheck,
    ModelReconciliationInput,
    ModelTransformationEvidence,
    ModelTransformationReceipt,
    calculate_transformation_receipt_hash,
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TransformationReceiptBuilder:
    """Compare canonical evidence and emit an immutable all-dimension receipt."""

    def build(
        self,
        request: ModelReconciliationInput,
    ) -> ModelTransformationReceipt:
        """Build a PASS only when every required invariant is proven."""
        contract = request.contract
        source = request.source
        target = request.target
        continuity = request.continuity
        self._require_atomic_connection_identity(source, target)
        comparisons = self._comparisons(contract, source, target, continuity)
        checks = tuple(
            self._check(dimension, *comparisons[dimension])
            for dimension in EnumReceiptDimension
        )
        status = (
            EnumReceiptStatus.PASS
            if all(check.passed for check in checks)
            else EnumReceiptStatus.FAIL
        )
        receipt_id = uuid4()
        generated_at = datetime.now(UTC)
        family_contract_hash = _digest(contract.model_dump(mode="json"))
        receipt_hash = calculate_transformation_receipt_hash(
            receipt_id=receipt_id,
            family_id=contract.family_id,
            family_contract_hash=family_contract_hash,
            generated_at=generated_at,
            source=source,
            target=target,
            continuity=continuity,
            checks=checks,
            status=status,
        )
        return ModelTransformationReceipt(
            receipt_id=receipt_id,
            family_id=contract.family_id,
            idempotency_key=request.idempotency_key,
            family_contract_hash=family_contract_hash,
            generated_at=generated_at,
            source=source,
            target=target,
            continuity=continuity,
            checks=checks,
            status=status,
            receipt_hash=receipt_hash,
        )

    @staticmethod
    def _require_atomic_connection_identity(
        source: ModelTransformationEvidence,
        target: ModelTransformationEvidence,
    ) -> None:
        """Refuse evidence not captured on one atomic connection snapshot.

        ``PostgresTransformationEvidenceCollector.collect_pair`` always stamps
        both sides of a genuine collection with the identical, server-verified
        connection identity read once inside a single repeatable-read
        transaction. Two independently or hand-assembled
        ``ModelTransformationEvidence`` objects -- even if every other
        dimension is internally self-consistent -- are refused the instant
        their identities diverge, closing the gap where a caller could pair
        evidence collected on two different connections (or two different
        instants) and have it accepted as one atomic snapshot.
        """
        if source.connection_identity != target.connection_identity:
            raise ValueError(
                "source and target evidence were not captured on the same "
                "atomic connection snapshot (connection_identity mismatch)"
            )

    def _comparisons(
        self,
        contract: ModelCutoverFamilyContract,
        source: ModelTransformationEvidence,
        target: ModelTransformationEvidence,
        continuity: ModelCutoverContinuityEvidence,
    ) -> dict[EnumReceiptDimension, tuple[object, object, bool, str]]:
        source_row_digest = _digest(source.transformed_row_hashes)
        target_row_digest = _digest(target.transformed_row_hashes)
        event_source, event_target, event_passed, event_detail = (
            self._event_offset_comparison(contract, continuity)
        )
        delta_source, delta_target, delta_passed, delta_detail = (
            self._control_plane_comparison(contract, continuity)
        )
        return {
            EnumReceiptDimension.EVIDENCE_CONTRACTS: (
                (
                    contract.source_evidence_contract_hash,
                    source.evidence_contract_hash,
                ),
                (
                    contract.target_evidence_contract_hash,
                    target.evidence_contract_hash,
                ),
                (
                    source.evidence_contract_hash
                    == contract.source_evidence_contract_hash
                    and target.evidence_contract_hash
                    == contract.target_evidence_contract_hash
                ),
                "source and target evidence queries match the registered contracts",
            ),
            EnumReceiptDimension.KEY_SET: (
                source.keys,
                target.keys,
                source.keys == target.keys,
                "canonical source and target key sets are equal",
            ),
            EnumReceiptDimension.ROW_COUNT: (
                source.row_count,
                target.row_count,
                source.row_count == target.row_count,
                "source and target row counts are equal",
            ),
            EnumReceiptDimension.TRANSFORMATION_HASH: (
                source_row_digest,
                target_row_digest,
                source_row_digest == target_row_digest,
                "transformation-aware row hashes are equal",
            ),
            EnumReceiptDimension.FOREIGN_KEYS: self._equal(
                source.foreign_keys, target.foreign_keys, "foreign keys"
            ),
            EnumReceiptDimension.SEQUENCES: self._equal(
                source.sequences, target.sequences, "sequence values and ownership"
            ),
            EnumReceiptDimension.OWNERS: self._equal(
                source.owners, target.owners, "object owners"
            ),
            EnumReceiptDimension.GRANTS: self._equal(
                source.grants, target.grants, "explicit grants"
            ),
            EnumReceiptDimension.POLICIES: self._equal(
                source.policies, target.policies, "RLS policy signatures"
            ),
            EnumReceiptDimension.VIEWS_FUNCTIONS: self._equal(
                source.views_functions,
                target.views_functions,
                "dependent view/function signatures",
            ),
            EnumReceiptDimension.EVENT_OFFSETS: (
                event_source,
                event_target,
                event_passed,
                event_detail,
            ),
            EnumReceiptDimension.CONTROL_PLANE_DELTA: (
                delta_source,
                delta_target,
                delta_passed,
                delta_detail,
            ),
            EnumReceiptDimension.COLLISIONS: (
                source.collision_keys,
                target.collision_keys,
                not source.collision_keys and not target.collision_keys,
                "transformed source and target collision scans are empty",
            ),
            EnumReceiptDimension.DEPENDENCIES: self._equal(
                source.dependencies,
                target.dependencies,
                "dependency signatures",
            ),
        }

    @staticmethod
    def _equal(
        source: object,
        target: object,
        label: str,
    ) -> tuple[object, object, bool, str]:
        return source, target, source == target, f"source and target {label} are equal"

    @staticmethod
    def _event_offset_comparison(
        contract: ModelCutoverFamilyContract,
        continuity: ModelCutoverContinuityEvidence,
    ) -> tuple[object, object, bool, str]:
        replays = continuity.projection_replays
        source = tuple(
            (
                item.projection_id,
                item.projection_label,
                item.projection_version,
                item.topic,
                item.partition,
                item.source_offset,
            )
            for item in replays
        )
        target = tuple(
            (
                item.projection_id,
                item.projection_label,
                item.projection_version,
                item.topic,
                item.partition,
                item.target_offset,
            )
            for item in replays
        )
        if contract.family_kind is EnumCutoverFamilyKind.PROJECTION:
            return (
                source,
                target,
                bool(replays) and source == target,
                "projection versions and authoritative event offsets are equal",
            )
        return (
            source,
            target,
            not replays,
            "control-plane families must not advertise projection replay evidence",
        )

    @staticmethod
    def _control_plane_comparison(
        contract: ModelCutoverFamilyContract,
        continuity: ModelCutoverContinuityEvidence,
    ) -> tuple[object, object, bool, str]:
        delta = continuity.control_plane_delta
        if delta is None:
            passed = contract.family_kind is EnumCutoverFamilyKind.PROJECTION
            return (), (), passed, "control-plane snapshot/final-delta evidence"
        source = (
            delta.source_snapshot_hash,
            delta.source_final_delta_hash,
            delta.source_watermark,
        )
        target = (
            delta.target_snapshot_hash,
            delta.target_final_delta_hash,
            delta.target_watermark,
        )
        passed = (
            contract.family_kind is EnumCutoverFamilyKind.CONTROL_PLANE
            and not continuity.projection_replays
            and source == target
        )
        return (
            source,
            target,
            passed,
            "control-plane snapshot, final delta, and watermarks are equal",
        )

    @staticmethod
    def _check(
        dimension: EnumReceiptDimension,
        source: object,
        target: object,
        passed: bool,
        success_detail: str,
    ) -> ModelReceiptCheck:
        detail = success_detail if passed else f"MISMATCH: {success_detail}"
        return ModelReceiptCheck(
            dimension=dimension,
            passed=passed,
            source_digest=_digest(source),
            target_digest=_digest(target),
            detail=detail,
        )


__all__ = ["TransformationReceiptBuilder"]
