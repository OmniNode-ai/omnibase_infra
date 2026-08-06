# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL implementation of the durable per-family cutover journal."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from importlib.resources import files
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.migration.cutover.enums import (
    EnumCutoverEventKind,
    EnumCutoverFamilyStatus,
    EnumPostCheckpointMode,
    EnumReceiptStatus,
)
from omnibase_infra.migration.cutover.models import (
    ModelApplicationPathWriteProof,
    ModelCutoverFamilyContract,
    ModelCutoverFamilyState,
    ModelCutoverJournalEvent,
    ModelCutoverJournalRequest,
    ModelReverseDeltaProof,
    ModelRollbackDecision,
    ModelTransformationReceipt,
)

_RECEIPT_CONTENT_FIELDS = (
    "family_contract_hash",
    "source",
    "target",
    "continuity",
    "checks",
    "status",
)

_ZERO_HASH = "0" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _receipt_content_signature(receipt: ModelTransformationReceipt) -> str:
    """Canonical signature of a receipt's deterministic content only.

    Excludes ``receipt_id``, ``generated_at``, and ``receipt_hash`` -- fields
    that are fresh on every build call even when the underlying reconciliation
    inputs are identical. Two receipts built from identical inputs must
    produce the same signature so a retried ``reconcile`` call can be proven
    idempotent instead of minting a second, distinct receipt.
    """
    dumped = receipt.model_dump(mode="json")
    return _canonical_json({field: dumped[field] for field in _RECEIPT_CONTENT_FIELDS})


class RepositoryPostgresCutoverJournal:
    """Persist immutable receipts and serialize family-local state transitions."""

    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    async def initialize(self) -> None:
        """Create the proof/journal schema only when explicitly requested."""
        bootstrap = (
            files("omnibase_infra.migration.cutover.sql")
            .joinpath("bootstrap.sql")
            .read_text(encoding="utf-8")
        )
        await self._connection.execute(bootstrap)

    async def register_family(self, contract: ModelCutoverFamilyContract) -> None:
        """Register an immutable contract; reject a same-id semantic rewrite."""
        contract_json = _canonical_json(contract.model_dump(mode="json"))
        contract_hash = _sha256(contract.model_dump(mode="json"))
        async with self._connection.transaction():
            await self._connection.execute(
                """
INSERT INTO omninode_internal.cutover_family_contracts
  (family_id, contract_json, contract_hash)
VALUES ($1, $2::jsonb, $3)
ON CONFLICT (family_id) DO NOTHING
""",
                contract.family_id,
                contract_json,
                contract_hash,
            )
            stored = await self._connection.fetchval(
                """
SELECT contract_hash
FROM omninode_internal.cutover_family_contracts
WHERE family_id = $1
FOR UPDATE
""",
                contract.family_id,
            )
            if stored != contract_hash:
                raise ValueError(
                    f"family contract drift for {contract.family_id!r}; "
                    "register a new family/version instead of rewriting history"
                )

    async def record_receipt(
        self, receipt: ModelTransformationReceipt
    ) -> ModelTransformationReceipt:
        """Persist a receipt and block only the mismatching family.

        Idempotent on ``(family_id, idempotency_key)``: a retried
        reconciliation carrying the same key and identical deterministic
        content returns the original persisted receipt (its own
        ``receipt_id``/``generated_at``/``receipt_hash``) instead of minting
        a second, distinct receipt for the same inputs.
        """
        receipt_json = _canonical_json(receipt.model_dump(mode="json"))
        content_signature = _receipt_content_signature(receipt)
        async with self._connection.transaction():
            family = await self._lock_family(receipt.family_id)

            keyed = await self._connection.fetchrow(
                """
SELECT receipt_json
FROM omninode_internal.transformation_receipts
WHERE family_id = $1 AND idempotency_key = $2
""",
                receipt.family_id,
                receipt.idempotency_key,
            )
            if keyed is not None:
                raw = keyed["receipt_json"]
                stored = ModelTransformationReceipt.model_validate_json(
                    raw if isinstance(raw, str) else _canonical_json(raw)
                )
                if _receipt_content_signature(stored) != content_signature:
                    raise ValueError(
                        "idempotency key already used with different "
                        "reconciliation content"
                    )
                return stored

            if receipt.family_contract_hash != family["contract_hash"]:
                raise ValueError(
                    "receipt is not bound to the registered family contract"
                )
            existing = await self._connection.fetchrow(
                """
SELECT family_id, receipt_hash
FROM omninode_internal.transformation_receipts
WHERE receipt_id = $1
""",
                receipt.receipt_id,
            )
            if existing is not None:
                if (
                    existing["family_id"] != receipt.family_id
                    or existing["receipt_hash"] != receipt.receipt_hash
                ):
                    raise ValueError(
                        "receipt UUID is already bound to different evidence"
                    )
                return receipt

            latest_generated_at = await self._connection.fetchval(
                """
SELECT max(generated_at)
FROM omninode_internal.transformation_receipts
WHERE family_id = $1
""",
                receipt.family_id,
            )
            if (
                latest_generated_at is not None
                and receipt.generated_at < latest_generated_at
            ):
                raise ValueError("stale receipt replay is forbidden")

            await self._connection.execute(
                """
INSERT INTO omninode_internal.transformation_receipts
  (receipt_id, family_id, status, receipt_hash, receipt_json, generated_at,
   idempotency_key)
VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
""",
                receipt.receipt_id,
                receipt.family_id,
                receipt.status.value,
                receipt.receipt_hash,
                receipt_json,
                receipt.generated_at,
                receipt.idempotency_key,
            )
            if receipt.status is EnumReceiptStatus.PASS:
                await self._connection.execute(
                    """
UPDATE omninode_internal.cutover_family_contracts
SET last_known_good_receipt_id = $2
WHERE family_id = $1
""",
                    receipt.family_id,
                    receipt.receipt_id,
                )
            else:
                await self._connection.execute(
                    """
UPDATE omninode_internal.cutover_family_contracts
SET status = 'blocked', blocked_receipt_id = $2
WHERE family_id = $1
""",
                    receipt.family_id,
                    receipt.receipt_id,
                )

            # Keep the row lock live until the receipt and state update commit.
            if family["family_id"] != receipt.family_id:
                raise AssertionError("locked family identity changed")
            return receipt

    async def append_event(
        self,
        family_id: UUID,
        request: ModelCutoverJournalRequest,
    ) -> ModelCutoverJournalEvent:
        """Validate, hash-chain, and append one family-local event atomically.

        Idempotent on ``(family_id, idempotency_key)``: a retried append with
        the same key and an identical request returns the already-durable
        event (its original ``event_id``/``sequence``/hashes) rather than
        minting a second event or advancing the sequence a second time.
        """
        request_json = _canonical_json(request.model_dump(mode="json"))
        async with self._connection.transaction():
            row = await self._lock_family(family_id)

            keyed = await self._connection.fetchrow(
                """
SELECT event_id, sequence, previous_event_hash, event_hash, request_json
FROM omninode_internal.cutover_journal
WHERE family_id = $1 AND idempotency_key = $2
""",
                family_id,
                request.idempotency_key,
            )
            if keyed is not None:
                raw = keyed["request_json"]
                stored_json = _canonical_json(
                    json.loads(raw) if isinstance(raw, str) else raw
                )
                if stored_json != request_json:
                    raise ValueError(
                        "idempotency key already used with a different request"
                    )
                return ModelCutoverJournalEvent(
                    event_id=keyed["event_id"],
                    family_id=family_id,
                    sequence=int(keyed["sequence"]),
                    previous_event_hash=keyed["previous_event_hash"],
                    event_hash=keyed["event_hash"],
                    request=request,
                )

            contract = self._contract_from_row(row)
            await self._validate_transition(row, contract, request)

            sequence = int(row["last_sequence"]) + 1
            previous_hash = str(row["last_event_hash"])
            event_id = uuid4()
            event_body = {
                "event_id": str(event_id),
                "family_id": family_id,
                "sequence": sequence,
                "previous_event_hash": previous_hash,
                "request": request.model_dump(mode="json"),
            }
            event_hash = _sha256(event_body)
            event = ModelCutoverJournalEvent(
                event_id=event_id,
                family_id=family_id,
                sequence=sequence,
                previous_event_hash=previous_hash,
                event_hash=event_hash,
                request=request,
            )
            await self._connection.execute(
                """
INSERT INTO omninode_internal.cutover_journal
  (event_id, family_id, sequence, event_kind, request_json, receipt_id,
   previous_event_hash, event_hash, occurred_at, idempotency_key)
VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10)
""",
                event.event_id,
                family_id,
                sequence,
                request.kind.value,
                request_json,
                request.receipt_id,
                previous_hash,
                event_hash,
                request.occurred_at,
                request.idempotency_key,
            )
            await self._project_transition(row, contract, event)
            return event

    async def record_reverse_delta_proof(
        self,
        proof: ModelReverseDeltaProof,
    ) -> None:
        """Persist contiguous reverse-delta coverage after writer quiescence."""
        async with self._connection.transaction():
            row = await self._lock_family(proof.family_id)
            contract = self._contract_from_row(row)
            if EnumCutoverFamilyStatus(str(row["status"])) is (
                EnumCutoverFamilyStatus.BLOCKED
            ):
                raise ValueError("blocked family cannot advertise reverse-delta proof")
            if (
                contract.post_checkpoint_mode
                is not EnumPostCheckpointMode.REVERSE_DELTA
            ):
                raise ValueError("forward-fix-only family cannot record reverse delta")
            if row["first_target_sequence"] is None:
                raise ValueError("no target-only write has been proven")
            if row["quiesced_target_sequence"] is None:
                raise ValueError("writer must be durably quiesced before proof")
            if proof.start_sequence != int(row["first_target_sequence"]):
                raise ValueError(
                    "reverse delta does not start at first target-only write"
                )
            if proof.end_sequence != int(row["quiesced_target_sequence"]):
                raise ValueError(
                    "reverse delta does not reach quiesced target sequence"
                )
            if proof.quiescence_event_id != row["quiescence_event_id"]:
                raise ValueError("reverse delta cites the wrong quiescence event")
            quiesced_at = await self._connection.fetchval(
                """
SELECT occurred_at
FROM omninode_internal.cutover_journal
WHERE event_id = $1 AND family_id = $2 AND event_kind = 'writer_quiesced'
""",
                proof.quiescence_event_id,
                proof.family_id,
            )
            if not isinstance(quiesced_at, datetime):
                raise ValueError("reverse delta cites no durable quiescence event")
            reconciled_at = await self._require_pass_receipt_after(
                proof.family_id,
                proof.reconciliation_receipt_id,
                quiesced_at,
            )
            if proof.proven_at < reconciled_at:
                raise ValueError("reverse-delta proof predates its reconciliation")

            readback_artifact = await self._connection.fetchrow(
                """
SELECT content_hash
FROM omninode_internal.reverse_delta_artifacts
WHERE family_id = $1 AND artifact_ref = $2
""",
                proof.family_id,
                proof.behavioral_readback_ref,
            )
            if readback_artifact is None:
                raise ValueError(
                    "behavioral readback ref does not dereference to a "
                    "durably registered artifact"
                )
            for entry in proof.entries:
                artifact = await self._connection.fetchrow(
                    """
SELECT content_hash
FROM omninode_internal.reverse_delta_artifacts
WHERE family_id = $1 AND artifact_ref = $2
""",
                    proof.family_id,
                    entry.inverse_artifact_ref,
                )
                if artifact is None:
                    raise ValueError(
                        f"reverse-delta entry at sequence {entry.target_sequence} "
                        "cites an inverse artifact that is not durably registered"
                    )
                if artifact["content_hash"] != entry.before_image_hash:
                    raise ValueError(
                        f"reverse-delta entry at sequence {entry.target_sequence} "
                        "inverse artifact does not hash-bind to its declared "
                        "before-image"
                    )

            await self._connection.execute(
                """
INSERT INTO omninode_internal.reverse_delta_proofs
  (proof_id, family_id, start_sequence, end_sequence, quiescence_event_id,
   reconciliation_receipt_id, proof_json, proven_at)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
""",
                proof.proof_id,
                proof.family_id,
                proof.start_sequence,
                proof.end_sequence,
                proof.quiescence_event_id,
                proof.reconciliation_receipt_id,
                _canonical_json(proof.model_dump(mode="json")),
                proof.proven_at,
            )
            for entry in proof.entries:
                await self._connection.execute(
                    """
INSERT INTO omninode_internal.reverse_delta_entries
  (entry_id, proof_id, family_id, target_sequence, entry_json)
VALUES ($1, $2, $3, $4, $5::jsonb)
""",
                    entry.entry_id,
                    proof.proof_id,
                    proof.family_id,
                    entry.target_sequence,
                    _canonical_json(entry.model_dump(mode="json")),
                )

    async def register_reverse_delta_artifact(
        self,
        family_id: UUID,
        artifact_ref: str,
        content: dict[str, object],
    ) -> str:
        """Durably register one dereferenceable artifact; return its hash.

        A reverse-delta entry's ``inverse_artifact_ref`` and a proof's
        ``behavioral_readback_ref`` must dereference to an artifact registered
        here before ``record_reverse_delta_proof`` accepts them -- a bare
        string ref with no durable, hash-bound backing is refused.
        """
        content_json = _canonical_json(content)
        content_hash = _sha256(content)
        async with self._connection.transaction():
            await self._lock_family(family_id)
            await self._connection.execute(
                """
INSERT INTO omninode_internal.reverse_delta_artifacts
  (family_id, artifact_ref, content_hash, content_json)
VALUES ($1, $2, $3, $4::jsonb)
ON CONFLICT (family_id, artifact_ref) DO NOTHING
""",
                family_id,
                artifact_ref,
                content_hash,
                content_json,
            )
            stored_hash = await self._connection.fetchval(
                """
SELECT content_hash
FROM omninode_internal.reverse_delta_artifacts
WHERE family_id = $1 AND artifact_ref = $2
""",
                family_id,
                artifact_ref,
            )
            if stored_hash != content_hash:
                raise ValueError(
                    f"artifact ref {artifact_ref!r} is already bound to "
                    "different content"
                )
            return content_hash

    async def get_state(self, family_id: UUID) -> ModelCutoverFamilyState:
        """Read the family-local state projection."""
        row = await self._connection.fetchrow(
            """
SELECT *
FROM omninode_internal.cutover_family_contracts
WHERE family_id = $1
""",
            family_id,
        )
        if row is None:
            raise KeyError(f"unknown cutover family {family_id!r}")
        return self._state_from_row(row)

    async def evaluate_direct_rollback(
        self,
        family_id: UUID,
    ) -> ModelRollbackDecision:
        """Refuse unsafe direct rollback after target-only authority exists."""
        state = await self.get_state(family_id)
        mode = state.contract.post_checkpoint_mode
        if state.status is EnumCutoverFamilyStatus.BLOCKED:
            return ModelRollbackDecision(
                family_id=family_id,
                allowed=False,
                direct_dsn_rollback=False,
                post_checkpoint_mode=mode,
                reason="family is blocked by a failed transformation receipt",
            )
        if state.dual_write_expires_at is not None:
            return ModelRollbackDecision(
                family_id=family_id,
                allowed=False,
                direct_dsn_rollback=False,
                post_checkpoint_mode=mode,
                reason="bounded dual-write is still open and must be quiesced",
            )
        if state.first_target_write_event_id is None:
            return ModelRollbackDecision(
                family_id=family_id,
                allowed=True,
                direct_dsn_rollback=True,
                post_checkpoint_mode=mode,
                reason=(
                    "no target-only authoritative write is proven; source remains current"
                ),
            )
        if mode is EnumPostCheckpointMode.FORWARD_FIX_ONLY:
            return ModelRollbackDecision(
                family_id=family_id,
                allowed=False,
                direct_dsn_rollback=False,
                post_checkpoint_mode=mode,
                reason=(
                    "target-only authority exists and the family is forward-fix-only"
                ),
            )
        if state.verified_reverse_delta_proof_id is None:
            return ModelRollbackDecision(
                family_id=family_id,
                allowed=False,
                direct_dsn_rollback=False,
                post_checkpoint_mode=mode,
                reason=(
                    "target-only authority exists; complete reverse delta, writer "
                    "quiescence, reconciliation, and behavioral readback are unproven"
                ),
            )
        return ModelRollbackDecision(
            family_id=family_id,
            allowed=True,
            direct_dsn_rollback=True,
            post_checkpoint_mode=mode,
            reason=(
                "writer is quiesced and complete reverse delta, reconciliation, "
                "and behavioral readback are durably proven"
            ),
            reverse_delta_proof_id=state.verified_reverse_delta_proof_id,
        )

    async def _lock_family(self, family_id: UUID) -> asyncpg.Record:
        row = await self._connection.fetchrow(
            """
SELECT *
FROM omninode_internal.cutover_family_contracts
WHERE family_id = $1
FOR UPDATE
""",
            family_id,
        )
        if row is None:
            raise KeyError(f"unknown cutover family {family_id!r}")
        return row

    @staticmethod
    def _contract_from_row(row: asyncpg.Record) -> ModelCutoverFamilyContract:
        raw = row["contract_json"]
        if isinstance(raw, str):
            return ModelCutoverFamilyContract.model_validate_json(raw)
        return ModelCutoverFamilyContract.model_validate_json(_canonical_json(raw))

    def _state_from_row(self, row: asyncpg.Record) -> ModelCutoverFamilyState:
        return ModelCutoverFamilyState(
            contract=self._contract_from_row(row),
            status=EnumCutoverFamilyStatus(str(row["status"])),
            last_known_good_receipt_id=row["last_known_good_receipt_id"],
            blocked_receipt_id=row["blocked_receipt_id"],
            checkpoint_event_id=row["checkpoint_event_id"],
            first_target_write_event_id=row["first_target_write_event_id"],
            first_target_sequence=row["first_target_sequence"],
            quiescence_event_id=row["quiescence_event_id"],
            quiesced_target_sequence=row["quiesced_target_sequence"],
            verified_reverse_delta_proof_id=row["verified_reverse_delta_proof_id"],
            dual_write_expires_at=row["dual_write_expires_at"],
            observation_ends_at=row["observation_ends_at"],
            last_event_at=row["last_event_at"],
            last_sequence=int(row["last_sequence"]),
            last_event_hash=str(row["last_event_hash"]),
        )

    async def _require_pass_receipt(
        self,
        family_id: UUID,
        receipt_id: UUID,
    ) -> None:
        status = await self._connection.fetchval(
            """
SELECT status
FROM omninode_internal.transformation_receipts
WHERE receipt_id = $1 AND family_id = $2
""",
            receipt_id,
            family_id,
        )
        if status != EnumReceiptStatus.PASS.value:
            raise ValueError("journal transition requires a PASS family receipt")

    async def _require_resolution_receipt(
        self,
        family_id: UUID,
        receipt_id: UUID,
        blocked_receipt_id: UUID | None,
    ) -> None:
        row = await self._connection.fetchrow(
            """
SELECT repair.status AS repair_status,
       repair.generated_at AS repair_generated_at,
       blocked.generated_at AS blocked_generated_at
FROM omninode_internal.transformation_receipts repair
JOIN omninode_internal.transformation_receipts blocked
  ON blocked.receipt_id = $3 AND blocked.family_id = repair.family_id
WHERE repair.receipt_id = $1 AND repair.family_id = $2
""",
            receipt_id,
            family_id,
            blocked_receipt_id,
        )
        if row is None or row["repair_status"] != EnumReceiptStatus.PASS.value:
            raise ValueError("mismatch resolution requires a PASS family receipt")
        if row["repair_generated_at"] <= row["blocked_generated_at"]:
            raise ValueError("mismatch resolution receipt must postdate the failure")

    async def _require_pass_receipt_after(
        self,
        family_id: UUID,
        receipt_id: UUID,
        not_before: datetime,
    ) -> datetime:
        generated_at = await self._connection.fetchval(
            """
SELECT generated_at
FROM omninode_internal.transformation_receipts
WHERE receipt_id = $1 AND family_id = $2 AND status = 'pass'
""",
            receipt_id,
            family_id,
        )
        if not isinstance(generated_at, datetime) or generated_at <= not_before:
            raise ValueError("post-write proof requires a fresh reconciliation receipt")
        return generated_at

    async def _validate_transition(
        self,
        row: asyncpg.Record,
        contract: ModelCutoverFamilyContract,
        request: ModelCutoverJournalRequest,
    ) -> None:
        kind = request.kind
        status = EnumCutoverFamilyStatus(str(row["status"]))
        previous_raw = row["last_event_kind"]
        previous = EnumCutoverEventKind(previous_raw) if previous_raw else None
        if (
            row["last_event_at"] is not None
            and request.occurred_at < row["last_event_at"]
        ):
            raise ValueError("journal event time precedes the durable prior event")

        if status is EnumCutoverFamilyStatus.BLOCKED:
            if kind is not EnumCutoverEventKind.MISMATCH_RESOLVED:
                raise ValueError("family is blocked; silent fallback is forbidden")
            if request.receipt_id is None:
                raise ValueError("mismatch resolution requires a PASS receipt")
            await self._require_resolution_receipt(
                contract.family_id,
                request.receipt_id,
                row["blocked_receipt_id"],
            )
            return

        if kind is EnumCutoverEventKind.MISMATCH_RESOLVED:
            raise ValueError("family is not blocked")

        dual_expiry = row["dual_write_expires_at"]
        if (
            dual_expiry is not None
            and kind is not EnumCutoverEventKind.DUAL_WRITE_ENDED
        ):
            raise ValueError("bounded dual-write must end before another transition")

        if request.receipt_id is not None:
            await self._require_pass_receipt(contract.family_id, request.receipt_id)

        if kind is EnumCutoverEventKind.BACKFILL_STARTED:
            if previous not in (None, EnumCutoverEventKind.PRE_CHECKPOINT_ROLLBACK):
                raise ValueError("backfill start is out of order")
        elif kind is EnumCutoverEventKind.BACKFILL_COMPLETED:
            self._require_previous(previous, EnumCutoverEventKind.BACKFILL_STARTED)
        elif kind is EnumCutoverEventKind.DUAL_WRITE_STARTED:
            self._require_previous(previous, EnumCutoverEventKind.BACKFILL_COMPLETED)
            self._validate_dual_write_window(contract, request)
        elif kind is EnumCutoverEventKind.DUAL_WRITE_ENDED:
            self._require_previous(previous, EnumCutoverEventKind.DUAL_WRITE_STARTED)
            if dual_expiry is None:
                raise ValueError("dual-write is not open")
            if request.occurred_at > dual_expiry:
                raise ValueError("dual-write exceeded its declared hard deadline")
        elif kind is EnumCutoverEventKind.FINAL_DELTA_APPLIED:
            if previous not in (
                EnumCutoverEventKind.BACKFILL_COMPLETED,
                EnumCutoverEventKind.DUAL_WRITE_ENDED,
            ):
                raise ValueError("final delta requires completed backfill/dual-write")
        elif kind is EnumCutoverEventKind.WRITER_CHECKPOINT:
            self._require_previous(previous, EnumCutoverEventKind.FINAL_DELTA_APPLIED)
            if request.source_binding_ref != contract.source_binding_ref:
                raise ValueError("checkpoint source binding differs from contract")
            if request.target_binding_ref != contract.target_binding_ref:
                raise ValueError("checkpoint target binding differs from contract")
        elif kind is EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN:
            self._require_previous(previous, EnumCutoverEventKind.WRITER_CHECKPOINT)
            await self._require_durable_write_proof(
                contract.family_id, request.application_path_write_proof
            )
        elif kind is EnumCutoverEventKind.READER_CUTOVER:
            self._require_previous(
                previous,
                EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
            )
        elif kind is EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED:
            self._require_previous(previous, EnumCutoverEventKind.READER_CUTOVER)
            minimum_end = request.occurred_at + timedelta(
                seconds=contract.observation_window_seconds
            )
            if request.observation_ends_at is None:
                raise ValueError("observation end is missing")
            if request.observation_ends_at < minimum_end:
                raise ValueError("observation window is shorter than its contract")
        elif kind is EnumCutoverEventKind.OBSERVATION_WINDOW_COMPLETED:
            self._require_previous(
                previous,
                EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED,
            )
            observation_ends_at = row["observation_ends_at"]
            if observation_ends_at is None:
                raise ValueError("declared observation deadline is not durable")
            # Server-clock-authoritative: gate on the database's own
            # clock_timestamp(), never on the caller-supplied occurred_at.
            # A caller can claim any occurred_at; only the server's live
            # clock proves the window has actually elapsed in real time.
            server_now = await self._connection.fetchval("SELECT clock_timestamp()")
            if server_now < observation_ends_at:
                raise ValueError(
                    "observation window has not reached its deadline per the "
                    "server clock"
                )
        elif kind is EnumCutoverEventKind.WRITER_QUIESCED:
            if row["first_target_write_event_id"] is None:
                raise ValueError("writer cannot quiesce before target-only write proof")
            if request.target_sequence is None:
                raise ValueError("writer quiescence sequence is missing")
            if request.target_sequence < int(row["first_target_sequence"]):
                raise ValueError("quiescence sequence precedes first target write")
        elif kind is EnumCutoverEventKind.REVERSE_DELTA_PROVEN:
            await self._validate_reverse_delta_event(row, contract, request)
        elif kind is EnumCutoverEventKind.FORWARD_FIX_RECORDED:
            if (
                contract.post_checkpoint_mode
                is not EnumPostCheckpointMode.FORWARD_FIX_ONLY
            ):
                raise ValueError("reverse-delta family cannot claim forward-fix-only")
            if row["first_target_write_event_id"] is None:
                raise ValueError("forward fix is not post-checkpoint evidence")
            first_write_at = await self._connection.fetchval(
                """
SELECT occurred_at
FROM omninode_internal.cutover_journal
WHERE event_id = $1 AND family_id = $2
  AND event_kind = 'application_path_write_proven'
""",
                row["first_target_write_event_id"],
                contract.family_id,
            )
            if first_write_at is None or request.receipt_id is None:
                raise ValueError("forward fix lacks durable target-write evidence")
            await self._require_pass_receipt_after(
                contract.family_id,
                request.receipt_id,
                first_write_at,
            )
        elif kind is EnumCutoverEventKind.PRE_CHECKPOINT_ROLLBACK:
            if row["first_target_write_event_id"] is not None:
                raise ValueError(
                    "direct pre-checkpoint rollback after target write refused"
                )

    async def _require_durable_write_proof(
        self,
        family_id: UUID,
        proof: ModelApplicationPathWriteProof | None,
    ) -> None:
        """Refuse a write proof unless it matches a durably verified record.

        A shape-valid ``ModelApplicationPathWriteProof`` is not sufficient --
        it must dereference to the row
        ``PostgresTransformationEvidenceCollector.verify_application_path_write``
        wrote to ``omninode_internal.application_path_write_proofs`` and match
        it field-for-field. A hand-constructed proof that never passed
        through the collector has no durable row and is rejected here.
        """
        if proof is None:
            raise ValueError("application-path write proof is missing")
        if proof.family_id != family_id:
            raise ValueError("application-path write proof belongs to another family")
        durable = await self._connection.fetchrow(
            """
SELECT database_ref, principal, schema_ref, verification_query_hash,
       write_result_hash, backend_pid, collected_at
FROM omninode_internal.application_path_write_proofs
WHERE family_id = $1 AND target_sequence = $2
""",
            family_id,
            proof.target_sequence,
        )
        if durable is None:
            raise ValueError(
                "application-path write proof was never durably verified by "
                "the evidence collector"
            )
        mismatch = (
            durable["database_ref"] != proof.database_ref
            or durable["principal"] != proof.principal
            or durable["schema_ref"] != proof.schema_ref
            or durable["verification_query_hash"] != proof.verification_query_hash
            or durable["write_result_hash"] != proof.write_result_hash
            or int(durable["backend_pid"]) != proof.connection_identity.backend_pid
            or durable["collected_at"] != proof.connection_identity.collected_at
        )
        if mismatch:
            raise ValueError(
                "application-path write proof does not match its durably "
                "verified record"
            )

    @staticmethod
    def _require_previous(
        actual: EnumCutoverEventKind | None,
        expected: EnumCutoverEventKind,
    ) -> None:
        if actual is not expected:
            raise ValueError(
                f"cutover transition requires {expected.value}, found "
                f"{actual.value if actual else 'no prior event'}"
            )

    @staticmethod
    def _validate_dual_write_window(
        contract: ModelCutoverFamilyContract,
        request: ModelCutoverJournalRequest,
    ) -> None:
        if contract.dual_write_max_seconds == 0:
            raise ValueError("dual-write is disabled by the family contract")
        if request.dual_write_expires_at is None:
            raise ValueError("dual-write deadline is missing")
        if request.dual_write_expires_at <= request.occurred_at:
            raise ValueError("dual-write deadline must be in the future")
        maximum = request.occurred_at + timedelta(
            seconds=contract.dual_write_max_seconds
        )
        if request.dual_write_expires_at > maximum:
            raise ValueError("dual-write deadline exceeds the contract maximum")

    async def _validate_reverse_delta_event(
        self,
        row: asyncpg.Record,
        contract: ModelCutoverFamilyContract,
        request: ModelCutoverJournalRequest,
    ) -> None:
        if contract.post_checkpoint_mode is not EnumPostCheckpointMode.REVERSE_DELTA:
            raise ValueError("forward-fix-only family cannot prove reverse delta")
        if row["quiescence_event_id"] is None:
            raise ValueError("writer quiescence is unproven")
        proof = await self._connection.fetchrow(
            """
SELECT reconciliation_receipt_id, quiescence_event_id, proven_at
FROM omninode_internal.reverse_delta_proofs
WHERE proof_id = $1 AND family_id = $2
""",
            request.reverse_delta_proof_id,
            contract.family_id,
        )
        if proof is None:
            raise ValueError("reverse-delta proof is not durable")
        if proof["quiescence_event_id"] != row["quiescence_event_id"]:
            raise ValueError("reverse-delta proof predates current quiescence")
        if proof["reconciliation_receipt_id"] != request.receipt_id:
            raise ValueError("reverse-delta proof and journal receipt differ")
        if request.occurred_at < proof["proven_at"]:
            raise ValueError("reverse-delta journal event predates its durable proof")

    async def _project_transition(
        self,
        previous_row: asyncpg.Record,
        contract: ModelCutoverFamilyContract,
        event: ModelCutoverJournalEvent,
    ) -> None:
        request = event.request
        kind = request.kind
        projected: dict[str, object] = {
            "status": previous_row["status"],
            "blocked_receipt_id": previous_row["blocked_receipt_id"],
            "checkpoint_event_id": previous_row["checkpoint_event_id"],
            "first_target_write_event_id": previous_row["first_target_write_event_id"],
            "first_target_sequence": previous_row["first_target_sequence"],
            "quiescence_event_id": previous_row["quiescence_event_id"],
            "quiesced_target_sequence": previous_row["quiesced_target_sequence"],
            "verified_reverse_delta_proof_id": previous_row[
                "verified_reverse_delta_proof_id"
            ],
            "dual_write_expires_at": previous_row["dual_write_expires_at"],
            "observation_ends_at": previous_row["observation_ends_at"],
            "last_sequence": event.sequence,
            "last_event_hash": event.event_hash,
            "last_event_kind": previous_row["last_event_kind"],
            "last_event_at": event.request.occurred_at,
        }
        preserve_phase = kind is EnumCutoverEventKind.MISMATCH_RESOLVED
        if not preserve_phase:
            projected["last_event_kind"] = kind.value

        if kind is EnumCutoverEventKind.DUAL_WRITE_STARTED:
            projected["dual_write_expires_at"] = request.dual_write_expires_at
        elif kind is EnumCutoverEventKind.DUAL_WRITE_ENDED:
            projected["dual_write_expires_at"] = None
        elif kind is EnumCutoverEventKind.WRITER_CHECKPOINT:
            projected["checkpoint_event_id"] = event.event_id
            projected["status"] = EnumCutoverFamilyStatus.CHECKPOINTED.value
        elif kind is EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN:
            proof = request.application_path_write_proof
            if proof is None:
                raise AssertionError(
                    "application-path write proof is missing after validation"
                )
            projected["first_target_write_event_id"] = event.event_id
            projected["first_target_sequence"] = proof.target_sequence
        elif kind is EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED:
            projected["status"] = EnumCutoverFamilyStatus.OBSERVING.value
            projected["observation_ends_at"] = request.observation_ends_at
        elif kind is EnumCutoverEventKind.OBSERVATION_WINDOW_COMPLETED:
            projected["status"] = EnumCutoverFamilyStatus.COMPLETE.value
        elif kind is EnumCutoverEventKind.WRITER_QUIESCED:
            projected["quiescence_event_id"] = event.event_id
            projected["quiesced_target_sequence"] = request.target_sequence
        elif kind is EnumCutoverEventKind.REVERSE_DELTA_PROVEN:
            projected["verified_reverse_delta_proof_id"] = (
                request.reverse_delta_proof_id
            )
        elif kind is EnumCutoverEventKind.MISMATCH_RESOLVED:
            projected["blocked_receipt_id"] = None
            projected["status"] = self._status_after_resolution(previous_row).value

        await self._connection.execute(
            """
UPDATE omninode_internal.cutover_family_contracts
SET status = $2,
    blocked_receipt_id = $3,
    checkpoint_event_id = $4,
    first_target_write_event_id = $5,
    first_target_sequence = $6,
    quiescence_event_id = $7,
    quiesced_target_sequence = $8,
    verified_reverse_delta_proof_id = $9,
    dual_write_expires_at = $10,
    observation_ends_at = $11,
    last_sequence = $12,
    last_event_hash = $13,
    last_event_kind = $14,
    last_event_at = $15
WHERE family_id = $1
""",
            contract.family_id,
            projected["status"],
            projected["blocked_receipt_id"],
            projected["checkpoint_event_id"],
            projected["first_target_write_event_id"],
            projected["first_target_sequence"],
            projected["quiescence_event_id"],
            projected["quiesced_target_sequence"],
            projected["verified_reverse_delta_proof_id"],
            projected["dual_write_expires_at"],
            projected["observation_ends_at"],
            projected["last_sequence"],
            projected["last_event_hash"],
            projected["last_event_kind"],
            projected["last_event_at"],
        )

    @staticmethod
    def _status_after_resolution(row: asyncpg.Record) -> EnumCutoverFamilyStatus:
        previous = row["last_event_kind"]
        if previous == EnumCutoverEventKind.OBSERVATION_WINDOW_COMPLETED.value:
            return EnumCutoverFamilyStatus.COMPLETE
        if previous == EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED.value:
            return EnumCutoverFamilyStatus.OBSERVING
        if row["checkpoint_event_id"] is not None:
            return EnumCutoverFamilyStatus.CHECKPOINTED
        return EnumCutoverFamilyStatus.READY


__all__ = ["RepositoryPostgresCutoverJournal"]
