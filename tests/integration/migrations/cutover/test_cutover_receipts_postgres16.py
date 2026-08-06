# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real PostgreSQL 16 receipt, journal, and rollback proof for OMN-15420."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import socket
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import asyncpg
import pytest

from omnibase_infra.migration.cutover import (
    CutoverCoordinator,
    ModelApplicationPathWriteProof,
    ModelConnectionIdentity,
    ModelControlPlaneDeltaEvidence,
    ModelCutoverContinuityEvidence,
    ModelCutoverFamilyContract,
    ModelCutoverJournalRequest,
    ModelPostgresEvidenceQuerySet,
    ModelProjectionReplayEvidence,
    ModelReverseDeltaEntry,
    ModelReverseDeltaProof,
    PostgresTransformationEvidenceCollector,
    RepositoryPostgresCutoverJournal,
)
from omnibase_infra.migration.cutover.enums import (
    EnumCutoverEventKind,
    EnumCutoverFamilyKind,
    EnumCutoverFamilyStatus,
    EnumPostCheckpointMode,
    EnumReceiptStatus,
    EnumReverseDeltaOperation,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.serial,
    pytest.mark.asyncio(loop_scope="module"),
]


def _postgres_bin_dir() -> Path | None:
    initdb = shutil.which("initdb")
    candidates = [Path(initdb).parent] if initdb else []
    candidates.extend(
        sorted(Path("/opt/homebrew/opt").glob("postgresql@*/bin"), reverse=True)
    )
    candidates.extend(sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True))
    for candidate in candidates:
        if not all((candidate / name).is_file() for name in ("initdb", "pg_ctl")):
            continue
        version = subprocess.run(
            [str(candidate / "postgres"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if " 16." in version:
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def postgres_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    bin_dir = _postgres_bin_dir()
    if bin_dir is None:
        pytest.skip("PostgreSQL 16 initdb/pg_ctl are unavailable")
    root = tmp_path_factory.mktemp("omn15420-pg16")
    data = root / "data"
    port = _free_port()
    init = subprocess.run(
        [
            str(bin_dir / "initdb"),
            "-D",
            str(data),
            "-U",
            "postgres",
            "--auth=trust",
            "--no-sync",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if init.returncode != 0:
        pytest.fail(f"PostgreSQL 16 initdb failed: {init.stderr}")
    start = subprocess.run(
        [
            str(bin_dir / "pg_ctl"),
            "-D",
            str(data),
            "-o",
            f"-F -h 127.0.0.1 -p {port}",
            "-l",
            str(root / "postgres.log"),
            "-w",
            "start",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if start.returncode != 0:
        postgres_log = root / "postgres.log"
        log_text = (
            postgres_log.read_text(encoding="utf-8", errors="replace")
            if postgres_log.is_file()
            else ""
        )
        pytest.skip(
            "PostgreSQL 16 binaries are present but an ephemeral cluster could "
            f"not start; pg_ctl stderr={start.stderr!r}; postgres.log={log_text!r}"
        )
    try:
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            [
                str(bin_dir / "pg_ctl"),
                "-D",
                str(data),
                "-m",
                "immediate",
                "-w",
                "stop",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _family_id(family_key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"omninode-cutover:{family_key}")


def _contract(
    family_key: str,
    kind: EnumCutoverFamilyKind,
    mode: EnumPostCheckpointMode,
    source_evidence_contract_hash: str,
    target_evidence_contract_hash: str,
) -> ModelCutoverFamilyContract:
    return ModelCutoverFamilyContract(
        family_id=_family_id(family_key),
        family_key=family_key,
        family_kind=kind,
        source_binding_ref="application.legacy",
        target_binding_ref="application.target",
        source_evidence_contract_hash=source_evidence_contract_hash,
        target_evidence_contract_hash=target_evidence_contract_hash,
        post_checkpoint_mode=mode,
        reverse_delta_contract_ref=(
            "contracts/reverse-delta/usage.yaml"
            if mode is EnumPostCheckpointMode.REVERSE_DELTA
            else ""
        ),
        forward_fix_runbook_ref=(
            "runbooks/control-plane-forward-fix.md"
            if mode is EnumPostCheckpointMode.FORWARD_FIX_ONLY
            else ""
        ),
        dual_write_max_seconds=30,
        observation_window_seconds=2,
    )


def _source_queries() -> ModelPostgresEvidenceQuerySet:
    return ModelPostgresEvidenceQuerySet(
        label="legacy-transformed",
        keys_sql="""
SELECT mapping.tenant_uuid::text || ':' || usage.id::text
FROM legacy_fixture.usage
JOIN legacy_fixture.tenant_mapping mapping
  ON mapping.legacy_slug = usage.tenant_slug
""",
        rows_sql="""
SELECT jsonb_build_array(usage.id, mapping.tenant_uuid, usage.amount)::text
FROM legacy_fixture.usage
JOIN legacy_fixture.tenant_mapping mapping
  ON mapping.legacy_slug = usage.tenant_slug
""",
        foreign_keys_sql="""
SELECT 'tenant_id->tenants.id:' || constraint_row.confdeltype::text
FROM pg_constraint constraint_row
WHERE constraint_row.conrelid = 'legacy_fixture.usage'::regclass
  AND constraint_row.contype = 'f'
""",
        sequences_sql="SELECT 'usage_id:' || last_value::text FROM legacy_fixture.usage_id_seq",
        owners_sql="""
SELECT 'usage:' || role_row.rolname
FROM pg_class class_row
JOIN pg_roles role_row ON role_row.oid = class_row.relowner
WHERE class_row.oid = 'legacy_fixture.usage'::regclass
""",
        grants_sql="""
SELECT lower(grantee) || ':' || lower(privilege_type)
FROM information_schema.role_table_grants
WHERE table_schema = 'legacy_fixture'
  AND table_name = 'usage'
  AND grantee = 'tenant_writer'
""",
        policies_sql="""
SELECT policy_row.polname || ':' || policy_row.polpermissive::text || ':' ||
       policy_row.polcmd::text || ':' ||
       pg_get_expr(policy_row.polqual, policy_row.polrelid)
FROM pg_policy policy_row
WHERE policy_row.polrelid = 'legacy_fixture.usage'::regclass
""",
        views_functions_sql="""
SELECT 'usage_view:' ||
       (coalesce(class_row.reloptions, ARRAY[]::text[]) @>
        ARRAY['security_invoker=true'])::text
FROM pg_class class_row
WHERE class_row.oid = 'legacy_fixture.usage_view'::regclass
UNION ALL
SELECT 'usage_count:' || procedure_row.prosecdef::text
FROM pg_proc procedure_row
WHERE procedure_row.oid = 'legacy_fixture.usage_count()'::regprocedure
""",
        dependencies_sql="""
SELECT 'usage_count'
WHERE to_regprocedure('legacy_fixture.usage_count()') IS NOT NULL
UNION ALL
SELECT 'usage_view'
WHERE to_regclass('legacy_fixture.usage_view') IS NOT NULL
""",
        collisions_sql="""
WITH transformed AS (
  SELECT mapping.tenant_uuid::text || ':' || usage.id::text AS canonical_key
  FROM legacy_fixture.usage
  JOIN legacy_fixture.tenant_mapping mapping
    ON mapping.legacy_slug = usage.tenant_slug
)
SELECT canonical_key
FROM transformed
GROUP BY canonical_key
HAVING count(*) > 1
""",
    )


def _target_queries() -> ModelPostgresEvidenceQuerySet:
    return ModelPostgresEvidenceQuerySet(
        label="target",
        keys_sql="SELECT tenant_id::text || ':' || id::text FROM tenant.usage",
        rows_sql="SELECT jsonb_build_array(id, tenant_id, amount)::text FROM tenant.usage",
        foreign_keys_sql="""
SELECT 'tenant_id->tenants.id:' || constraint_row.confdeltype::text
FROM pg_constraint constraint_row
WHERE constraint_row.conrelid = 'tenant.usage'::regclass
  AND constraint_row.contype = 'f'
""",
        sequences_sql="SELECT 'usage_id:' || last_value::text FROM tenant.usage_id_seq",
        owners_sql="""
SELECT 'usage:' || role_row.rolname
FROM pg_class class_row
JOIN pg_roles role_row ON role_row.oid = class_row.relowner
WHERE class_row.oid = 'tenant.usage'::regclass
""",
        grants_sql="""
SELECT lower(grantee) || ':' || lower(privilege_type)
FROM information_schema.role_table_grants
WHERE table_schema = 'tenant'
  AND table_name = 'usage'
  AND grantee = 'tenant_writer'
""",
        policies_sql="""
SELECT policy_row.polname || ':' || policy_row.polpermissive::text || ':' ||
       policy_row.polcmd::text || ':' ||
       pg_get_expr(policy_row.polqual, policy_row.polrelid)
FROM pg_policy policy_row
WHERE policy_row.polrelid = 'tenant.usage'::regclass
""",
        views_functions_sql="""
SELECT 'usage_view:' ||
       (coalesce(class_row.reloptions, ARRAY[]::text[]) @>
        ARRAY['security_invoker=true'])::text
FROM pg_class class_row
WHERE class_row.oid = 'tenant.usage_view'::regclass
UNION ALL
SELECT 'usage_count:' || procedure_row.prosecdef::text
FROM pg_proc procedure_row
WHERE procedure_row.oid = 'tenant.usage_count()'::regprocedure
""",
        dependencies_sql="""
SELECT 'usage_count'
WHERE to_regprocedure('tenant.usage_count()') IS NOT NULL
UNION ALL
SELECT 'usage_view'
WHERE to_regclass('tenant.usage_view') IS NOT NULL
""",
        collisions_sql="""
SELECT tenant_id::text || ':' || id::text AS canonical_key
FROM tenant.usage
GROUP BY tenant_id, id
HAVING count(*) > 1
""",
    )


async def _seed_transformed_family(connection: asyncpg.Connection) -> None:
    await connection.execute(
        """
CREATE ROLE tenant_writer NOLOGIN;
CREATE SCHEMA legacy_fixture;
CREATE SCHEMA tenant;
CREATE TABLE legacy_fixture.tenants (slug TEXT PRIMARY KEY);
CREATE TABLE legacy_fixture.tenant_mapping (
  legacy_slug TEXT PRIMARY KEY REFERENCES legacy_fixture.tenants(slug),
  tenant_uuid UUID NOT NULL UNIQUE
);
CREATE TABLE legacy_fixture.usage (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  tenant_slug TEXT NOT NULL REFERENCES legacy_fixture.tenants(slug),
  amount INTEGER NOT NULL
);
CREATE TABLE tenant.tenants (id UUID PRIMARY KEY);
CREATE TABLE tenant.usage (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant.tenants(id),
  amount INTEGER NOT NULL
);
INSERT INTO legacy_fixture.tenants VALUES ('alpha'), ('beta');
INSERT INTO legacy_fixture.tenant_mapping VALUES
  ('alpha', '00000000-0000-0000-0000-000000000001'),
  ('beta', '00000000-0000-0000-0000-000000000002');
INSERT INTO legacy_fixture.usage (id, tenant_slug, amount) VALUES
  (1, 'alpha', 10), (2, 'beta', 20);
INSERT INTO tenant.tenants VALUES
  ('00000000-0000-0000-0000-000000000001'),
  ('00000000-0000-0000-0000-000000000002');
INSERT INTO tenant.usage (id, tenant_id, amount) VALUES
  (1, '00000000-0000-0000-0000-000000000001', 10),
  (2, '00000000-0000-0000-0000-000000000002', 20);
SELECT setval('legacy_fixture.usage_id_seq', 2, true);
SELECT setval('tenant.usage_id_seq', 2, true);
GRANT SELECT ON legacy_fixture.usage TO tenant_writer;
GRANT SELECT ON tenant.usage TO tenant_writer;
ALTER TABLE legacy_fixture.usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE legacy_fixture.usage FORCE ROW LEVEL SECURITY;
ALTER TABLE tenant.usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant.usage FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON legacy_fixture.usage USING (true);
CREATE POLICY tenant_isolation ON tenant.usage USING (true);
CREATE VIEW legacy_fixture.usage_view WITH (security_invoker=true)
  AS SELECT * FROM legacy_fixture.usage;
CREATE VIEW tenant.usage_view WITH (security_invoker=true)
  AS SELECT * FROM tenant.usage;
CREATE FUNCTION legacy_fixture.usage_count() RETURNS BIGINT
  LANGUAGE SQL STABLE AS 'SELECT count(*) FROM legacy_fixture.usage';
CREATE FUNCTION tenant.usage_count() RETURNS BIGINT
  LANGUAGE SQL STABLE AS 'SELECT count(*) FROM tenant.usage';
REVOKE ALL ON FUNCTION legacy_fixture.usage_count() FROM PUBLIC;
REVOKE ALL ON FUNCTION tenant.usage_count() FROM PUBLIC;
"""
    )


async def test_postgres16_receipts_journal_and_rollback_boundaries(
    postgres_dsn: str,
) -> None:
    connection = await asyncpg.connect(postgres_dsn)
    try:
        await _seed_transformed_family(connection)
        repository = RepositoryPostgresCutoverJournal(connection)
        await repository.initialize()
        await repository.initialize()
        coordinator = CutoverCoordinator(repository)
        collector = PostgresTransformationEvidenceCollector(connection)
        source, target = await collector.collect_pair(
            _source_queries(),
            "application.legacy",
            _target_queries(),
            "application.target",
        )
        assert (
            source.connection_identity.database == target.connection_identity.database
        )
        assert (
            source.connection_identity.backend_pid
            == target.connection_identity.backend_pid
        )
        assert source.binding_ref == "application.legacy"
        assert target.binding_ref == "application.target"

        projection = _contract(
            "tenant.usage",
            EnumCutoverFamilyKind.PROJECTION,
            EnumPostCheckpointMode.REVERSE_DELTA,
            source.evidence_contract_hash,
            target.evidence_contract_hash,
        )
        control = _contract(
            "tenant.control-plane",
            EnumCutoverFamilyKind.CONTROL_PLANE,
            EnumPostCheckpointMode.FORWARD_FIX_ONLY,
            source.evidence_contract_hash,
            target.evidence_contract_hash,
        )
        await coordinator.register_family(projection)
        await coordinator.register_family(control)
        now = datetime.now(UTC) - timedelta(minutes=1)

        projection_continuity = ModelCutoverContinuityEvidence(
            projection_replays=(
                ModelProjectionReplayEvidence(
                    projection_id=uuid5(
                        NAMESPACE_URL,
                        "projection:usage_projection",
                    ),
                    projection_label="usage_projection",
                    projection_version="v7",
                    topic="onex.evt.usage-recorded.v1",
                    partition=0,
                    source_offset=42,
                    target_offset=42,
                ),
            )
        )
        projection_receipt = await coordinator.reconcile(
            projection,
            source,
            target,
            projection_continuity,
            "idem:reconcile/projection/initial",
        )
        assert projection_receipt.status is EnumReceiptStatus.PASS

        # Idempotency: retrying reconcile with the same key and identical
        # inputs returns the original persisted receipt, not a fresh one.
        replayed_projection_receipt = await coordinator.reconcile(
            projection,
            source,
            target,
            projection_continuity,
            "idem:reconcile/projection/initial",
        )
        assert replayed_projection_receipt.receipt_id == projection_receipt.receipt_id
        assert (
            replayed_projection_receipt.generated_at == projection_receipt.generated_at
        )
        with pytest.raises(ValueError, match="different reconciliation content"):
            await coordinator.reconcile(
                projection,
                source,
                target.model_copy(update={"owners": ("usage:different_owner",)}),
                projection_continuity,
                "idem:reconcile/projection/initial",
            )

        digest = _hash("control-plane")
        control_continuity = ModelCutoverContinuityEvidence(
            control_plane_delta=ModelControlPlaneDeltaEvidence(
                snapshot_id=uuid4(),
                source_snapshot_hash=digest,
                target_snapshot_hash=digest,
                final_delta_id=uuid4(),
                source_final_delta_hash=digest,
                target_final_delta_hash=digest,
                source_watermark="42",
                target_watermark="42",
            )
        )
        old_control_receipt = await coordinator.reconcile(
            control,
            source,
            target,
            control_continuity,
            "idem:reconcile/control/old",
        )
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await connection.execute(
                """
INSERT INTO omninode_internal.cutover_journal
  (event_id, family_id, sequence, event_kind, request_json, receipt_id,
   previous_event_hash, event_hash, occurred_at, idempotency_key)
VALUES ($1, $2, 99, 'backfill_completed', '{}'::jsonb, $3,
        repeat('0', 64), repeat('f', 64), clock_timestamp(), 'idem:fk-violation-probe')
""",
                uuid4(),
                projection.family_id,
                old_control_receipt.receipt_id,
            )
        failed_target = target.model_copy(update={"owners": ("usage:wrong_owner",)})
        failed_control_receipt = await coordinator.reconcile(
            control,
            source,
            failed_target,
            control_continuity,
            "idem:reconcile/control/failed",
        )
        assert failed_control_receipt.status is EnumReceiptStatus.FAIL
        assert (await repository.get_state(control.family_id)).status is (
            EnumCutoverFamilyStatus.BLOCKED
        )
        assert (await repository.get_state(projection.family_id)).status is (
            EnumCutoverFamilyStatus.READY
        )
        with pytest.raises(ValueError, match="silent fallback"):
            await coordinator.append(
                control.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.BACKFILL_STARTED,
                    occurred_at=now,
                    evidence_ref="proof/control/backfill",
                    idempotency_key="idem:proof/control/backfill",
                ),
            )
        with pytest.raises(ValueError, match="postdate the failure"):
            await coordinator.append(
                control.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.MISMATCH_RESOLVED,
                    occurred_at=now,
                    evidence_ref="proof/control/stale-pass-replay",
                    idempotency_key="idem:proof/control/stale-pass-replay",
                    receipt_id=old_control_receipt.receipt_id,
                ),
            )

        control_receipt = await coordinator.reconcile(
            control,
            source,
            target,
            control_continuity,
            "idem:reconcile/control/resolved",
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.MISMATCH_RESOLVED,
                occurred_at=now,
                evidence_ref="proof/control/mismatch-resolved",
                idempotency_key="idem:proof/control/mismatch-resolved",
                receipt_id=control_receipt.receipt_id,
            ),
        )

        assert (
            await coordinator.evaluate_direct_rollback(projection.family_id)
        ).allowed
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.BACKFILL_STARTED,
                occurred_at=now,
                evidence_ref="proof/projection/backfill-start",
                idempotency_key="idem:proof/projection/backfill-start",
            ),
        )
        with pytest.raises(ValueError, match="precedes the durable prior event"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.BACKFILL_COMPLETED,
                    occurred_at=now - timedelta(seconds=1),
                    evidence_ref="proof/projection/time-travel",
                    idempotency_key="idem:proof/projection/time-travel",
                    receipt_id=projection_receipt.receipt_id,
                ),
            )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.BACKFILL_COMPLETED,
                occurred_at=now + timedelta(seconds=1),
                evidence_ref="proof/projection/backfill-complete",
                idempotency_key="idem:proof/projection/backfill-complete",
                receipt_id=projection_receipt.receipt_id,
            ),
        )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.DUAL_WRITE_STARTED,
                occurred_at=now + timedelta(seconds=2),
                evidence_ref="proof/projection/dual-write-telemetry",
                idempotency_key="idem:proof/projection/dual-write-telemetry",
                dual_write_expires_at=now + timedelta(seconds=12),
            ),
        )
        dual_write_rollback = await coordinator.evaluate_direct_rollback(
            projection.family_id
        )
        assert not dual_write_rollback.allowed
        assert "dual-write" in dual_write_rollback.reason
        with pytest.raises(ValueError, match="must end"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.FINAL_DELTA_APPLIED,
                    occurred_at=now + timedelta(seconds=3),
                    evidence_ref="proof/projection/final-delta-early",
                    idempotency_key="idem:proof/projection/final-delta-early",
                    receipt_id=projection_receipt.receipt_id,
                ),
            )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.DUAL_WRITE_ENDED,
                occurred_at=now + timedelta(seconds=4),
                evidence_ref="proof/projection/dual-write-ended",
                idempotency_key="idem:proof/projection/dual-write-ended",
            ),
        )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.FINAL_DELTA_APPLIED,
                occurred_at=now + timedelta(seconds=5),
                evidence_ref="proof/projection/final-delta",
                idempotency_key="idem:proof/projection/final-delta",
                receipt_id=projection_receipt.receipt_id,
            ),
        )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.WRITER_CHECKPOINT,
                occurred_at=now + timedelta(seconds=6),
                evidence_ref="proof/projection/writer-checkpoint",
                idempotency_key="idem:proof/projection/writer-checkpoint",
                receipt_id=projection_receipt.receipt_id,
                source_binding_ref=projection.source_binding_ref,
                target_binding_ref=projection.target_binding_ref,
            ),
        )
        before_write = await coordinator.evaluate_direct_rollback(projection.family_id)
        assert before_write.allowed and before_write.direct_dsn_rollback

        # Gap 1 RED controls: verify_application_path_write refuses a
        # mutating query, a query with no rows, and a schema that does not
        # exist on the verifying connection -- it never trusts a caller
        # string.
        with pytest.raises(ValueError, match="SELECT or WITH"):
            await collector.verify_application_path_write(
                projection.family_id,
                "UPDATE tenant.usage SET amount = amount",
                "tenant",
                7,
            )
        with pytest.raises(ValueError, match="read-only"):
            await collector.verify_application_path_write(
                projection.family_id,
                "SELECT count(*) FROM tenant.usage WHERE 1=1 OR DELETE",
                "tenant",
                7,
            )
        with pytest.raises(ValueError, match="no rows"):
            await collector.verify_application_path_write(
                projection.family_id,
                "SELECT id FROM tenant.usage WHERE id = -1",
                "tenant",
                7,
            )
        with pytest.raises(ValueError, match="does not exist"):
            await collector.verify_application_path_write(
                projection.family_id,
                "SELECT 1",
                "does_not_exist",
                7,
            )

        # GAP 1 forgery-refusal RED control: a shape-valid, hand-constructed
        # ModelApplicationPathWriteProof that never passed through
        # verify_application_path_write (and so has no durable row in
        # omninode_internal.application_path_write_proofs) must be refused
        # by append_event, not accepted on shape validity alone.
        forged_write_proof = ModelApplicationPathWriteProof(
            family_id=projection.family_id,
            database_ref="totally_fake_db",
            principal="attacker",
            schema_ref="nonexistent_schema",
            target_sequence=7,
            verification_query_hash="a" * 64,
            write_result_hash="b" * 64,
            connection_identity=ModelConnectionIdentity(
                database="totally_fake_db",
                backend_pid=1,
                collected_at=datetime.now(UTC),
            ),
        )
        with pytest.raises(ValueError, match="never durably verified"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
                    occurred_at=now + timedelta(seconds=7),
                    evidence_ref="proof/projection/forged-application-path-write",
                    idempotency_key="idem:proof/projection/forged-write",
                    application_path_write_proof=forged_write_proof,
                ),
            )

        # GREEN: an independently, server-verified write proof -- database
        # and principal come from a live current_database()/current_user
        # readback on the connection, never from a caller-typed string.
        write_proof = await collector.verify_application_path_write(
            projection.family_id,
            "SELECT id, tenant_id, amount FROM tenant.usage ORDER BY id",
            "tenant",
            7,
        )
        assert write_proof.database_ref == write_proof.connection_identity.database
        assert write_proof.principal
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
                occurred_at=now + timedelta(seconds=7),
                evidence_ref="proof/projection/real-application-path-write",
                idempotency_key="idem:proof/projection/real-application-path-write",
                application_path_write_proof=write_proof,
            ),
        )
        after_write = await coordinator.evaluate_direct_rollback(projection.family_id)
        assert not after_write.allowed
        assert "complete reverse delta" in after_write.reason
        with pytest.raises(ValueError, match="after target write refused"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.PRE_CHECKPOINT_ROLLBACK,
                    occurred_at=now + timedelta(seconds=8),
                    evidence_ref="proof/projection/unsafe-direct-rollback",
                    idempotency_key="idem:proof/projection/unsafe-direct-rollback",
                ),
            )
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.READER_CUTOVER,
                occurred_at=now + timedelta(seconds=8),
                evidence_ref="proof/projection/reader-cutover",
                idempotency_key="idem:proof/projection/reader-cutover",
            ),
        )
        # Gap 2: server-clock-authoritative observation deadline. Anchor the
        # window to real wall-clock time (not the backdated `now`) so a too-
        # early completion attempt is genuinely refused, and only a real
        # elapsed sleep proves the window, per the database's own
        # clock_timestamp() -- never the caller-supplied occurred_at.
        observation_started_at = datetime.now(UTC)
        observation_ends_at = observation_started_at + timedelta(seconds=2)
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED,
                occurred_at=observation_started_at,
                evidence_ref="proof/projection/observation-start",
                idempotency_key="idem:proof/projection/observation-start",
                observation_ends_at=observation_ends_at,
            ),
        )
        with pytest.raises(ValueError, match="server clock"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.OBSERVATION_WINDOW_COMPLETED,
                    # A caller lying that the deadline has already passed
                    # must still be refused: only real elapsed server time
                    # proves the window.
                    occurred_at=observation_ends_at + timedelta(seconds=1),
                    evidence_ref="proof/projection/observation-too-early",
                    idempotency_key="idem:proof/projection/observation-too-early",
                ),
            )
        await asyncio.sleep(2.1)
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.OBSERVATION_WINDOW_COMPLETED,
                occurred_at=datetime.now(UTC),
                evidence_ref="proof/projection/observation-complete",
                idempotency_key="idem:proof/projection/observation-complete",
            ),
        )
        quiescence = await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.WRITER_QUIESCED,
                occurred_at=datetime.now(UTC),
                evidence_ref="proof/projection/writer-quiesced",
                idempotency_key="idem:proof/projection/writer-quiesced",
                target_sequence=8,
            ),
        )

        # Idempotency at the journal-append layer: retrying the same
        # (family_id, idempotency_key) with identical content returns the
        # already-durable event rather than advancing the sequence again;
        # the same key with different content is refused.
        replayed_quiescence = await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.WRITER_QUIESCED,
                occurred_at=quiescence.request.occurred_at,
                evidence_ref="proof/projection/writer-quiesced",
                idempotency_key="idem:proof/projection/writer-quiesced",
                target_sequence=8,
            ),
        )
        assert replayed_quiescence.event_id == quiescence.event_id
        assert replayed_quiescence.sequence == quiescence.sequence
        with pytest.raises(ValueError, match="different request"):
            await coordinator.append(
                projection.family_id,
                ModelCutoverJournalRequest(
                    kind=EnumCutoverEventKind.WRITER_QUIESCED,
                    occurred_at=datetime.now(UTC),
                    evidence_ref="proof/projection/writer-quiesced-retry-mismatch",
                    idempotency_key="idem:proof/projection/writer-quiesced",
                    target_sequence=8,
                ),
            )

        reverse_reconciliation_receipt = await coordinator.reconcile(
            projection,
            source,
            target,
            projection_continuity,
            "idem:reconcile/projection/reverse-delta",
        )
        # Gap 4: reverse-delta entries must dereference + hash-bind to a
        # durably registered artifact -- a bare ref string is refused.
        before_image_hashes: dict[int, str] = {}
        for sequence in (7, 8):
            before_image_hashes[
                sequence
            ] = await coordinator.register_reverse_delta_artifact(
                projection.family_id,
                f"proof/reverse-delta/{sequence}",
                {"target_sequence": sequence, "before": f"before-{sequence}"},
            )
        entries = tuple(
            ModelReverseDeltaEntry(
                entry_id=uuid4(),
                family_id=projection.family_id,
                target_sequence=sequence,
                relation="tenant.usage",
                operation=EnumReverseDeltaOperation.INSERT,
                primary_key_hash=_hash(f"pk-{sequence}"),
                before_image_hash=before_image_hashes[sequence],
                after_image_hash=_hash(f"after-{sequence}"),
                inverse_artifact_ref=f"proof/reverse-delta/{sequence}",
            )
            for sequence in (7, 8)
        )
        unregistered_readback_proof = ModelReverseDeltaProof(
            proof_id=uuid4(),
            family_id=projection.family_id,
            start_sequence=7,
            end_sequence=8,
            entries=entries,
            quiescence_event_id=quiescence.event_id,
            reconciliation_receipt_id=reverse_reconciliation_receipt.receipt_id,
            behavioral_readback_ref="proof/reverse-delta/never-registered",
            proven_at=datetime.now(UTC),
        )
        with pytest.raises(ValueError, match="behavioral readback"):
            await coordinator.record_reverse_delta(unregistered_readback_proof)

        mismatched_entries = tuple(
            entry.model_copy(
                update={"before_image_hash": _hash(f"tampered-{entry.target_sequence}")}
            )
            for entry in entries
        )
        mismatched_proof = ModelReverseDeltaProof(
            proof_id=uuid4(),
            family_id=projection.family_id,
            start_sequence=7,
            end_sequence=8,
            entries=mismatched_entries,
            quiescence_event_id=quiescence.event_id,
            reconciliation_receipt_id=reverse_reconciliation_receipt.receipt_id,
            behavioral_readback_ref="proof/reverse-delta/behavioral-readback",
            proven_at=datetime.now(UTC),
        )
        await coordinator.register_reverse_delta_artifact(
            projection.family_id,
            "proof/reverse-delta/behavioral-readback",
            {"family_id": str(projection.family_id), "readback": "matches-source"},
        )
        with pytest.raises(ValueError, match="hash-bind"):
            await coordinator.record_reverse_delta(mismatched_proof)

        reverse_proof = ModelReverseDeltaProof(
            proof_id=uuid4(),
            family_id=projection.family_id,
            start_sequence=7,
            end_sequence=8,
            entries=entries,
            quiescence_event_id=quiescence.event_id,
            reconciliation_receipt_id=reverse_reconciliation_receipt.receipt_id,
            behavioral_readback_ref="proof/reverse-delta/behavioral-readback",
            proven_at=datetime.now(UTC),
        )
        await coordinator.record_reverse_delta(reverse_proof)
        assert not (
            await coordinator.evaluate_direct_rollback(projection.family_id)
        ).allowed
        await coordinator.append(
            projection.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.REVERSE_DELTA_PROVEN,
                occurred_at=datetime.now(UTC),
                evidence_ref="proof/reverse-delta/complete",
                idempotency_key="idem:proof/reverse-delta/complete",
                receipt_id=reverse_reconciliation_receipt.receipt_id,
                reverse_delta_proof_id=reverse_proof.proof_id,
            ),
        )
        reverse_allowed = await coordinator.evaluate_direct_rollback(
            projection.family_id
        )
        assert reverse_allowed.allowed and reverse_allowed.direct_dsn_rollback
        assert reverse_allowed.reverse_delta_proof_id == reverse_proof.proof_id

        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.BACKFILL_STARTED,
                occurred_at=now,
                evidence_ref="proof/control/backfill-start",
                idempotency_key="idem:proof/control/backfill-start",
            ),
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.BACKFILL_COMPLETED,
                occurred_at=now + timedelta(seconds=1),
                evidence_ref="proof/control/backfill-complete",
                idempotency_key="idem:proof/control/backfill-complete",
                receipt_id=control_receipt.receipt_id,
            ),
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.FINAL_DELTA_APPLIED,
                occurred_at=now + timedelta(seconds=2),
                evidence_ref="proof/control/final-delta",
                idempotency_key="idem:proof/control/final-delta",
                receipt_id=control_receipt.receipt_id,
            ),
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.WRITER_CHECKPOINT,
                occurred_at=now + timedelta(seconds=3),
                evidence_ref="proof/control/checkpoint",
                idempotency_key="idem:proof/control/checkpoint",
                receipt_id=control_receipt.receipt_id,
                source_binding_ref=control.source_binding_ref,
                target_binding_ref=control.target_binding_ref,
            ),
        )
        control_write_proof = await collector.verify_application_path_write(
            control.family_id,
            "SELECT id, tenant_id, amount FROM tenant.usage ORDER BY id",
            "tenant",
            1,
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN,
                occurred_at=now + timedelta(seconds=4),
                evidence_ref="proof/control/real-application-path-write",
                idempotency_key="idem:proof/control/real-application-path-write",
                application_path_write_proof=control_write_proof,
            ),
        )
        forward_fix_receipt = await coordinator.reconcile(
            control,
            source,
            target,
            control_continuity,
            "idem:reconcile/control/forward-fix",
        )
        await coordinator.append(
            control.family_id,
            ModelCutoverJournalRequest(
                kind=EnumCutoverEventKind.FORWARD_FIX_RECORDED,
                occurred_at=now + timedelta(seconds=5),
                evidence_ref="proof/control/forward-fix",
                idempotency_key="idem:proof/control/forward-fix",
                receipt_id=forward_fix_receipt.receipt_id,
            ),
        )
        forward_only = await coordinator.evaluate_direct_rollback(control.family_id)
        assert not forward_only.allowed
        assert "forward-fix-only" in forward_only.reason

        journal_rows = await connection.fetch(
            """
SELECT sequence, previous_event_hash, event_hash
FROM omninode_internal.cutover_journal
WHERE family_id = $1
ORDER BY sequence
""",
            projection.family_id,
        )
        assert journal_rows[0]["previous_event_hash"] == "0" * 64
        for previous, current in pairwise(journal_rows):
            assert current["previous_event_hash"] == previous["event_hash"]
    finally:
        await connection.close()

    # Reconnect to prove the journal projection is durable, not process memory.
    reopened = await asyncpg.connect(postgres_dsn)
    try:
        state = await RepositoryPostgresCutoverJournal(reopened).get_state(
            _family_id("tenant.usage")
        )
        assert state.verified_reverse_delta_proof_id is not None
        assert state.observation_ends_at == observation_ends_at
        assert state.last_sequence > 0
    finally:
        await reopened.close()
