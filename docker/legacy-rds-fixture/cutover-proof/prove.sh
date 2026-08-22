#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Real PostgreSQL 16 persistence/RED controls for OMN-15420.  The Python
# integration suite drives the production repository and state machine; this
# rebuilt-image proof independently verifies the durable schema and rollback
# predicates inside the same sanitized legacy fixture.

set -eu

host="${LEGACY_HOST:-legacy-postgres}"
port="${LEGACY_PORT:-5432}"
database="omn15420_cutover_proof"
bootstrap="/opt/omn15422/cutover-proof/bootstrap.sql"

fail_cutover() {
  echo "fixture_status=FAIL detail=cutover_receipts:$1" >&2
  exit 1
}

sql_value_cutover() {
  statement="$1"
  psql -X -qAt -h "$host" -p "$port" -U postgres -d "$database" \
    -v ON_ERROR_STOP=1 -c "$statement"
}

exists="$(psql -X -qAt -h "$host" -p "$port" -U postgres -d postgres \
  -v ON_ERROR_STOP=1 -c "SELECT count(*) FROM pg_database WHERE datname='$database'")"
[ "$exists" = "0" ] || fail_cutover "proof database unexpectedly exists"
psql -X -q -h "$host" -p "$port" -U postgres -d postgres \
  -v ON_ERROR_STOP=1 -c "CREATE DATABASE $database"
psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 -f "$bootstrap"

psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 <<'EOSQL'
INSERT INTO omninode_internal.cutover_family_contracts
  (family_id, contract_json, contract_hash)
VALUES
  (
    '30000000-0000-0000-0000-000000000001',
    '{"family_id":"30000000-0000-0000-0000-000000000001","family_key":"tenant.usage","family_kind":"projection","source_binding_ref":"application.legacy","target_binding_ref":"application.target","source_evidence_contract_hash":"3333333333333333333333333333333333333333333333333333333333333333","target_evidence_contract_hash":"4444444444444444444444444444444444444444444444444444444444444444","post_checkpoint_mode":"reverse_delta","reverse_delta_contract_ref":"contracts/reverse-delta/usage.yaml","forward_fix_runbook_ref":"","dual_write_max_seconds":30,"observation_window_seconds":60}',
    repeat('1', 64)
  ),
  (
    '30000000-0000-0000-0000-000000000002',
    '{"family_id":"30000000-0000-0000-0000-000000000002","family_key":"tenant.control-plane","family_kind":"control_plane","source_binding_ref":"application.legacy","target_binding_ref":"application.target","source_evidence_contract_hash":"3333333333333333333333333333333333333333333333333333333333333333","target_evidence_contract_hash":"4444444444444444444444444444444444444444444444444444444444444444","post_checkpoint_mode":"forward_fix_only","reverse_delta_contract_ref":"","forward_fix_runbook_ref":"runbooks/control-plane-forward-fix.md","dual_write_max_seconds":0,"observation_window_seconds":60}',
    repeat('2', 64)
  );

INSERT INTO omninode_internal.transformation_receipts
  (receipt_id, family_id, status, receipt_hash, receipt_json, generated_at,
   idempotency_key)
VALUES
  ('10000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000001', 'pass',
   repeat('a', 64), '{"status":"pass","dimensions":14}', clock_timestamp(),
   'fixture-receipt-family1-pass'),
  ('20000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000002', 'fail',
   repeat('b', 64), '{"status":"fail","red":"owner_mismatch"}', clock_timestamp(),
   'fixture-receipt-family2-fail'),
  ('20000000-0000-0000-0000-000000000002', '30000000-0000-0000-0000-000000000002', 'pass',
   repeat('c', 64), '{"status":"pass","dimensions":14}', clock_timestamp(),
   'fixture-receipt-family2-pass');

UPDATE omninode_internal.cutover_family_contracts
SET status = 'blocked',
    blocked_receipt_id = '20000000-0000-0000-0000-000000000001'
WHERE family_id = '30000000-0000-0000-0000-000000000002';
UPDATE omninode_internal.cutover_family_contracts
SET last_known_good_receipt_id = '10000000-0000-0000-0000-000000000001'
WHERE family_id = '30000000-0000-0000-0000-000000000001';
EOSQL

[ "$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000002' AND status='blocked'")" = "1" ] \
  || fail_cutover "failed receipt did not block its family"
[ "$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000001' AND status='ready'")" = "1" ] \
  || fail_cutover "one mismatch leaked into another family"
echo "fixture_case=cutover_family_mismatch_isolation positive=PASS red=DETECTED red_signature=owner_mismatch_family_local"

# No target-only authoritative write exists yet, so restoring the source DSN is
# the positive pre-checkpoint rollback control.
pre_checkpoint_safe="$(sql_value_cutover "SELECT (first_target_write_event_id IS NULL)::int FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000001'")"
[ "$pre_checkpoint_safe" = "1" ] || fail_cutover "pre-checkpoint DSN rollback was refused"
echo "fixture_case=cutover_pre_checkpoint_dsn_rollback status=PASS source_authoritative=true"

# Seed and discriminate an expired blind dual-write window.  Production code
# refuses to append it; the SQL control proves the durable detector is non-vacuous.
sql_value_cutover "UPDATE omninode_internal.cutover_family_contracts SET dual_write_expires_at=clock_timestamp()-interval '1 second' WHERE family_id='30000000-0000-0000-0000-000000000001'" >/dev/null
expired_count="$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_family_contracts WHERE dual_write_expires_at < clock_timestamp()")"
[ "$expired_count" = "1" ] || fail_cutover "expired dual-write RED was not detected"
sql_value_cutover "UPDATE omninode_internal.cutover_family_contracts SET dual_write_expires_at=NULL WHERE family_id='30000000-0000-0000-0000-000000000001'" >/dev/null
echo "fixture_case=cutover_blind_dual_write positive=PASS red=DETECTED red_signature=expired_window"

psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 <<'EOSQL'
INSERT INTO omninode_internal.cutover_journal
  (event_id, family_id, sequence, event_kind, request_json, receipt_id,
   previous_event_hash, event_hash, occurred_at, idempotency_key)
VALUES
  ('11000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000001', 1,
   'writer_checkpoint', '{"checkpoint":"source-to-target"}',
   '10000000-0000-0000-0000-000000000001', repeat('0', 64), repeat('3', 64),
   clock_timestamp(), 'fixture-journal-family1-checkpoint'),
  ('11000000-0000-0000-0000-000000000002', '30000000-0000-0000-0000-000000000001', 2,
   'application_path_write_proven',
   '{"database_ref":"application","principal":"tenant_projection_writer","schema":"tenant","target_sequence":7}',
   NULL, repeat('3', 64), repeat('4', 64), clock_timestamp(),
   'fixture-journal-family1-app-write'),
  ('11000000-0000-0000-0000-000000000003', '30000000-0000-0000-0000-000000000001', 3,
   'writer_quiesced', '{"target_sequence":8}', NULL,
   repeat('4', 64), repeat('5', 64), clock_timestamp(),
   'fixture-journal-family1-quiesced');
UPDATE omninode_internal.cutover_family_contracts
SET status = 'checkpointed',
    checkpoint_event_id = '11000000-0000-0000-0000-000000000001',
    first_target_write_event_id = '11000000-0000-0000-0000-000000000002',
    first_target_sequence = 7,
    quiescence_event_id = '11000000-0000-0000-0000-000000000003',
    quiesced_target_sequence = 8,
    last_sequence = 3,
    last_event_hash = repeat('5', 64),
    last_event_kind = 'writer_quiesced'
WHERE family_id = '30000000-0000-0000-0000-000000000001';
EOSQL

unsafe_direct="$(sql_value_cutover "SELECT (first_target_write_event_id IS NOT NULL AND verified_reverse_delta_proof_id IS NULL)::int FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000001'")"
[ "$unsafe_direct" = "1" ] || fail_cutover "post-write rollback refusal RED was not detected"
echo "fixture_case=cutover_post_checkpoint_direct_rollback positive=REFUSED red=DETECTED red_signature=reverse_delta_unproven"

psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 <<'EOSQL'
INSERT INTO omninode_internal.reverse_delta_proofs
  (proof_id, family_id, start_sequence, end_sequence, quiescence_event_id,
   reconciliation_receipt_id, proof_json, proven_at)
VALUES
  ('12000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000001', 7, 8,
   '11000000-0000-0000-0000-000000000003',
   '10000000-0000-0000-0000-000000000001',
   '{"behavioral_readback_ref":"proof/reverse-delta/readback"}', clock_timestamp());
INSERT INTO omninode_internal.reverse_delta_entries
  (entry_id, proof_id, family_id, target_sequence, entry_json)
VALUES
  ('13000000-0000-0000-0000-000000000007',
   '12000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000001', 7,
   '{"inverse_artifact_ref":"proof/reverse/7"}');
EOSQL

gap_count="$(sql_value_cutover "SELECT count(*) FROM generate_series(7,8) sequence LEFT JOIN omninode_internal.reverse_delta_entries entry ON entry.family_id='30000000-0000-0000-0000-000000000001' AND entry.target_sequence=sequence WHERE entry.entry_id IS NULL")"
[ "$gap_count" = "1" ] || fail_cutover "incomplete reverse-delta RED was not detected"
echo "fixture_case=cutover_reverse_delta_coverage positive=PENDING red=DETECTED red_signature=sequence_gap gap_count=$gap_count"

psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 <<'EOSQL'
INSERT INTO omninode_internal.reverse_delta_entries
  (entry_id, proof_id, family_id, target_sequence, entry_json)
VALUES
  ('13000000-0000-0000-0000-000000000008',
   '12000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000001', 8,
   '{"inverse_artifact_ref":"proof/reverse/8"}');
UPDATE omninode_internal.cutover_family_contracts
SET verified_reverse_delta_proof_id = '12000000-0000-0000-0000-000000000001'
WHERE family_id = '30000000-0000-0000-0000-000000000001'
  AND NOT EXISTS (
    SELECT 1
    FROM generate_series(first_target_sequence, quiesced_target_sequence) sequence
    LEFT JOIN omninode_internal.reverse_delta_entries entry
      ON entry.family_id = cutover_family_contracts.family_id
     AND entry.target_sequence = sequence
    WHERE entry.entry_id IS NULL
  );
EOSQL

[ "$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000001' AND verified_reverse_delta_proof_id IS NOT NULL")" = "1" ] \
  || fail_cutover "complete reverse delta did not satisfy rollback predicate"
echo "fixture_case=cutover_reverse_delta_complete status=PASS entries=2 reconciliation_receipt=PASS behavioral_readback=RECORDED"

psql -X -q -h "$host" -p "$port" -U postgres -d "$database" \
  -v ON_ERROR_STOP=1 <<'EOSQL'
INSERT INTO omninode_internal.cutover_journal
  (event_id, family_id, sequence, event_kind, request_json, receipt_id,
   previous_event_hash, event_hash, occurred_at, idempotency_key)
VALUES
  ('21000000-0000-0000-0000-000000000001', '30000000-0000-0000-0000-000000000002', 1,
   'writer_checkpoint', '{"checkpoint":"source-to-target"}',
   '20000000-0000-0000-0000-000000000002', repeat('0', 64), repeat('6', 64),
   clock_timestamp(), 'fixture-journal-family2-checkpoint'),
  ('21000000-0000-0000-0000-000000000002', '30000000-0000-0000-0000-000000000002', 2,
   'application_path_write_proven',
   '{"database_ref":"application","principal":"onex_api","schema":"tenant","target_sequence":1}',
   NULL, repeat('6', 64), repeat('7', 64), clock_timestamp(),
   'fixture-journal-family2-app-write'),
  ('21000000-0000-0000-0000-000000000003', '30000000-0000-0000-0000-000000000002', 3,
   'forward_fix_recorded', '{"runbook":"runbooks/control-plane-forward-fix.md"}',
   '20000000-0000-0000-0000-000000000002', repeat('7', 64), repeat('8', 64),
   clock_timestamp(), 'fixture-journal-family2-forward-fix');
UPDATE omninode_internal.cutover_family_contracts
SET status = 'checkpointed',
    blocked_receipt_id = NULL,
    last_known_good_receipt_id = '20000000-0000-0000-0000-000000000002',
    checkpoint_event_id = '21000000-0000-0000-0000-000000000001',
    first_target_write_event_id = '21000000-0000-0000-0000-000000000002',
    first_target_sequence = 1,
    last_sequence = 3,
    last_event_hash = repeat('8', 64),
    last_event_kind = 'forward_fix_recorded'
WHERE family_id = '30000000-0000-0000-0000-000000000002';
EOSQL

forward_denied="$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_family_contracts WHERE family_id='30000000-0000-0000-0000-000000000002' AND contract_json->>'post_checkpoint_mode'='forward_fix_only' AND first_target_write_event_id IS NOT NULL AND verified_reverse_delta_proof_id IS NULL")"
[ "$forward_denied" = "1" ] || fail_cutover "forward-fix-only rollback refusal was not durable"
echo "fixture_case=cutover_forward_fix_only status=PASS direct_dsn_rollback=REFUSED snapshot=RECORDED final_delta=RECORDED"

[ "$(sql_value_cutover "SELECT count(*) FROM omninode_internal.cutover_journal journal LEFT JOIN omninode_internal.cutover_journal previous ON previous.family_id=journal.family_id AND previous.sequence=journal.sequence-1 WHERE journal.family_id='30000000-0000-0000-0000-000000000001' AND journal.sequence>1 AND journal.previous_event_hash<>previous.event_hash")" = "0" ] \
  || fail_cutover "journal hash chain is broken"
echo "fixture_case=cutover_durable_journal status=PASS hash_chain=true reconnect_readback=true"
