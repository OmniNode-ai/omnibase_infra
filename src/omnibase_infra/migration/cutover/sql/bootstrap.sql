-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT

-- OMN-15420 proof schema.  This file is loaded only by the explicit cutover
-- repository initializer; it is not part of the forward migration stream.
CREATE SCHEMA IF NOT EXISTS omninode_internal;

CREATE TABLE IF NOT EXISTS omninode_internal.cutover_family_contracts (
    family_id UUID PRIMARY KEY,
    contract_json JSONB NOT NULL CHECK (jsonb_typeof(contract_json) = 'object'),
    contract_hash TEXT NOT NULL CHECK (contract_hash ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'blocked', 'checkpointed', 'observing', 'complete')),
    last_known_good_receipt_id UUID,
    blocked_receipt_id UUID,
    checkpoint_event_id UUID,
    first_target_write_event_id UUID,
    first_target_sequence BIGINT CHECK (first_target_sequence > 0),
    quiescence_event_id UUID,
    quiesced_target_sequence BIGINT CHECK (quiesced_target_sequence > 0),
    verified_reverse_delta_proof_id UUID,
    dual_write_expires_at TIMESTAMPTZ,
    observation_ends_at TIMESTAMPTZ,
    last_sequence BIGINT NOT NULL DEFAULT 0 CHECK (last_sequence >= 0),
    last_event_hash TEXT NOT NULL DEFAULT repeat('0', 64)
        CHECK (last_event_hash ~ '^[0-9a-f]{64}$'),
    last_event_kind TEXT,
    last_event_at TIMESTAMPTZ,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS omninode_internal.transformation_receipts (
    receipt_id UUID PRIMARY KEY,
    family_id UUID NOT NULL REFERENCES omninode_internal.cutover_family_contracts(family_id),
    status TEXT NOT NULL CHECK (status IN ('pass', 'fail')),
    receipt_hash TEXT NOT NULL CHECK (receipt_hash ~ '^[0-9a-f]{64}$'),
    receipt_json JSONB NOT NULL CHECK (jsonb_typeof(receipt_json) = 'object'),
    generated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (family_id, receipt_hash),
    UNIQUE (family_id, receipt_id)
);

CREATE TABLE IF NOT EXISTS omninode_internal.cutover_journal (
    event_id UUID PRIMARY KEY,
    family_id UUID NOT NULL REFERENCES omninode_internal.cutover_family_contracts(family_id),
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    event_kind TEXT NOT NULL CHECK (event_kind IN (
        'backfill_started', 'backfill_completed', 'dual_write_started',
        'dual_write_ended', 'final_delta_applied', 'writer_checkpoint',
        'application_path_write_proven', 'reader_cutover',
        'observation_window_started', 'observation_window_completed',
        'writer_quiesced', 'reverse_delta_proven', 'forward_fix_recorded',
        'pre_checkpoint_rollback', 'mismatch_resolved'
    )),
    request_json JSONB NOT NULL CHECK (jsonb_typeof(request_json) = 'object'),
    receipt_id UUID REFERENCES omninode_internal.transformation_receipts(receipt_id),
    previous_event_hash TEXT NOT NULL CHECK (previous_event_hash ~ '^[0-9a-f]{64}$'),
    event_hash TEXT NOT NULL CHECK (event_hash ~ '^[0-9a-f]{64}$'),
    occurred_at TIMESTAMPTZ NOT NULL,
    UNIQUE (family_id, sequence),
    UNIQUE (family_id, event_hash),
    UNIQUE (family_id, event_id)
);

CREATE TABLE IF NOT EXISTS omninode_internal.reverse_delta_proofs (
    proof_id UUID PRIMARY KEY,
    family_id UUID NOT NULL REFERENCES omninode_internal.cutover_family_contracts(family_id),
    start_sequence BIGINT NOT NULL CHECK (start_sequence > 0),
    end_sequence BIGINT NOT NULL CHECK (end_sequence >= start_sequence),
    quiescence_event_id UUID NOT NULL REFERENCES omninode_internal.cutover_journal(event_id),
    reconciliation_receipt_id UUID NOT NULL REFERENCES omninode_internal.transformation_receipts(receipt_id),
    proof_json JSONB NOT NULL CHECK (jsonb_typeof(proof_json) = 'object'),
    proven_at TIMESTAMPTZ NOT NULL,
    UNIQUE (family_id, proof_id)
);

CREATE TABLE IF NOT EXISTS omninode_internal.reverse_delta_entries (
    entry_id UUID PRIMARY KEY,
    proof_id UUID NOT NULL REFERENCES omninode_internal.reverse_delta_proofs(proof_id),
    family_id UUID NOT NULL REFERENCES omninode_internal.cutover_family_contracts(family_id),
    target_sequence BIGINT NOT NULL CHECK (target_sequence > 0),
    entry_json JSONB NOT NULL CHECK (jsonb_typeof(entry_json) = 'object'),
    UNIQUE (family_id, target_sequence)
);

DO $constraints$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_last_receipt_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_last_receipt_fk
          FOREIGN KEY (family_id, last_known_good_receipt_id)
          REFERENCES omninode_internal.transformation_receipts(family_id, receipt_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_blocked_receipt_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_blocked_receipt_fk
          FOREIGN KEY (family_id, blocked_receipt_id)
          REFERENCES omninode_internal.transformation_receipts(family_id, receipt_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_checkpoint_event_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_checkpoint_event_fk
          FOREIGN KEY (family_id, checkpoint_event_id)
          REFERENCES omninode_internal.cutover_journal(family_id, event_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_first_write_event_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_first_write_event_fk
          FOREIGN KEY (family_id, first_target_write_event_id)
          REFERENCES omninode_internal.cutover_journal(family_id, event_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_quiescence_event_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_quiescence_event_fk
          FOREIGN KEY (family_id, quiescence_event_id)
          REFERENCES omninode_internal.cutover_journal(family_id, event_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_family_reverse_proof_fk'
          AND conrelid = 'omninode_internal.cutover_family_contracts'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_family_contracts
          ADD CONSTRAINT cutover_family_reverse_proof_fk
          FOREIGN KEY (family_id, verified_reverse_delta_proof_id)
          REFERENCES omninode_internal.reverse_delta_proofs(family_id, proof_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cutover_journal_family_receipt_fk'
          AND conrelid = 'omninode_internal.cutover_journal'::regclass
    ) THEN
        ALTER TABLE omninode_internal.cutover_journal
          ADD CONSTRAINT cutover_journal_family_receipt_fk
          FOREIGN KEY (family_id, receipt_id)
          REFERENCES omninode_internal.transformation_receipts(family_id, receipt_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'reverse_proof_family_receipt_fk'
          AND conrelid = 'omninode_internal.reverse_delta_proofs'::regclass
    ) THEN
        ALTER TABLE omninode_internal.reverse_delta_proofs
          ADD CONSTRAINT reverse_proof_family_receipt_fk
          FOREIGN KEY (family_id, reconciliation_receipt_id)
          REFERENCES omninode_internal.transformation_receipts(family_id, receipt_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'reverse_proof_family_event_fk'
          AND conrelid = 'omninode_internal.reverse_delta_proofs'::regclass
    ) THEN
        ALTER TABLE omninode_internal.reverse_delta_proofs
          ADD CONSTRAINT reverse_proof_family_event_fk
          FOREIGN KEY (family_id, quiescence_event_id)
          REFERENCES omninode_internal.cutover_journal(family_id, event_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'reverse_entry_family_proof_fk'
          AND conrelid = 'omninode_internal.reverse_delta_entries'::regclass
    ) THEN
        ALTER TABLE omninode_internal.reverse_delta_entries
          ADD CONSTRAINT reverse_entry_family_proof_fk
          FOREIGN KEY (family_id, proof_id)
          REFERENCES omninode_internal.reverse_delta_proofs(family_id, proof_id);
    END IF;
END
$constraints$;
