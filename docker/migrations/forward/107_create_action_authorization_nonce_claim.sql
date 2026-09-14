-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Dedicated, redacted nonce-claim boundary for canonical pre-execution action
-- authorizations (OMN-17486). This migration owns only the
-- action_authorization_claim schema and its nonce_claims table. It does not
-- inspect or alter any prior ledger, outbox, queue, or shared schema.
--
-- One call to claim_action_authorization validates, binds, inserts, and claims
-- one canonical request in one transaction. A durable CLAIMED row is claim
-- evidence only; it does not enable action execution or a consumer.

DO $role$
DECLARE
    claim_role pg_catalog.pg_roles%ROWTYPE;
BEGIN
    SELECT *
      INTO claim_role
      FROM pg_catalog.pg_roles
     WHERE rolname = 'rsd_action_authorization_claim';

    IF NOT FOUND THEN
        CREATE ROLE rsd_action_authorization_claim
            NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT
            NOBYPASSRLS NOREPLICATION;
    ELSIF claim_role.rolsuper
       OR claim_role.rolcreatedb
       OR claim_role.rolcreaterole
       OR claim_role.rolcanlogin
       OR claim_role.rolinherit
       OR claim_role.rolbypassrls
       OR claim_role.rolreplication THEN
        RAISE EXCEPTION
            'rsd_action_authorization_claim must remain a non-owner restricted role';
    END IF;
END;
$role$;

CREATE SCHEMA action_authorization_claim;

REVOKE ALL ON SCHEMA action_authorization_claim FROM PUBLIC;
REVOKE ALL ON SCHEMA action_authorization_claim FROM rsd_action_authorization_claim;

CREATE TABLE action_authorization_claim.nonce_claims (
    authorization_id               TEXT NOT NULL,
    ticket_id                      TEXT NOT NULL,
    contract_path                  TEXT NOT NULL,
    contract_commit_sha            TEXT NOT NULL,
    contract_sha256                TEXT NOT NULL,
    action_id                      TEXT NOT NULL,
    source_sha                     TEXT NOT NULL,
    artifact_sha256                TEXT NOT NULL,
    target_database                TEXT NOT NULL,
    target_schema                  TEXT NOT NULL,
    target_service                 TEXT NOT NULL,
    target_principal               TEXT NOT NULL,
    execute_enabled                BOOLEAN NOT NULL,
    issuer                         TEXT NOT NULL,
    nonce_digest                   TEXT NOT NULL,
    issued_at                      TIMESTAMPTZ NOT NULL,
    expires_at                     TIMESTAMPTZ NOT NULL,
    one_time_use                   BOOLEAN NOT NULL,
    reason                         TEXT NOT NULL,
    request_digest                 TEXT NOT NULL,
    state                          TEXT NOT NULL,
    redacted_receipt_digest        TEXT NOT NULL,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    claimed_at                     TIMESTAMPTZ,
    expired_at                     TIMESTAMPTZ,
    version                        BIGINT NOT NULL DEFAULT 1,
    CONSTRAINT pk_action_authorization_nonce_claims
        PRIMARY KEY (authorization_id, nonce_digest),
    CONSTRAINT uq_action_authorization_nonce_claims_authorization
        UNIQUE (authorization_id),
    CONSTRAINT uq_action_authorization_nonce_claims_nonce
        UNIQUE (nonce_digest),
    CONSTRAINT uq_action_authorization_nonce_claims_request
        UNIQUE (request_digest),
    CONSTRAINT ck_action_authorization_authorization_id
        CHECK (authorization_id ~ '^action-auth-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    CONSTRAINT ck_action_authorization_ticket_id
        CHECK (ticket_id ~ '^OMN-[1-9][0-9]*$'),
    CONSTRAINT ck_action_authorization_contract_path
        CHECK (contract_path = 'contracts/' || ticket_id || '.yaml'),
    CONSTRAINT ck_action_authorization_contract_commit_sha
        CHECK (contract_commit_sha ~ '^[0-9a-f]{40}$'),
    CONSTRAINT ck_action_authorization_contract_sha256
        CHECK (contract_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT ck_action_authorization_action_id
        CHECK (action_id ~ '^[a-z][a-z0-9-]{2,63}$'),
    CONSTRAINT ck_action_authorization_source_sha
        CHECK (source_sha ~ '^[0-9a-f]{40}$'),
    CONSTRAINT ck_action_authorization_artifact_sha256
        CHECK (artifact_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT ck_action_authorization_target_database
        CHECK (target_database ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    CONSTRAINT ck_action_authorization_target_schema
        CHECK (target_schema ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    CONSTRAINT ck_action_authorization_target_service
        CHECK (target_service ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    CONSTRAINT ck_action_authorization_target_principal
        CHECK (target_principal ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    CONSTRAINT ck_action_authorization_execute_disabled
        CHECK (execute_enabled IS FALSE),
    CONSTRAINT ck_action_authorization_issuer
        CHECK (issuer ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'),
    CONSTRAINT ck_action_authorization_nonce_digest
        CHECK (nonce_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_action_authorization_expiry_order
        CHECK (expires_at > issued_at),
    CONSTRAINT ck_action_authorization_one_time_use
        CHECK (one_time_use IS TRUE),
    CONSTRAINT ck_action_authorization_reason
        CHECK (btrim(reason) <> '' AND length(reason) <= 1024),
    CONSTRAINT ck_action_authorization_request_digest
        CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_action_authorization_redacted_receipt_digest
        CHECK (redacted_receipt_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_action_authorization_state
        CHECK (state IN ('CLAIMED', 'EXPIRED')),
    CONSTRAINT ck_action_authorization_version
        CHECK (version = 1),
    CONSTRAINT ck_action_authorization_state_shape
        CHECK (
            (state = 'CLAIMED' AND claimed_at IS NOT NULL AND expired_at IS NULL)
            OR (state = 'EXPIRED' AND claimed_at IS NULL AND expired_at IS NOT NULL)
        )
);

REVOKE ALL ON TABLE action_authorization_claim.nonce_claims FROM PUBLIC;
REVOKE ALL ON TABLE action_authorization_claim.nonce_claims FROM rsd_action_authorization_claim;

CREATE FUNCTION action_authorization_claim.enforce_nonce_claim_immutability()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, action_authorization_claim
AS $function$
BEGIN
    RAISE EXCEPTION 'action authorization nonce claims are immutable';
END;
$function$;

CREATE TRIGGER trg_action_authorization_nonce_claim_immutable
    BEFORE UPDATE OR DELETE ON action_authorization_claim.nonce_claims
    FOR EACH ROW
    EXECUTE FUNCTION action_authorization_claim.enforce_nonce_claim_immutability();

CREATE FUNCTION action_authorization_claim.claim_action_authorization(
    p_authorization_id TEXT,
    p_ticket_id TEXT,
    p_contract_path TEXT,
    p_contract_commit_sha TEXT,
    p_contract_sha256 TEXT,
    p_action_id TEXT,
    p_source_sha TEXT,
    p_artifact_sha256 TEXT,
    p_target_database TEXT,
    p_target_schema TEXT,
    p_target_service TEXT,
    p_target_principal TEXT,
    p_execute_enabled BOOLEAN,
    p_issuer TEXT,
    p_nonce_digest TEXT,
    p_issued_at TIMESTAMPTZ,
    p_expires_at TIMESTAMPTZ,
    p_one_time_use BOOLEAN,
    p_reason TEXT,
    p_request_digest TEXT,
    p_redacted_receipt_digest TEXT
)
RETURNS TABLE (
    outcome TEXT,
    state TEXT,
    version BIGINT,
    redacted_receipt_digest TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, action_authorization_claim
AS $function$
DECLARE
    existing action_authorization_claim.nonce_claims%ROWTYPE;
    claim_time TIMESTAMPTZ;
BEGIN
    IF p_authorization_id IS NULL
       OR p_ticket_id IS NULL
       OR p_contract_path IS NULL
       OR p_contract_commit_sha IS NULL
       OR p_contract_sha256 IS NULL
       OR p_action_id IS NULL
       OR p_source_sha IS NULL
       OR p_artifact_sha256 IS NULL
       OR p_target_database IS NULL
       OR p_target_schema IS NULL
       OR p_target_service IS NULL
       OR p_target_principal IS NULL
       OR p_issuer IS NULL
       OR p_nonce_digest IS NULL
       OR p_issued_at IS NULL
       OR p_expires_at IS NULL
       OR p_reason IS NULL
       OR p_request_digest IS NULL
       OR p_redacted_receipt_digest IS NULL THEN
        RAISE EXCEPTION 'action authorization request contains a null canonical field'
            USING ERRCODE = '22023';
    END IF;

    IF p_authorization_id !~ '^action-auth-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
       OR p_ticket_id !~ '^OMN-[1-9][0-9]*$'
       OR p_contract_path <> 'contracts/' || p_ticket_id || '.yaml'
       OR p_contract_commit_sha !~ '^[0-9a-f]{40}$'
       OR p_contract_sha256 !~ '^sha256:[0-9a-f]{64}$'
       OR p_action_id !~ '^[a-z][a-z0-9-]{2,63}$'
       OR p_source_sha !~ '^[0-9a-f]{40}$'
       OR p_artifact_sha256 !~ '^sha256:[0-9a-f]{64}$'
       OR p_target_database !~ '^[a-z0-9][a-z0-9_-]{0,62}$'
       OR p_target_schema !~ '^[a-z0-9][a-z0-9_-]{0,62}$'
       OR p_target_service !~ '^[a-z0-9][a-z0-9_-]{0,62}$'
       OR p_target_principal !~ '^[a-z0-9][a-z0-9_-]{0,62}$'
       OR p_issuer !~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'
       OR p_nonce_digest !~ '^[0-9a-f]{64}$'
       OR p_request_digest !~ '^[0-9a-f]{64}$'
       OR p_redacted_receipt_digest !~ '^[0-9a-f]{64}$'
       OR btrim(p_reason) = ''
       OR length(p_reason) > 1024
       OR p_expires_at <= p_issued_at
       OR p_execute_enabled IS DISTINCT FROM FALSE
       OR p_one_time_use IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'action authorization request violates canonical constraints'
            USING ERRCODE = '22023';
    END IF;

    LOOP
        SELECT *
          INTO existing
          FROM action_authorization_claim.nonce_claims
         WHERE authorization_id = p_authorization_id
            OR nonce_digest = p_nonce_digest
            OR request_digest = p_request_digest
         FOR UPDATE;

        IF FOUND THEN
            IF existing.authorization_id <> p_authorization_id
               OR existing.nonce_digest <> p_nonce_digest
               OR existing.ticket_id <> p_ticket_id
               OR existing.contract_path <> p_contract_path
               OR existing.contract_commit_sha <> p_contract_commit_sha
               OR existing.contract_sha256 <> p_contract_sha256
               OR existing.action_id <> p_action_id
               OR existing.source_sha <> p_source_sha
               OR existing.artifact_sha256 <> p_artifact_sha256
               OR existing.target_database <> p_target_database
               OR existing.target_schema <> p_target_schema
               OR existing.target_service <> p_target_service
               OR existing.target_principal <> p_target_principal
               OR existing.execute_enabled <> p_execute_enabled
               OR existing.issuer <> p_issuer
               OR existing.issued_at <> p_issued_at
               OR existing.expires_at <> p_expires_at
               OR existing.one_time_use <> p_one_time_use
               OR existing.reason <> p_reason
               OR existing.request_digest <> p_request_digest
               OR existing.redacted_receipt_digest <> p_redacted_receipt_digest THEN
                RETURN QUERY SELECT 'MISMATCH', existing.state, existing.version,
                    existing.redacted_receipt_digest;
                RETURN;
            END IF;
            IF existing.state = 'EXPIRED' THEN
                RETURN QUERY SELECT 'EXPIRED', existing.state, existing.version,
                    existing.redacted_receipt_digest;
            ELSE
                RETURN QUERY SELECT 'ALREADY_CONSUMED', existing.state, existing.version,
                    existing.redacted_receipt_digest;
            END IF;
        END IF;

        claim_time := clock_timestamp();
        BEGIN
            IF claim_time >= p_expires_at THEN
                INSERT INTO action_authorization_claim.nonce_claims (
                    authorization_id, ticket_id, contract_path, contract_commit_sha,
                    contract_sha256, action_id, source_sha, artifact_sha256,
                    target_database, target_schema, target_service, target_principal,
                    execute_enabled, issuer, nonce_digest, issued_at, expires_at,
                    one_time_use, reason, request_digest, state,
                    redacted_receipt_digest, expired_at
                ) VALUES (
                    p_authorization_id, p_ticket_id, p_contract_path, p_contract_commit_sha,
                    p_contract_sha256, p_action_id, p_source_sha, p_artifact_sha256,
                    p_target_database, p_target_schema, p_target_service, p_target_principal,
                    p_execute_enabled, p_issuer, p_nonce_digest, p_issued_at, p_expires_at,
                    p_one_time_use, p_reason, p_request_digest, 'EXPIRED',
                    p_redacted_receipt_digest, claim_time
                );
                RETURN QUERY SELECT 'EXPIRED', 'EXPIRED', 1::BIGINT,
                    p_redacted_receipt_digest;
                RETURN;
            END IF;

            INSERT INTO action_authorization_claim.nonce_claims (
                authorization_id, ticket_id, contract_path, contract_commit_sha,
                contract_sha256, action_id, source_sha, artifact_sha256,
                target_database, target_schema, target_service, target_principal,
                execute_enabled, issuer, nonce_digest, issued_at, expires_at,
                one_time_use, reason, request_digest, state,
                redacted_receipt_digest, claimed_at
            ) VALUES (
                p_authorization_id, p_ticket_id, p_contract_path, p_contract_commit_sha,
                p_contract_sha256, p_action_id, p_source_sha, p_artifact_sha256,
                p_target_database, p_target_schema, p_target_service, p_target_principal,
                p_execute_enabled, p_issuer, p_nonce_digest, p_issued_at, p_expires_at,
                p_one_time_use, p_reason, p_request_digest, 'CLAIMED',
                p_redacted_receipt_digest, claim_time
            );
            RETURN QUERY SELECT 'CLAIMED', 'CLAIMED', 1::BIGINT,
                p_redacted_receipt_digest;
            RETURN;
        EXCEPTION
            WHEN unique_violation THEN
                -- A concurrent winner committed the same nonce, authorization,
                -- or request identity. Loop back to lock and classify its row.
                NULL;
        END;
    END LOOP;
END;
$function$;

REVOKE ALL ON FUNCTION action_authorization_claim.enforce_nonce_claim_immutability() FROM PUBLIC;
REVOKE ALL ON FUNCTION action_authorization_claim.claim_action_authorization(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT,
    BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, TIMESTAMPTZ, BOOLEAN, TEXT, TEXT, TEXT
) FROM PUBLIC;

GRANT USAGE ON SCHEMA action_authorization_claim TO rsd_action_authorization_claim;
GRANT EXECUTE ON FUNCTION action_authorization_claim.claim_action_authorization(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT,
    BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, TIMESTAMPTZ, BOOLEAN, TEXT, TEXT, TEXT
) TO rsd_action_authorization_claim;
