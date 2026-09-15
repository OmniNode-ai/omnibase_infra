-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Backout for 107_create_action_authorization_nonce_claim (OMN-17486).
-- Disable the restricted atomic claim interface while retaining the dedicated
-- schema, function, and durable claim history. No unrelated object is changed.

REVOKE EXECUTE ON FUNCTION action_authorization_claim.claim_action_authorization(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT,
    BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, TIMESTAMPTZ, BOOLEAN, TEXT, TEXT, TEXT
) FROM rsd_action_authorization_claim;
REVOKE USAGE ON SCHEMA action_authorization_claim FROM rsd_action_authorization_claim;
