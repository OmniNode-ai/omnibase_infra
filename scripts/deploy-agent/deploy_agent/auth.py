# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HMAC-SHA256 envelope authentication for deploy-agent commands."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ENV_KEY = "DEPLOY_AGENT_HMAC_SECRET"

SIGNATURE_FIELD = "_signature"

# OMN-24397: the label of the check value of the signing key. omnimarket's
# node_deploy_agent_key_check_effect holds the same label and vector; both sides pin it.
_KEY_CHECK_LABEL = b"deploy-agent-key-check-v1"


def canonical_signing_body(envelope: dict[str, Any]) -> bytes:
    """The exact bytes the signature covers: ONE definition for signer and verifier (OMN-16442).

    ``sort_keys`` makes field order irrelevant; ``_signature`` is excluded.
    """
    body_dict = {k: v for k, v in envelope.items() if k != SIGNATURE_FIELD}
    return json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()


def compute_signature(envelope: dict[str, Any], secret: str) -> str:
    """Return the hex HMAC-SHA256 signature for ``envelope`` under ``secret``."""
    return hmac.new(
        secret.encode(), canonical_signing_body(envelope), hashlib.sha256
    ).hexdigest()


def hmac_key_check() -> str | None:
    """One-way check value of the key in the environment (16 hex), or None without a key."""
    secret = os.environ.get(_ENV_KEY, "")
    if not secret:
        return None
    return hmac.new(secret.encode(), _KEY_CHECK_LABEL, hashlib.sha256).hexdigest()[:16]


def sign_envelope(envelope: dict[str, Any], secret: str) -> dict[str, Any]:
    """Return ``envelope`` plus the ``_signature`` field ``verify_command`` accepts."""
    body_dict = {k: v for k, v in envelope.items() if k != SIGNATURE_FIELD}
    return {**body_dict, SIGNATURE_FIELD: compute_signature(body_dict, secret)}


def verify_command(envelope: dict[str, Any]) -> bool:
    """Verify HMAC-SHA256 signature on a rebuild command envelope.

    The signature is computed over the JSON-serialised envelope (sort_keys,
    no spaces) *after* removing the ``_signature`` field itself.  The caller
    must include ``_signature`` in the envelope; its absence is treated as an
    authentication failure.

    Returns True only when the signature is valid.  Logs a warning and returns
    False on any failure so callers can emit a rejection event with the reason.
    """
    secret = os.environ.get(_ENV_KEY)
    if not secret:
        logger.error(
            "DEPLOY_AGENT_HMAC_SECRET not set — rejecting all commands. "
            "Generate one with: openssl rand -hex 32"
        )
        return False

    # Pop signature before computing expected value (non-destructive: work on a copy)
    signature = envelope.get(SIGNATURE_FIELD)
    if not signature:
        logger.warning("Rebuild command rejected: missing _signature field")
        return False

    expected = compute_signature(envelope, secret)

    if not hmac.compare_digest(signature, expected):
        logger.warning("Rebuild command rejected: invalid HMAC signature")
        return False

    return True
