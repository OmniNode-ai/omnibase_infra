# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HMAC-SHA256 envelope authentication for deploy-agent commands."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

logger = logging.getLogger(__name__)

_ENV_KEY = "DEPLOY_AGENT_HMAC_SECRET"


def verify_command(envelope: dict) -> bool:
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
    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    signature = envelope.get("_signature")
    if not signature:
        logger.warning("Rebuild command rejected: missing _signature field")
        return False

    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        logger.warning("Rebuild command rejected: invalid HMAC signature")
        return False

    return True
