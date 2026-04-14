# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HMAC-SHA256 command authentication."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import patch

from deploy_agent.auth import verify_command

_SECRET = "test-secret-abc123"


def _sign(envelope: dict, secret: str = _SECRET) -> dict:
    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {**envelope, "_signature": sig}


_BASE_ENVELOPE = {
    "correlation_id": "00000000-0000-0000-0000-000000000001",
    "requested_by": "test",
    "scope": "full",
    "services": [],
    "git_ref": "origin/main",
}


def test_valid_signature_passes() -> None:
    signed = _sign(_BASE_ENVELOPE)
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is True


def test_invalid_signature_rejected() -> None:
    signed = _sign(_BASE_ENVELOPE)
    signed["_signature"] = "deadbeef" * 8  # wrong but same length
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is False


def test_missing_signature_rejected() -> None:
    envelope = dict(_BASE_ENVELOPE)  # no _signature key
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(envelope) is False


def test_missing_secret_rejects_all() -> None:
    signed = _sign(_BASE_ENVELOPE)
    with patch.dict("os.environ", {}, clear=True):
        # Ensure key is absent even if set in outer env
        import os

        os.environ.pop("DEPLOY_AGENT_HMAC_SECRET", None)
        assert verify_command(signed) is False


def test_tampered_body_rejected() -> None:
    signed = _sign(_BASE_ENVELOPE)
    # Mutate a field after signing
    signed["scope"] = "core"
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is False


def test_wrong_secret_rejected() -> None:
    signed = _sign(_BASE_ENVELOPE, secret="correct-secret")  # noqa: S106
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": "wrong-secret"}):
        assert verify_command(signed) is False
