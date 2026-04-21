# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify that the trigger helper script produces signatures accepted by auth.py.

This test does NOT publish to Kafka.  It exercises the signing logic that
deploy-agent-trigger.sh delegates to its embedded Python snippet, then feeds
the resulting envelope to auth.py::verify_command() to confirm end-to-end
compatibility.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import patch

import pytest
from deploy_agent.auth import verify_command

_SECRET = "test-trigger-helper-secret-abc123"


def _sign_like_trigger_sh(envelope: dict, secret: str) -> dict:
    """Reproduce the signing logic from deploy-agent-trigger.sh exactly."""
    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {**body_dict, "_signature": sig}


_SAMPLE_ENVELOPE = {
    "correlation_id": "aaaaaaaa-0000-0000-0000-000000000001",
    "git_ref": "origin/main",
    "reason": "manual trigger by operator",
    "requested_at": "2026-04-21T12:00:00+00:00",
    "requested_by": "operator-manual",
    "scope": "runtime",
    "services": [],
}


def test_trigger_helper_signature_accepted_by_auth() -> None:
    """Signature produced by trigger helper is accepted by auth.py."""
    signed = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, _SECRET)
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is True


def test_trigger_helper_signature_field_present() -> None:
    """Signed envelope always contains _signature field."""
    signed = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, _SECRET)
    assert "_signature" in signed
    assert len(signed["_signature"]) == 64  # hex-encoded SHA-256


def test_trigger_helper_signature_excluded_from_body() -> None:
    """_signature is not included in the body that is signed (no circular dep)."""
    signed = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, _SECRET)
    body_dict = {k: v for k, v in signed.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert signed["_signature"] == expected


def test_tampered_envelope_rejected_by_auth() -> None:
    """Tampering any field after signing causes auth.py to reject."""
    signed = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, _SECRET)
    signed["git_ref"] = "origin/evil-branch"
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is False


def test_wrong_secret_rejected_by_auth() -> None:
    """Envelope signed with a different secret is rejected by auth.py."""
    signed = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, "wrong-secret")
    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        assert verify_command(signed) is False


def test_sort_keys_order_is_canonical() -> None:
    """Field insertion order doesn't affect signature — sort_keys is deterministic."""
    envelope_shuffled = {
        "services": [],
        "scope": "runtime",
        "requested_by": "operator-manual",
        "requested_at": "2026-04-21T12:00:00+00:00",
        "reason": "manual trigger by operator",
        "git_ref": "origin/main",
        "correlation_id": "aaaaaaaa-0000-0000-0000-000000000001",
    }
    signed_normal = _sign_like_trigger_sh(_SAMPLE_ENVELOPE, _SECRET)
    signed_shuffled = _sign_like_trigger_sh(envelope_shuffled, _SECRET)
    assert signed_normal["_signature"] == signed_shuffled["_signature"]
