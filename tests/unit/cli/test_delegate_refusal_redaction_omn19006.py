# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A refusal's receipt keeps the cause and drops the credentials (OMN-19006)."""

from __future__ import annotations

import pytest

from omnibase_infra.cli.cli_delegate import _redact_refusal_text as redact_refusal_text


def test_a_message_that_merely_names_a_token_keeps_every_word() -> None:
    """``sanitize_error_string`` withholds this whole message because of ``token``."""
    message = (
        "--caller-lane 'two words' is not a lane token: expected a letter or "
        "digit, then letters, digits, '.', '_', ':' or '-'"
    )
    assert redact_refusal_text(message) == message


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        pytest.param(
            "--kafka-bootstrap kafka://ops:hunter2@broker:9092 is unreachable",
            "hunter2",
            id="url-userinfo",
        ),
        pytest.param("refused with api_key=sk-abc123 set", "sk-abc123", id="key-pair"),
        pytest.param("PASSWORD=hunter2 was rejected", "hunter2", id="password-pair"),
        pytest.param(
            "bad key -----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----",
            "MIIB",
            id="key-material",
        ),
    ],
)
def test_a_credential_in_the_message_is_removed(message: str, secret: str) -> None:
    redacted = redact_refusal_text(message)
    assert secret not in redacted
    assert "REDACTED" in redacted


def test_the_flag_a_credential_was_given_to_is_still_named() -> None:
    redacted = redact_refusal_text(
        "--kafka-bootstrap kafka://ops:hunter2@broker:9092 is unreachable"
    )
    assert redacted.startswith("--kafka-bootstrap kafka://[REDACTED]@broker:9092")
    assert redacted.endswith("is unreachable")


def test_a_very_long_message_is_bounded() -> None:
    assert redact_refusal_text("x" * 5000).endswith("... [truncated]")
