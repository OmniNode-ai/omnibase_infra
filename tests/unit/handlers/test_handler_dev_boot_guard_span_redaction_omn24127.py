# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Span redaction of the dev boot guard's kept container log (OMN-24127).

``sanitize_error_string`` withholds a WHOLE text containing any sensitive word,
which is how a real dev boot log was kept as the 39-byte
``[REDACTED - potentially sensitive data]``. ``redact_credential_spans`` removes
only the credential-shaped spans. Every test pairs "the credential is gone" with
"the surrounding text survived".
"""

from __future__ import annotations

import pytest

from omnibase_infra.handlers.handler_dev_boot_guard import redact_credential_spans
from omnibase_infra.utils.util_log_credential_redaction import LOG_REDACTION_MARKER


@pytest.mark.unit
class TestRedactCredentialSpans:
    """``redact_credential_spans`` removes spans and keeps every other line.

    ``sanitize_error_string`` withholds a WHOLE text containing any sensitive
    word, which is how a real dev boot log was kept as the 39-byte
    ``[REDACTED - potentially sensitive data]`` (OMN-24127). Every test here
    pairs "the credential is gone" with "the surrounding text survived", the
    same pairing the filter tests above make.
    """

    def test_planted_token_replaced_and_surrounding_lines_survive(self) -> None:
        token = "sk-" + "a" * 24
        text = f"boot started\nattach upstream with {token}\nboot finished"
        redacted = redact_credential_spans(text)
        assert token not in redacted
        assert LOG_REDACTION_MARKER in redacted
        assert "boot started" in redacted
        assert "boot finished" in redacted

    def test_url_userinfo_replaced_keeping_scheme_and_host(self) -> None:
        assert (
            redact_credential_spans("redis://user:hunter2@host:6379")
            == "redis://[REDACTED]@host:6379"
        )

    def test_bare_sensitive_words_without_values_are_unchanged(self) -> None:
        """The exact text ``sanitize_error_string`` destroys: the words are
        there but no credential-shaped value, so nothing is replaced."""
        text = "the token expired and the password was rejected"
        assert redact_credential_spans(text) == text

    def test_secret_key_value_pair_keeps_key_and_loses_value(self) -> None:
        assert redact_credential_spans("password=hunter2") == "password=[REDACTED]"

    def test_pem_key_material_replaced_and_text_around_it_kept(self) -> None:
        pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQ\n"
            "-----END PRIVATE KEY-----\n"
        )
        text = f"loaded key\n{pem}service up"
        redacted = redact_credential_spans(text)
        assert "MIIEvQIBADANBgkqhkiG9w0BAQ" not in redacted
        assert "-----BEGIN PRIVATE KEY-----" not in redacted
        assert "[REDACTED key material]" in redacted
        assert "loaded key" in redacted
        assert "service up" in redacted

    def test_empty_string_returns_empty_string(self) -> None:
        assert redact_credential_spans("") == ""


@pytest.mark.unit
def test_guard_value_shapes_equal_the_shared_credential_value_patterns() -> None:
    """Seam: the guard holds its own copy of the credential value shapes.

    The deploy agent runs the guard against a published omnibase_infra that can
    predate ``util_dlq_credential_redaction``, so the guard cannot import the
    shared list. A shape added, dropped or changed on either side fails here, so
    the kept boot log is cleaned of the same shapes the rest of the system treats
    as credentials.
    """
    from omnibase_infra.handlers import handler_dev_boot_guard as guard
    from omnibase_infra.utils.util_dlq_credential_redaction import (
        CREDENTIAL_VALUE_PATTERNS,
    )

    assert [(p.pattern, p.flags) for p in guard._CREDENTIAL_VALUE_SHAPES] == [
        (p.pattern, p.flags) for p in CREDENTIAL_VALUE_PATTERNS
    ]


@pytest.mark.unit
def test_guard_marker_equals_the_shared_log_redaction_marker() -> None:
    from omnibase_infra.handlers import handler_dev_boot_guard as guard

    assert guard._REDACTION_MARKER == LOG_REDACTION_MARKER
