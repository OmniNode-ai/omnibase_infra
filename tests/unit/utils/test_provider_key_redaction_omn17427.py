# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Synthetic BYOK shapes must not escape the runtime's redaction paths."""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from omnibase_infra.cli.cli_delegate import _redact_refusal_text
from omnibase_infra.gate.executor import redact_if_secret_bearing
from omnibase_infra.runtime.service_kernel import configure_logging
from omnibase_infra.utils.util_dlq_credential_redaction import (
    DLQ_REDACTION_MARKER,
    redact_credential_fields_in_text,
)
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_error_string,
)
from omnibase_infra.utils.util_log_credential_redaction import (
    LOG_REDACTION_MARKER,
    CredentialRedactionFilter,
    redact_credential_patterns,
)

pytestmark = pytest.mark.unit

# Constructed, deliberately repetitive fixtures; none is a real credential.
# This list is independent of the implementation so dropping a matcher fails.
PROVIDER_KEYS = [
    pytest.param("sk-" + "testOnly0" * 6, id="openai"),
    pytest.param("sk-proj-" + "testOnly_-0" * 16, id="openai-project"),
    pytest.param("sk-ant-api03-" + "testOnly_-0" * 9, id="anthropic"),
    pytest.param("sk-ant-oat01-" + "testOnly_-0" * 9, id="anthropic-oauth"),
    pytest.param("sk-or-v1-" + "0123456789abcdef" * 4, id="openrouter"),
    pytest.param("AIza" + "test0_-" * 5, id="gemini"),
    pytest.param("0123456789abcdef" * 2 + "." + "testOnly0" * 2, id="glm"),
    pytest.param("Bearer " + "opaqueTestValue0" * 3, id="glm-opaque-bearer"),
]


@pytest.mark.parametrize("key", PROVIDER_KEYS)
def test_rendered_provider_key_is_redacted(key: str) -> None:
    assert redact_credential_patterns(f"before {key} after") == (
        f"before {LOG_REDACTION_MARKER} after"
    )


@pytest.mark.parametrize("key", PROVIDER_KEYS)
@pytest.mark.parametrize("mode", ["message", "args", "nested", "exception", "cached"])
def test_runtime_handler_redacts_provider_keys(key: str, mode: str) -> None:
    """Exercise the same configure_logging -> handler -> formatter path as PID 1."""
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    # The installer also covers existing non-propagating logger islands.
    # Restore their filters so this bootstrap probe cannot affect later tests.
    loggers = [
        root,
        *(
            logger
            for logger in root.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ),
    ]
    saved_filters = [(h, h.filters[:]) for logger in loggers for h in logger.handlers]
    try:
        root.handlers = [handler]
        configure_logging()
        logger = logging.getLogger("provider-redaction-test")
        if mode == "message":
            logger.warning(f"before {key} after")
        elif mode == "args":
            logger.warning("before %s after", key)
        elif mode == "nested":
            logger.warning("before %s after", {"note": [key], "tenant": "example"})
        elif mode == "exception":
            try:
                raise ValueError(f"before {key} after")
            except ValueError:
                logger.warning("operation failed", exc_info=sys.exc_info())
        else:
            record = logger.makeRecord(
                logger.name, logging.WARNING, __file__, 1, "operation failed", (), None
            )
            record.exc_text = f"before {key} after"
            logger.handle(record)
        rendered = output.getvalue()
        assert key not in rendered
        assert LOG_REDACTION_MARKER in rendered
        assert "before" in rendered and "after" in rendered
    finally:
        root.handlers, root.level = saved_handlers, saved_level
        for saved_handler, filters in saved_filters:
            saved_handler.filters = filters
        handler.close()


@pytest.mark.parametrize("key", PROVIDER_KEYS)
@pytest.mark.parametrize("mode", ["json", "list", "scalar", "text"])
def test_dlq_body_redacts_provider_keys_and_records_paths(key: str, mode: str) -> None:
    if mode == "json":
        text = json.dumps({"note": f"before {key} after", "tenant": "example"})
        expected_path = "note"
    elif mode == "list":
        text = json.dumps({"items": [f"before {key} after"]})
        expected_path = "items[0]"
    elif mode == "scalar":
        text = json.dumps(f"before {key} after")
        expected_path = "<root>"
    else:
        text = f"before {key} after"
        expected_path = "<root>"
    redacted, paths = redact_credential_fields_in_text(text)
    assert key not in redacted
    assert DLQ_REDACTION_MARKER in redacted
    assert "before" in redacted and "after" in redacted
    assert expected_path in paths
    assert redact_credential_fields_in_text(redacted) == (redacted, ())


@pytest.mark.parametrize("key", PROVIDER_KEYS)
def test_error_and_receipt_paths_redact_provider_keys_before_truncation(
    key: str,
) -> None:
    text = f"before {key} after"
    for rendered in (
        sanitize_error_string(text),
        sanitize_error_message(ValueError(text)),
        _redact_refusal_text(text),
        redact_if_secret_bearing(text),
    ):
        assert rendered is not None
        assert key not in rendered
        assert "redact" in rendered.lower()
    # Cutting a key first can leave a credential fragment that no matcher sees.
    assert "testOnly" not in sanitize_error_string(text, max_length=24)
    assert "testOnly" not in sanitize_error_message(ValueError(text), max_length=24)


def test_opaque_glm_value_is_redacted_by_field_name() -> None:
    text = json.dumps({"api_key": "synthetic-opaque-value", "note": "ordinary"})
    redacted, paths = redact_credential_fields_in_text(text)
    assert json.loads(redacted) == {"api_key": DLQ_REDACTION_MARKER, "note": "ordinary"}
    assert paths == ("api_key",)


@pytest.mark.parametrize(
    "text",
    [
        "ordinary text: task completed in 42 ms",
        "sk-short AIza-example onxk_short",
        "--caller-lane 'two words' is not a lane token",
        "0123456789abcdef0123456789abcdef",
        '{"note": "ordinary", "tenant": "example"}',
    ],
)
def test_ordinary_text_is_unchanged(text: str) -> None:
    assert redact_credential_patterns(text) == text
    assert redact_credential_fields_in_text(text) == (text, ())
    assert _redact_refusal_text(text) == text
    assert redact_if_secret_bearing(text) == text
    record = logging.LogRecord("test", logging.INFO, __file__, 1, text, (), None)
    assert CredentialRedactionFilter().filter(record)
    assert record.getMessage() == text


def test_safe_error_control_is_unchanged() -> None:
    text = "ordinary text: task completed in 42 ms"
    assert sanitize_error_string(text) == text
    assert sanitize_error_message(ValueError(text)) == f"ValueError: {text}"
