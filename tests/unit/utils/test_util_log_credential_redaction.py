# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the shared log credential redaction filter (OMN-17423).

Tests cover:
    - Pass 1 (structural): credential-keyed values in ``LogRecord.args``
    - Pass 2 (shape): already-serialised credentials in the rendered message
    - Pass 3 (exception): credentials inside a traceback
    - Reference names (``api_key_count``, ``secret_ref``) staying readable
    - Installation: root coverage, non-propagating island loggers, idempotency
    - ``configure_logging()`` installing the filter on the runtime bootstrap

Every redaction test is paired with a NEGATIVE assertion that the surrounding
context survived. A filter that blanked the whole record would pass a
"secret is absent" assertion while destroying the log.

Related:
    - OMN-17423: API keys and session tokens must not reach service logs
    - OMN-18385: the dead-letter path this module shares its vocabulary with
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from omnibase_infra.utils.util_log_credential_redaction import (
    LOG_REDACTION_MARKER,
    CredentialRedactionFilter,
    install_credential_redaction_filter,
    redact_credential_patterns,
)

# A syntactically real JWT shape (three base64url segments). Not a live token:
# the payload decodes to {"sub":"omn17423-test"} and it is unsigned garbage.
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvbW4xNzQyMy10ZXN0In0.ZmFrZXNpZ25hdHVyZQ"
FAKE_API_KEY = "onxk_omn17423testkeyvaluenotreal0123456789"


def _make_record(msg: str, args: Any = None, **kwargs: Any) -> logging.LogRecord:
    """Build a record the way ``Logger._log`` does.

    ``Logger.info("msg %s", payload)`` reaches ``LogRecord`` as the TUPLE
    ``(payload,)``; LogRecord then unwraps a lone Mapping back to the bare
    dict. Passing a bare dict here instead would construct a record no logging
    call can produce -- and would test the filter against the wrong shape.
    """
    if args is not None and not isinstance(args, tuple):
        args = (args,)
    return logging.LogRecord(
        name="test.omn17423",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=kwargs.get("exc_info"),
    )


def _apply(record: logging.LogRecord) -> logging.LogRecord:
    assert CredentialRedactionFilter().filter(record) is True
    return record


@pytest.fixture
def clean_logging() -> Iterator[None]:
    """Snapshot and restore global logging state.

    These tests attach filters to real handlers. Without restoration a failure
    here would silently change the behaviour of every later test in the worker.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_filters = {id(h): list(h.filters) for h in root.handlers}
    saved_dict = dict(root.manager.loggerDict)
    try:
        yield
    finally:
        root.handlers = saved_handlers
        for handler in root.handlers:
            handler.filters = saved_filters.get(id(handler), [])
        root.manager.loggerDict.clear()
        root.manager.loggerDict.update(saved_dict)


# ---------------------------------------------------------------------------
# Pass 1 - structural
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStructuralPass:
    def test_dict_arg_credential_key_redacted(self) -> None:
        record = _apply(
            _make_record("event %s", {"api_key": FAKE_API_KEY, "tenant": "acme"})
        )
        assert record.args == {"api_key": LOG_REDACTION_MARKER, "tenant": "acme"}

    def test_plaintext_key_redacted(self) -> None:
        """The dashboard key-creation field this ticket exists for."""
        record = _apply(_make_record("created %s", {"plaintext_key": FAKE_API_KEY}))
        assert record.args == {"plaintext_key": LOG_REDACTION_MARKER}

    @pytest.mark.parametrize(
        "field",
        [
            "access_token",
            "refresh_token",
            "session_token",
            "gateway_token",
            "client_secret",
            "password",
            "authorization",
            "x-api-key",
            "apiKey",
        ],
    )
    def test_every_credential_field_shape_redacted(self, field: str) -> None:
        record = _apply(_make_record("event %s", {field: "sensitive-value"}))
        assert record.args == {field: LOG_REDACTION_MARKER}

    @pytest.mark.parametrize(
        "field",
        ["api_key_count", "token_count", "secret_ref", "api_key_id", "key_hash"],
    )
    def test_reference_fields_stay_readable(self, field: str) -> None:
        """Non-vacuity: a filter that redacted everything would pass the
        positive tests above. References carry forensic value and no secret."""
        record = _apply(_make_record("event %s", {field: "readable-value"}))
        assert record.args == {field: "readable-value"}

    def test_nested_containers_are_walked(self) -> None:
        record = _apply(
            _make_record("event %s", {"a": {"b": [{"access_token": FAKE_JWT}]}})
        )
        assert record.args == {"a": {"b": [{"access_token": LOG_REDACTION_MARKER}]}}

    def test_tuple_args_are_walked(self) -> None:
        record = _apply(_make_record("%s %s", ("acme", {"api_key": FAKE_API_KEY})))
        assert record.args == ("acme", {"api_key": LOG_REDACTION_MARKER})

    def test_depth_limit_fails_closed(self) -> None:
        """Beyond the depth limit nothing is inspected, so the subtree is
        replaced rather than passed through unredacted."""
        deep: Any = {"api_key": FAKE_API_KEY}
        for _ in range(20):
            deep = {"nest": deep}
        record = _apply(_make_record("event %s", deep))
        rendered = record.getMessage()
        assert FAKE_API_KEY not in rendered
        assert LOG_REDACTION_MARKER in rendered


# ---------------------------------------------------------------------------
# Pass 2 - shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShapePass:
    def test_serialised_jwt_in_message_redacted(self) -> None:
        record = _apply(_make_record(f'{{"access_token": "{FAKE_JWT}"}}'))
        message = record.getMessage()
        assert FAKE_JWT not in message
        assert LOG_REDACTION_MARKER in message

    def test_serialised_api_key_in_message_redacted(self) -> None:
        record = _apply(_make_record(f"provisioned key {FAKE_API_KEY} for acme"))
        message = record.getMessage()
        assert FAKE_API_KEY not in message
        # The surrounding context must survive -- this is what separates
        # redaction from destroying the line.
        assert "provisioned key" in message
        assert "for acme" in message

    def test_credential_shape_inside_positional_arg_redacted(self) -> None:
        record = _apply(_make_record("body=%s", (f"token={FAKE_JWT}",)))
        assert FAKE_JWT not in record.getMessage()

    def test_ordinary_message_untouched(self) -> None:
        record = _apply(_make_record("tenant acme resolved in 12ms"))
        assert record.getMessage() == "tenant acme resolved in 12ms"

    def test_short_onxk_lookalike_not_redacted(self) -> None:
        """The prefix alone is not a key; the pattern requires 20+ chars of
        body so an identifier like ``onxk_dev`` stays readable."""
        assert redact_credential_patterns("onxk_dev") == "onxk_dev"


# ---------------------------------------------------------------------------
# Pass 3 - exception text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionPass:
    def test_credential_in_traceback_redacted(self) -> None:
        try:
            raise ValueError(f"upstream rejected {FAKE_JWT}")
        except ValueError:
            import sys

            record = _make_record("call failed", exc_info=sys.exc_info())
        _apply(record)
        assert record.exc_text is not None
        assert FAKE_JWT not in record.exc_text
        assert "upstream rejected" in record.exc_text
        # Cleared so the formatter uses our scrubbed copy rather than
        # re-deriving the raw traceback from the live tuple.
        assert record.exc_info is None

    def test_formatted_output_carries_no_credential(self) -> None:
        """End-to-end through a real Formatter -- the surface a pod log is."""
        try:
            raise RuntimeError(f"bad token {FAKE_JWT}")
        except RuntimeError:
            import sys

            record = _make_record("call failed", exc_info=sys.exc_info())
        _apply(record)
        formatted = logging.Formatter("%(message)s").format(record)
        assert FAKE_JWT not in formatted
        assert "call failed" in formatted

    def test_precached_exc_text_redacted(self) -> None:
        record = _make_record("call failed")
        record.exc_text = f"Traceback ... {FAKE_JWT}"
        _apply(record)
        assert record.exc_text is not None
        assert FAKE_JWT not in record.exc_text


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInstallation:
    def test_installs_on_root_handlers(self, clean_logging: None) -> None:
        root = logging.getLogger()
        handler = logging.NullHandler()
        root.handlers = [handler]

        assert install_credential_redaction_filter() == 1
        assert any(isinstance(f, CredentialRedactionFilter) for f in handler.filters)

    def test_idempotent(self, clean_logging: None) -> None:
        root = logging.getLogger()
        handler = logging.NullHandler()
        root.handlers = [handler]

        install_credential_redaction_filter()
        assert install_credential_redaction_filter() == 0
        assert (
            sum(isinstance(f, CredentialRedactionFilter) for f in handler.filters) == 1
        )

    def test_covers_non_propagating_island_logger(self, clean_logging: None) -> None:
        """The uvicorn shape, discovered rather than hardcoded: a logger with
        its own handlers and ``propagate = False`` never reaches root."""
        root = logging.getLogger()
        root.handlers = [logging.NullHandler()]

        island = logging.getLogger("omn17423.island")
        island_handler = logging.NullHandler()
        island.handlers = [island_handler]
        island.propagate = False

        installed = install_credential_redaction_filter()

        assert installed == 2
        assert any(
            isinstance(f, CredentialRedactionFilter) for f in island_handler.filters
        )

    def test_propagating_child_is_not_double_filtered(
        self, clean_logging: None
    ) -> None:
        """A child that propagates is covered by root's handler. Attaching a
        second filter there would scrub the same record twice for nothing."""
        root = logging.getLogger()
        root.handlers = [logging.NullHandler()]

        child = logging.getLogger("omn17423.child")
        child_handler = logging.NullHandler()
        child.handlers = [child_handler]
        child.propagate = True

        install_credential_redaction_filter()

        assert not any(
            isinstance(f, CredentialRedactionFilter) for f in child_handler.filters
        )

    def test_named_logger_covered_on_request(self, clean_logging: None) -> None:
        root = logging.getLogger()
        root.handlers = [logging.NullHandler()]
        named = logging.getLogger("omn17423.named")
        named_handler = logging.NullHandler()
        named.handlers = [named_handler]
        named.propagate = True

        install_credential_redaction_filter(extra_logger_names=("omn17423.named",))

        assert any(
            isinstance(f, CredentialRedactionFilter) for f in named_handler.filters
        )

    def test_record_reaches_handler_scrubbed(self, clean_logging: None) -> None:
        """The whole point, exercised through a real emit path."""
        emitted: list[str] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                emitted.append(self.format(record))

        handler = CapturingHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)

        install_credential_redaction_filter()
        logging.getLogger("omn17423.emit").info(
            "attach for %s", {"access_token": FAKE_JWT, "edge": "edge-1"}
        )

        assert len(emitted) == 1
        assert FAKE_JWT not in emitted[0]
        assert "edge-1" in emitted[0]


@pytest.mark.unit
class TestRuntimeBootstrapWiring:
    def test_configure_logging_installs_the_filter(self, clean_logging: None) -> None:
        """AC2's actual claim: the shared runtime bootstrap -- the one every
        runtime service reaches via the ``onex-runtime`` entrypoint -- installs
        the filter, so no service has to remember to."""
        from omnibase_infra.runtime.service_kernel import configure_logging

        root = logging.getLogger()
        root.handlers = [logging.NullHandler()]

        configure_logging()

        assert root.handlers, "configure_logging left no handler to filter"
        assert all(
            any(isinstance(f, CredentialRedactionFilter) for f in handler.filters)
            for handler in root.handlers
        )
