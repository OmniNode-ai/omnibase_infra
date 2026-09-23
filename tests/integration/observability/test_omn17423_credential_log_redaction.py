# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end credential redaction across the shared runtime logging path (OMN-17423).

The unit tests exercise the filter's three passes in isolation. These exercise
the SEAMS the unit tests cannot see:

1. ``configure_logging()`` -- the real runtime bootstrap every service reaches
   via the ``onex-runtime`` entrypoint -- installs the filter, and a credential
   logged afterwards comes out of a real handler + Formatter scrubbed. AC2's
   claim is about the bootstrap, not about the class.
2. The filter and the dead-letter path agree on what a credential field is.
   They share ``is_credential_field_name`` precisely so they cannot drift; a
   test that pins the two together is what makes that guarantee observable
   rather than aspirational.
3. The AC4 gate reports zero violations over the real ``src/omnibase_infra``
   tree -- the same tree AC2 covers.

Related:
    - OMN-17423: API keys and session tokens must not reach service logs
    - OMN-18385: the dead-letter redaction path sharing the vocabulary
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest

from omnibase_infra.utils.util_dlq_credential_redaction import (
    is_credential_field_name,
    redact_credential_fields,
)
from omnibase_infra.utils.util_log_credential_redaction import (
    LOG_REDACTION_MARKER,
    CredentialRedactionFilter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Syntactically real shapes, deliberately not live values: the JWT payload
# decodes to {"sub":"omn17423-test"} over an unsigned garbage segment.
#
# Named for their SHAPE, not for what they impersonate. These tests have to
# push a credential-shaped value through a real logger to prove the filter
# scrubs it, and CodeQL's clear-text-logging query classifies a source by its
# NAME -- a constant called FAKE_API_KEY logged here is a high-severity alert
# on every run, for a value that is not a credential and a line whose whole
# purpose is to prove it never reaches the stream.
JWT_SHAPED_SENTINEL = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvbW4xNzQyMy10ZXN0In0.ZmFrZXNpZ25hdHVyZQ"
)
ONXK_SHAPED_SENTINEL = "onxk_omn17423integrationkeyvaluenotreal0123"


class Bootstrapped(NamedTuple):
    """What the bootstrap produced: the handler it filtered, and what it wrote."""

    handler: logging.Handler
    stream: StringIO


@pytest.fixture
def bootstrapped_logging() -> Iterator[Bootstrapped]:
    """Run the real bootstrap, capture what a handler would actually write.

    ``configure_logging`` calls ``basicConfig``, which is a no-op when root
    already has a handler -- so the capture handler is installed FIRST and the
    bootstrap then filters it, which is exactly the ordering in a live process
    (handlers exist, the filter is attached to them).
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_filters = {id(h): list(h.filters) for h in saved_handlers}

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    from omnibase_infra.runtime.service_kernel import configure_logging

    configure_logging()
    try:
        yield Bootstrapped(handler=handler, stream=stream)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        for existing in saved_handlers:
            existing.filters = saved_filters.get(id(existing), [])


@pytest.mark.integration
class TestBootstrapInstallsTheFilter:
    def test_configure_logging_attaches_the_filter(
        self, bootstrapped_logging: Bootstrapped
    ) -> None:
        """Asserted on the handler the bootstrap saw, not on every handler on
        the root logger: pytest's own logging plugin attaches and detaches
        capture handlers around each test, and a live process has none of them.
        """
        assert any(
            isinstance(f, CredentialRedactionFilter)
            for f in bootstrapped_logging.handler.filters
        )

    def test_structured_credential_never_reaches_the_stream(
        self, bootstrapped_logging: Bootstrapped
    ) -> None:
        """A service logger, not the root logger -- records propagate up to the
        handler the filter is attached to, which is the live shape."""
        logging.getLogger("omnibase_infra.runtime.gateway").info(
            "attach accepted %s",
            {"access_token": JWT_SHAPED_SENTINEL, "edge_instance_id": "edge-omn17423"},
        )
        written = bootstrapped_logging.stream.getvalue()

        assert JWT_SHAPED_SENTINEL not in written
        assert LOG_REDACTION_MARKER in written
        # Non-vacuity: the line was emitted and kept its context. A filter that
        # dropped the record would satisfy the assertion above.
        assert "edge-omn17423" in written
        assert "omnibase_infra.runtime.gateway" in written

    def test_serialised_credential_never_reaches_the_stream(
        self, bootstrapped_logging: Bootstrapped
    ) -> None:
        """No field name survives an f-string; only the shape pass can catch it."""
        logging.getLogger("omnibase_infra.runtime.provision").warning(
            f"provisioned {ONXK_SHAPED_SENTINEL} for tenant acme"
        )
        written = bootstrapped_logging.stream.getvalue()

        assert ONXK_SHAPED_SENTINEL not in written
        assert "for tenant acme" in written

    def test_credential_in_a_traceback_never_reaches_the_stream(
        self, bootstrapped_logging: Bootstrapped
    ) -> None:
        """A Formatter appends exception text AFTER the message, outside the
        message passes -- the surface that leaked before pass 3 existed."""
        logger = logging.getLogger("omnibase_infra.runtime.effects")
        try:
            raise RuntimeError(f"introspection rejected {JWT_SHAPED_SENTINEL}")
        except RuntimeError:
            logger.exception("gateway introspect failed")

        written = bootstrapped_logging.stream.getvalue()
        assert JWT_SHAPED_SENTINEL not in written
        assert "introspection rejected" in written
        assert "Traceback" in written

    def test_ordinary_runtime_logging_is_unchanged(
        self, bootstrapped_logging: Bootstrapped
    ) -> None:
        """The filter is on every handler in every runtime service. If it
        altered ordinary lines, it would corrupt every log on the platform."""
        logging.getLogger("omnibase_infra.runtime.service_kernel").info(
            "ONEX Kernel v%s initializing...", "0.1.0"
        )
        written = bootstrapped_logging.stream.getvalue()
        assert "ONEX Kernel v0.1.0 initializing..." in written
        assert LOG_REDACTION_MARKER not in written


@pytest.mark.integration
class TestVocabularyIsShared:
    """The log filter and the dead-letter path must not drift apart.

    Both defer to ``is_credential_field_name``. These assertions fail the
    moment one of them grows a private denylist.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "api_key",
            "plaintext_key",
            "access_token",
            "refresh_token",
            "session_token",
            "gateway_token",
            "client_secret",
            "password",
        ],
    )
    def test_both_paths_redact_the_same_fields(
        self, field: str, bootstrapped_logging: Bootstrapped
    ) -> None:
        assert is_credential_field_name(field)

        redacted, paths = redact_credential_fields({field: JWT_SHAPED_SENTINEL})
        assert paths == (field,)
        assert isinstance(redacted, dict)
        assert redacted[field] != JWT_SHAPED_SENTINEL

        logging.getLogger("omnibase_infra.runtime.shared").info(
            "event %s", {field: JWT_SHAPED_SENTINEL}
        )
        assert JWT_SHAPED_SENTINEL not in bootstrapped_logging.stream.getvalue()

    @pytest.mark.parametrize("field", ["api_key_count", "api_key_id", "token_count"])
    def test_both_paths_leave_references_readable(
        self, field: str, bootstrapped_logging: Bootstrapped
    ) -> None:
        """Symmetry in the other direction. Redacting a reference destroys
        forensic value and protects nothing."""
        assert not is_credential_field_name(field)

        _, paths = redact_credential_fields({field: "readable-value"})
        assert paths == ()

        logging.getLogger("omnibase_infra.runtime.shared").info(
            "event %s", {field: "readable-value"}
        )
        assert "readable-value" in bootstrapped_logging.stream.getvalue()


def _load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_omn17423_gate", REPO_ROOT / "scripts" / "ci" / "check_no_credential_in_log.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
class TestGateOverTheRealTree:
    def test_shared_runtime_path_has_no_credential_log_site(self) -> None:
        """AC4 over the same tree AC2 covers."""
        assert _load_gate().run_normal(REPO_ROOT / "src" / "omnibase_infra") == 0

    def test_gate_is_non_vacuous_against_the_committed_fixture(self) -> None:
        gate = _load_gate()
        fixture = (
            REPO_ROOT
            / "scripts"
            / "ci"
            / "tests"
            / "fixtures"
            / "credential_in_log_fixture.py"
        )
        assert gate.run_self_test(fixture, 9) == 0
        assert gate.run_self_test(fixture, 8) == 1
