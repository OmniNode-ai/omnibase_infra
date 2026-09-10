# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for registration E2E logger isolation."""

from __future__ import annotations

import logging

import pytest

from tests.integration.registration.e2e.logging_fixture import configured_e2e_logging


def _restore_logger(
    logger: logging.Logger,
    *,
    level: int,
    propagate: bool,
    handlers: tuple[logging.Handler, ...],
) -> None:
    """Return a test-mutated logger to its exact prior state."""
    for handler in tuple(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.mark.unit
def test_configured_e2e_logging_restores_later_debug_capture(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A temporary E2E INFO level cannot suppress a later unrelated DEBUG log."""
    e2e_logger = logging.getLogger("tests.integration.registration.e2e")
    infra_logger = logging.getLogger("omnibase_infra")
    unrelated_logger = logging.getLogger("omnibase_infra.fixture_isolation_probe")
    e2e_state = (e2e_logger.level, e2e_logger.propagate, tuple(e2e_logger.handlers))
    infra_state = (
        infra_logger.level,
        infra_logger.propagate,
        tuple(infra_logger.handlers),
    )
    existing_handler = logging.StreamHandler()
    e2e_logger.addHandler(existing_handler)
    e2e_logger.setLevel(logging.ERROR)
    infra_logger.setLevel(logging.DEBUG)

    try:
        with configured_e2e_logging():
            assert e2e_logger.level == logging.DEBUG
            assert infra_logger.level == logging.INFO
            assert existing_handler in e2e_logger.handlers
            assert e2e_logger.handlers.count(existing_handler) == 1

            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                unrelated_logger.debug("suppressed while E2E fixture is active")
            assert "suppressed while E2E fixture is active" not in caplog.text

        assert e2e_logger.level == logging.ERROR
        assert infra_logger.level == logging.DEBUG
        assert e2e_logger.propagate == e2e_state[1]
        assert infra_logger.propagate == infra_state[1]
        assert existing_handler in e2e_logger.handlers
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            unrelated_logger.debug("captured after E2E fixture cleanup")
        assert "captured after E2E fixture cleanup" in caplog.text
    finally:
        _restore_logger(
            e2e_logger,
            level=e2e_state[0],
            propagate=e2e_state[1],
            handlers=e2e_state[2],
        )
        _restore_logger(
            infra_logger,
            level=infra_state[0],
            propagate=infra_state[1],
            handlers=infra_state[2],
        )


@pytest.mark.unit
def test_configured_e2e_logging_restores_on_exception_unwind() -> None:
    """Configuration is restored even when an E2E test body raises."""
    e2e_logger = logging.getLogger("tests.integration.registration.e2e")
    infra_logger = logging.getLogger("omnibase_infra")
    e2e_state = (e2e_logger.level, e2e_logger.propagate, tuple(e2e_logger.handlers))
    infra_state = (
        infra_logger.level,
        infra_logger.propagate,
        tuple(infra_logger.handlers),
    )
    existing_handler = logging.StreamHandler()
    e2e_logger.addHandler(existing_handler)
    e2e_logger.setLevel(logging.ERROR)
    infra_logger.setLevel(logging.WARNING)

    try:
        with pytest.raises(RuntimeError, match="fixture-body-failure"):
            with configured_e2e_logging():
                raise RuntimeError("fixture-body-failure")

        assert e2e_logger.level == logging.ERROR
        assert infra_logger.level == logging.WARNING
        assert e2e_logger.propagate == e2e_state[1]
        assert infra_logger.propagate == infra_state[1]
        assert existing_handler in e2e_logger.handlers
        assert e2e_logger.handlers.count(existing_handler) == 1
    finally:
        _restore_logger(
            e2e_logger,
            level=e2e_state[0],
            propagate=e2e_state[1],
            handlers=e2e_state[2],
        )
        _restore_logger(
            infra_logger,
            level=infra_state[0],
            propagate=infra_state[1],
            handlers=infra_state[2],
        )


@pytest.mark.unit
def test_configured_e2e_logging_removes_and_closes_its_owned_handler() -> None:
    """The no-existing-handler branch owns and cleans up precisely one handler."""
    e2e_logger = logging.getLogger("tests.integration.registration.e2e")
    e2e_state = (e2e_logger.level, e2e_logger.propagate, tuple(e2e_logger.handlers))
    removed_stream_handlers = [
        handler
        for handler in e2e_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    for handler in removed_stream_handlers:
        e2e_logger.removeHandler(handler)

    try:
        before = tuple(e2e_logger.handlers)
        with configured_e2e_logging():
            added = [
                handler for handler in e2e_logger.handlers if handler not in before
            ]
            assert len(added) == 1
            owned_handler = added[0]

        assert owned_handler not in e2e_logger.handlers
        assert owned_handler._closed is True  # type: ignore[attr-defined]
    finally:
        _restore_logger(
            e2e_logger,
            level=e2e_state[0],
            propagate=e2e_state[1],
            handlers=e2e_state[2],
        )


@pytest.mark.unit
def test_configured_e2e_logging_restores_after_setup_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handler-install failure cannot leak either temporary logger level."""
    e2e_logger = logging.getLogger("tests.integration.registration.e2e")
    infra_logger = logging.getLogger("omnibase_infra")
    e2e_state = (e2e_logger.level, e2e_logger.propagate, tuple(e2e_logger.handlers))
    infra_state = (
        infra_logger.level,
        infra_logger.propagate,
        tuple(infra_logger.handlers),
    )
    removed_stream_handlers = [
        handler
        for handler in e2e_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    for handler in removed_stream_handlers:
        e2e_logger.removeHandler(handler)
    e2e_logger.setLevel(logging.ERROR)
    infra_logger.setLevel(logging.WARNING)

    def raise_on_add(_handler: logging.Handler) -> None:
        raise RuntimeError("handler-install-failure")

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(e2e_logger, "addHandler", raise_on_add)
            with pytest.raises(RuntimeError, match="handler-install-failure"):
                with configured_e2e_logging():
                    pytest.fail("fixture setup must fail before its body runs")

        assert e2e_logger.level == logging.ERROR
        assert infra_logger.level == logging.WARNING
        assert e2e_logger.propagate == e2e_state[1]
        assert infra_logger.propagate == infra_state[1]
    finally:
        _restore_logger(
            e2e_logger,
            level=e2e_state[0],
            propagate=e2e_state[1],
            handlers=e2e_state[2],
        )
        _restore_logger(
            infra_logger,
            level=infra_state[0],
            propagate=infra_state[1],
            handlers=infra_state[2],
        )
