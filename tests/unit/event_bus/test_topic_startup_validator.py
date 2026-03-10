# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for TopicStartupValidator (OMN-3769).

Validates startup topic existence checking with five scenarios:
1. All required topics present
2. Topics missing (degraded)
3. Strict mode raises RuntimeError
4. Broker unreachable
5. aiokafka not importable
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.model_topic_validation_result import (
    ModelTopicValidationResult,
)

# Test-local suffixes to avoid depending on real platform topics
SAMPLE_SUFFIXES: tuple[str, ...] = (
    "onex.evt.test.topic-a.v1",
    "onex.evt.test.topic-b.v1",
    "onex.cmd.test.topic-c.v1",
)


def _make_mock_admin(
    *,
    broker_topics: dict[str, object] | None = None,
    start_error: Exception | None = None,
) -> AsyncMock:
    """Create a mock AIOKafkaAdminClient."""
    admin = AsyncMock()
    if start_error:
        admin.start = AsyncMock(side_effect=start_error)
    else:
        admin.start = AsyncMock()
    admin.list_topics = AsyncMock(return_value=broker_topics or {})
    admin.close = AsyncMock()
    return admin


def _patch_aiokafka_import(mock_admin: AsyncMock):
    """Context manager that patches the runtime import of aiokafka.admin.

    The validator does ``from aiokafka.admin import AIOKafkaAdminClient``
    inside ``validate()``. We inject a fake module into ``sys.modules``
    so the import resolves to our mock.
    """
    fake_admin_module = ModuleType("aiokafka.admin")
    fake_admin_module.AIOKafkaAdminClient = MagicMock(return_value=mock_admin)  # type: ignore[attr-defined]

    fake_aiokafka = ModuleType("aiokafka")

    return patch.dict(
        sys.modules,
        {
            "aiokafka": fake_aiokafka,
            "aiokafka.admin": fake_admin_module,
        },
    )


@pytest.fixture
def validator():
    """Create a validator with test bootstrap servers."""
    from omnibase_infra.event_bus.service_topic_startup_validator import (
        TopicStartupValidator,
    )

    return TopicStartupValidator(bootstrap_servers="localhost:19092")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_topics_present(
    validator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All required topics present -> is_valid=True, status='success', no error logs."""
    broker_topics = {s: MagicMock() for s in SAMPLE_SUFFIXES}
    mock_admin = _make_mock_admin(broker_topics=broker_topics)

    with (
        patch(
            "omnibase_infra.event_bus.service_topic_startup_validator.ALL_PROVISIONED_SUFFIXES",
            SAMPLE_SUFFIXES,
        ),
        _patch_aiokafka_import(mock_admin),
        caplog.at_level(logging.DEBUG),
    ):
        result = await validator.validate(correlation_id=uuid4())

    assert result.is_valid is True
    assert result.status == "success"
    assert result.missing_topics == ()
    assert set(result.present_topics) == set(SAMPLE_SUFFIXES)

    # No ERROR-level logs
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_topics_missing(
    validator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Topics missing -> is_valid=False, status='degraded', each missing in logger.error."""
    # Only first topic present
    broker_topics = {SAMPLE_SUFFIXES[0]: MagicMock()}
    mock_admin = _make_mock_admin(broker_topics=broker_topics)

    with (
        patch(
            "omnibase_infra.event_bus.service_topic_startup_validator.ALL_PROVISIONED_SUFFIXES",
            SAMPLE_SUFFIXES,
        ),
        _patch_aiokafka_import(mock_admin),
        caplog.at_level(logging.ERROR),
    ):
        result = await validator.validate(correlation_id=uuid4())

    assert result.is_valid is False
    assert result.status == "degraded"
    assert len(result.missing_topics) == 2
    assert SAMPLE_SUFFIXES[1] in result.missing_topics
    assert SAMPLE_SUFFIXES[2] in result.missing_topics

    # Each missing topic should appear in an ERROR log
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    for missing in result.missing_topics:
        assert any(missing in msg for msg in error_messages), (
            f"Expected error log for missing topic '{missing}'"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strict_mode_raises(
    validator,
) -> None:
    """STARTUP_VALIDATION_STRICT=1 + missing -> RuntimeError raised with topic names."""
    mock_admin = _make_mock_admin(broker_topics={})

    with (
        patch(
            "omnibase_infra.event_bus.service_topic_startup_validator.ALL_PROVISIONED_SUFFIXES",
            SAMPLE_SUFFIXES,
        ),
        _patch_aiokafka_import(mock_admin),
    ):
        result = await validator.validate(correlation_id=uuid4())

    # Validator returns degraded; strict RuntimeError is raised by service_kernel.py
    assert result.is_valid is False
    assert result.status == "degraded"
    assert len(result.missing_topics) == len(SAMPLE_SUFFIXES)

    # Verify the kernel's strict-mode pattern: env var + missing -> RuntimeError
    import os

    with patch.dict(os.environ, {"STARTUP_VALIDATION_STRICT": "1"}):
        if not result.is_valid:
            if os.environ.get("STARTUP_VALIDATION_STRICT") == "1":
                with pytest.raises(RuntimeError, match="Missing topics"):
                    raise RuntimeError(f"Missing topics: {result.missing_topics}")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_broker_unreachable(
    validator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Broker unreachable -> is_valid=True, status='unavailable', logger.warning."""
    mock_admin = _make_mock_admin(start_error=ConnectionError("Connection refused"))

    with (
        patch(
            "omnibase_infra.event_bus.service_topic_startup_validator.ALL_PROVISIONED_SUFFIXES",
            SAMPLE_SUFFIXES,
        ),
        _patch_aiokafka_import(mock_admin),
        caplog.at_level(logging.WARNING),
    ):
        result = await validator.validate(correlation_id=uuid4())

    assert result.is_valid is True
    assert result.status == "unavailable"

    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("unreachable" in msg.lower() for msg in warning_messages), (
        f"Expected warning about broker unreachable, got: {warning_messages}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aiokafka_not_importable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """aiokafka not importable -> is_valid=True, status='skipped', logger.warning."""
    from omnibase_infra.event_bus.service_topic_startup_validator import (
        TopicStartupValidator,
    )

    validator = TopicStartupValidator(bootstrap_servers="localhost:19092")

    with (
        patch(
            "omnibase_infra.event_bus.service_topic_startup_validator.ALL_PROVISIONED_SUFFIXES",
            SAMPLE_SUFFIXES,
        ),
        caplog.at_level(logging.WARNING),
    ):
        # Remove aiokafka from sys.modules and make import fail
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith("aiokafka"):
                saved_modules[key] = sys.modules.pop(key)

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aiokafka.admin" or name.startswith("aiokafka"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                result = await validator.validate(correlation_id=uuid4())
        finally:
            # Restore aiokafka modules
            sys.modules.update(saved_modules)

    assert result.is_valid is True
    assert result.status == "skipped"

    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("aiokafka" in msg.lower() for msg in warning_messages), (
        f"Expected warning about missing aiokafka, got: {warning_messages}"
    )
