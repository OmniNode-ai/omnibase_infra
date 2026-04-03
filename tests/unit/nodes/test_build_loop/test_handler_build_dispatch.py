# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerBuildDispatch — Kafka delegation and filesystem fallback.

Related:
    - OMN-7381: Wire handler_build_dispatch to delegation orchestrator
    - OMN-7318: node_build_dispatch_effect
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_buildability import EnumBuildability
from omnibase_infra.event_bus.topic_constants import TOPIC_DELEGATION_REQUEST
from omnibase_infra.nodes.node_build_dispatch_effect.handlers.handler_build_dispatch import (
    _DELEGATION_EVENT_TYPE,
    HandlerBuildDispatch,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)


def _target(ticket_id: str = "OMN-1234", title: str = "Fix widget") -> ModelBuildTarget:
    return ModelBuildTarget(
        ticket_id=ticket_id,
        title=title,
        buildability=EnumBuildability.AUTO_BUILDABLE,
    )


# ------------------------------------------------------------------
# Kafka publisher path
# ------------------------------------------------------------------


@pytest.mark.unit
class TestKafkaDispatch:
    """Tests for the primary Kafka delegation path."""

    @pytest.mark.asyncio
    async def test_publishes_delegation_request(self) -> None:
        publisher = AsyncMock(return_value=True)
        handler = HandlerBuildDispatch(publisher=publisher)

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
        )

        assert result.total_dispatched == 1
        assert result.total_failed == 0
        publisher.assert_called_once()

        call_kwargs = publisher.call_args.kwargs
        assert call_kwargs["event_type"] == _DELEGATION_EVENT_TYPE
        assert call_kwargs["topic"] == TOPIC_DELEGATION_REQUEST
        assert "OMN-1234" in call_kwargs["payload"]["prompt"]

    @pytest.mark.asyncio
    async def test_publishes_multiple_targets(self) -> None:
        publisher = AsyncMock(return_value=True)
        handler = HandlerBuildDispatch(publisher=publisher)

        targets = (
            _target("OMN-1001", "First"),
            _target("OMN-1002", "Second"),
            _target("OMN-1003", "Third"),
        )
        result = await handler.handle(correlation_id=uuid4(), targets=targets)

        assert result.total_dispatched == 3
        assert result.total_failed == 0
        assert publisher.call_count == 3

    @pytest.mark.asyncio
    async def test_publish_failure_records_error(self) -> None:
        publisher = AsyncMock(return_value=False)
        handler = HandlerBuildDispatch(publisher=publisher)

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
        )

        assert result.total_dispatched == 0
        assert result.total_failed == 1
        assert result.outcomes[0].dispatched is False
        assert "not delivered" in (result.outcomes[0].error or "")

    @pytest.mark.asyncio
    async def test_publish_exception_records_error(self) -> None:
        publisher = AsyncMock(side_effect=RuntimeError("Kafka down"))
        handler = HandlerBuildDispatch(publisher=publisher)

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
        )

        assert result.total_dispatched == 0
        assert result.total_failed == 1
        assert "Kafka down" in (result.outcomes[0].error or "")

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self) -> None:
        """One publish failure should not block subsequent tickets."""
        call_count = 0

        async def flaky_publisher(**kwargs: object) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("transient")
            return True

        handler = HandlerBuildDispatch(publisher=flaky_publisher)
        targets = (
            _target("OMN-1001", "A"),
            _target("OMN-1002", "B"),
            _target("OMN-1003", "C"),
        )
        result = await handler.handle(correlation_id=uuid4(), targets=targets)

        assert result.total_dispatched == 2
        assert result.total_failed == 1

    @pytest.mark.asyncio
    async def test_payload_shape(self) -> None:
        """Ensure the published payload matches ModelDelegationRequest fields."""
        publisher = AsyncMock(return_value=True)
        handler = HandlerBuildDispatch(publisher=publisher)
        cid = uuid4()

        await handler.handle(correlation_id=cid, targets=(_target(),))

        payload = publisher.call_args.kwargs["payload"]
        assert payload["task_type"] == "research"
        assert payload["correlation_id"] == str(cid)
        assert payload["max_tokens"] == 4096
        assert "emitted_at" in payload


# ------------------------------------------------------------------
# Dry-run
# ------------------------------------------------------------------


@pytest.mark.unit
class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_skips_publish(self) -> None:
        publisher = AsyncMock(return_value=True)
        handler = HandlerBuildDispatch(publisher=publisher)

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
            dry_run=True,
        )

        assert result.total_dispatched == 1
        publisher.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_no_publisher(self) -> None:
        handler = HandlerBuildDispatch()

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
            dry_run=True,
        )

        assert result.total_dispatched == 1


# ------------------------------------------------------------------
# Filesystem fallback
# ------------------------------------------------------------------


@pytest.mark.unit
class TestFilesystemFallback:
    @pytest.mark.asyncio
    async def test_writes_manifest_when_no_publisher(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pathlib

        state_dir = pathlib.Path(str(tmp_path))
        monkeypatch.setenv("ONEX_STATE_DIR", str(state_dir))

        handler = HandlerBuildDispatch()  # no publisher
        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
        )

        assert result.total_dispatched == 1
        manifest = state_dir / "autopilot" / "dispatch" / "OMN-1234.json"
        assert manifest.exists()

    @pytest.mark.asyncio
    async def test_raises_without_state_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ONEX_STATE_DIR", raising=False)
        handler = HandlerBuildDispatch()  # no publisher

        with pytest.raises(RuntimeError, match="ONEX_STATE_DIR"):
            await handler.handle(
                correlation_id=uuid4(),
                targets=(_target(),),
            )


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    @pytest.mark.asyncio
    async def test_duplicate_ticket_ids_rejected(self) -> None:
        handler = HandlerBuildDispatch(publisher=AsyncMock(return_value=True))

        with pytest.raises(ValueError, match="Duplicate"):
            await handler.handle(
                correlation_id=uuid4(),
                targets=(_target("OMN-1001"), _target("OMN-1001")),
            )
