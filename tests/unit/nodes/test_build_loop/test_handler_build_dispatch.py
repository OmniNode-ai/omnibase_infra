# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerBuildDispatch — delegation payloads and filesystem fallback.

Related:
    - OMN-7381: Wire handler_build_dispatch to delegation orchestrator
    - OMN-7318: node_build_dispatch_effect
"""

from __future__ import annotations

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
# Delegation payload path (primary — orchestrator publishes)
# ------------------------------------------------------------------


@pytest.mark.unit
class TestDelegationPayloads:
    """Tests for the primary delegation payload path."""

    @pytest.mark.asyncio
    async def test_builds_delegation_payload(self) -> None:
        handler = HandlerBuildDispatch()

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
        )

        assert result.total_dispatched == 1
        assert result.total_failed == 0
        assert len(result.delegation_payloads) == 1

        dp = result.delegation_payloads[0]
        assert dp.event_type == _DELEGATION_EVENT_TYPE
        assert dp.topic == TOPIC_DELEGATION_REQUEST
        assert "OMN-1234" in dp.payload["prompt"]

    @pytest.mark.asyncio
    async def test_builds_multiple_payloads(self) -> None:
        handler = HandlerBuildDispatch()

        targets = (
            _target("OMN-1001", "First"),
            _target("OMN-1002", "Second"),
            _target("OMN-1003", "Third"),
        )
        result = await handler.handle(correlation_id=uuid4(), targets=targets)

        assert result.total_dispatched == 3
        assert result.total_failed == 0
        assert len(result.delegation_payloads) == 3

    @pytest.mark.asyncio
    async def test_payload_shape(self) -> None:
        """Ensure the delegation payload matches ModelDelegationRequest fields."""
        handler = HandlerBuildDispatch()
        cid = uuid4()

        result = await handler.handle(correlation_id=cid, targets=(_target(),))

        dp = result.delegation_payloads[0]
        assert dp.payload["task_type"] == "research"
        assert dp.payload["correlation_id"] == str(cid)
        assert dp.payload["max_tokens"] == 4096
        assert "emitted_at" in dp.payload
        assert dp.correlation_id == str(cid)


# ------------------------------------------------------------------
# Dry-run
# ------------------------------------------------------------------


@pytest.mark.unit
class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_skips_payload_build(self) -> None:
        handler = HandlerBuildDispatch()

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
            dry_run=True,
        )

        assert result.total_dispatched == 1
        assert len(result.delegation_payloads) == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_fallback(self) -> None:
        handler = HandlerBuildDispatch()

        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
            dry_run=True,
            use_filesystem_fallback=True,
        )

        assert result.total_dispatched == 1
        assert len(result.delegation_payloads) == 0


# ------------------------------------------------------------------
# Filesystem fallback
# ------------------------------------------------------------------


@pytest.mark.unit
class TestFilesystemFallback:
    @pytest.mark.asyncio
    async def test_writes_manifest_when_fallback(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pathlib

        state_dir = pathlib.Path(str(tmp_path))
        monkeypatch.setenv("ONEX_STATE_DIR", str(state_dir))

        handler = HandlerBuildDispatch()
        result = await handler.handle(
            correlation_id=uuid4(),
            targets=(_target(),),
            use_filesystem_fallback=True,
        )

        assert result.total_dispatched == 1
        assert len(result.delegation_payloads) == 0
        manifest = state_dir / "autopilot" / "dispatch" / "OMN-1234.json"
        assert manifest.exists()

    @pytest.mark.asyncio
    async def test_raises_without_state_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ONEX_STATE_DIR", raising=False)
        handler = HandlerBuildDispatch()

        with pytest.raises(RuntimeError, match="ONEX_STATE_DIR"):
            await handler.handle(
                correlation_id=uuid4(),
                targets=(_target(),),
                use_filesystem_fallback=True,
            )


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    @pytest.mark.asyncio
    async def test_duplicate_ticket_ids_rejected(self) -> None:
        handler = HandlerBuildDispatch()

        with pytest.raises(ValueError, match="Duplicate"):
            await handler.handle(
                correlation_id=uuid4(),
                targets=(_target("OMN-1001"), _target("OMN-1001")),
            )
