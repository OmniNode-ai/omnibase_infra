# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parity tests for the typed-request handle() of node_session_state_effect.

The five handlers took scalar or model arguments next to a correlation id; each
now takes one request model that carries both. These tests drive
``HandlerRunContextRead`` (the node's representative handler) through the new
call shape. Against the merge-base handler the call raises ``TypeError``, which
each test turns into its own assertion failure.

The scenarios double as the candidate inputs the adequacy receipt under
``scripts/ci/adequacy_receipts/`` was recorded over.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_session_state_effect.handlers.handler_run_context_read import (
    HandlerRunContextRead,
)
from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionStateResult,
)

pytestmark = pytest.mark.unit


class ModelRunReadScenario(BaseModel):
    """One run-context read plus the file the state directory holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    run_id: str
    file_state: Literal["absent", "valid", "corrupt"]

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[ModelRunReadScenario, ...] = (
    ModelRunReadScenario(seed=1, run_id="run-a", file_state="absent"),
    ModelRunReadScenario(seed=2, run_id="run-b", file_state="valid"),
    ModelRunReadScenario(seed=3, run_id="run-c", file_state="corrupt"),
    ModelRunReadScenario(seed=4, run_id="../escape", file_state="absent"),
)


def _state_dir_for(scenario: ModelRunReadScenario, root: Path) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True)
    run_file = runs / f"{scenario.run_id}.json"
    if scenario.file_state == "valid":
        run_file.write_text(
            json.dumps(ModelRunContext(run_id=scenario.run_id).model_dump(mode="json")),
            encoding="utf-8",
        )
    elif scenario.file_state == "corrupt":
        run_file.write_text("{not json", encoding="utf-8")
    return root


async def run_scenario(
    scenario: ModelRunReadScenario,
) -> tuple[ModelRunContext | None, ModelSessionStateResult]:
    """Call the handler the way the runtime does: one request argument."""
    try:
        from omnibase_infra.nodes.node_session_state_effect.models import (
            ModelRunContextReadRequest,
        )
    except ImportError as exc:
        pytest.fail(f"the typed read request model is missing: {exc}")
    request = ModelRunContextReadRequest(
        run_id=scenario.run_id, correlation_id=scenario.correlation_id
    )
    with tempfile.TemporaryDirectory(prefix="run-read-") as tmp:
        handler = HandlerRunContextRead(_state_dir_for(scenario, Path(tmp)))
        try:
            return await handler.handle(request)
        except TypeError as exc:
            pytest.fail(f"handle() does not accept the single request model: {exc}")


class TestRunContextReadHandleTypedRequest:
    """HandlerRunContextRead.handle takes one request model."""

    @pytest.mark.asyncio
    async def test_absent_run_succeeds_with_no_context(self) -> None:
        ctx, result = await run_scenario(SCENARIOS[0])
        assert ctx is None
        assert result.success is True
        assert result.correlation_id == SCENARIOS[0].correlation_id

    @pytest.mark.asyncio
    async def test_valid_run_returns_the_context(self) -> None:
        ctx, result = await run_scenario(SCENARIOS[1])
        assert ctx is not None
        assert ctx.run_id == "run-b"
        assert result.success is True
        assert result.correlation_id == SCENARIOS[1].correlation_id

    @pytest.mark.asyncio
    async def test_corrupt_run_reports_a_parse_error(self) -> None:
        ctx, result = await run_scenario(SCENARIOS[2])
        assert ctx is None
        assert result.success is False
        assert result.error_code == "RUN_CONTEXT_PARSE_ERROR"

    @pytest.mark.asyncio
    async def test_unsafe_run_id_is_rejected(self) -> None:
        ctx, result = await run_scenario(SCENARIOS[3])
        assert ctx is None
        assert result.success is False
        assert result.error_code == "RUN_CONTEXT_INVALID_ID"
        assert result.correlation_id == SCENARIOS[3].correlation_id
