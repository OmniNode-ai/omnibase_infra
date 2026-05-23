# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test-only fixture handler for RuntimeLocal --input flag wiring.

This module is NOT production code. It exists solely to exercise the JSON
payload → model validation → handler invocation pipeline introduced by
OMN-8938 (`onex node <name> --input <path>`). Production handlers MUST NOT
import from or depend on this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ModelProofNoopRequest(BaseModel):
    """Test-only input model for RuntimeLocal --input flag wiring. Not a production type."""

    name: str = ""
    count: int = 0


class ModelProofNoopResult(BaseModel):
    """Test-only output model paired with ModelProofNoopRequest."""

    status: str = "success"
    echoed_name: str
    echoed_count: int


class HandlerProofNoop:
    """Minimal handler used by test_runtime_local_input_flag to verify payload wiring.

    On invocation, writes the received request to ``state_root / "echo.json"``
    and returns a paired result model.
    """

    def __init__(self, state_root: Path = Path(".onex_state")) -> None:
        self._state_root = state_root

    def handle(self, request: ModelProofNoopRequest) -> ModelProofNoopResult:
        state_root = self._state_root
        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / "echo.json").write_text(
            json.dumps({"name": request.name, "count": request.count}),
            encoding="utf-8",
        )
        return ModelProofNoopResult(
            status="success",
            echoed_name=request.name,
            echoed_count=request.count,
        )
