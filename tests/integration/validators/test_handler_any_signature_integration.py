# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-10820 handler Any signature gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators.handler_any_signature import main


@pytest.mark.integration
def test_handler_any_signature_gate_runs_against_fixture_tree(tmp_path: Path) -> None:
    handler = tmp_path / "handler_fixture.py"
    handler.write_text(
        """
class HandlerFixture:
    def handle(self, payload: str) -> str:
        return payload
""",
        encoding="utf-8",
    )

    assert main(["--max-violations", "0", str(tmp_path)]) == 0
