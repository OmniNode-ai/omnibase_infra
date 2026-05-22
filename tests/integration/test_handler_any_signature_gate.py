# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Root integration coverage for the handler Any signature gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators.handler_any_signature import main


@pytest.mark.integration
def test_handler_any_signature_gate_accepts_typed_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "handler_fixture.py"
    fixture.write_text(
        "class HandlerFixture:\n"
        "    def handle(self, payload: str) -> str:\n"
        "        return payload\n",
        encoding="utf-8",
    )

    assert main(["--max-violations", "0", str(tmp_path)]) == 0
