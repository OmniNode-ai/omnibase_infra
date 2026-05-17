# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage marker for delegation quality contract dispatch fields."""

from __future__ import annotations

from pathlib import Path


def test_quality_contract_dispatch_fields_are_declared() -> None:
    source = Path(
        "src/omnibase_infra/runtime/service_delegation_dispatch_port.py"
    ).read_text()

    assert "quality_contract_mode" in source
    assert "acceptance_criteria" in source
