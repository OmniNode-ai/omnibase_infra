# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dispatch-parity workflow names the committed selection oracle (OMN-18648)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = "tests/fixtures/dispatch_parity/selection-oracle-v2.json"
WORKFLOW = ROOT / ".github/workflows/dispatch-parity-gate.yml"


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/required_status_checks.json"
)
def test_workflow_oracle_path_exists_and_recorded_contexts_are_strings(
    recorded_response: dict[str, Any],
) -> None:
    assert ORACLE in WORKFLOW.read_text()
    assert (ROOT / ORACLE).is_file()
    contexts = recorded_response["response"]["contexts"]
    assert contexts
    assert all(isinstance(c, str) for c in contexts)
