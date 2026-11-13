# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex ledger query --stdin`` with no lane declaration, end to end (OMN-24202).

The command, the lane selection and the declaration reader run together with nothing
replaced: the workspace is an empty directory, so no lane declaration can be found.
Before the change the command printed only ``Work-ledger read refused
(DelegateLaneSelectionError)`` and a lane that read it concluded that no working read
existed, when one setting (the workspace config root) was missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_ledger import ledger_group

pytestmark = pytest.mark.integration


def test_read_without_a_lane_declaration_names_the_missing_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path))
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    request = json.dumps({"correlation_id": str(uuid4()), "query": "rulings"})
    result = CliRunner().invoke(ledger_group, ["query", "--stdin"], input=request)
    assert result.exit_code == 1
    assert not result.stdout
    assert "refused (DelegateLaneSelectionError):" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr
    assert "ci_bus_lanes.yaml" in result.stderr
