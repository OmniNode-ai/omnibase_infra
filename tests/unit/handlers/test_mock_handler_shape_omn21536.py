# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reshaped mock handlers pass K3; restoring local state fails it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "handler_path",
    [
        "registration_storage/handler_registration_storage_mock.py",
        "service_discovery/handler_service_discovery_mock.py",
    ],
)
def test_handler_shape_rejects_restored_state_mutation(
    handler_path: str, tmp_path: Path
) -> None:
    handlers = Path(__file__).resolve().parents[3] / "src/omnibase_infra/handlers"
    source = (handlers / handler_path).read_text()
    assert source.count("self._store.reset()") == 1
    target = tmp_path / "src/handlers" / handler_path
    target.parent.mkdir(parents=True)
    report = tmp_path / "report.json"

    for mutated in (False, True):
        target.write_text(
            source.replace("self._store.reset()", "self._records = {}")
            if mutated
            else source
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "omnibase_core.nodes.node_validation_report_write_effect.runtime_validator_run",
                "omnibase_core.validation.validator_handler_shape:ValidatorHandlerShape",
                str(target),
                "--report-json",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == int(mutated), result.stdout + result.stderr
        findings = json.loads(report.read_text())["findings"]
        assert (
            any(finding["rule_id"] == "mutating_public_method" for finding in findings)
            is mutated
        )
