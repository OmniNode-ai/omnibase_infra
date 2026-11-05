# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the installed teardown adapter through its composition root."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]


def test_teardown_plan_report_uses_a_document_schema(tmp_path: Path) -> None:
    # The CLI imports its real handler and models. FakeHost supplies command
    # responses only; no Docker daemon, broker or shared runtime is contacted.
    program = """
import json
import sys
from pathlib import Path
from tests.unit.scripts.test_prepr_teardown_slot_omn18896 import FakeHost, SLOT1, teardown

report = Path(sys.argv[1])
code = teardown.run_teardown(
    SLOT1,
    runner=FakeHost(),
    staging_root=report.parent / "slot-1",
    base_env={"POSTGRES_USER": "postgres"},
    plan_only=True,
    report_path=report,
)
raise SystemExit(code)
"""
    report_path = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, "-c", program, str(report_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads(report_path.read_text())
    assert report["schema"] == "urn:onex:prepr:slot-teardown:v1"
    assert report["verdict"] == "PLAN"
    assert report["errors"] == []
    assert report["slot"] == 1
    assert report["planned"]["containers"]
