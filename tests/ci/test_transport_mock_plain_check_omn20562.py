# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""transport-mock-lint is a plain blocking check with no exception list (OMN-20562).

The OMN-13026 baseline had drained to ``{}`` and six inline suppressions were the
only exceptions left. OMN-20562 gave those six mocks a real ``spec=`` and deleted the
baseline. These tests keep both gone: no baseline file, no ``--baseline`` argument in
the hook or the CI step, no inline suppression marker anywhere under ``tests/``, and
a planted bare transport mock still fails the lint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.validators import transport_mock_lint

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
BASELINE = REPO_ROOT / "config" / "validation" / "transport_mock_baseline.yaml"
# Spelled in two halves so this file does not itself carry the marker it forbids.
SUPPRESSION_MARKER = "transport-mock" + "-ok"


def test_baseline_file_is_gone() -> None:
    assert not BASELINE.exists()


def test_precommit_hook_has_no_baseline_argument() -> None:
    repos = yaml.safe_load(PRECOMMIT_CONFIG.read_text())["repos"]
    hooks = [hook for repo in repos for hook in repo.get("hooks", [])]
    (hook,) = [h for h in hooks if h.get("id") == "transport-mock-lint"]
    assert not any("baseline" in str(arg) for arg in hook.get("args", []))


def test_ci_step_has_no_baseline_argument() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    runs = [
        str(step.get("run", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "transport_mock_lint" in str(step.get("run", ""))
    ]
    assert len(runs) == 1
    assert "--baseline" not in runs[0]
    assert runs[0].rstrip().endswith("tests/")


def test_no_inline_suppression_under_tests() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "tests").rglob("*.py")
        if SUPPRESSION_MARKER in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders == [], (
        "give the mock a spec= or spec_set= of the real type instead of "
        f"suppressing the lint: {offenders}"
    )


def test_planted_bare_transport_mock_fails(tmp_path: Path) -> None:
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "from unittest.mock import AsyncMock\n\n\n"
        "def test_planted() -> None:\n"
        "    bus = AsyncMock()\n"
        "    assert bus\n"
    )
    assert transport_mock_lint.main([str(planted)]) == 1


def test_spec_mock_passes(tmp_path: Path) -> None:
    clean = tmp_path / "test_clean.py"
    clean.write_text(
        "from unittest.mock import AsyncMock\n\n\n"
        "class ProtocolBus:\n"
        "    async def publish(self) -> None: ...\n\n\n"
        "def test_clean() -> None:\n"
        "    bus = AsyncMock(spec=ProtocolBus)\n"
        "    assert bus\n"
    )
    assert transport_mock_lint.main([str(clean)]) == 0
