# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the nightly selector with pytest, without an LLM or e2e stack."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
LIVE_SUITE = (
    "tests/integration/adapters/llm/"
    "test_adapter_documentation_generation_integration.py"
)


def _selection() -> list[str]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/nightly-integration.yml").read_text()
    )
    steps = workflow["jobs"]["integration-tests"]["steps"]
    script = next(s["run"] for s in steps if s.get("name") == "Run integration tests")
    command = script[script.index("uv run pytest tests/integration/") :]
    argv = shlex.split(command.replace("\\\n", " "))
    # Replay the shipped marker expression and every deselection, without the
    # runner's timeout and artifact-output options.
    selection = []
    for index, arg in enumerate(argv):
        if arg in {"-m", "--deselect"}:
            selection.extend((arg, argv[index + 1]))
    return selection


@pytest.mark.parametrize("nightly", [True, False], ids=["nightly", "explicit-live"])
def test_pytest_selects_only_the_dependencies_this_stack_owns(
    tmp_path: Path, nightly: bool
) -> None:
    """The nightly keeps ordinary slow tests; explicit live selection fails closed."""
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    integration\n    bastion\n    slow\n"
    )
    live = tmp_path / LIVE_SUITE
    live.parent.mkdir(parents=True)
    live.write_text(
        "import pytest\n"
        "pytestmark = [pytest.mark.integration, pytest.mark.slow]\n"
        "@pytest.fixture(autouse=True)\n"
        "def endpoint():\n"
        "    pytest.fail('Set LLM_DEEPSEEK_R1_URL to run the live adapter suite')\n"
        "def test_live_adapter():\n"
        "    pass\n"
    )
    (tmp_path / "tests/integration/test_other.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.integration\n"
        "@pytest.mark.slow\n"
        "def test_other_slow_integration():\n"
        "    pass\n"
        "@pytest.mark.bastion\n"
        "def test_bastion():\n"
        "    pytest.fail('bastion has its own workflow')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/" if nightly else LIVE_SUITE,
            *(_selection() if nightly else []),
            "-q",
            "--tb=short",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    if nightly:
        assert result.returncode == 0, output
        assert "1 passed, 2 deselected" in output, output
    else:
        assert result.returncode == 1, output
        assert "Set LLM_DEEPSEEK_R1_URL" in output, output
