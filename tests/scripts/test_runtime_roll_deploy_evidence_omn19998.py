# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC39 deploy evidence must reject broken behavior even with its message intact."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("swallow_failure", [False, True])
def test_deploy_evidence_exercises_the_roll_failure(
    tmp_path: Path, swallow_failure: bool
) -> None:
    contract = yaml.safe_load((ROOT / "contracts/OMN-19998.yaml").read_text())
    evidence = next(
        item
        for item in contract["dod_evidence"]
        if item["id"] == "dod-omn19998-ac39-runtime-roll-deploy-source"
    )
    # Reproduce the reviewed false positive: published source contains the
    # diagnostic only as a comment. No network or GitHub credentials are used.
    content = base64.b64encode(b"# Runtime roll stopped at\n").decode()
    gh = tmp_path / "gh"
    gh.write_text(f"#!{sys.executable}\nprint({content!r})\n")
    gh.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    if swallow_failure:
        # Mutate the actual effect at the executor's loader boundary. Keep
        # source text untouched: evidence must detect behavior, not a string.
        (tmp_path / "swallow_roll_failure.py").write_text(
            "import pytest\n"
            "@pytest.fixture(autouse=True)\n"
            "def swallow_roll_failure(monkeypatch):\n"
            "    from deploy_agent.executor import load_dev_boot_guard\n"
            "    guard = load_dev_boot_guard()\n"
            "    original = guard.run_compose_up_steps\n"
            "    def swallow(*args, **kwargs):\n"
            "        result, error = original(*args, **kwargs)\n"
            "        return result, None\n"
            "    monkeypatch.setattr(guard, 'run_compose_up_steps', swallow)\n"
        )
        env["PYTEST_PLUGINS"] = "swallow_roll_failure"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(tmp_path), str(ROOT / "src"), str(ROOT / "scripts/deploy-agent")]
        )
    result = subprocess.run(
        ["bash", "-c", evidence["checks"][0]["check_value"]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    output = result.stdout + result.stderr
    if swallow_failure:
        assert result.returncode != 0, (
            "deploy evidence accepted a swallowed roll failure"
        )
        assert "DID NOT RAISE" in output, output
    else:
        assert result.returncode == 0, output
