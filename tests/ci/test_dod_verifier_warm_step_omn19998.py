# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The CI test splits install the pinned DoD verifier before pytest times anything (OMN-19998).

Nine open omnibase_infra pull requests went red on 2026-11-03 in whichever
``Tests (Split n/m)`` job held ``tests/ci/test_ticket_contract_prepush_omn20870.py``
and ``tests/ci/test_ticket_contract_prerequisite_parity_omn21642.py``. Those
tests call ``install_verifier``, which on an empty runner cache does a
``git fetch`` of the omnimarket tag and a ``uv sync`` into a Python 3.13 venv.
The splits run pytest with ``--timeout=60 --timeout-method=signal``: the install
outlasted it, the signal killed the install's parent, and the orphaned install
kept writing under the cache entry that the next test's ``rm -rf`` removed, so
the remaining tests died with ``FileNotFoundError`` on ``venv/bin/python``.

The install belongs before pytest, in a step with its own ceiling. The tests then
find the installed marker and return at once. ``test-parallel`` runs that step
ahead of both of its pytest steps; these tests run the step's own Python body.
"""

from __future__ import annotations

import re
import runpy
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import check_deploy_scope_dod as hook

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_WARM_STEP = "Install the pinned DoD verifier before pytest times it (OMN-19998)"
_HEREDOC = re.compile(r"python - <<'PY'\n(?P<body>.*?)\n\s*PY\s*$", re.DOTALL)


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["test-parallel"]["steps"]
    return steps


def _warm_step() -> dict[str, Any]:
    named = [s for s in _steps() if s.get("name") == _WARM_STEP]
    assert len(named) == 1, "exactly one step installs the verifier ahead of pytest"
    return named[0]


def _run_step_body(tmp_path: Path) -> None:
    match = _HEREDOC.search(_warm_step()["run"])
    assert match, "the step runs an inline Python heredoc"
    script = tmp_path / "warm_step.py"
    script.write_text(textwrap.dedent(match["body"]), encoding="utf-8")
    runpy.run_path(str(script), run_name="__main__")


def test_the_step_installs_the_tag_the_caller_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed: list[str] = []

    def fake_install(tag: str) -> Path:
        installed.append(tag)
        return Path("/cache") / tag / "venv" / "bin" / "python"

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(hook, "install_verifier", fake_install)

    _run_step_body(tmp_path)

    assert installed == [hook.read_verifier_tag(REPO_ROOT)]
    assert installed[0] in capsys.readouterr().out


def test_the_step_fails_when_the_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_install(tag: str) -> Path:
        raise hook.DeployScopeHookError(f"installing omnimarket tag {tag} failed")

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(hook, "install_verifier", failing_install)

    with pytest.raises(hook.DeployScopeHookError, match="failed"):
        _run_step_body(tmp_path)


def test_the_test_splits_install_the_verifier_before_either_pytest_step() -> None:
    steps = _steps()
    warm = [i for i, s in enumerate(steps) if s.get("name") == _WARM_STEP]
    pytest_steps = [
        i
        for i, s in enumerate(steps)
        if str(s.get("name", "")).startswith("Run pytest")
    ]

    assert len(warm) == 1
    assert len(pytest_steps) == 2, "smart selection and full suite"
    assert warm[0] < min(pytest_steps)


def test_the_step_runs_no_pytest_and_has_its_own_ceiling() -> None:
    step = _warm_step()

    assert "pytest" not in step["run"]
    assert 0 < int(step["timeout-minutes"]) <= 15
