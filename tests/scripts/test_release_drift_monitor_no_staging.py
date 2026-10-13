# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20112: the release/deploy drift monitor reads no staging-plane result.

Operator ruling 2026-09-29: nothing outside the cloud infra repository may
require or read staging. The monitor used to carry one deploy target, the staging
deploy workflow of that repository, so a red or unreachable staging deploy turned
the scheduled run red. The monitor's remaining probes (PyPI, tags, main/dev,
release and rebuild workflow health) cover this organisation's published
packages and stay.

The replay hands the monitor a red staging deploy verdict for every probe that
falls outside the repositories it monitors, and compares its report with the
report produced when those probes answer nothing. Identical reports, and no
request outside the monitored repositories, prove the monitor does not consult
the staging plane.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load_shell() -> Any:
    mod_name = "release_drift_monitor_under_test_no_staging"
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            mod_name, _SCRIPTS / "release_drift_monitor.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPTS))
    return mod


_shell = _load_shell()

_RED_STAGING_RUN = {
    "name": "deploy-staging.yml",
    "conclusion": "failure",
    "head_sha": "0" * 40,
    "created_at": "2026-09-28T18:37:00Z",
}


def _monitored_prefixes() -> tuple[str, ...]:
    return tuple(f"repos/{_shell._ORG}/{cfg.repo}/" for cfg in _shell.DEFAULT_REPOS)


def _outside_monitored(paths: list[str]) -> list[str]:
    prefixes = _monitored_prefixes()
    return [p for p in paths if not p.startswith(prefixes)]


def _run_monitor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    outside_answer: Any,
) -> tuple[str, list[str]]:
    requested: list[str] = []

    def fake_gh_api(path: str, jq: str | None = None) -> tuple[Any, str | None]:
        requested.append(path)
        if _outside_monitored([path]):
            return outside_answer, None
        return None, "stubbed: no data"

    monkeypatch.setattr(_shell, "_gh_api", fake_gh_api)
    monkeypatch.setattr(
        _shell, "_pypi_latest", lambda package: (None, "stubbed: unreachable")
    )
    _shell.main(["--json", "--no-emit", "--warn-only"])
    report = json.loads(capsys.readouterr().out)
    report.pop("generated_at", None)
    return json.dumps(report, sort_keys=True), requested


def test_no_deploy_target_is_configured() -> None:
    assert _shell.DEPLOY_TARGETS == ()


def test_report_is_independent_of_a_red_staging_deploy_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    red_report, red_requests = _run_monitor(
        monkeypatch, capsys, outside_answer=json.dumps(_RED_STAGING_RUN)
    )
    quiet_report, _ = _run_monitor(monkeypatch, capsys, outside_answer=None)

    assert red_report == quiet_report
    assert _outside_monitored(red_requests) == []
    assert red_requests, "the monitor made no probe at all"


def test_positive_control_a_probe_outside_the_monitored_repos_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorder in ``_run_monitor`` sees an outside probe when one is made."""
    seen: list[str] = []

    def fake_gh_api(path: str, jq: str | None = None) -> tuple[Any, str | None]:
        seen.append(path)
        return None, "stubbed: no data"

    monkeypatch.setattr(_shell, "_gh_api", fake_gh_api)
    cfg = _shell.DeployTargetConfig(
        repo="some_other_repo",
        target="staging",
        branch="dev",
        workflow="deploy-staging.yml",
        deploy_paths=("k8s/staging",),
    )
    _shell.gather_deploy_facts(cfg)
    assert _outside_monitored(seen) != []
