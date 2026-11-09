# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed controls around the prod-promotion grant registry (OMN-23362).

The registry is committed with no entries. These controls pin what must stay true
around it: a stability refresh over the committed registry needs no
acknowledgement, a dev refresh needs no registry and no grant, a registry that is
absent, blank or unparseable is still UNREADABLE, and the prod lane is not passed
by the registry being present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "preflight_lane_deploy_attribution.py"


def _load_module() -> Any:
    mod_name = "preflight_lane_deploy_attribution"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()

_REGISTRY = _REPO_ROOT / _mod.GRANTS_REPO_RELPATH
_STABILITY_LANE = next(iter(_mod.GRANT_INTERLOCK_LANES))
_REASON = "OMN-23362 stability refresh to latest dev"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ONEX_DEPLOY_") and key != "OMNI_HOME"
    }
    env["ONEX_DEPLOY_REASON"] = _REASON
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


def _record(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return payload


def _args(lane: str, grants_file: Path | None) -> list[str]:
    args = [
        "--lane",
        lane,
        "--source",
        "pytest",
        "--now",
        "2026-11-09T12:00:00Z",
        "--check-only",
        "--json",
    ]
    if grants_file is not None:
        args += ["--grants-file", str(grants_file)]
    return args


@pytest.mark.unit
def test_grant_registry_present_stability_refresh_needs_no_acknowledgement() -> None:
    """AC2 stand-in: the stability preflight over the committed registry passes bare."""
    result = _run(_args(_STABILITY_LANE, _REGISTRY))
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(result)
    assert record["grant_guard"]["verdict"] == "CLEAR"
    assert record["grant_guard"]["live_grants"] == []
    assert record["grant_guard"]["acknowledged"] is False
    assert record["result"] == "ALLOW"
    assert "REFUSED" not in result.stdout + result.stderr


@pytest.mark.unit
def test_grant_registry_present_dev_needs_no_registry_and_no_grant(
    tmp_path: Path,
) -> None:
    """A dev refresh never reads the registry: an unreachable one does not refuse it."""
    result = _run(
        [
            "--lane",
            "dev",
            "--source",
            "pytest",
            "--grants-repo",
            str(tmp_path / "no-such-clone"),
            "--check-only",
            "--json",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _record(result)["grant_guard"]["verdict"] == "NOT_APPLICABLE"


@pytest.mark.unit
def test_grant_registry_present_absent_file_still_fails_closed(tmp_path: Path) -> None:
    """Control: the fix is the file, not a looser preflight (the old failure is kept)."""
    result = _run(_args(_STABILITY_LANE, tmp_path / "missing.yaml"))
    assert result.returncode == 1
    assert _record(result)["grant_guard"]["verdict"] == "UNREADABLE"


@pytest.mark.unit
@pytest.mark.parametrize("content", ["", "entries:\n", "entries: [[[", "{}\n"])
def test_grant_registry_present_blank_or_broken_registry_fails_closed(
    tmp_path: Path, content: str
) -> None:
    """Control: only a real ``entries`` list reads as zero live grants."""
    registry = tmp_path / "grants.yaml"
    registry.write_text(content, encoding="utf-8")
    result = _run(_args(_STABILITY_LANE, registry))
    assert result.returncode == 1
    assert _record(result)["grant_guard"]["verdict"] == "UNREADABLE"


@pytest.mark.unit
def test_grant_registry_present_prod_never_passes_on_registry_presence() -> None:
    """Prod is refused without a stated reason whatever the registry says, and the
    interlock lane set did not grow to cover it by accident."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ONEX_DEPLOY_") and key != "OMNI_HOME"
    }
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *_args("prod", _REGISTRY)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert _record(result)["result"] == "REFUSE"
    assert frozenset({_STABILITY_LANE}) == _mod.GRANT_INTERLOCK_LANES
    assert "prod" not in _mod.GRANT_INTERLOCK_LANES
    assert "prod" in _mod.GOVERNED_LANES
