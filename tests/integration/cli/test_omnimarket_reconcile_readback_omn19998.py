# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise the guard with real Git HEADs and installed distribution metadata."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.cli.omnimarket_drift_guard import (
    OmnimarketDriftError,
    check_omnimarket_drift,
)
from omnibase_infra.cli.workspace_reconcile import ModelReconcileOutcome

pytestmark = pytest.mark.integration
_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_omnimarket_reconcile_readback_effect/contract.yaml"
)


def _git(clone: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(clone), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    clone = tmp_path / "omnimarket"
    clone.mkdir()
    _git(clone, "init", "-q", "-b", "dev")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "config", "user.email", "test@example.com")
    (clone / "README.md").write_text("initial\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-qm", "initial")
    site = tmp_path / "site"
    dist = site / "omnimarket-0.0.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.0.0\n", encoding="utf-8"
    )
    direct_url = dist / "direct_url.json"
    _installed(direct_url, "f" * 40)
    monkeypatch.syspath_prepend(str(site))
    return clone, direct_url


def _installed(direct_url: Path, commit: str) -> None:
    direct_url.write_text(
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": commit}}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("after", ["match", "mismatch", "unreadable", "absent"])
def test_guard_rechecks_both_live_identities_after_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: str
) -> None:
    clone, direct_url = _workspace(tmp_path, monkeypatch)
    old_head = _git(clone, "rev-parse", "HEAD")
    calls = []
    refreshed = []

    def reconcile() -> ModelReconcileOutcome:
        calls.append(True)
        (clone / "README.md").write_text("advanced\n", encoding="utf-8")
        _git(clone, "add", "README.md")
        _git(clone, "commit", "-qm", "advanced")
        head = _git(clone, "rev-parse", "HEAD")
        refreshed.append(head)
        assert head != old_head
        _installed(direct_url, old_head if after == "unreadable" else head)
        if after == "mismatch":
            _installed(direct_url, "e" * 40)
        if after == "unreadable":
            (clone / ".git").rename(clone / "unavailable-git")
        if after == "absent":
            direct_url.unlink()
        return ModelReconcileOutcome(
            ok=True,
            run_id="local-readback",
            command="local-reconcile",
            detail="readback",
        )

    if after == "match":
        assert check_omnimarket_drift(str(tmp_path), reconcile=reconcile) is None
    else:
        with pytest.raises(OmnimarketDriftError, match="STILL drifted") as error:
            check_omnimarket_drift(str(tmp_path), reconcile=reconcile)
        expected = "UNREADABLE" if after == "unreadable" else refreshed[0][:12]
        assert f"HEAD {expected}. That is" in str(error.value)
    assert calls == [True]


async def test_contract_routes_to_handler_that_reads_real_metadata_and_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone, direct_url = _workspace(tmp_path, monkeypatch)
    head = _git(clone, "rev-parse", "HEAD")
    _installed(direct_url, head)
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    route = contract["handler_routing"]["handlers"][0]["handler"]
    handler_class = getattr(importlib.import_module(route["module"]), route["name"])
    model = contract["input_model"]
    request_class = getattr(importlib.import_module(model["module"]), model["name"])
    handler = handler_class(ModelONEXContainer())

    result = await handler.handle(request_class(workspace_root=str(tmp_path)))

    assert result.installed == result.canonical == head
    assert result.matches is True
    direct_url.unlink()
    (clone / ".git").rename(clone / "unavailable-git")
    unreadable = await handler.handle(request_class(workspace_root=str(tmp_path)))
    assert unreadable.installed == unreadable.canonical == ""
    assert unreadable.matches is False
