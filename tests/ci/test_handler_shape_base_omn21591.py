# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The handler ratchet compares a PR with its ancestor when dev moves ahead."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        env=scrub_git_location_env(os.environ),
    ).strip()


def _step() -> dict[str, Any]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return next(
        step
        for step in workflow["jobs"]["canonical-handler-shape-gate"]["steps"]
        if step.get("name", "").startswith("Canonical handler-shape ratchet")
    )


@pytest.mark.unit
@pytest.mark.parametrize("event_key", ["PR_BASE", "MQ_BASE", "PUSH_BEFORE"])
def test_later_base_retirement_is_not_growth(tmp_path: Path, event_key: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Regression")
    _git(repo, "config", "user.email", "regression@example.invalid")
    baseline = repo / "baseline.json"
    baseline.write_text('["existing"]')
    _git(repo, "add", "baseline.json")
    _git(repo, "commit", "-qm", "ancestor")
    ancestor = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "feature")
    (repo / "product").write_text("feature")
    _git(repo, "add", "product")
    _git(repo, "commit", "-qm", "feature")
    feature = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "dev", ancestor)
    baseline.write_text("[]")
    _git(repo, "add", "baseline.json")
    _git(repo, "commit", "-qm", "retire on dev")
    base_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/dev", base_tip)
    _git(repo, "checkout", "-q", "feature")

    # Capture the validator invocation, executing the workflow's real shell and Git.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "args.json"
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    uv.chmod(0o755)
    env = scrub_git_location_env(os.environ)
    for key in ("PR_BASE", "MQ_BASE", "PUSH_BEFORE"):
        env.pop(key, None)
    env.update({event_key: base_tip, "PATH": f"{bin_dir}:{env['PATH']}"})
    subprocess.run(["bash", "-c", _step()["run"]], cwd=repo, env=env, check=True)
    args = json.loads(capture.read_text())
    selected = args[args.index("--base-ref") + 1]
    assert selected == ancestor
    assert feature != ancestor != base_tip
    assert "--full" in args
    assert (
        args[args.index("--baseline") + 1]
        == "scripts/ci/canonical_handler_shape_baseline.py"
    )
    assert args[args.index("--receipts-dir") + 1] == "scripts/ci/adequacy_receipts"
    # The selected input preserves the ancestor's frozen set. A new head entry
    # still differs from it; the later retirement on dev does not enter the diff.
    frozen = set(json.loads(_git(repo, "show", f"{selected}:baseline.json")))
    head = set(json.loads(baseline.read_text()))
    assert head - frozen == set()
    assert (head | {"new_noncanonical"}) - frozen == {"new_noncanonical"}
    assert head - set(json.loads(_git(repo, "show", f"{base_tip}:baseline.json"))) == {
        "existing"
    }


@pytest.mark.unit
def test_unresolvable_base_stops_before_validator(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    env = {**scrub_git_location_env(os.environ), "PR_BASE": "missing-base"}
    result = subprocess.run(
        ["bash", "-c", _step()["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"missing-base" in result.stderr


@pytest.mark.unit
@pytest.mark.live_contact("tests/ci/recordings/handler_shape_base_omn21591.json")
def test_recorded_baseline_retirement_incident(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the actual divergent baseline inputs and pinned detector readback."""
    head = set(recorded_response["head_baseline"])
    ancestor = set(recorded_response["ancestor_baseline"])
    later_base = set(recorded_response["later_base_baseline"])
    assert head == ancestor
    assert head - later_base == {"omnibase_infra.nodes.node_registration_orchestrator"}
    red = recorded_response["red"]
    green = recorded_response["green"]
    assert red["exit_code"] == 1
    assert "Baseline GROWTH" in red["stderr"]
    assert green["exit_code"] == 0
    assert "new=0 unproven_flips=0" in green["stdout"]
    assert '--base-ref "$shape_base"' in _step()["run"]
