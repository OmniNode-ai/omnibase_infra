# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: execute the train follower's tag selection and pin writes."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/omnimarket-contract-pin-refresh.yml"
PIN = Path(".github/omnimarket-contract-pin.yaml")
SIBLINGS = Path(".github/sibling-pins.yaml")
RESOLVER = Path("scripts/resolve_omnimarket_contract_pin.py")


def _step_script(name: str) -> str:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return next(
        step["run"]
        for step in document["jobs"]["refresh"]["steps"]
        if step["name"] == name
    )


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Keep every local git operation isolated from the invoking worktree."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )


def _write_pin(repo: Path, tag: str) -> None:
    (repo / PIN).parent.mkdir(parents=True, exist_ok=True)
    (repo / PIN).write_text(
        f"repository: OmniNode-ai/omnimarket\nomnimarket_release_tag: {tag}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("current", "changed"), [("v0.4.318", "true"), ("v0.4.319", "false")]
)
def test_newest_release_tag_follows_numeric_order(
    tmp_path: Path, current: str, changed: str
) -> None:
    upstream = tmp_path / "omnimarket"
    upstream.mkdir()
    _git("init", "-q", "-b", "main", cwd=upstream)
    _git("config", "user.email", "ci@omninode.ai", cwd=upstream)
    _git("config", "user.name", "ci", cwd=upstream)
    _git(
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "base",
        cwd=upstream,
    )
    for tag in (
        "v0.4.9",
        "v0.4.10",
        "v0.4.319",
        "v0.5.0-rc1",
        "dev-20261023000000-abcdef012345",
    ):
        _git("tag", tag, cwd=upstream)

    repo = tmp_path / "infra"
    _write_pin(repo, current)
    (repo / RESOLVER).parent.mkdir()
    shutil.copyfile(ROOT / RESOLVER, repo / RESOLVER)
    output = tmp_path / "github-output"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            _step_script("Resolve the current pin and the train's newest release tag"),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={
            **scrub_git_location_env(os.environ),
            "OMNIMARKET_REMOTE": str(upstream),
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["current"] == current
    assert outputs["target"] == "v0.4.319"
    assert outputs["changed"] == changed


@pytest.mark.parametrize("target", ["v0.4.319", "a" * 40])
def test_tag_writer_updates_every_reader_and_refuses_a_sha(
    tmp_path: Path, target: str
) -> None:
    _write_pin(tmp_path, "v0.4.318")
    (tmp_path / SIBLINGS).write_text(
        "pins:\n  omnimarket: v0.4.318\n  other: unchanged\n", encoding="utf-8"
    )
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir()
    for name in ("one.yml", "two.yml"):
        (workflows / name).write_text(
            "          ref: v0.4.318 # .github/sibling-pins.yaml\n",
            encoding="utf-8",
        )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    completed = subprocess.run(
        ["bash", "-c", _step_script("Write the tag into the pin files")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(os.environ), "TARGET_TAG": target},
    )
    if target == "v0.4.319":
        assert completed.returncode == 0, completed.stdout + completed.stderr
        for path, original in before.items():
            assert path.read_bytes() == original.replace(b"v0.4.318", b"v0.4.319")
    else:
        assert completed.returncode != 0
        assert "release tag" in completed.stderr
        assert {path: path.read_bytes() for path in before} == before


def test_the_follower_states_its_staleness_budget_and_ordering_gate() -> None:
    """OMN-17292 AC3/AC4: the trade is recorded where the writer lives."""
    header = WORKFLOW.read_text(encoding="utf-8").split("\nname: ", 1)[0]
    assert "STALENESS (OMN-15701)" in header
    assert "one train cycle" in header
    assert "OMN-17156 remains the upstream ordering gate" in header


@pytest.mark.parametrize(
    ("event", "trigger", "conclusion", "allowed"),
    [
        ("workflow_run", "schedule", "failure", False),
        ("workflow_run", "schedule", "cancelled", False),
        ("workflow_run", "schedule", "timed_out", False),
        ("workflow_run", "schedule", "success", True),
        ("workflow_run", "workflow_dispatch", "success", False),
        ("workflow_dispatch", "", "", True),
    ],
)
def test_only_a_successful_scheduled_train_or_manual_dispatch_can_refresh(
    event: str, trigger: str, conclusion: str, allowed: bool
) -> None:
    """Replay the workflow's actual boolean guard over terminal train events."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    guard = document["jobs"]["refresh"]["if"]
    for field, value in (
        ("github.event.workflow_run.conclusion", conclusion),
        ("github.event.workflow_run.event", trigger),
        ("github.event_name", event),
    ):
        guard = guard.replace(field, f"'{value}'")
    result = subprocess.run(
        ["bash", "-c", f"if [[ {guard} ]]; then exit 0; else exit 1; fi"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert (result.returncode == 0) is allowed


def test_remote_lookup_failure_cannot_publish_a_target(tmp_path: Path) -> None:
    """An unreachable remote fails before a target or changed flag is published."""
    repo = tmp_path / "infra"
    _write_pin(repo, "v0.4.318")
    (repo / RESOLVER).parent.mkdir()
    shutil.copyfile(ROOT / RESOLVER, repo / RESOLVER)
    output = tmp_path / "github-output"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            _step_script("Resolve the current pin and the train's newest release tag"),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={
            **scrub_git_location_env(os.environ),
            "OMNIMARKET_REMOTE": str(tmp_path / "nonexistent-remote"),
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert completed.returncode != 0
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs == {"tag": "v0.4.318"}
    assert "fatal:" in completed.stderr


@pytest.mark.live_contact(
    "tests/ci/fixtures/release_train_refresh_conclusions_omn17292.json"
)
def test_recorded_train_completions_enforce_refresh_admission(
    recorded_response: dict[str, object],
) -> None:
    """Replay real completed runs through the workflow's actual job guard."""
    runs = recorded_response["workflow_runs"]
    assert isinstance(runs, list)
    seen: set[tuple[str, str]] = set()
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for run in runs:
        assert run["status"] == "completed"
        assert run["path"] == ".github/workflows/release-train-nightly.yml"
        seen.add((run["event"], run["conclusion"]))
        guard = document["jobs"]["refresh"]["if"]
        for field, value in (
            ("github.event.workflow_run.conclusion", run["conclusion"]),
            ("github.event.workflow_run.event", run["event"]),
            ("github.event_name", "workflow_run"),
        ):
            guard = guard.replace(field, f"'{value}'")
        result = subprocess.run(
            ["bash", "-c", f"if [[ {guard} ]]; then exit 0; else exit 1; fi"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in (0, 1), result.stderr
        allowed = run["event"] == "schedule" and run["conclusion"] == "success"
        assert (result.returncode == 0) is allowed, run["html_url"]
    assert seen == {
        ("schedule", "success"),
        ("schedule", "failure"),
        ("workflow_dispatch", "success"),
    }
