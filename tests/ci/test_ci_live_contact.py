# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admission falsifiers and recorded API replays for OMN-18648."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_ci_live_contact import (
    HandlerCILiveContact,
)
from omnibase_infra.models.model_ci_live_contact_request import (
    ModelCILiveContactRequest,
)


@pytest.fixture
def change(tmp_path: Path) -> Path:
    subprocess.run(
        ["git", "init", "--quiet", str(tmp_path)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    (tmp_path / "scripts/ci").mkdir(parents=True)
    (tmp_path / "scripts/ci/check.py").write_text("print('gate')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_check.py").write_text(
        "def test_seam(monkeypatch):\n    monkeypatch.setattr('subprocess.run', lambda *a, **k: None)\n"
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return tmp_path


async def test_seam_only_ci_tooling_pr_is_refused(change: Path) -> None:
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    assert not result.success, result
    assert "live-contact" in result.reason


async def test_explicit_reason_is_accepted(change: Path) -> None:
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(
            repository=str(change),
            pr_body="Live-contact exception: This changes a workflow display label; there is no API or runner behaviour to replay.",
        )
    )
    assert result.success, result


ROOT = Path(__file__).resolve().parents[2]
CHECK_RUNS = "tests/ci/fixtures/omn15496_merge_time_external_check_runs.json"
PROTECTION = "tests/ci/fixtures/ci_live_contact/required_status_checks.json"
CONTRACT = ROOT / "src/omnibase_infra/nodes/node_ci_live_contact_effect/contract.yaml"


def add_recorded_test(
    change: Path, *, mock: bool = False, artifact: bool = True, provenance: bool = True
) -> None:
    recorded = change / "tests/recording.json"
    if artifact:
        value = {"response": {"check_runs": []}}
        if provenance:
            value["_provenance"] = {
                "source": "recorded API response",
                "captured_utc": "2026-10-08",
            }
        recorded.write_text(json.dumps(value))
    parameter = "recorded_response, monkeypatch" if mock else "recorded_response"
    (change / "tests/test_recorded.py").write_text(
        "import pytest\n@pytest.mark.live_contact('tests/recording.json')\n"
        f"def test_replay({parameter}):\n    assert recorded_response['response'] is not None\n"
    )
    subprocess.run(
        ["git", "-C", str(change), "add", "."],
        check=True,
        env=scrub_git_location_env(os.environ),
    )


async def test_recorded_test_is_admitted(change: Path) -> None:
    add_recorded_test(change)
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    assert result.success, result
    assert result.live_contact_tests == ("tests/test_recorded.py::test_replay",)


@pytest.mark.parametrize(
    "case", ["missing", "unproven", "mock", "unused", "comment", "unstaged", "skipped"]
)
async def test_decorative_or_broken_marker_is_refused(change: Path, case: str) -> None:
    add_recorded_test(
        change,
        artifact=case != "missing",
        provenance=case != "unproven",
        mock=case == "mock",
    )
    path = change / "tests/test_recorded.py"
    if case == "skipped":
        path.write_text(
            path.read_text().replace(
                "def test_replay",
                "@pytest.mark.skip(reason='never contacts anything')\ndef test_replay",
            )
        )
    if case == "unused":
        path.write_text(
            path.read_text().replace(
                "assert recorded_response['response'] is not None", "assert True"
            )
        )
    if case == "comment":
        path.write_text(
            path.read_text().replace(
                "@pytest.mark.live_contact", "# @pytest.mark.live_contact"
            )
        )
    if case == "unstaged":
        subprocess.run(
            ["git", "-C", str(change), "rm", "--cached", str(path)],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
    else:
        subprocess.run(
            ["git", "-C", str(change), "add", "."],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    assert not result.success, result


@pytest.mark.parametrize(
    "body",
    [
        "",
        "Live-contact exception:",
        "Live-contact exception: impossible",
        "```\nLive-contact exception: This is only a copied example and does not declare an exception.\n```",
    ],
)
async def test_empty_or_example_exception_is_refused(change: Path, body: str) -> None:
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change), pr_body=body)
    )
    assert not result.success, result


async def test_no_tooling_change_is_a_positive_control(change: Path) -> None:
    subprocess.run(
        ["git", "-C", str(change), "rm", "--cached", "scripts/ci/check.py"],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    assert result.success and not result.changed_tooling, result


async def test_unknown_base_is_refused(change: Path) -> None:
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change), base_ref="missing-base")
    )
    assert not result.success and "Cannot establish" in result.reason, result


async def test_committed_head_does_not_borrow_unstaged_evidence(change: Path) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(change),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "seam only",
        ],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    add_recorded_test(change)
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change), base_ref="HEAD~1")
    )
    assert not result.success, result


def install_gh_replay(
    tmp_path: Path,
    response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    denied: bool = False,
) -> None:
    """Replay at the executable boundary; retain the production gh --jq call.

    jq applies the production expression to recorded bytes in another process.
    The denied case is a labelled transport-failure control, not a live token claim.
    """
    data = tmp_path / "api.json"
    data.write_text(json.dumps(response))
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport subprocess,sys\n"
        + (
            "sys.stderr.write('replay transport denied required_status_checks read\\n');sys.exit(1)\n"
            if denied
            else f"sys.exit(subprocess.run(['jq','-c','-r',sys.argv[-1],{str(data)!r}],check=False).returncode)\n"
        )
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


@pytest.fixture
def replay_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return lambda response, denied=False: install_gh_replay(
        tmp_path, response, monkeypatch, denied=denied
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn15496_merge_time_external_check_runs.json"
)
def test_recorded_check_listing_exercises_real_fetch_projection_and_ordering(
    recorded_response, replay_gh
) -> None:
    from scripts.ci import ci_summary_gate, release_train

    rows = recorded_response["pull_requests"]["2567"]["check_runs"]
    replay_gh({"check_runs": rows})
    fetched = release_train.default_check_runs("omnibase_infra", "recorded-head")
    assert len(fetched) == len(rows)
    assert [row["started_at"] for row in fetched] == [row["started_at"] for row in rows]
    winner = ci_summary_gate.latest_check_run_rows(fetched)["Canonical Inference Gate"]
    assert winner["started_at"] == "2026-07-30T14:23:29Z"
    assert (
        ci_summary_gate.latest_check_run_rows(list(reversed(fetched)))[
            "Canonical Inference Gate"
        ]
        == winner
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/required_status_checks.json"
)
def test_recorded_protection_exercises_real_required_context_fetch(
    recorded_response, replay_gh
) -> None:
    from scripts.ci import release_train

    response = recorded_response["response"]
    replay_gh(response)
    assert (
        release_train.default_required_contexts("recorded-repo", "main")
        == response["contexts"]
    )
    assert "CI Summary" in response["contexts"]


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/required_status_checks.json"
)
def test_protection_read_denied_is_a_refusal_not_empty_green(
    recorded_response, replay_gh
) -> None:
    from scripts.ci import release_train

    replay_gh(recorded_response["response"], denied=True)
    reason, detail = release_train.classify_ci_green(
        "recorded-repo",
        "recorded-head",
        "main",
        required_contexts=release_train.default_required_contexts,
        check_runs=release_train.default_check_runs,
        gating_sha=release_train.default_gating_sha,
    )
    assert reason is release_train.EnumTrainReason.CI_PROTECTION_UNREADABLE
    assert "could not read required status checks" in detail


@pytest.mark.parametrize("recorded", [False, True])
def test_bus_runtime_refuses_seam_only_and_passes_recorded_pair(
    change: Path, tmp_path: Path, recorded: bool
) -> None:
    if recorded:
        add_recorded_test(change)
    request = tmp_path / "request.json"
    request.write_text(
        ModelCILiveContactRequest(repository=str(change)).model_dump_json()
    )
    from omnibase_core.runtime.runtime_local import RuntimeLocal

    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=tmp_path / "state",
        input_path=request,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    assert runtime.exit_code == (0 if recorded else 1), runtime.last_error
    assert runtime._handlers_wired, "the proof must cross the contract-declared bus"


def test_recording_source_bytes_are_pinned() -> None:
    document = json.loads((ROOT / PROTECTION).read_text())
    provenance = document["_provenance"]
    assert (
        hashlib.sha256((ROOT / provenance["source_file"]).read_bytes()).hexdigest()
        == provenance["source_sha256"]
    )
    assert (
        document["response"]
        == json.loads((ROOT / provenance["source_file"]).read_text())[
            "required_status_checks"
        ]
    )


def test_ci_and_hook_use_the_bus_node_and_required_context() -> None:
    import tomllib

    import yaml

    from scripts.ci.ci_summary_gate import EXPECTED_EXTERNAL_CONTEXTS

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci-live-contact.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert {"pull_request", "push", "merge_group"} <= triggers.keys()
    assert "edited" in triggers["pull_request"]["types"]
    assert not {"paths", "paths-ignore"} & triggers["pull_request"].keys()
    job = workflow["jobs"]["admission"]
    assert job["name"] in EXPECTED_EXTERNAL_CONTEXTS
    assert "if" not in job and "continue-on-error" not in job
    steps = job["steps"]
    assert any(
        "onex node node_ci_live_contact_effect" in step.get("run", "")
        and "event_bus=inmemory" in step["run"]
        for step in steps
    )
    assert any("'live_contact'" in step.get("run", "") for step in steps)
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        h
        for repo in hooks["repos"]
        for h in repo.get("hooks", [])
        if h["id"] == "ci-live-contact"
    )
    assert (
        "onex node node_ci_live_contact_effect" in hook["entry"] and hook["always_run"]
    )
    assert "event_bus=inmemory" in hook["entry"]
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["runtime_profiles"] == ["main"]
    assert (
        contract["handler_routing"]["handlers"][0]["handler"]["module"]
        == "omnibase_infra.handlers.handler_ci_live_contact"
    )
    runtime_topics = yaml.safe_load(
        (ROOT / "src/omnibase_infra/runtime/topics.yaml").read_text()
    )["topics"]
    command_topic = contract["event_bus"]["subscribe_topics"][0]
    assert command_topic in runtime_topics
    assert command_topic not in contract["event_bus"]["publish_topics"]
    wiring_hook = next(
        h
        for repo in hooks["repos"]
        for h in repo.get("hooks", [])
        if h["id"] == "subscribe-wiring-health"
    )
    import re

    assert re.search(wiring_hook["files"], "src/omnibase_infra/runtime/topics.yaml")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        "node_ci_live_contact_effect"
        in project["project"]["entry-points"]["onex.nodes"]
    )


def test_admission_context_missing_red_and_green_controls() -> None:
    from scripts.ci.ci_summary_gate import evaluate_external_contexts

    name = "CI Live Contact (OMN-18648)"
    bad, missing = evaluate_external_contexts([], (name,))
    assert not bad and missing == [name]
    row = {"name": name, "status": "completed", "conclusion": "failure"}
    bad, missing = evaluate_external_contexts([row], (name,))
    assert bad and not missing
    row["conclusion"] = "success"
    assert evaluate_external_contexts([row], (name,)) == ([], [])


async def test_renaming_tooling_out_of_scope_still_requires_evidence(
    change: Path,
) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(change),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial tooling",
        ],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    subprocess.run(
        ["git", "-C", str(change), "mv", "scripts/ci/check.py", "scripts/check.py"],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    assert not result.success, result
    assert "scripts/ci/check.py" in result.changed_tooling
