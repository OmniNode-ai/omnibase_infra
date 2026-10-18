# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17427, OMN-20413 - the hook push rehearsal runs the pre-push set as a worker's host does.

A controller worker's push runs this repository's pre-push hook set, and a hook change that makes
that push fail loses the worker's fix. The rehearsal pushes a sample commit through that set on a
pull request that changes a hook path. A worker's host holds the sibling clones under OMNI_HOME
and two pre-push hooks read them (architecture layers: an omnibase_core source tree; deploy-scope
DoD: omniclaude's deploy-gate validator). A hosted runner holds none, so the rehearsal refused
every hook-changing pull request until it cloned them. These tests hold the job to that: the
siblings it clones are the ones the hooks read, the clone step exports OMNI_HOME before the push
step, and the hooks' resolvers find what the step lays out.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.check_deploy_scope_dod import (
    _CANONICAL_VALIDATOR_RELPATH,
    resolve_omni_home,
)
from scripts.ci.ci_summary_gate import (
    EXPECTED_EXTERNAL_CONTEXTS,
    evaluate_external_contexts,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hook-push-rehearsal.yml"
CHECK_ARCHITECTURE = REPO_ROOT / "scripts" / "check_architecture.sh"
JOB_ID = "hook-push-rehearsal"
CLONE_STEP = "Clone the sibling repositories a worker's OMNI_HOME holds"
PUSH_STEP = "Install the pre-push hook and push a sample commit through it"


def _document() -> dict[Any, Any]:
    return dict(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def _job() -> dict[str, Any]:
    return dict(_document()["jobs"][JOB_ID])


def _triggers() -> dict[str, Any]:
    document = _document()
    # PyYAML reads the bare key `on` as the boolean True.
    return dict(document.get("on") or document[True])


def _step(name: str) -> dict[str, Any]:
    for step in _job()["steps"]:
        if step.get("name") == name:
            return dict(step)
    raise AssertionError(f"no step named {name!r}")


def _sibling_repos() -> list[str]:
    return str(_job()["env"]["SIBLING_REPOS"]).split()


def test_the_rehearsal_runs_on_pull_requests_and_merge_queue_entries() -> None:
    assert set(_triggers()) == {"pull_request", "merge_group"}


def test_the_job_keeps_the_check_context_and_reads_contents_only() -> None:
    job = _job()
    assert job["name"] == "hook-push-rehearsal / rehearse"
    assert job["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["PUSH_TIMEOUT_S"] == "1800"


def test_the_uv_installer_is_pinned_to_an_immutable_commit() -> None:
    assert re.fullmatch(r"astral-sh/setup-uv@[0-9a-f]{40}", _step("Install uv")["uses"])


@pytest.mark.parametrize(
    "conclusion", ["success", "failure", "skipped", "neutral", None]
)
def test_ci_summary_requires_a_successful_rehearsal(conclusion: str | None) -> None:
    """A missing or skipped rehearsal must not retire the push proof silently."""
    context = _job()["name"]
    assert context in EXPECTED_EXTERNAL_CONTEXTS
    payload = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in EXPECTED_EXTERNAL_CONTEXTS
        if name != context
    ]
    if conclusion is not None:
        payload.append(
            {"name": context, "status": "completed", "conclusion": conclusion}
        )
    failures, pending = evaluate_external_contexts(payload, EXPECTED_EXTERNAL_CONTEXTS)
    if conclusion == "success":
        assert (failures, pending) == ([], [])
    elif conclusion is None:
        assert failures == []
        assert pending == [context]
    else:
        assert failures == [context]
        assert pending == []


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/hook_push_rehearsal_admission_omn20413.json"
)
def test_the_rehearsal_reported_before_merge_on_sixteen_recent_dev_prs(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the watcher rows used to admit the existing external context."""
    pull_requests = recorded_response["pull_requests"]
    assert len(pull_requests) == 16
    for pr, entry in pull_requests.items():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["head_sha"]), pr
        context, status, conclusion, reported_at = entry["check_run"]
        assert context == _job()["name"], pr
        assert (status, conclusion) == ("completed", "success"), pr
        assert reported_at <= entry["merged_at"], pr
        assert entry["detail"][0][0] == context, pr
        assert entry["detail"][0][2] <= entry["merged_at"], pr


def test_the_path_regex_matches_hook_paths_and_the_rehearsal_itself() -> None:
    pattern = re.compile(_job()["env"]["PATH_REGEX"])
    for changed in (
        ".pre-commit-config.yaml",
        ".pre-commit-hooks.yaml",
        ".pre-commit-hooks/reject-deploy-gate-skip-token.sh",
        "scripts/hooks/example.sh",
        ".github/workflows/hook-push-rehearsal.yml",
    ):
        assert pattern.search(changed), changed
    for unrelated in (
        "README.md",
        "src/omnibase_infra/utils/correlation.py",
        ".github/workflows/ci.yml",
    ):
        assert not pattern.search(unrelated), unrelated


def test_the_sample_file_is_a_tracked_python_source() -> None:
    sample = _job()["env"]["SAMPLE_FILE"]
    assert sample.endswith(".py")
    assert (REPO_ROOT / sample).is_file()


def test_the_siblings_cloned_are_the_ones_the_pre_push_hooks_read() -> None:
    siblings = set(_sibling_repos())
    # Deploy-scope DoD imports <OMNI_HOME>/omniclaude/.github/actions/deploy-gate/...
    assert _CANONICAL_VALIDATOR_RELPATH.parts[0] in siblings
    # Architecture layers resolves ${OMNI_HOME}/omnibase_core/src/omnibase_core.
    assert (
        "${OMNI_HOME}/omnibase_core/src/omnibase_core"
        in CHECK_ARCHITECTURE.read_text(encoding="utf-8")
    )
    assert "omnibase_core" in siblings


def test_the_clone_step_runs_before_the_push_under_the_same_condition() -> None:
    names = [step.get("name") for step in _job()["steps"]]
    assert names.index(CLONE_STEP) < names.index(PUSH_STEP)
    assert _step(CLONE_STEP)["if"] == _step(PUSH_STEP)["if"]
    assert "OMNI_HOME=" in _step(CLONE_STEP)["run"]
    assert "GITHUB_ENV" in _step(CLONE_STEP)["run"]


def test_the_clone_step_lays_out_a_sibling_root_the_deploy_scope_hook_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the clone step's script with a stub git and resolve the validator from its OMNI_HOME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    # `git clone ... <url> <dest>` makes the destination (with the validator for omniclaude);
    # `git -C <dir> rev-parse HEAD` prints a sha.
    validator = _CANONICAL_VALIDATOR_RELPATH.relative_to("omniclaude")
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = clone ]; then\n'
        '  url="${@: -2:1}"; dest="${@: -1}"; mkdir -p "$dest"\n'
        '  case "$dest" in */omniclaude)\n'
        f'    mkdir -p "$dest/{validator.parent}" && touch "$dest/{validator}";;\n'
        "  esac\n"
        '  echo "$url" >> "$CLONE_LOG"; exit 0\n'
        "fi\n"
        "echo 0000000000000000000000000000000000000000\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    github_env = tmp_path / "github_env"
    clone_log = tmp_path / "clone_log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_ENV": str(github_env),
        "SIBLING_REPOS": str(_job()["env"]["SIBLING_REPOS"]),
        "CLONE_LOG": str(clone_log),
    }
    subprocess.run(
        ["bash", "-c", _step(CLONE_STEP)["run"]],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert clone_log.read_text(encoding="utf-8").split() == [
        f"https://github.com/OmniNode-ai/{repo}.git" for repo in _sibling_repos()
    ]
    exported = dict(
        line.split("=", 1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    sibling_root = Path(exported["OMNI_HOME"])
    assert sibling_root == runner_temp / "siblings"

    workspace = tmp_path / "workspace" / "omnibase_infra"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("OMNI_HOME", str(sibling_root))
    assert resolve_omni_home(workspace) == sibling_root


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/hook_push_rehearsal_siblings_omn20413.json"
)
def test_the_recorded_siblings_carry_what_the_pre_push_hooks_read(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the siblings' default-branch trees as the clone step checks them out."""
    siblings = recorded_response["siblings"]
    assert set(siblings) == set(_sibling_repos())
    clone_run = _step(CLONE_STEP)["run"]
    architecture = CHECK_ARCHITECTURE.read_text(encoding="utf-8")
    for repo, recorded in siblings.items():
        assert re.fullmatch(r"[0-9a-f]{40}", recorded["commit"])
        assert recorded["default_branch"] == "origin/dev"
        url = "https://github.com/OmniNode-ai/${repo}.git"
        assert url in clone_run
        assert recorded["origin_url"] == url.replace("${repo}", repo)
    # check_architecture.sh accepts the omnibase_core clone's origin URL as canonical.
    assert f"{siblings['omnibase_core']['origin_url']}|" in architecture
    core_paths = {
        line.split("\t", 1)[1] for line in siblings["omnibase_core"]["ls_tree"]
    }
    assert {"pyproject.toml", "src/omnibase_core/__init__.py"} <= core_paths
    claude_paths = {
        line.split("\t", 1)[1] for line in siblings["omniclaude"]["ls_tree"]
    }
    assert str(_CANONICAL_VALIDATOR_RELPATH.relative_to("omniclaude")) in claude_paths
