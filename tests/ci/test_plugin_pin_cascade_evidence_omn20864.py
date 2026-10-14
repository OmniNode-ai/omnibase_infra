# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20864: a plugin pin cascade PR carries its own passing-test evidence.

The operator ruled that a pin bump PR whose tests pass needs no further
evidence. The cascade opens a PR that changes only the plugin pins in
``docker/Dockerfile.runtime``. That file is a deploy-scoped path, and the
Repo Evidence Gate refuses a PR that cites a ticket without a
``contracts/<ticket>.yaml`` that differs from the merge base, a changed
test-side path, and a bound check that fails at the merge base. Before this
change a lane wrote that contract and a test by hand for every cascade PR
(omnibase_infra#4799).

The cascade now renders them, as ``omnimarket-contract-pin-refresh.yml``
does for its pin (OMN-20068). The render is gated on the diff being pin-only
by its changed paths and lines, never on the author: a cascade diff that
also changes any other path or any non-pin line of the Dockerfile renders
nothing and fails the step.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "plugin-pin-cascade.yml"
DOCKERFILE = Path("docker/Dockerfile.runtime")
PIN_TEST = Path("tests/unit/docker/test_runtime_plugin_pin_floors_omn20864.py")
PIN_FIXTURE = Path("tests/ci/fixtures/runtime_plugin_pin_floors_omn20864.yaml")
PIN_CHECK = (
    "env -u PYTHONPATH uv run --frozen pytest "
    "tests/unit/docker/test_runtime_plugin_pin_floors_omn20864.py -q"
)
STEP_NAME = "Render the pin evidence (OMN-20864)"
# TEST_SIDE from omnibase_core receipt-gate.yml.
TEST_SIDE = r"^tests?/|/tests?/|(^|/)conftest\.py$|(^|/)test_[^/]*\.py$|_test\.py$"

BASE_DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install \\
    "omninode-claude>=0.27.0,<1.0.0" \\
    "omninode-memory>=0.18.4,<1.0.0" \\
    "omninode-intelligence>=0.24.0,<1.0.0"
"""
HEAD_DOCKERFILE = BASE_DOCKERFILE.replace(
    '"omninode-memory>=0.18.4,<1.0.0"', '"omninode-memory>=0.19.0,<1.0.0"'
)
MEMORY_HEAD_PIN = '"omninode-memory>=0.19.0,<1.0.0"'
ENV = {"PKG": "omninode-memory", "VER": "v0.19.0", "TICKET": "OMN-18596"}


def _steps() -> list[dict[str, object]]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    (job,) = data["jobs"].values()
    return list(job["steps"])


def _render_step() -> dict[str, object]:
    (step,) = [s for s in _steps() if s.get("name") == STEP_NAME]
    return step


def _git(cwd: Path, *argv: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *argv],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
    )


@pytest.fixture
def cascade_repo(tmp_path: Path) -> Path:
    """A repository whose HEAD carries the base Dockerfile and the pin test."""
    _git(tmp_path, "init", "-q", "-b", "dev")
    (tmp_path / DOCKERFILE).parent.mkdir(parents=True)
    (tmp_path / DOCKERFILE).write_text(BASE_DOCKERFILE, encoding="utf-8")
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / "OMN-18596.yaml").write_text("old\n", encoding="utf-8")
    (tmp_path / PIN_FIXTURE).parent.mkdir(parents=True)
    (tmp_path / PIN_TEST).parent.mkdir(parents=True)
    shutil.copyfile(REPO / PIN_TEST, tmp_path / PIN_TEST)
    shutil.copyfile(REPO / PIN_FIXTURE, tmp_path / PIN_FIXTURE)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _run_render(
    repo: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    script = str(_render_step()["run"])
    return subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        env={**scrub_git_location_env(os.environ), **(env or ENV)},
        check=False,
        capture_output=True,
        text=True,
    )


def _pytest_argv(repo: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(repo),
        "-c",
        "/dev/null",
        "-o",
        "markers=unit: Unit tests",
        str(repo / PIN_TEST),
    ]


def test_render_step_runs_before_the_commit_and_the_commit_adds_it() -> None:
    names = [str(s.get("name")) for s in _steps()]
    assert names.index("Check for changes") < names.index(STEP_NAME)
    assert names.index(STEP_NAME) < names.index("Commit and push")
    assert names.index(STEP_NAME) < names.index("Open PR")
    step = _render_step()
    assert step["if"] == "steps.diff.outputs.changed == 'true'"
    env = step["env"]
    assert set(env) == {"PKG", "VER", "TICKET"}
    assert env["TICKET"] == "${{ steps.inputs.outputs.ticket }}"
    (commit,) = [s for s in _steps() if s.get("name") == "Commit and push"]
    add = re.search(r"git add ((?:.*\\\n)*.*)", str(commit["run"]))
    assert add
    assert "docker/Dockerfile.runtime" in add.group(1)
    assert "contracts/${{ steps.inputs.outputs.ticket }}.yaml" in add.group(1)
    assert PIN_FIXTURE.as_posix() in add.group(1)


def test_rendered_evidence_passes_at_head_and_fails_at_base(cascade_repo: Path) -> None:
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    result = _run_render(cascade_repo)
    assert result.returncode == 0, result.stdout + result.stderr

    contract = yaml.safe_load(
        (cascade_repo / "contracts" / "OMN-18596.yaml").read_text(encoding="utf-8")
    )
    assert contract["ticket_id"] == "OMN-18596"
    assert "omninode-memory" in contract["title"]
    assert "0.19.0" in contract["title"]
    (evidence,) = contract["evidence_requirements"]
    assert evidence["kind"] == "tests"
    assert evidence["command"] == PIN_CHECK
    tests_item, probe_item = contract["dod_evidence"]
    (tests_check,) = tests_item["checks"]
    assert tests_check["check_type"] == "test_passes"
    assert tests_check["check_value"] == PIN_CHECK
    assert tests_check["cwd"] == "${OMNI_HOME}/omnibase_infra"
    assert tests_item["binds_ac"] == ["AC1"]
    (probe_check,) = probe_item["checks"]
    assert probe_check["check_type"] == "command"
    assert probe_check["check_value"].startswith("gh api ")
    assert MEMORY_HEAD_PIN in probe_check["check_value"]
    assert probe_item["binds_ac"] == ["AC1"]
    assert re.search(TEST_SIDE, PIN_FIXTURE.as_posix())

    fixture = yaml.safe_load((cascade_repo / PIN_FIXTURE).read_text(encoding="utf-8"))
    assert fixture["floors"] == {
        "omninode-claude": "0.27.0",
        "omninode-intelligence": "0.24.0",
        "omninode-memory": "0.19.0",
    }

    argv = _pytest_argv(cascade_repo)
    at_head = subprocess.run(
        argv, cwd=cascade_repo, check=False, capture_output=True, text=True
    )
    assert at_head.returncode == 0, at_head.stdout + at_head.stderr
    # The merge base's Dockerfile with the rendered test side overlaid.
    (cascade_repo / DOCKERFILE).write_text(BASE_DOCKERFILE, encoding="utf-8")
    at_base = subprocess.run(
        argv, cwd=cascade_repo, check=False, capture_output=True, text=True
    )
    assert at_base.returncode != 0, at_base.stdout + at_base.stderr


def test_rendered_contract_differs_from_the_merge_base(cascade_repo: Path) -> None:
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    assert _run_render(cascade_repo).returncode == 0
    head = (cascade_repo / "contracts" / "OMN-18596.yaml").read_text(encoding="utf-8")
    assert head != "old\n"
    assert "0.19.0" in head


def test_rendered_deploy_probe_is_falsifiable(
    cascade_repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The probe reads the Dockerfile through ``gh api`` and fails at the base."""
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    assert _run_render(cascade_repo).returncode == 0
    contract = yaml.safe_load(
        (cascade_repo / "contracts" / "OMN-18596.yaml").read_text(encoding="utf-8")
    )
    probe = contract["dod_evidence"][1]["checks"][0]["check_value"]

    bin_dir = tmp_path_factory.mktemp("fakebin")
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/bin/sh\nbase64 < \"$PROBE_DOCKERFILE\" | tr -d '\\n'\n", encoding="utf-8"
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    def run_probe(dockerfile_text: str) -> int:
        copy = cascade_repo / "probe-dockerfile"
        copy.write_text(dockerfile_text, encoding="utf-8")
        return subprocess.run(
            ["bash", "-c", probe],
            cwd=cascade_repo,
            env={**env, "PROBE_DOCKERFILE": str(copy)},
            check=False,
            capture_output=True,
        ).returncode

    assert run_probe(HEAD_DOCKERFILE) == 0
    assert run_probe(BASE_DOCKERFILE) != 0


def test_a_cascade_diff_that_also_changes_the_dockerfile_elsewhere_is_not_pin_only(
    cascade_repo: Path,
) -> None:
    (cascade_repo / DOCKERFILE).write_text(
        HEAD_DOCKERFILE + "ENV EXTRA=1\n", encoding="utf-8"
    )
    result = _run_render(cascade_repo)
    assert result.returncode != 0, result.stdout
    assert (cascade_repo / "contracts" / "OMN-18596.yaml").read_text() == "old\n"
    assert (cascade_repo / PIN_FIXTURE).read_text() == (REPO / PIN_FIXTURE).read_text()


def test_a_cascade_diff_that_also_changes_code_is_not_pin_only(
    cascade_repo: Path,
) -> None:
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    code = cascade_repo / "src" / "module.py"
    code.parent.mkdir(parents=True)
    code.write_text("x = 1\n", encoding="utf-8")
    result = _run_render(cascade_repo)
    assert result.returncode != 0, result.stdout
    assert (cascade_repo / "contracts" / "OMN-18596.yaml").read_text() == "old\n"


def test_a_cascade_that_changed_a_tracked_code_file_is_not_pin_only(
    cascade_repo: Path,
) -> None:
    tracked = cascade_repo / "src" / "module.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("x = 1\n", encoding="utf-8")
    _git(cascade_repo, "add", "-A")
    _git(cascade_repo, "commit", "-q", "-m", "code")
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    tracked.write_text("x = 2\n", encoding="utf-8")
    assert _run_render(cascade_repo).returncode != 0


def test_the_committed_floors_hold_against_the_shipped_dockerfile(
    cascade_repo: Path,
) -> None:
    """The committed fixture's floors hold against the shipped Dockerfile."""
    shipped = (REPO / DOCKERFILE).read_text(encoding="utf-8")
    (cascade_repo / DOCKERFILE).write_text(shipped, encoding="utf-8")
    ok = subprocess.run(
        _pytest_argv(cascade_repo),
        cwd=cascade_repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.live_contact(
    "tests/ci/fixtures/bot_refresh_contract_omn20413_dod_verify_annotations.json"
)
def test_the_recorded_gate_refusal_is_answered_by_the_rendered_contract(
    cascade_repo: Path, recorded_response: dict[str, object]
) -> None:
    """The live gate's recorded pin-only verdict applies to the cascade's path.

    The recorded dod-verify run on a bot pin PR says the writer app is not
    exempt because its changed path is not a dependency manifest or lockfile,
    then asks for ``contracts/<ticket>.yaml`` at the head. The cascade changes
    ``docker/Dockerfile.runtime``, a path outside the same two basenames, so the
    same verdict applies to it and the rendered contract is what answers it.
    """
    recording = recorded_response["response"]
    assert isinstance(recording, dict)
    messages = [a["message"] for a in recording["omnibase_infra#4816"]["annotations"]]
    (verdict,) = [
        m
        for m in messages
        if "dependency-pin-only: changed path is not a dependency manifest" in m
    ]
    named_path = re.search(r"or lockfile: (\S+) \(", verdict)
    assert named_path
    assert Path(named_path.group(1)).name not in {"pyproject.toml", "uv.lock"}
    assert DOCKERFILE.name not in {"pyproject.toml", "uv.lock"}
    (ticket,) = [
        m.group(1)
        for m in (
            re.fullmatch(r"Include contracts/(OMN-[0-9]+)\.yaml at this PR head.*", n)
            for n in messages
        )
        if m
    ]
    (cascade_repo / DOCKERFILE).write_text(HEAD_DOCKERFILE, encoding="utf-8")
    result = _run_render(cascade_repo, {**ENV, "TICKET": ticket})
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = cascade_repo / "contracts" / f"{ticket}.yaml"
    assert yaml.safe_load(rendered.read_text(encoding="utf-8"))["ticket_id"] == ticket


def test_pre_commit_and_ci_run_the_same_suites() -> None:
    pre_commit = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for path in (
        "tests/ci/test_plugin_pin_cascade_evidence_omn20864.py",
        "tests/unit/docker/test_runtime_plugin_pin_floors_omn20864.py",
    ):
        assert path in pre_commit
        assert path in ci
    assert "id: plugin-pin-cascade-evidence" in pre_commit
