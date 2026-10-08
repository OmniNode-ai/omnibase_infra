# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106: hosted dod_verify controls need the pre-change Git history.

Exercise the workflow's clone policy through the verifier's shared-clone and
detached-checkout sequence. A shallow source can fetch the earlier commit and
still fail that sequence: the shared clone cannot resolve the detached commit.
The valid twin runs the overlaid test against earlier code and observes an
assertion failure, then observes the same test passing against current code.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/evidence-autoclose-sweep.yml"
TEST_BODY = (
    "from feature import value\n\ndef test_value():\n    assert value() == 'after'\n"
)


def _steps() -> list[dict[str, Any]]:
    return cast(
        "list[dict[str, Any]]",
        yaml.safe_load(WORKFLOW.read_text())["jobs"]["evidence-autoclose-sweep"][
            "steps"
        ],
    )


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "origin"
    source.mkdir()
    _git(source, "init", "--quiet", "--initial-branch=dev")
    for value in ("before", "after"):
        (source / "feature.py").write_text(f"def value():\n    return {value!r}\n")
        _git(source, "add", "feature.py")
        _git(
            source,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            value,
        )
    return source, _git(source, "rev-parse", "HEAD^").stdout.strip()


def _shared_checkout(
    source: Path, tree: Path, sha: str
) -> subprocess.CompletedProcess[str]:
    _git(
        tree.parent,
        "clone",
        "--quiet",
        "--shared",
        "--no-checkout",
        str(source),
        str(tree),
    )
    return _git(tree, "checkout", "--quiet", "--detach", sha, check=False)


def _run_test(tree: Path) -> subprocess.CompletedProcess[str]:
    (tree / "test_feature.py").write_text(TEST_BODY)
    env = scrub_git_location_env(os.environ)
    env["PYTHONPATH"] = str(tree)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_feature.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.live_contact("tests/ci/fixtures/autoclose_history_omn16106.json")
def test_shallow_source_reproduces_control_checkout_failure(
    history: tuple[Path, str], tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    """Fetching the earlier SHA into a shallow source does not fix the shared clone."""
    origin, earlier = history
    source = tmp_path / "shallow"
    _git(tmp_path, "clone", "--quiet", "--depth", "1", origin.as_uri(), str(source))
    _git(source, "fetch", "--quiet", "--no-tags", "origin", earlier)
    _git(source, "cat-file", "-e", f"{earlier}^{{commit}}")
    result = _shared_checkout(source, tmp_path / "control", earlier)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "reference is not a tree" in result.stderr
    recording = cast("dict[str, object]", recorded_response["response"])
    assert result.returncode == recording["returncode"]
    assert (
        result.stderr.replace(earlier, "<pre-change sha>")
        == recording["stderr_template"]
    )


@pytest.mark.parametrize(
    "repository",
    ["omnibase_infra", "omnimarket", "onex_change_control"],
)
def test_checkout_history_can_execute_the_pre_change_control(
    repository: str, history: tuple[Path, str], tmp_path: Path
) -> None:
    origin, earlier = history
    checkout = next(
        step
        for step in _steps()
        if str(step.get("uses", "")).startswith("actions/checkout@")
        and step.get("with", {}).get("repository", "OmniNode-ai/omnibase_infra")
        == f"OmniNode-ai/{repository}"
    )
    depth = checkout.get("with", {}).get("fetch-depth", 1)
    source = tmp_path / "checkout"
    args = ["clone", "--quiet"]
    if depth:
        args.extend(["--depth", str(depth)])
    _git(tmp_path, *args, origin.as_uri(), str(source))
    assert depth == 0, f"{repository}: historical must-fail controls need full history"
    _prove_control(source, earlier, tmp_path)


def test_derived_clone_history_can_execute_the_pre_change_control(
    history: tuple[Path, str], tmp_path: Path
) -> None:
    origin, earlier = history
    materialise = next(
        step
        for step in _steps()
        if "Derive and materialise the cwd repo set" in step.get("name", "")
    )
    match = re.search(r"elif (git clone .*?); then", materialise["run"], re.DOTALL)
    assert match is not None, (
        "the derived product-repository clone command is unreadable"
    )
    argv = shlex.split(match.group(1).split("2>", 1)[0].replace("\\\n", " "))
    source = tmp_path / "derived"
    replacements = {
        "${branch}": "dev",
        "https://github.com/OmniNode-ai/${repo}.git": origin.as_uri(),
        "${src}": str(source),
    }
    args = [replacements.get(arg, arg) for arg in argv[1:]]
    assert not any("${" in arg for arg in args), args
    _git(tmp_path, *args)
    _prove_control(source, earlier, tmp_path)


def _prove_control(source: Path, earlier: str, tmp_path: Path) -> None:
    head = _git(source, "rev-parse", "HEAD").stdout.strip()
    result = _shared_checkout(source, tmp_path / "control", earlier)
    assert result.returncode == 0, result.stdout + result.stderr
    negative = _run_test(tmp_path / "control")
    assert negative.returncode == 1, negative.stdout + negative.stderr
    assert "AssertionError" in negative.stdout
    assert "1 failed" in negative.stdout
    positive = _run_test(source)
    assert positive.returncode == 0, positive.stdout + positive.stderr
    assert "1 passed" in positive.stdout
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == head


def test_history_regression_is_a_required_ci_and_precommit_check() -> None:
    target = "tests/ci/test_autoclose_history_omn16106.py"
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = ci["jobs"]["ci-evidence-policy"]
    step = next(step for step in job["steps"] if target in step.get("run", ""))
    assert not step.get("if") and not step.get("continue-on-error")
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in hooks["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "autoclose-history"
    )
    assert target in hook["entry"]
    assert hook["pass_filenames"] is False
    pattern = re.compile(hook["files"])
    for path in (
        ".github/workflows/evidence-autoclose-sweep.yml",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
        "tests/ci/fixtures/autoclose_history_omn16106.json",
        target,
    ):
        assert pattern.search(path), f"history regression hook does not cover {path}"
