# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Hostile Reviewer reads omniintelligence by its newest dev tag and passes the Claude key (OMN-20910).

Operator ruling 2026-10-18: lab consumers read tags, not pins. omniintelligence
tags every dev merge ``dev-<UTC>-<sha12>``, so the review tooling is resolved at
run time from ``refs/tags/dev-*`` and logged; a branch tip stays refused
(OMN-20331). The Claude API voters' key is the org Actions secret
``HOSTILE_REVIEW_ANTHROPIC_API_KEY``, handed to the review step alone. The
voter code reads the variable its overlay entry names in ``api_key_env`` (and
``ANTHROPIC_API_KEY`` when none is named), and never substitutes another
variable for it.

The clone step is executed against a fake ``git`` rather than text-matched, so
the tag selection is the one the workflow runs.
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

pytestmark = [pytest.mark.unit]

_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "hostile-reviewer.yml"
)
_KEY_EXPR = "${{ secrets.HOSTILE_REVIEW_ANTHROPIC_API_KEY }}"
_KEY_ENV_NAMES = ("ANTHROPIC_API_KEY", "REVIEW_ANTHROPIC_API_KEY")
_NEWEST = "dev-20261018085529-fe44102f0bf5"
_LS_REMOTE = "".join(
    f"{sha * 40}\trefs/tags/{name}\n"
    for sha, name in (
        ("a", "dev-20261018081705-96d68f37a133"),
        ("b", _NEWEST),
        ("c", "dev-20261018081711-c0cdfceade8b"),
        ("d", "dev-latest"),
        ("f", "v0.99.0"),
    )
)

_FAKE_GIT = """#!/bin/sh
case "$1" in
  ls-remote) echo "$@" >> "$FAKE_LOG"; cat "$FAKE_LS_REMOTE" ;;
  clone) echo "$@" >> "$FAKE_LOG" ;;
  -C) echo "$@" >> "$FAKE_LOG"; [ "$3" = rev-parse ] && echo 0123456789abcdef0123456789abcdef01234567; true ;;
esac
"""


def _steps() -> list[dict[str, Any]]:
    parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return list(parsed["jobs"]["hostile-review"]["steps"])


def _step(name: str) -> dict[str, Any]:
    (found,) = [s for s in _steps() if s.get("name") == name]
    return found


def _run_clone_step(
    tmp_path: Path, ls_remote: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool, body in (("git", _FAKE_GIT), ("sleep", "#!/bin/sh\nexit 0\n")):
        path = bindir / tool
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    listing = tmp_path / "ls-remote.txt"
    listing.write_text(ls_remote, encoding="utf-8")
    log = tmp_path / "git.log"
    log.write_text("", encoding="utf-8")
    work = tmp_path / "ws"
    work.mkdir()
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_LS_REMOTE": str(listing),
        "FAKE_LOG": str(log),
    }
    done = subprocess.run(
        ["bash", "-e", "-c", str(_step("Clone omniintelligence")["run"])],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done, log.read_text(encoding="utf-8").splitlines()


def test_clone_step_reads_the_newest_dev_tag_and_logs_it(tmp_path: Path) -> None:
    done, calls = _run_clone_step(tmp_path, _LS_REMOTE)
    assert done.returncode == 0, done.stderr
    assert "refs/tags/dev-*" in calls[0]
    assert any(f"--branch {_NEWEST}" in c for c in calls if c.startswith("clone"))
    assert any(c.endswith(f"checkout --quiet refs/tags/{_NEWEST}") for c in calls)
    assert (
        f"omniintelligence newest dev tag: {_NEWEST} -> "
        "0123456789abcdef0123456789abcdef01234567"
    ) in done.stdout


def test_clone_step_fails_closed_without_a_dev_tag(tmp_path: Path) -> None:
    only_others = "".join(
        f"{'a' * 40}\trefs/tags/{name}\n" for name in ("v0.99.0", "dev-latest")
    )
    done, calls = _run_clone_step(tmp_path, only_others)
    assert done.returncode == 1
    assert not any(c.startswith("clone") for c in calls)
    assert "could not resolve and clone omniintelligence's newest" in done.stdout


def test_clone_step_never_reads_a_branch_tip_or_a_pin() -> None:
    run = str(_step("Clone omniintelligence")["run"])
    assert "git ls-remote --tags --refs" in run
    assert "'refs/tags/dev-*'" in run
    assert not re.search(r"--branch\s+(dev|main)\b", run)
    assert not re.search(r"checkout\s+(--quiet\s+)?(origin/)?(dev|main)\b", run)
    assert not re.search(r"\b[0-9a-f]{40}\b", run)
    assert "sibling-pins.yaml" not in run
    # Positive control: the checker's branch-tip pattern does catch a branch read.
    assert re.search(r"--branch\s+(dev|main)\b", "git clone --branch dev url dir")


def test_claude_key_reaches_only_the_review_step() -> None:
    holders = [
        s["name"]
        for s in _steps()
        if "HOSTILE_REVIEW_ANTHROPIC_API_KEY" in yaml.safe_dump(s)
    ]
    assert holders == ["Run adversarial review"]
    env = _step("Run adversarial review")["env"]
    for name in _KEY_ENV_NAMES:
        assert env[name] == _KEY_EXPR
    # Positive control: the preflight holds the other secret and not this one.
    preflight = _step("Preflight — probe review endpoint reachability")["env"]
    assert "LOCAL_LLM_SHARED_SECRET" in preflight
    assert not set(_KEY_ENV_NAMES) & set(preflight)


def test_key_is_a_secret_expression_never_a_literal() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert not re.search(r"sk-ant-[A-Za-z0-9_-]+", text)
    lines = [
        ln
        for ln in text.splitlines()
        if re.match(r"\s+(REVIEW_)?ANTHROPIC_API_KEY:", ln)
    ]
    assert len(lines) == len(_KEY_ENV_NAMES)
    for line in lines:
        assert line.strip().endswith(_KEY_EXPR), line


def test_dev_tag_and_key_tests_run_as_ci_step_and_hook() -> None:
    """Paired with the review-voter-model-from-overlay hook and lint step."""
    this = "tests/ci/test_hostile_reviewer_dev_tag_key_omn20910.py"
    root = _WORKFLOW.parents[2]
    ci = yaml.safe_load((root / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    assert any(this in str(s.get("run", "")) for s in ci["jobs"]["lint"]["steps"])
    repos = yaml.safe_load((root / ".pre-commit-config.yaml").read_text("utf-8"))
    hooks = [h for repo in repos["repos"] for h in repo["hooks"]]
    wired = [h for h in hooks if this in str(h.get("entry", ""))]
    assert wired
    assert all(re.search(h["files"], this) for h in wired)
