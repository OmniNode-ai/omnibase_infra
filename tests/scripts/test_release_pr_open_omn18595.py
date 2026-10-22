# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The release train rebuilds an open release PR whose candidate is superseded.

The nightly cut job used to skip whenever ANY release PR was open for the
branch, without reading which candidate that PR was cut from. omnibase_infra#5256
(0.38.76, cut from candidate 39e52ccb2078677a6755450c7f3c44d32c862a8f) was
therefore never replaced after dev moved on, although it could not go green at
its own candidate. The decision now compares the PR body's recorded candidate
with the run's candidate (OMN-18595, AC5-AC8).

Everything here runs on fakes: no network, no gh. The pure decision is tested
directly, and the shipped workflow step is executed in two halves under a fake
``gh`` and a fake ``git`` -- the part before branch creation (what the step
decides) and the part after the commit (what it pushes and which PR write it
makes) -- so the lease push and the PR edit are observed, not read off the YAML.

Test names carry the falsifier selector of the acceptance criterion they bind:
``release_pr_refresh_superseded_candidate``, ``release_pr_same_candidate_noop``,
``release_pr_orphan_branch_rebuilt`` and ``release_pr_unreadable_candidate_refuses``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"

# The incident's recorded values: omnibase_infra#5256's candidate, and the dev
# head at the time this was written.
_INCIDENT_PR = 5256
_INCIDENT_OLD_CANDIDATE = "39e52ccb2078677a6755450c7f3c44d32c862a8f"
_DEV_HEAD = "ef393dd930631b1d4a9f4da90b5b85ea78d0a148"
_OTHER_SHA = "a" * 40
_BRANCH_SHA = "b" * 40
_BRANCH = "automation/release-0.38.76-omn-18595"


def _load() -> Any:
    name = "release_train_under_test_release_pr"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:
        msg = f"cannot load {_MODULE}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


def _body(candidate_line: str) -> str:
    """A PR body in the shape the workflow writes, with the candidate line given."""
    return (
        "Opened by the scheduled release train (OMN-18595).\n\n"
        "- Repository: `OmniNode-ai/omnibase_infra`\n"
        f"{candidate_line}\n"
        "- Version: `0.38.76`\n"
    )


def _pr(number: int, candidate: str) -> Any:
    return rt.ModelOpenReleasePr(
        number=number, body=_body(f"- Candidate commit: `{candidate}`")
    )


class TestReleasePrRefreshSupersededCandidate:
    """AC5: a superseded candidate rebuilds the release branch and the PR body."""

    def test_release_pr_refresh_superseded_candidate(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_OTHER_SHA,
            open_prs=[_pr(7, _INCIDENT_OLD_CANDIDATE)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFRESH
        assert plan.pr_number == 7
        assert plan.previous_candidate == _INCIDENT_OLD_CANDIDATE
        assert plan.lease_sha == _BRANCH_SHA

    def test_release_pr_refresh_superseded_candidate_incident_5256(self) -> None:
        """Red-first, from the incident's recorded values.

        The logic this replaced exited 0 on any open PR, so it could not return
        refresh for #5256 whatever the decision's candidate was.
        """
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD,
            open_prs=[_pr(_INCIDENT_PR, _INCIDENT_OLD_CANDIDATE)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFRESH
        assert plan.pr_number == _INCIDENT_PR
        assert plan.previous_candidate == _INCIDENT_OLD_CANDIDATE

    def test_release_pr_refresh_superseded_candidate_uppercase_body_sha(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_OTHER_SHA,
            open_prs=[_pr(7, _INCIDENT_OLD_CANDIDATE.upper())],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFRESH
        assert plan.previous_candidate == _INCIDENT_OLD_CANDIDATE

    def test_release_pr_refresh_superseded_candidate_shell_rebuilds_and_edits(
        self, tmp_path: Path
    ) -> None:
        head = _run_step_head(
            tmp_path,
            candidate=_DEV_HEAD,
            prs=[{"number": _INCIDENT_PR, "body": _body(_candidate_line_old())}],
            remote_sha=_BRANCH_SHA,
        )
        assert head.result.returncode == 0, head.result.stderr
        assert "REACHED_BRANCH_CREATION" in head.result.stdout
        assert "refresh:" in head.result.stdout

        tail = _run_step_tail(
            tmp_path,
            action="refresh",
            pr_number=str(_INCIDENT_PR),
            remote_sha=_BRANCH_SHA,
        )
        assert tail.result.returncode == 0, tail.result.stderr
        assert [
            "push",
            f"--force-with-lease={_BRANCH}:{_BRANCH_SHA}",
            "origin",
            _BRANCH,
        ] in tail.git_calls
        edit = _gh_call(tail, "pr", "edit")
        assert edit[2] == str(_INCIDENT_PR)
        assert f"- Candidate commit: `{_DEV_HEAD}`" in edit[edit.index("--body") + 1]
        assert _gh_call(tail, "pr", "create", required=False) is None


class TestReleasePrSameCandidateNoop:
    """AC6: the same candidate pushes nothing and edits nothing."""

    def test_release_pr_same_candidate_noop(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_INCIDENT_OLD_CANDIDATE,
            open_prs=[_pr(_INCIDENT_PR, _INCIDENT_OLD_CANDIDATE)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.NOOP
        assert plan.pr_number == _INCIDENT_PR
        assert "already open" in plan.reason

    def test_release_pr_same_candidate_noop_case_insensitive(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_INCIDENT_OLD_CANDIDATE.upper(),
            open_prs=[_pr(_INCIDENT_PR, _INCIDENT_OLD_CANDIDATE)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.NOOP

    def test_release_pr_same_candidate_noop_shell_stops_before_any_write(
        self, tmp_path: Path
    ) -> None:
        head = _run_step_head(
            tmp_path,
            candidate=_INCIDENT_OLD_CANDIDATE,
            prs=[{"number": _INCIDENT_PR, "body": _body(_candidate_line_old())}],
            remote_sha=_BRANCH_SHA,
        )
        assert head.result.returncode == 0, head.result.stderr
        assert "already open" in head.result.stdout
        assert "REACHED_BRANCH_CREATION" not in head.result.stdout
        assert head.git_calls == [
            ["ls-remote", "--heads", "origin", f"refs/heads/{_BRANCH}"]
        ]
        assert [c[:2] for c in head.gh_calls] == [["pr", "list"]]


class TestReleasePrOrphanBranchRebuilt:
    """AC7: a branch on the remote with no open PR is rebuilt with a lease push."""

    def test_release_pr_orphan_branch_rebuilt(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD, open_prs=[], remote_branch_sha=_BRANCH_SHA
        )
        assert plan.action is rt.EnumReleasePrAction.OPEN
        assert plan.pr_number is None
        assert plan.lease_sha == _BRANCH_SHA

    def test_release_pr_orphan_branch_rebuilt_absent_branch_is_a_plain_open(
        self,
    ) -> None:
        """The discriminating half: no remote branch means no lease."""
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD, open_prs=[], remote_branch_sha=""
        )
        assert plan.action is rt.EnumReleasePrAction.OPEN
        assert plan.lease_sha == ""

    def test_release_pr_orphan_branch_rebuilt_shell_lease_pushes_and_creates(
        self, tmp_path: Path
    ) -> None:
        head = _run_step_head(
            tmp_path, candidate=_DEV_HEAD, prs=[], remote_sha=_BRANCH_SHA
        )
        assert head.result.returncode == 0, head.result.stderr
        assert "REACHED_BRANCH_CREATION" in head.result.stdout
        assert "open:" in head.result.stdout

        tail = _run_step_tail(
            tmp_path, action="open", pr_number="", remote_sha=_BRANCH_SHA
        )
        assert tail.result.returncode == 0, tail.result.stderr
        assert [
            "push",
            f"--force-with-lease={_BRANCH}:{_BRANCH_SHA}",
            "origin",
            _BRANCH,
        ] in tail.git_calls
        create = _gh_call(tail, "pr", "create")
        assert create[create.index("--head") + 1] == _BRANCH
        assert _gh_call(tail, "pr", "edit", required=False) is None

    def test_release_pr_orphan_branch_rebuilt_shell_clean_open_is_a_plain_push(
        self, tmp_path: Path
    ) -> None:
        head = _run_step_head(tmp_path, candidate=_DEV_HEAD, prs=[], remote_sha="")
        assert "REACHED_BRANCH_CREATION" in head.result.stdout
        tail = _run_step_tail(tmp_path, action="open", pr_number="", remote_sha="")
        assert tail.result.returncode == 0, tail.result.stderr
        assert ["push", "origin", _BRANCH] in tail.git_calls
        assert not any(
            arg.startswith("--force") for call in tail.git_calls for arg in call
        )


class TestReleasePrUnreadableCandidateRefuses:
    """AC8: a body with no readable candidate is refused and never force-pushed."""

    @pytest.mark.parametrize(
        "candidate_line",
        [
            pytest.param("", id="no-candidate-line"),
            pytest.param("- Candidate commit: `39e52ccb2`", id="short-sha"),
            pytest.param(
                "- Candidate commit: `not-a-sha-not-a-sha-not-a-sha-not-a-sha-`",
                id="non-hex",
            ),
            pytest.param(f"- Candidate commit: {_OTHER_SHA}", id="no-backticks"),
            pytest.param(f"- Candidate commit: `{_OTHER_SHA}0`", id="41-hex"),
            pytest.param("- Candidate commit: ``", id="empty-value"),
        ],
    )
    def test_release_pr_unreadable_candidate_refuses(self, candidate_line: str) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD,
            open_prs=[rt.ModelOpenReleasePr(number=9, body=_body(candidate_line))],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFUSE
        assert "#9" in plan.reason
        assert "Candidate commit" in plan.reason
        assert plan.lease_sha == ""

    def test_release_pr_unreadable_candidate_refuses_two_different_candidates(
        self,
    ) -> None:
        body = _body(
            f"- Candidate commit: `{_OTHER_SHA}`\n"
            f"- Candidate commit: `{_INCIDENT_OLD_CANDIDATE}`"
        )
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD,
            open_prs=[rt.ModelOpenReleasePr(number=9, body=body)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFUSE

    def test_release_pr_unreadable_candidate_refuses_several_open_prs(self) -> None:
        plan = rt.decide_release_pr(
            candidate_sha=_DEV_HEAD,
            open_prs=[_pr(1, _OTHER_SHA), _pr(2, _OTHER_SHA)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFUSE
        assert "#1" in plan.reason
        assert "#2" in plan.reason

    def test_release_pr_unreadable_candidate_refuses_a_malformed_decision_sha(
        self,
    ) -> None:
        plan = rt.decide_release_pr(
            candidate_sha="39e52ccb2",
            open_prs=[_pr(1, _OTHER_SHA)],
            remote_branch_sha=_BRANCH_SHA,
        )
        assert plan.action is rt.EnumReleasePrAction.REFUSE

    def test_release_pr_unreadable_candidate_refuses_shell_fails_the_step(
        self, tmp_path: Path
    ) -> None:
        head = _run_step_head(
            tmp_path,
            candidate=_DEV_HEAD,
            prs=[{"number": 9, "body": "Opened by hand, no candidate line."}],
            remote_sha=_BRANCH_SHA,
        )
        assert head.result.returncode == 1
        assert "::error::" in head.result.stdout
        assert "#9" in head.result.stdout
        assert "REACHED_BRANCH_CREATION" not in head.result.stdout
        assert head.git_calls == [
            ["ls-remote", "--heads", "origin", f"refs/heads/{_BRANCH}"]
        ]


class TestTheWorkflowWritesWhatTheDecisionReads:
    """The body template and the parser share one shape, so a PR the train opens
    is always readable by the next run."""

    def test_the_body_template_parses_back_to_its_candidate(self) -> None:
        tail = _step_run().split("# publish the rebuilt branch", 1)[1]
        template_line = next(
            line.strip()
            for line in tail.splitlines()
            if line.strip().startswith("- Candidate commit:")
        )
        assert template_line == "- Candidate commit: \\`${CANDIDATE_SHA}\\`"
        rendered = template_line.replace("\\`", "`").replace(
            "${CANDIDATE_SHA}", _DEV_HEAD
        )
        assert rt.parse_recorded_candidate(rendered) == _DEV_HEAD

    def test_a_refresh_rewrites_the_body_and_never_the_title(self) -> None:
        run = _step_run()
        assert 'gh pr edit "${pr_number}"' in run
        edit = run.split('gh pr edit "${pr_number}"', 1)[1].split("else", 1)[0]
        assert "--title" not in edit

    def test_there_is_no_force_or_skip_input(self) -> None:
        parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        raw = parsed.get(True, parsed.get("on"))
        inputs = set(raw["workflow_dispatch"].get("inputs", {}))
        assert not {i for i in inputs if "force" in i or "skip" in i}

    def test_the_only_force_push_is_the_leased_one(self) -> None:
        run = _step_run()
        assert run.count("--force-with-lease=") == 1
        assert "--force " not in run
        assert " -f " not in run


# --------------------------------------------------------------------------- #
# Running the shipped step under fakes.                                         #
# --------------------------------------------------------------------------- #
class _Ran:
    def __init__(self, result: subprocess.CompletedProcess[str], log_dir: Path) -> None:
        self.result = result
        self.git_calls = _read_calls(log_dir / "git.log")
        self.gh_calls = _read_calls(log_dir / "gh.log")


def _read_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _gh_call(ran: _Ran, *verb: str, required: bool = True) -> Any:
    for call in ran.gh_calls:
        if tuple(call[: len(verb)]) == verb:
            return call
    if required:
        raise AssertionError(f"no gh {' '.join(verb)} call in {ran.gh_calls}")
    return None


def _candidate_line_old() -> str:
    return f"- Candidate commit: `{_INCIDENT_OLD_CANDIDATE}`"


def _step_run() -> str:
    parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = [
        step
        for step in parsed["jobs"]["cut"]["steps"]
        if step.get("name") == "Open the release pull request"
    ]
    assert len(steps) == 1, "the cut job must declare exactly one release PR step"
    run: str = steps[0]["run"]
    return run


def _write_fakes(bin_dir: Path) -> None:
    """A fake gh and git that log their argv as JSON and answer from env vars.

    ``gh pr list`` with ``--jq length`` answers the open count and without it the
    JSON list, so the same fake drives the shipped step and the logic it replaced.
    """
    bin_dir.mkdir()
    fake_py = (
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "tool = os.path.basename(sys.argv[0])\n"
        "args = sys.argv[1:]\n"
        "with open(os.path.join(os.environ['FAKE_LOG_DIR'], tool + '.log'), 'a') as f:\n"
        "    f.write(json.dumps(args) + '\\n')\n"
        "if tool == 'gh' and args[:2] == ['pr', 'list']:\n"
        "    prs = json.loads(os.environ['FAKE_OPEN_PRS'])\n"
        "    print(len(prs) if '--jq' in args else json.dumps(prs))\n"
        "elif tool == 'git' and args[:1] == ['ls-remote']:\n"
        "    sha = os.environ.get('FAKE_REMOTE_SHA', '')\n"
        "    if sha:\n"
        "        print(sha + '\\t' + args[-1])\n"
        "elif tool == 'git' and args[:1] == ['push']:\n"
        "    pass\n"
        "elif tool == 'gh' and args[:1] == ['pr']:\n"
        "    pass\n"
        "else:\n"
        "    sys.exit(1)\n"
    )
    for tool in ("gh", "git"):
        script = bin_dir / tool
        script.write_text(fake_py, encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _base_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    train = tmp_path / "_train"
    if not train.exists():
        train.symlink_to(_REPO_ROOT)
    (tmp_path / "_target").mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        _write_fakes(bin_dir)
    return {
        **os.environ,
        "FAKE_LOG_DIR": str(tmp_path),
        "GH_TOKEN": "unused",
        "TARGET_REPO": "omnibase_infra",
        "VERSION": "0.38.76",
        "NEEDS_BUMP": "false",
        "BASE_BRANCH": "dev",
        "TICKET": "OMN-18595",
        "GITHUB_WORKSPACE": str(tmp_path),
        "GITHUB_WORKFLOW": "release-train-nightly",
        "GITHUB_SERVER_URL": "https://example.invalid",
        "GITHUB_REPOSITORY": "OmniNode-ai/omnibase_infra",
        "GITHUB_RUN_ID": "1",
        "RUNNER_TEMP": str(tmp_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        **extra,
    }


def _run_step_head(
    tmp_path: Path, *, candidate: str, prs: list[dict[str, Any]], remote_sha: str
) -> _Ran:
    """Run the shipped step up to branch creation; a sentinel marks reaching it."""
    run = _step_run()
    boundary = 'git checkout -b "${branch}"'
    assert run.count(boundary) == 1, "branch creation must appear exactly once"
    script = run.split(boundary, 1)[0] + "\necho REACHED_BRANCH_CREATION\n"
    env = _base_env(
        tmp_path,
        CANDIDATE_SHA=candidate,
        FAKE_OPEN_PRS=json.dumps(prs),
        FAKE_REMOTE_SHA=remote_sha,
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return _Ran(result, tmp_path)


def _run_step_tail(
    tmp_path: Path, *, action: str, pr_number: str, remote_sha: str
) -> _Ran:
    """Run the shipped step from the publish marker to its end."""
    run = _step_run()
    marker = "# publish the rebuilt branch"
    assert run.count(marker) == 1
    # Fresh logs: the head half's calls are asserted on separately.
    for name in ("git.log", "gh.log"):
        (tmp_path / name).unlink(missing_ok=True)
    prelude = (
        "set -euo pipefail\n"
        f"branch={_BRANCH}\n"
        f"action={action}\n"
        f"pr_number={pr_number}\n"
        f"remote_sha={remote_sha}\n"
    )
    script = prelude + marker + run.split(marker, 1)[1]
    env = _base_env(tmp_path, CANDIDATE_SHA=_DEV_HEAD)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path / "_target",
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return _Ran(result, tmp_path)
