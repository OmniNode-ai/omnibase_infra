# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replays for the Lint job's one-process runner and diff scope (OMN-19613).

Both scripts change HOW a guard runs, not what it decides, so the way either
can fail is by making a real violation invisible: a false green. Each case
drives the real script over bytes captured from this repository's own history,
not a reconstruction.

CASE 1, the one-process runner, replays OMN-14141. The flat
``handler_class``/``handler_module`` routing schema parsed to zero dispatchers
and phantom-wired the subscribed topic while the node reported WIRED and
committed offsets: the WI-14 root cause. ``node_session_state_effect`` carried
that shape until 340c539c. The artifact is its contract.yaml at the parent of
that fix. Bundling fourteen validators into one process must not let a red one
be swallowed by the greens around it, so the replay places the flat-schema gate
FIRST, follows it with two gates that pass on this repository's own tree, and
requires the whole step to fail, with an error annotation naming that gate.
The discriminator drives the identical list over the same contract as fixed in
340c539c and requires success, so a runner that fails everything cannot
satisfy the case.

CASE 2, the diff scope, replays OMN-10122. ``PluginEmitDaemon`` was a Plugin*
class that owned a daemon lifecycle; it was added by OMN-7640, had to be
deleted by OMN-10122 and is the reason the Plugin* guard exists (OMN-10123).
The artifact is that file at the parent of the deletion. A PR adding it is
merged onto a base that has moved on, exactly as GitHub's pull_request checkout
is, and the scope the real script chooses must still hand this file to the
real guard, which must reject it. A scope that diffed the wrong range (the
second parent, or the base branch's own new commits) would drop the file and
the guard would say OK. The discriminator is a PR that touches only an
unrelated file: the same scope must leave the captured daemon out of the range.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from omnibase_core.validators import no_plugin_daemon_classes
from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.ci import precommit_diff_scope as scope_mod
from scripts.ci import run_validators_in_process as runner

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19613"
FLAT_CONTRACT = FIXTURES / "node_session_state_effect.contract.yaml.flat.captured"
NESTED_CONTRACT = FIXTURES / "node_session_state_effect.contract.yaml.nested.captured"
EMIT_DAEMON = FIXTURES / "plugin_emit_daemon.py.captured"

SHA256 = {
    FLAT_CONTRACT: "3a50b4edfadbc69f357a071d2b389a771e03c36dae0da02bc4d928254286cf44",
    NESTED_CONTRACT: "9d783707ca5a7a2a505c652e3422af63f5920672fa0df1c4f0926cd97246f806",
    EMIT_DAEMON: "b8eabf6d555889dec8286f0b2c95124d2d9abbe0adc44347b4aca83915e03197",
}


@pytest.mark.parametrize("fixture", sorted(SHA256, key=str))
def test_the_captured_bytes_are_unmodified(fixture: Path) -> None:
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == SHA256[fixture]


# --------------------------------------------------------------------------- #
# CASE 1: the runner still fails the step on the real OMN-14141 contract
# --------------------------------------------------------------------------- #


def _run_list(contract_bytes: bytes, tmp_path: Path) -> tuple[int, str]:
    node = tmp_path / "node_session_state_effect"
    node.mkdir()
    (node / "contract.yaml").write_bytes(contract_bytes)
    specs = runner.parse_specs(
        f"handler_routing flat-schema gate (OMN-14141) | "
        f"omnibase_infra.validators.handler_routing_schema {node}\n"
        f"Check operation_match fan-out gate (OMN-16088) | "
        f"omnibase_infra.validators.operation_match_fanout {REPO_ROOT / 'src' / 'omnibase_infra'}\n"
        f"Check no baseline refreeze (OMN-18013) | "
        f"omnibase_infra.validators.no_baseline_refreeze {REPO_ROOT}\n"
    )
    out = io.StringIO()
    results = runner.run_all(specs, stdout=out, stderr=io.StringIO())
    failed = any(r.exit_code != 0 for r in results)
    return (1 if failed else 0), out.getvalue()


def test_the_runner_fails_the_step_on_the_real_flat_schema_contract(
    tmp_path: Path,
) -> None:
    code, out = _run_list(FLAT_CONTRACT.read_bytes(), tmp_path)
    assert code == 1
    errors = [line for line in out.splitlines() if line.startswith("::error")]
    assert len(errors) == 1
    assert errors[0].startswith(
        "::error title=handler_routing flat-schema gate (OMN-14141)::"
    )
    # The gates after the red one still ran.
    assert "::group::Check no baseline refreeze (OMN-18013)" in out


def test_the_same_runner_passes_the_same_contract_once_fixed(tmp_path: Path) -> None:
    code, out = _run_list(NESTED_CONTRACT.read_bytes(), tmp_path)
    assert code == 0
    assert "::error" not in out


# --------------------------------------------------------------------------- #
# CASE 2: the diff scope still hands the real PluginEmitDaemon to the guard
# --------------------------------------------------------------------------- #


def _git_in(repo: Path) -> scope_mod.GitRunner:
    def git(args: Sequence[str]) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
        if proc.returncode != 0:
            raise scope_mod.GitError(proc.stderr)
        return proc.stdout

    return git


def _merged_pr(tmp_path: Path, pr_path: str, pr_bytes: bytes) -> Path:
    """A repo whose HEAD merges a PR onto a base that moved after it branched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _git_in(repo)
    git(["init", "-q", "-b", "dev"])
    git(["config", "user.email", "t@example.invalid"])
    git(["config", "user.name", "t"])
    (repo / "src" / "omnibase_infra" / "plugins").mkdir(parents=True)
    (repo / "src" / "omnibase_infra" / "plugins" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    git(["add", "."])
    git(["commit", "-q", "-m", "base"])
    git(["checkout", "-q", "-b", "pr"])
    target = repo / pr_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pr_bytes)
    git(["add", "."])
    git(["commit", "-q", "-m", "pr"])
    git(["checkout", "-q", "dev"])
    (repo / "src" / "omnibase_infra" / "moved.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    git(["add", "."])
    git(["commit", "-q", "-m", "base moved"])
    git(["merge", "-q", "--no-ff", "-m", "merge pr", "pr"])
    return repo


def _guard_over_scope(repo: Path) -> tuple[int, list[str]]:
    git = _git_in(repo)
    scope = scope_mod.decide("pull_request", git)
    assert scope.args[0] == "--from-ref", scope.reason
    _, base, _, head = scope.args
    # The range pre-commit computes for --from-ref/--to-ref.
    files = [
        f for f in git(["diff", "--name-only", f"{base}...{head}"]).splitlines() if f
    ]
    py_files = [repo / f for f in files if f.endswith(".py")]
    if not py_files:
        return 0, files
    findings = no_plugin_daemon_classes.validate_paths(py_files)
    return (1 if findings else 0), files


def test_the_scope_hands_the_real_plugin_emit_daemon_to_the_guard(
    tmp_path: Path,
) -> None:
    daemon = "src/omnibase_infra/plugins/plugin_emit_daemon.py"
    repo = _merged_pr(tmp_path, daemon, EMIT_DAEMON.read_bytes())
    verdict, files = _guard_over_scope(repo)
    assert daemon in files
    assert "src/omnibase_infra/moved.py" not in files, (
        "the base's own commits are not the PR"
    )
    assert verdict == 1


def test_the_same_scope_leaves_the_daemon_out_of_an_unrelated_pr(
    tmp_path: Path,
) -> None:
    repo = _merged_pr(tmp_path, "src/omnibase_infra/unrelated.py", b"y = 2\n")
    # The daemon is already on the base here, the way the whole tree is on push.
    (repo / "src" / "omnibase_infra" / "plugins" / "plugin_emit_daemon.py").write_bytes(
        EMIT_DAEMON.read_bytes()
    )
    verdict, files = _guard_over_scope(repo)
    assert files == ["src/omnibase_infra/unrelated.py"]
    assert verdict == 0
