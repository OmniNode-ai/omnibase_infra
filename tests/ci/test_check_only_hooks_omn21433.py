# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""T2.6 canary: the pre-commit hooks that rewrote files are check-only (OMN-21433).

A hook that edits the working tree during a commit races the stash/restore step
and hides the defect it fixed. Each hook below now refuses and leaves every byte
of the tree as it found it. The behavioural tests run the repo's own hook
definition through ``pre-commit`` against a throwaway git repository holding the
defect, then compare the tree byte for byte before and after the run. Every
refusal has a positive control: a clean input the same hook accepts, so a hook
that refuses everything cannot pass.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

# Hook ids that wrote to the tree before this change; none may come back.
RETIRED_WRITING_IDS = (
    "end-of-file-fixer",
    "clean-tmp-directory",
    "strip-ai-coauthor-trailer",
)
# ruff keeps its ids: architecture-handshakes/validator-requirements.yaml names them.
RUFF_FORMAT_ID = "ruff-format"
RUFF_CHECK_ID = "ruff-fix"
EOF_ID = "check-end-of-file"
TMP_ID = "refuse-staged-tmp-files"
TRAILER_ID = "refuse-ai-coauthor-trailer"
SELF_TEST_ID = "check-no-credential-in-log-self-test"

WRITING_TOKENS = frozenset(
    {"--fix", "--exit-non-zero-on-fix", "rm", "perl", "sed", "-i", "-pi"}
)


def _hooks() -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    found: dict[str, dict[str, Any]] = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            found[hook["id"]] = {**hook, "_repo": repo["repo"]}
    return found


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _local_hook(hook_id: str) -> dict[str, Any]:
    """The repo's hook definition, with ``uv run`` swapped for this interpreter."""
    hook = {k: v for k, v in _hooks()[hook_id].items() if k != "_repo"}
    entry = str(hook["entry"])
    if entry.startswith("uv run --frozen ruff"):
        python = shlex.quote(sys.executable)
        entry = entry.replace("uv run --frozen ruff", f"{python} -m ruff", 1)
    hook["entry"] = entry
    return hook


def _run_hook(
    tmp_path: Path,
    hook_id: str,
    files: dict[str, str],
    *,
    commit_msg: str | None = None,
) -> tuple[int, str, dict[str, str], dict[str, str]]:
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=work,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    for name, text in files.items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode())
    subprocess.run(
        ["git", "add", "-A"],
        cwd=work,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    config = tmp_path / "hook-config.yaml"
    config.write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [_local_hook(hook_id)]}]}),
        encoding="utf-8",
    )
    command = ["pre-commit", "run", hook_id, "-c", str(config)]
    if commit_msg is None:
        command += ["--files", *files]
    else:
        msg_file = work / "COMMIT_EDITMSG_TEST"
        msg_file.write_bytes(commit_msg.encode())
        command += [
            "--hook-stage",
            "commit-msg",
            "--commit-msg-filename",
            str(msg_file),
        ]
    before = _snapshot(work)
    # No `check=`: the exit code is the observation under test.
    done = subprocess.run(
        command,
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": _tool_path(),
            "HOME": str(tmp_path),
            "PRE_COMMIT_HOME": str(tmp_path / "pc"),
            "RUFF_CACHE_DIR": str(tmp_path / "ruff-cache"),
        },
    )
    return done.returncode, done.stdout + done.stderr, before, _snapshot(work)


def _tool_path() -> str:
    return f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"


class TestConfigShape:
    def test_retired_writing_ids_are_gone(self) -> None:
        present = [h for h in RETIRED_WRITING_IDS if h in _hooks()]
        assert present == []

    @pytest.mark.parametrize(
        "hook_id",
        [RUFF_FORMAT_ID, RUFF_CHECK_ID, EOF_ID, TMP_ID, TRAILER_ID, SELF_TEST_ID],
    )
    def test_hook_entry_cannot_write(self, hook_id: str) -> None:
        hook = _hooks()[hook_id]
        text = " ".join([str(hook["entry"]), *map(str, hook.get("args", []))])
        assert WRITING_TOKENS.intersection(shlex.split(text)) == set()

    def test_ruff_format_is_check_mode(self) -> None:
        assert "--check" in str(_hooks()[RUFF_FORMAT_ID]["entry"]).split()

    def test_ruff_check_is_no_fix_mode(self) -> None:
        assert "--no-fix" in str(_hooks()[RUFF_CHECK_ID]["entry"]).split()


class TestRuffFormat:
    def test_unformatted_file_is_refused_and_untouched(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, RUFF_FORMAT_ID, {"sample.py": "x   =   1\n"}
        )
        assert rc != 0, out
        assert before == after

    def test_formatted_file_passes(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, RUFF_FORMAT_ID, {"sample.py": "x = 1\n"}
        )
        assert rc == 0, out
        assert before == after


class TestRuffCheck:
    def test_fixable_lint_is_refused_and_untouched(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, RUFF_CHECK_ID, {"sample.py": "import os\n"}
        )
        assert rc != 0, out
        assert before == after

    def test_clean_file_passes(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, RUFF_CHECK_ID, {"sample.py": "x = 1\n"}
        )
        assert rc == 0, out
        assert before == after


class TestEndOfFile:
    @pytest.mark.parametrize("text", ["no newline", "two newlines\n\n", "a\n\n\n"])
    def test_bad_ending_is_refused_and_untouched(
        self, tmp_path: Path, text: str
    ) -> None:
        rc, out, before, after = _run_hook(tmp_path, EOF_ID, {"notes.txt": text})
        assert rc != 0, out
        assert before == after

    @pytest.mark.parametrize("text", ["one newline\n", "", "multi\nline\n"])
    def test_good_ending_passes(self, tmp_path: Path, text: str) -> None:
        rc, out, before, after = _run_hook(tmp_path, EOF_ID, {"notes.txt": text})
        assert rc == 0, out
        assert before == after

    def test_captured_artifacts_stay_excluded(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, EOF_ID, {"fixtures/body.captured": "verbatim, no newline"}
        )
        assert rc == 0, out
        assert before == after


class TestStagedTmp:
    def test_staged_tmp_file_is_refused_and_kept(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, TMP_ID, {"tmp/scratch.txt": "scratch\n"}
        )
        assert rc != 0, out
        assert before == after
        assert "tmp/scratch.txt" in before

    def test_other_paths_pass(self, tmp_path: Path) -> None:
        rc, out, before, after = _run_hook(
            tmp_path, TMP_ID, {"docs/tmp/keep.md": "keep\n"}
        )
        assert rc == 0, out
        assert before == after


class TestCoAuthorTrailer:
    def test_ai_trailer_is_refused_and_message_untouched(self, tmp_path: Path) -> None:
        message = (
            "feat(OMN-21433): subject\n\nbody\n\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>\n"
        )
        rc, out, before, after = _run_hook(
            tmp_path, TRAILER_ID, {"keep.txt": "x\n"}, commit_msg=message
        )
        assert rc != 0, out
        assert before == after

    def test_human_trailer_passes(self, tmp_path: Path) -> None:
        message = (
            "feat(OMN-21433): subject\n\nCo-authored-by: A Person <a@example.com>\n"
        )
        rc, out, before, after = _run_hook(
            tmp_path, TRAILER_ID, {"keep.txt": "x\n"}, commit_msg=message
        )
        assert rc == 0, out
        assert before == after


class TestCredentialSelfTestNeverWrote:
    """The plan lists this hook as REWRITES FILES; its entry only reads a fixture."""

    def test_running_it_leaves_the_checkout_byte_identical(self) -> None:
        hook = _hooks()[SELF_TEST_ID]
        argv = shlex.split(
            str(hook["entry"]).replace("uv run --frozen python", sys.executable, 1)
        )
        fixture = REPO_ROOT / "scripts/ci/tests/fixtures/credential_in_log_fixture.py"
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        done = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == digest
