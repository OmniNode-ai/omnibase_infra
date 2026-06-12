# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/preflight_hotpatch_ledger.py (OMN-13014, retro B-1)."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "preflight_hotpatch_ledger.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "preflight_hotpatch_ledger", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return result.stdout.strip()


def _make_repo(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "dev")
    (repo / "f.txt").write_text("one\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "c1")
    return repo


def _commit(repo: Path, content: str) -> str:
    (repo / "f.txt").write_text(content)
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", content)
    return _git(repo, "rev-parse", "HEAD")


def _write_ledger(
    path: Path,
    rows: list[dict[str, object]],
    schema: int = 1,
) -> Path:
    path.write_text(yaml.safe_dump({"schema": schema, "rows": rows}))
    return path


def _row(
    container: str,
    repo: str,
    commit: str,
    *,
    lane: str = "stability-test",
    file: str = "/app/x.py",
    prepatch: str = "/app/x.py.prepatch",
) -> dict[str, object]:
    return {
        "container": container,
        "lane": lane,
        "file": file,
        "prepatch_path": prepatch,
        "source_repo": repo,
        "source_pr": f"OmniNode-ai/{repo}#1",
        "merge_commit": commit,
        "merged": True,
    }


def _fake_docker(bin_dir: Path, prepatch_lines: list[str], rc: int = 0) -> str:
    """Create a stub docker executable whose `exec` prints the given paths."""
    script = bin_dir / "docker"
    body = "\n".join(prepatch_lines)
    script.write_text(
        f'#!/bin/sh\nif [ "$1" = "exec" ]; then\n'
        f'cat <<"EOF"\n{body}\nEOF\nexit {rc}\nfi\nexit 0\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.mark.unit
class TestAncestorGate:
    def test_pass_when_merge_commit_in_build_ref(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path, "omnimarket")
        sha = _commit(repo, "two")
        ledger = _write_ledger(
            tmp_path / "ledger.yaml", [_row("c1", "omnimarket", sha)]
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                f"omnimarket={_git(repo, 'rev-parse', 'HEAD')}",
                "--skip-tripwire",
            ]
        )
        assert rc == 0

    def test_fail_when_merge_commit_not_in_build_ref(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path, "omnimarket")
        base = _git(repo, "rev-parse", "HEAD")
        unmerged = _commit(repo, "two")  # build ref = base, patch commit after it
        ledger = _write_ledger(
            tmp_path / "ledger.yaml", [_row("c1", "omnimarket", unmerged)]
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                f"omnimarket={base}",
                "--skip-tripwire",
            ]
        )
        assert rc == 1

    def test_unknown_commit_is_config_error(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path, "omnimarket")
        ledger = _write_ledger(
            tmp_path / "ledger.yaml",
            [_row("c1", "omnimarket", "0" * 40)],
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                f"omnimarket={_git(repo, 'rev-parse', 'HEAD')}",
                "--skip-tripwire",
            ]
        )
        assert rc == 2

    def test_build_ref_defaults_to_clone_head(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path, "omnimarket")
        sha = _commit(repo, "two")
        ledger = _write_ledger(
            tmp_path / "ledger.yaml", [_row("c1", "omnimarket", sha)]
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--skip-tripwire",
            ]
        )
        assert rc == 0

    def test_lane_scope_selects_rows(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path, "omnimarket")
        base = _git(repo, "rev-parse", "HEAD")
        unmerged = _commit(repo, "two")
        ledger = _write_ledger(
            tmp_path / "ledger.yaml",
            [
                _row("c1", "omnimarket", unmerged, lane="stability-test"),
                _row("c2", "omnimarket", base, lane="prod"),
            ],
        )
        # prod lane only sees the merged row -> pass
        rc = MODULE.main(
            [
                "--lane",
                "prod",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                f"omnimarket={base}",
                "--skip-tripwire",
            ]
        )
        assert rc == 0
        # stability lane sees the unmerged row -> fail
        rc = MODULE.main(
            [
                "--lane",
                "stability-test",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                f"omnimarket={base}",
                "--skip-tripwire",
            ]
        )
        assert rc == 1


@pytest.mark.unit
class TestLedgerLoading:
    def test_missing_ledger_is_config_error(self, tmp_path: Path) -> None:
        _make_repo(tmp_path, "omnimarket")
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(tmp_path / "nope.yaml"),
                "--skip-tripwire",
            ]
        )
        assert rc == 2

    def test_unsupported_schema_is_config_error(self, tmp_path: Path) -> None:
        _make_repo(tmp_path, "omnimarket")
        ledger = _write_ledger(tmp_path / "ledger.yaml", [], schema=99)
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--skip-tripwire",
            ]
        )
        assert rc == 2

    def test_empty_rows_pass(self, tmp_path: Path) -> None:
        _make_repo(tmp_path, "omnimarket")
        ledger = _write_ledger(tmp_path / "ledger.yaml", [])
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--skip-tripwire",
            ]
        )
        assert rc == 0


@pytest.mark.unit
class TestTripwire:
    def _ledger_with_merged_row(
        self, tmp_path: Path, prepatch: str
    ) -> tuple[Path, str]:
        repo = _make_repo(tmp_path, "omnimarket")
        sha = _commit(repo, "two")
        ledger = _write_ledger(
            tmp_path / "ledger.yaml",
            [_row("c1", "omnimarket", sha, prepatch=prepatch)],
        )
        return ledger, sha

    def test_ledgered_prepatch_passes_pre_rebuild(self, tmp_path: Path) -> None:
        ledger, _ = self._ledger_with_merged_row(tmp_path, "/app/x.py.prepatch")
        docker = _fake_docker(tmp_path, ["/app/x.py.prepatch"])
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--docker-cmd",
                docker,
            ]
        )
        assert rc == 0

    def test_unledgered_prepatch_fails(self, tmp_path: Path) -> None:
        ledger, _ = self._ledger_with_merged_row(tmp_path, "/app/x.py.prepatch")
        docker = _fake_docker(
            tmp_path, ["/app/x.py.prepatch", "/app/rogue.py.prepatch"]
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--docker-cmd",
                docker,
            ]
        )
        assert rc == 1

    def test_post_rebuild_fails_on_any_prepatch(self, tmp_path: Path) -> None:
        ledger, _ = self._ledger_with_merged_row(tmp_path, "/app/x.py.prepatch")
        docker = _fake_docker(tmp_path, ["/app/x.py.prepatch"])
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--docker-cmd",
                docker,
                "--post-rebuild",
            ]
        )
        assert rc == 1

    def test_post_rebuild_passes_when_clean(self, tmp_path: Path) -> None:
        ledger, _ = self._ledger_with_merged_row(tmp_path, "/app/x.py.prepatch")
        docker = _fake_docker(tmp_path, [])
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--docker-cmd",
                docker,
                "--post-rebuild",
            ]
        )
        assert rc == 0

    def test_exec_failure_fails_gate(self, tmp_path: Path) -> None:
        ledger, _ = self._ledger_with_merged_row(tmp_path, "/app/x.py.prepatch")
        docker = _fake_docker(tmp_path, [], rc=1)
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--docker-cmd",
                docker,
            ]
        )
        assert rc == 1


@pytest.mark.unit
class TestBypass:
    def test_valid_receipt_bypasses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "HOTPATCH_PREFLIGHT_BYPASS", "# skip-token-allowed: receipt-abc-123"
        )
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(tmp_path / "absent.yaml"),
            ]
        )
        assert rc == 0

    def test_malformed_bypass_is_hard_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOTPATCH_PREFLIGHT_BYPASS", "because I said so")
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(tmp_path / "absent.yaml"),
            ]
        )
        assert rc == 2

    def test_no_bypass_var_runs_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without bypass, missing ledger is a config error (proves gate ran).
        monkeypatch.delenv("HOTPATCH_PREFLIGHT_BYPASS", raising=False)
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(tmp_path / "absent.yaml"),
            ]
        )
        assert rc == 2


@pytest.mark.unit
class TestBuildRefParsing:
    def test_malformed_build_ref_is_config_error(self, tmp_path: Path) -> None:
        _make_repo(tmp_path, "omnimarket")
        ledger = _write_ledger(tmp_path / "ledger.yaml", [])
        rc = MODULE.main(
            [
                "--container",
                "c1",
                "--clones-root",
                str(tmp_path),
                "--ledger",
                str(ledger),
                "--build-ref",
                "missing-equals",
                "--skip-tripwire",
            ]
        )
        assert rc == 2
