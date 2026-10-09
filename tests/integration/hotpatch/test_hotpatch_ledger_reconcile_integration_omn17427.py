# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the reconcile node as production does: ``onex node`` on the packaged contract.

The command is the one the rebuild preflight prints when it refuses a row, so
this drives the whole loop: refuse, reconcile through the runtime, pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = REPO_ROOT / "scripts" / "preflight_hotpatch_ledger.py"
ONEX = Path(sys.executable).parent / "onex"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=t@e.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


class Host:
    """A clone holding a merged fix, a docker stub, and a ledger with one stale row."""

    def __init__(self, root: Path, *, prepatch: str = "") -> None:
        self.root = root
        self.clones = root / "clones"
        repo = self.clones / "omnibase_infra"
        repo.mkdir(parents=True)
        _git(repo, "init", "-q", "-b", "dev")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
        (repo / "f.txt").write_text("fix\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-q", "-m", "canary deadline (#4640)")
        self.fix = _git(repo, "rev-parse", "HEAD")
        self.docker = root / "docker"
        self.docker.write_text(
            f'#!/bin/sh\nif [ "$1" = "exec" ]; then printf "%s" "{prepatch}"; fi\nexit 0\n'
        )
        self.docker.chmod(0o755)
        self.ledger = root / "ledger.yaml"
        self.ledger.write_text(
            yaml.safe_dump(
                {
                    "schema": 1,
                    "rows": [
                        {
                            "container": "forwarder",
                            "lane": "dev",
                            "file": "/app/x.py",
                            "prepatch_path": "/app/x.py.prepatch",
                            "source_repo": "omnibase_infra",
                            "source_pr": "OmniNode-ai/omnibase_infra#4640",
                            "merge_commit": None,
                            "merged": False,
                        }
                    ],
                }
            )
        )

    def preflight(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PREFLIGHT),
                "--lane",
                "dev",
                "--ledger",
                str(self.ledger),
                "--clones-root",
                str(self.clones),
                "--docker-cmd",
                str(self.docker),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def reconcile(self, merge_commit: str) -> subprocess.CompletedProcess[str]:
        payload = self.root / "input.json"
        payload.write_text(
            json.dumps(
                {
                    "ledger_path": str(self.ledger),
                    "clones_root": str(self.clones),
                    "container": "forwarder",
                    "file": "/app/x.py",
                    "merge_commit": merge_commit,
                    "docker_cmd": str(self.docker),
                }
            )
        )
        return subprocess.run(
            [
                str(ONEX),
                "node",
                "node_hotpatch_ledger_reconcile_effect",
                "--input",
                str(payload),
                "--backend",
                "event_bus=inmemory",
                "--state-root",
                str(self.root / "state"),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )

    def refusal_reason(self) -> str:
        loaded = json.loads((self.root / "state" / "workflow_result.json").read_text())
        reason = loaded["handler_result"]["reason"]
        assert isinstance(reason, str)
        return reason


def test_preflight_refusal_names_a_command_that_clears_it(tmp_path: Path) -> None:
    host = Host(tmp_path)
    assert ONEX.is_file(), ONEX

    before = host.preflight()
    assert before.returncode == 2, before.stderr
    command = next(
        line.removeprefix("HOTPATCH-PREFLIGHT RECONCILE: ")
        for line in before.stderr.splitlines()
        if line.startswith("HOTPATCH-PREFLIGHT RECONCILE: ")
    )
    assert "node_hotpatch_ledger_reconcile_effect" in command
    assert host.fix in command  # suggested from the PR number on the build ref

    ran = subprocess.run(
        ["/bin/sh", "-c", command],  # the printed command is a shell line by contract
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    row = yaml.safe_load(host.ledger.read_text())["rows"][0]
    assert row["status"] == "reconciled"
    assert row["merge_commit"] == host.fix
    assert list(tmp_path.glob("ledger.yaml.bak-reconcile-*"))

    after = host.preflight()
    assert after.returncode == 0, after.stderr
    assert "SKIP (reconciled" in after.stdout


def test_refusal_exits_nonzero_and_leaves_the_ledger_alone(tmp_path: Path) -> None:
    host = Host(tmp_path, prepatch="/app/x.py.prepatch")
    original = host.ledger.read_bytes()

    ran = host.reconcile(host.fix)

    assert ran.returncode == 1, ran.stderr
    reason = host.refusal_reason()
    assert "still carries .prepatch" in reason
    assert host.ledger.read_bytes() == original
    assert not list(tmp_path.glob("ledger.yaml.bak-*"))
