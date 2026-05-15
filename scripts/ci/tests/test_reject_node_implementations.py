#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the infra node handler ownership gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "reject-node-implementations.sh"
ALLOWLIST = Path("scripts/ci/infra-node-allowlist.txt")


def _write_allowlist(repo: Path, content: str) -> None:
    allowlist = repo / ALLOWLIST
    allowlist.parent.mkdir(parents=True, exist_ok=True)
    allowlist.write_text(content, encoding="utf-8")


def _write_handler(repo: Path, relative_path: str) -> None:
    handler = repo / relative_path
    handler.parent.mkdir(parents=True, exist_ok=True)
    handler.write_text("def handle() -> None:\n    return None\n", encoding="utf-8")


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def test_all_mode_passes_when_handler_is_allowlisted(tmp_path: Path) -> None:
    handler_path = (
        "src/omnibase_infra/nodes/node_registry_effect/handlers/"
        "handler_postgres_upsert.py"
    )
    _write_handler(tmp_path, handler_path)
    _write_allowlist(
        tmp_path,
        f"{handler_path}  # infra registry persistence handler remains in infra\n",
    )

    result = _run(tmp_path, "--all")

    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_all_mode_rejects_unallowlisted_handler(tmp_path: Path) -> None:
    handler_path = (
        "src/omnibase_infra/nodes/node_new_business_compute/handlers/"
        "handler_business.py"
    )
    _write_handler(tmp_path, handler_path)
    _write_allowlist(tmp_path, "")

    result = _run(tmp_path, "--all")

    assert result.returncode == 1
    assert "Node handlers belong in omnimarket" in result.stderr
    assert handler_path in result.stderr


def test_all_mode_rejects_stale_allowlist_entry(tmp_path: Path) -> None:
    handler_path = (
        "src/omnibase_infra/nodes/node_deleted_effect/handlers/handler_deleted.py"
    )
    _write_allowlist(
        tmp_path,
        f"{handler_path}  # stale entry should be removed with deleted handler\n",
    )

    result = _run(tmp_path, "--all")

    assert result.returncode == 1
    assert "stale infra node handler allowlist" in result.stderr
    assert handler_path in result.stderr


def test_allowlist_entries_require_inline_justification(tmp_path: Path) -> None:
    handler_path = (
        "src/omnibase_infra/nodes/node_registry_effect/handlers/"
        "handler_postgres_upsert.py"
    )
    _write_handler(tmp_path, handler_path)
    _write_allowlist(tmp_path, f"{handler_path}\n")

    result = _run(tmp_path, "--all")

    assert result.returncode == 1
    assert "missing inline justification comment" in result.stderr


def test_staged_mode_rejects_staged_unallowlisted_handler(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    handler_path = (
        "src/omnibase_infra/nodes/node_new_business_compute/handlers/"
        "handler_business.py"
    )
    _write_handler(tmp_path, handler_path)
    _write_allowlist(tmp_path, "")
    subprocess.run(["git", "add", handler_path], cwd=tmp_path, check=True)

    result = _run(tmp_path, "--staged")

    assert result.returncode == 1
    assert "Node handlers belong in omnimarket" in result.stderr
    assert handler_path in result.stderr


def test_staged_mode_uses_staged_allowlist(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    handler_path = (
        "src/omnibase_infra/nodes/node_new_business_compute/handlers/"
        "handler_business.py"
    )
    _write_allowlist(tmp_path, "")
    subprocess.run(["git", "add", str(ALLOWLIST)], cwd=tmp_path, check=True)

    _write_handler(tmp_path, handler_path)
    _write_allowlist(
        tmp_path,
        f"{handler_path}  # unstaged allowlist edits must not satisfy hook\n",
    )
    subprocess.run(["git", "add", handler_path], cwd=tmp_path, check=True)

    result = _run(tmp_path, "--staged")

    assert result.returncode == 1
    assert "Node handlers belong in omnimarket" in result.stderr
    assert handler_path in result.stderr
