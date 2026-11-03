# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise floor authorization across real Git and package metadata boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SRC = Path(__file__).resolve().parents[3] / "src"
_PROBE = """
import json, sys
from omnibase_infra.cli.omnimarket_drift_guard import (
    check_omnimarket_drift, OmnimarketDriftError,
)
try:
    result = check_omnimarket_drift(sys.argv[1])
except OmnimarketDriftError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
print(json.dumps(result.as_receipt_fields() if result is not None else None))
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        env={
            "PATH": os.defpath,
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "floor-test",
            "GIT_AUTHOR_EMAIL": "floor@example.invalid",
            "GIT_COMMITTER_NAME": "floor-test",
            "GIT_COMMITTER_EMAIL": "floor@example.invalid",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _clone(home: Path) -> tuple[Path, str, str]:
    root = home / "omnimarket"
    root.mkdir()
    _git(root, "init", "-q", "-b", "dev")
    commits = []
    for text in ("anchor", "next"):
        (root / "content").write_text(text, encoding="utf-8")
        _git(root, "add", "content")
        _git(root, "commit", "-q", "-m", text)
        commits.append(_git(root, "rev-parse", "HEAD"))
    _git(root, "checkout", "-q", "--detach", commits[0])
    return root, commits[0], commits[1]


def _metadata(home: Path, installed: str) -> Path:
    # Importlib reads this distribution's actual PEP 610 metadata in the child.
    site = home / "site"
    dist = site / "omnimarket-0.0.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.0.0\n", encoding="utf-8"
    )
    (dist / "direct_url.json").write_text(
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": installed}}),
        encoding="utf-8",
    )
    return site


def _run(home: Path, installed: str) -> subprocess.CompletedProcess[str]:
    site = _metadata(home, installed)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(site), str(_SRC))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _PROBE, str(home)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "marker", ["exact", "uppercase", "elsewhere", "missing", "short", "invalid-utf8"]
)
def test_floor_marker_authorizes_only_its_full_commit(
    tmp_path: Path, marker: str
) -> None:
    _, anchor, other = _clone(tmp_path)
    path = tmp_path / ".onex-workspace-floor.json"
    if marker == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif marker != "missing":
        named = {"exact": anchor, "uppercase": anchor.upper(), "elsewhere": other}.get(
            marker, anchor[:12]
        )
        path.write_text(json.dumps({"omnimarket_commit": named}), encoding="utf-8")
    result = _run(tmp_path, anchor)
    if marker in ("exact", "uppercase"):
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) is None
    else:
        assert result.returncode == 1, result.stderr
        assert "DETACHED HEAD" in result.stderr
        assert "converge-canonical-clone.sh omnimarket --execute" in result.stderr


def test_floor_marker_preserves_registry_reference_receipt(tmp_path: Path) -> None:
    root, anchor, reference = _clone(tmp_path)
    _git(root, "update-ref", "refs/remotes/origin/dev", reference)
    _git(root, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/dev")
    (tmp_path / ".onex-workspace-floor.json").write_text(
        json.dumps({"omnimarket_commit": anchor}), encoding="utf-8"
    )
    result = _run(tmp_path, reference)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["mode"] == "registry-reference"
    assert receipt["installed_commit"] == reference
    assert receipt["reference_commit"] == reference
    assert receipt["checkout_commit"] == anchor
    assert receipt["commits_behind"] == 0


def test_floor_marker_does_not_authorize_an_unrelated_install(tmp_path: Path) -> None:
    _, anchor, _ = _clone(tmp_path)
    (tmp_path / ".onex-workspace-floor.json").write_text(
        json.dumps({"omnimarket_commit": anchor}), encoding="utf-8"
    )
    result = _run(tmp_path, "f" * 40)
    assert result.returncode == 1
    assert "STALE" in result.stderr


if __name__ == "__main__":
    # The contract's local CLI readback uses the same real Git/package setup.
    home = Path(sys.argv[1])
    _, anchor, _ = _clone(home)
    _metadata(home, anchor)
    (home / ".onex-workspace-floor.json").write_text(
        json.dumps({"omnimarket_commit": anchor}), encoding="utf-8"
    )
