# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent is its own installer, and its dev self-install leaves the prod unit alone (OMN-26106 AC4).

There is no separate dev installer: the agent re-installs its own unit through
``unit_drift.sync_own_unit`` (copy the tracked bytes over the one drifting unit,
reload, never restart). AC4 reads the prod unit's sha256 before and after that
install. The test runs the real manifest's dev entry over a home that also holds the
prod unit, so the before and after reads are real bytes, not a recorded constant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from deploy_agent.unit_drift import ModelUnitEntry, load_manifest, sync_own_unit

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST = "labhost"
_DEV = "deploy-agent-dev.service"
_PROD = "deploy-agent.service"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entries() -> list[ModelUnitEntry]:
    """The real manifest's entries, declared for the test host."""
    manifest = load_manifest(_REPO_ROOT / "deploy" / "unit-drift-manifest.yaml")
    return [
        entry.model_copy(update={"hosts": [_HOST]})
        for entry in manifest
        if entry.name in (_DEV, _PROD)
    ]


def _lab(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A repo copy holding both tracked units and a home holding both stale copies."""
    repo = tmp_path / "repo"
    tracked_dir = repo / "scripts" / "deploy-agent" / "deploy"
    tracked_dir.mkdir(parents=True)
    for name in (_DEV, _PROD):
        src = _REPO_ROOT / "scripts" / "deploy-agent" / "deploy" / name
        (tracked_dir / name).write_bytes(src.read_bytes())
    units = tmp_path / "home" / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    dev_installed = units / _DEV
    prod_installed = units / _PROD
    dev_installed.write_text("# stale dev unit\n")
    prod_installed.write_text("# prod unit as installed, differs from tracked\n")
    return repo, tmp_path / "home", dev_installed, prod_installed


def test_dev_self_install_leaves_the_prod_unit_sha256_unchanged(
    tmp_path: Path,
) -> None:
    repo, home, dev_installed, prod_installed = _lab(tmp_path)
    entries = _entries()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    prod_before = _sha256(prod_installed)
    dev_before = _sha256(dev_installed)

    changed = sync_own_unit(
        entries,
        unit_name=_DEV,
        repo_root=repo,
        hostname=_HOST,
        home=home,
        runner=runner,
        stamp="20261126T0000Z",
    )

    assert changed is True
    assert _sha256(dev_installed) != dev_before
    assert _sha256(dev_installed) == _sha256(
        repo / "scripts" / "deploy-agent" / "deploy" / _DEV
    )
    assert _sha256(prod_installed) == prod_before
    assert calls == [["systemctl", "--user", "daemon-reload"]]
    assert not list(prod_installed.parent.glob(f"{_PROD}.bak-*"))


def test_the_prod_digest_read_is_a_known_bad_when_the_prod_unit_is_installed(
    tmp_path: Path,
) -> None:
    """Control: the same read goes red when the install targets the prod unit."""
    repo, home, _dev, prod_installed = _lab(tmp_path)
    entries = _entries()
    prod_before = _sha256(prod_installed)

    assert sync_own_unit(
        entries,
        unit_name=_PROD,
        repo_root=repo,
        hostname=_HOST,
        home=home,
        runner=lambda _cmd: 0,
        stamp="20261126T0000Z",
    )

    assert _sha256(prod_installed) != prod_before
