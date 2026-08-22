# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The catalog CLI must read the env file operators are told to edit.

OMN-16187. ``install.sh`` and ``make setup`` create a repo-local ``.env`` from
``.env.example`` and tell the operator to fill it in, but the catalog CLI only
ever read ``~/.omnibase/.env``. A new operator who followed the documented steps
exactly still hit "Cannot start: missing required env vars", because the file
they edited was never loaded.

Resolution order is first-wins per key, so an existing operator's
``~/.omnibase/.env`` keeps precedence over any repo-local ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog import cli as catalog_cli


@pytest.fixture(autouse=True)
def _clear_probe_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OMN16187_PROBE", "OMN16187_PRECEDENCE"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.unit
def test_repo_local_env_is_loaded_when_home_env_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The repo-local ``.env`` that install.sh creates must actually be read."""
    repo_env = tmp_path / "repo.env"
    repo_env.write_text("OMN16187_PROBE=from-repo\n", encoding="utf-8")
    monkeypatch.setattr(catalog_cli, "_HOME_ENV", tmp_path / "absent-home.env")
    monkeypatch.setattr(catalog_cli, "_REPO_ENV", repo_env)

    catalog_cli._load_omnibase_env()

    assert os.environ["OMN16187_PROBE"] == "from-repo"


@pytest.mark.unit
def test_home_env_wins_over_repo_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing operators keep their ``~/.omnibase/.env`` precedence."""
    home_env = tmp_path / "home.env"
    home_env.write_text("OMN16187_PRECEDENCE=from-home\n", encoding="utf-8")
    repo_env = tmp_path / "repo.env"
    repo_env.write_text("OMN16187_PRECEDENCE=from-repo\n", encoding="utf-8")
    monkeypatch.setattr(catalog_cli, "_HOME_ENV", home_env)
    monkeypatch.setattr(catalog_cli, "_REPO_ENV", repo_env)

    catalog_cli._load_omnibase_env()

    assert os.environ["OMN16187_PRECEDENCE"] == "from-home"


@pytest.mark.unit
def test_ambient_environment_wins_over_every_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loading files must never clobber a value already exported by the caller."""
    monkeypatch.setenv("OMN16187_PRECEDENCE", "from-ambient")
    home_env = tmp_path / "home.env"
    home_env.write_text("OMN16187_PRECEDENCE=from-home\n", encoding="utf-8")
    monkeypatch.setattr(catalog_cli, "_HOME_ENV", home_env)
    monkeypatch.setattr(catalog_cli, "_REPO_ENV", tmp_path / "absent-repo.env")

    catalog_cli._load_omnibase_env()

    assert os.environ["OMN16187_PRECEDENCE"] == "from-ambient"
