# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: the boot guard reads an unhealthy runtime as running no revision.

Runs the real ``subprocess`` path of ``running_ref_if_healthy`` against a ``docker``
executable on ``PATH`` that answers ``docker inspect`` the way the daemon does.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_boot_guard import (
    healthy_ref_reader,
    running_ref_if_healthy,
)

pytestmark = pytest.mark.integration

SHA = "a" * 40


def _docker_on_path(
    directory: Path, monkeypatch: pytest.MonkeyPatch, health: str
) -> None:
    script = directory / "docker"
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{health}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{directory}:/usr/bin:/bin")


def _read_ref(container: str, repository: str) -> str | None:
    return SHA


def test_unhealthy_runtime_reads_as_no_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _docker_on_path(tmp_path, monkeypatch, "unhealthy")
    assert running_ref_if_healthy("stability-main", "omnimarket", _read_ref) == ""


def test_healthy_runtime_reads_its_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _docker_on_path(tmp_path, monkeypatch, "healthy")
    reader = healthy_ref_reader(lambda: "stability-main", _read_ref)
    assert reader("omnimarket") == SHA
