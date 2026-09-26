# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the release dependency-advance trigger (OMN-19655).

The captured artifact is PyPI's JSON document for omnibase-infra 0.38.57, the
newest published infra on 2026-09-25 while omnibase_infra dev pinned
omnibase-core==0.47.23 and omnimarket's release failed on every dev push. The
real guard, fed those bytes and dev's pin, must call the state stranded; fed
the same bytes and the pin the release carries, it must not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.ci.release_dependency_advance as advance

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO_ROOT / "tests/fixtures/omn19655/pypi-omnibase-infra-0.38.57.json.captured"
)
_SHA256 = "5e29161ece38aa044977c2bcc1cf34cf496ae64b67422e1376c785dbce869d0d"


def _captured(package: str) -> dict[str, Any]:
    assert package == "omnibase-infra"
    loaded: dict[str, Any] = json.loads(_FIXTURE.read_bytes())
    return loaded


def _dev_pyproject(tmp_path: Path, core_pin: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "omnibase-infra"\nversion = "0.38.58"\n'
        f'dependencies = [\n    "omnibase-core=={core_pin}",\n'
        '    "omnibase-spi==0.23.5",\n    "omnibase-compat==0.5.7",\n]\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_the_real_guard_calls_the_2026_09_25_state_stranded(tmp_path: Path) -> None:
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_dev_pyproject(tmp_path, "0.47.23"),
        fetch=_captured,
    )
    assert verdict.published_version == "0.38.57"
    assert verdict.stranded is True
    assert [(a.name, a.published, a.dev) for a in verdict.advances] == [
        ("omnibase-core", "0.47.22", "0.47.23")
    ]


@pytest.mark.unit
def test_the_same_guard_accepts_the_pins_the_release_carries(tmp_path: Path) -> None:
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_dev_pyproject(tmp_path, "0.47.22"),
        fetch=_captured,
    )
    assert verdict.stranded is False


@pytest.mark.unit
def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _SHA256
