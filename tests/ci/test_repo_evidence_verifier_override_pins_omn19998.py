# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: repo evidence verifier admits the omnibase-core override pin."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER_PATH = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"

# First release admitting tool.uv.override-dependencies, omnimarket#3694.
_OVERRIDE_PIN_CLASSIFIER_FLOOR = (0, 4, 310)


def _admits_override_pins(version: str) -> bool:
    return (
        tuple(int(part) for part in version.split("."))
        >= _OVERRIDE_PIN_CLASSIFIER_FLOOR
    )


def test_core_pin_lives_in_override_dependencies() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    overrides = data["tool"]["uv"]["override-dependencies"]
    assert any(
        re.split(r"[=<>!~;\[ ]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        == "omnibase-core"
        for requirement in overrides
    ), "omnibase-core must be pinned in tool.uv.override-dependencies (OMN-19655)"


def test_caller_verifier_admits_override_dependency_pins() -> None:
    data = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    version = data["jobs"]["repo-evidence"]["with"]["verifier-version"]
    assert _admits_override_pins(version), (
        "verifier-version must ship omnimarket#3694 to admit "
        "tool.uv.override-dependencies "
        f"(>= {'.'.join(map(str, _OVERRIDE_PIN_CLASSIFIER_FLOOR))}); found {version}"
    )


def test_floor_refuses_the_release_that_refused_the_core_bumps() -> None:
    assert not _admits_override_pins("0.4.305")
    assert not _admits_override_pins("0.4.309")
    assert _admits_override_pins("0.4.310")
    assert _admits_override_pins("0.5.0")
