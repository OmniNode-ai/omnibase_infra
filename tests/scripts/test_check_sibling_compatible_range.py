# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hold OMN-19655's published range shape; CI runs core's tag/override/lock rules."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

import scripts.ci.check_sibling_compatible_range as check

pytestmark = pytest.mark.unit
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _data(deps: list[str]) -> dict[str, Any]:
    return {"project": {"dependencies": deps}}


def test_the_ruled_shape_passes() -> None:
    assert (
        check.violations(_data(["omnibase-core>=0.47.23,<0.48.0"]), "omnibase-core")
        == []
    )


@pytest.mark.parametrize("operator", ["==", "==="])
def test_the_pre_ruling_exact_pin_is_refused(operator: str) -> None:
    found = check.violations(
        _data([f"omnibase-core{operator}0.47.23"]), "omnibase-core"
    )
    assert len(found) == 1
    assert "exact pin" in found[0]


@pytest.mark.parametrize(
    "spec",
    [
        "omnibase-core>=0.47.23,<0.49.0",
        "omnibase-core>=0.47.23",
        "omnibase-core>=0.47.23,<1.0.0",
        "omnibase-core~=0.47.23",
        "omnibase-core>=0.47.23,<0.48.0,!=0.47.24",
        "omnibase-core>=0.47.23,<0.48.1",
    ],
)
def test_any_other_range_shape_is_refused(spec: str) -> None:
    found = check.violations(_data([spec]), "omnibase-core")
    assert found, spec
    assert "NEXT_MINOR" in found[0]


def test_a_missing_requirement_is_refused() -> None:
    found = check.violations(_data([]), "omnibase-core")
    assert len(found) == 1
    assert "[project.dependencies]" in found[0]


def test_duplicate_requirements_are_refused() -> None:
    found = check.violations(
        _data(["omnibase-core>=0.47.23,<0.48.0"] * 2), "omnibase-core"
    )
    assert len(found) == 1
    assert "found 2" in found[0]


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.toml"
    good.write_text(
        '[project]\ndependencies = ["omnibase-core>=0.47.23,<0.48.0"]\n',
        encoding="utf-8",
    )
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[project]\ndependencies = ["omnibase-core==0.47.23"]\n',
        encoding="utf-8",
    )
    assert check.main(["--pyproject", str(good), "--package", "omnibase-core"]) == 0
    assert check.main(["--pyproject", str(bad), "--package", "omnibase-core"]) == 1
    assert "exact pin" in capsys.readouterr().err


def test_missing_or_invalid_manifest_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    assert check.main(["--pyproject", str(path)]) == 2
    path.write_text("[[broken")
    assert check.main(["--pyproject", str(path)]) == 2


def test_this_repo_publishes_the_ruled_core_range() -> None:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert check.violations(data, "omnibase-core") == []


# Verbatim manifest released as v0.38.57 (bd51068d), whose exact published
# core pin stranded downstream floor raises on 2026-09-25 (OMN-19655).
_V0_38_57 = (
    _REPO_ROOT
    / "tests/fixtures/omn19655/omnibase-infra-v0.38.57-pyproject.toml.captured"
)


def test_the_guard_rejects_the_manifest_0_38_57_was_released_from() -> None:
    data = tomllib.loads(_V0_38_57.read_text(encoding="utf-8"))
    found = check.violations(data, "omnibase-core")
    assert len(found) == 1
    assert "omnibase-core==0.47.22, an exact pin" in found[0]


def test_the_same_guard_accepts_that_manifest_with_the_ruled_range() -> None:
    """Change only the published requirement in the captured manifest."""
    text = _V0_38_57.read_text(encoding="utf-8")
    project, marker, tool_uv = text.partition("\n[tool.uv]\n")
    assert marker, "the captured manifest carries a [tool.uv] table"
    published = '    "omnibase-core==0.47.22",\n'
    assert project.count(published) == 1
    data = tomllib.loads(
        project.replace(published, '    "omnibase-core>=0.47.22,<0.48.0",\n')
        + marker
        + tool_uv
    )
    assert check.violations(data, "omnibase-core") == []
