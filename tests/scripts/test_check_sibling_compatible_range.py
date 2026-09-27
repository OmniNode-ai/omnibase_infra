# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnibase_infra publishes a compatible omnibase-core range; the exact pin lives only in the override (OMN-19655).

Operator ruling 2026-09-25 ("yes to both", ledger RULING orchestrator-83): infra
publishes ``omnibase-core>=FLOOR,<NEXT_MINOR`` in ``[project.dependencies]`` and
keeps the exact version only in ``[tool.uv] override-dependencies``, so
``uv.lock`` and the runtime image are unchanged while a downstream core floor
raise inside the minor resolves against the published infra.

uv's override REPLACES the project requirement during resolution, so nothing in
uv notices when the two drift apart: a cascade that moves the override to
0.48.0 while the range still says ``<0.48.0`` locks and tests 0.48.0 and then
publishes metadata that forbids it. ``check_sibling_compatible_range.py`` holds
the shape, and the last test here runs it against this repo's own pyproject so
CI enforces it on every pull request.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

import scripts.ci.check_sibling_compatible_range as check

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _data(deps: list[str], overrides: list[str] | None) -> dict[str, Any]:
    data: dict[str, Any] = {"project": {"dependencies": deps}}
    if overrides is not None:
        data["tool"] = {"uv": {"override-dependencies": overrides}}
    return data


@pytest.mark.unit
def test_the_ruled_shape_passes() -> None:
    assert (
        check.violations(
            _data(["omnibase-core>=0.47.23,<0.48.0"], ["omnibase-core==0.47.23"]),
            "omnibase-core",
        )
        == []
    )


@pytest.mark.unit
def test_a_patch_override_inside_the_range_passes() -> None:
    """A cascade moves only the override on a patch release; that is admitted."""
    assert (
        check.violations(
            _data(["omnibase-core>=0.47.23,<0.48.0"], ["omnibase-core==0.47.25"]),
            "omnibase-core",
        )
        == []
    )


@pytest.mark.unit
def test_the_pre_ruling_exact_pin_is_refused() -> None:
    """Negative control: dev's shape before OMN-19655."""
    found = check.violations(
        _data(["omnibase-core==0.47.23"], ["omnibase-core==0.47.23"]),
        "omnibase-core",
    )
    assert len(found) == 1
    assert "exact pin" in found[0]


@pytest.mark.unit
def test_an_override_past_the_ceiling_is_refused() -> None:
    found = check.violations(
        _data(["omnibase-core>=0.47.23,<0.48.0"], ["omnibase-core==0.48.0"]),
        "omnibase-core",
    )
    assert len(found) == 1
    assert "0.48.0" in found[0]
    assert "does not admit" in found[0]


@pytest.mark.unit
def test_an_override_below_the_floor_is_refused() -> None:
    found = check.violations(
        _data(["omnibase-core>=0.47.23,<0.48.0"], ["omnibase-core==0.47.22"]),
        "omnibase-core",
    )
    assert len(found) == 1
    assert "does not admit" in found[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec",
    [
        "omnibase-core>=0.47.23,<0.49.0",
        "omnibase-core>=0.47.23",
        "omnibase-core>=0.47.23,<1.0.0",
        "omnibase-core~=0.47.23",
        "omnibase-core>=0.47.23,<0.48.0,!=0.47.24",
    ],
)
def test_any_other_range_shape_is_refused(spec: str) -> None:
    found = check.violations(_data([spec], ["omnibase-core==0.47.23"]), "omnibase-core")
    assert found, spec
    assert "NEXT_MINOR" in found[0]


@pytest.mark.unit
def test_a_range_with_no_override_pin_is_refused() -> None:
    """Without the override nothing holds uv.lock and the image to one version."""
    found = check.violations(
        _data(["omnibase-core>=0.47.23,<0.48.0"], None), "omnibase-core"
    )
    assert len(found) == 1
    assert "override-dependencies" in found[0]


@pytest.mark.unit
def test_a_missing_requirement_is_refused() -> None:
    found = check.violations(_data([], ["omnibase-core==0.47.23"]), "omnibase-core")
    assert len(found) == 1
    assert "[project.dependencies]" in found[0]


@pytest.mark.unit
def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.toml"
    good.write_text(
        '[project]\ndependencies = ["omnibase-core>=0.47.23,<0.48.0"]\n'
        '[tool.uv]\noverride-dependencies = ["omnibase-core==0.47.23"]\n',
        encoding="utf-8",
    )
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[project]\ndependencies = ["omnibase-core==0.47.23"]\n'
        '[tool.uv]\noverride-dependencies = ["omnibase-core==0.47.23"]\n',
        encoding="utf-8",
    )
    assert check.main(["--pyproject", str(good), "--package", "omnibase-core"]) == 0
    assert check.main(["--pyproject", str(bad), "--package", "omnibase-core"]) == 1
    assert "exact pin" in capsys.readouterr().err


@pytest.mark.unit
def test_this_repo_publishes_the_ruled_core_range() -> None:
    """The enforcement: this repo's own pyproject holds the ruled shape."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert check.violations(data, "omnibase-core") == []


# ---------------------------------------------------------------------------
# Incident replay (OMN-19655): the manifest omnibase-infra 0.38.57 was released
# from, verbatim (tag v0.38.57, bd51068d). Its exact core pin is what stranded
# omnimarket's release all day on 2026-09-25.
# ---------------------------------------------------------------------------

_V0_38_57 = (
    _REPO_ROOT
    / "tests/fixtures/omn19655/omnibase-infra-v0.38.57-pyproject.toml.captured"
)


@pytest.mark.unit
def test_the_guard_rejects_the_manifest_0_38_57_was_released_from() -> None:
    data = tomllib.loads(_V0_38_57.read_text(encoding="utf-8"))
    found = check.violations(data, "omnibase-core")
    assert len(found) == 1
    assert "omnibase-core==0.47.22, an exact pin" in found[0]


@pytest.mark.unit
def test_the_same_guard_accepts_that_manifest_with_the_ruled_range() -> None:
    """Discriminator: the captured bytes with only the published core line moved
    to the ruled range pass, so the guard is not rejecting everything."""
    text = _V0_38_57.read_text(encoding="utf-8")
    project, marker, tool_uv = text.partition("\n[tool.uv]\n")
    assert marker, "the captured manifest carries a [tool.uv] table"
    published = '    "omnibase-core==0.47.22",\n'
    assert project.count(published) == 1
    assert '"omnibase-core==0.47.22"' in tool_uv, "the override keeps the exact pin"
    data = tomllib.loads(
        project.replace(published, '    "omnibase-core>=0.47.22,<0.48.0",\n')
        + marker
        + tool_uv
    )
    assert check.violations(data, "omnibase-core") == []
