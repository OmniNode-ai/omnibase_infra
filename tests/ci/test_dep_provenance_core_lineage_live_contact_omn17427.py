# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay core's recorded GitHub tag decisions at infra's cascade handoff."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def _provenance_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/check_dep_provenance.py"
    spec = importlib.util.spec_from_file_location("dep_provenance_recorded", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.live_contact(
    "tests/fixtures/dep_provenance_core_lineage_recorded_omn17427.json"
)
def test_recorded_real_tag_can_yield_to_a_release(
    recorded_response: dict[str, object],
) -> None:
    cases = recorded_response["cases"]
    assert isinstance(cases, list)
    real = next(case for case in cases if case["label"] == "real")
    tag = real["tag"]
    assert tag == "dev-20261031000524-fc6159b98026"
    remote = real["ls_remote"]
    assert remote["exit_code"] == 0
    refs = {
        ref: commit
        for commit, ref in (line.split() for line in remote["stdout"].splitlines())
    }
    tag_ref = f"refs/tags/{tag}"
    commit = refs.get(f"{tag_ref}^{{}}", refs.get(tag_ref))
    assert commit is not None and len(commit) == 40
    assert commit.startswith(tag.rsplit("-", 1)[1])
    result = real["core"]
    assert result["exit_code"] == 0
    assert result["stdout"] == ""
    assert result["stderr"] == ""

    text = real["pyproject_toml"]
    line = real["source_line"]
    assert text.count(line) == 1
    script = _provenance_script()
    assert script.find_unmovable_cascade_targets(text, "omnibase-core") == []
    assert script.drop_dev_tag_override(text, "omnibase-core") == text.replace(
        line, "", 1
    )


@pytest.mark.live_contact(
    "tests/fixtures/dep_provenance_core_lineage_recorded_omn17427.json"
)
def test_recorded_absent_tag_stops_the_cascade(
    recorded_response: dict[str, object],
) -> None:
    cases = recorded_response["cases"]
    assert isinstance(cases, list)
    absent = next(case for case in cases if case["label"] == "absent")
    assert absent["tag"] == "dev-20261029143607-000000000000"
    remote = absent["ls_remote"]
    assert remote["exit_code"] == 0
    assert remote["stdout"] == ""
    result = absent["core"]
    assert result["exit_code"] == 1
    assert result["stdout"].startswith("pyproject.toml: tag_absent: omnibase-core:")
    assert absent["tag"] in result["stdout"]
    assert result["stderr"] == ""
