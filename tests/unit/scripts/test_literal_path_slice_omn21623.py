# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Path resolution behavior and literal_path coverage for OMN-21623."""

from pathlib import Path

import pytest
from deploy_agent.unit_drift import ModelUnitEntry, _installed_path

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from scripts.validation.check_claudemd_references import resolve_path

REPO_ROOT = Path(__file__).resolve().parents[3]

CLEARED_FILES = (
    "scripts/deploy-agent/deploy_agent/unit_drift.py",
    "scripts/validation/check_claudemd_references.py",
)


def _literal_path_issues(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate_file(path)
    return [issue.message for issue in result.issues if "literal_path" in str(issue)]


def test_rule_flags_a_planted_path_literal(tmp_path: Path) -> None:
    """Positive control: the same call reports a literal path, so a zero below means clean."""
    planted = tmp_path / "planted.py"
    planted.write_text('STATE_FILE = "/var/lib/example/state.json"\n', encoding="utf-8")
    assert len(_literal_path_issues(planted)) == 1


@pytest.mark.parametrize("relative", CLEARED_FILES)
def test_cleared_file_has_no_literal_path_finding(relative: str) -> None:
    path = REPO_ROOT / relative
    assert path.is_file()
    assert _literal_path_issues(path) == []


@pytest.mark.parametrize(
    ("value", "suffix"),
    [
        ("~", ""),
        ("~/", ""),
        ("~/x", "x"),
        ("~/units/nested/x.service", "units/nested/x.service"),
        ("~/./units/x.service", "units/x.service"),
        ("~/units/../x.service", "units/../x.service"),
    ],
)
def test_both_resolvers_expand_home_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, suffix: str
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    entry = ModelUnitEntry(
        name="x.service", tracked="tracked/x.service", installed=value, hosts=[]
    )
    expected = home / suffix
    assert _installed_path(entry, home) == expected
    assert resolve_path(value, tmp_path / "repo", tmp_path / "omni") == expected


@pytest.mark.parametrize("value", ["", "x.service", "units/x.service", "~other/x"])
def test_installed_path_keeps_non_home_relative_paths(
    tmp_path: Path, value: str
) -> None:
    entry = ModelUnitEntry(
        name="x.service", tracked="tracked/x.service", installed=value, hosts=[]
    )
    assert _installed_path(entry, tmp_path / "home") == Path(value)


def test_both_resolvers_keep_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "absent" / "x.service"
    entry = ModelUnitEntry(
        name="x.service",
        tracked="tracked/x.service",
        installed=str(absolute),
        hosts=[],
    )
    assert _installed_path(entry, tmp_path / "home") == absolute
    assert resolve_path(str(absolute), tmp_path / "repo", tmp_path / "omni") == absolute


@pytest.mark.parametrize("value", ["x.md", "docs/nested/x.md", "~other/x.md"])
def test_reference_path_prefers_repo_then_falls_back_to_omni_home(
    tmp_path: Path, value: str
) -> None:
    repo, omni = tmp_path / "repo", tmp_path / "omni"
    repo_file, omni_file = repo / value, omni / value
    assert resolve_path(value, repo, omni) is None

    omni_file.parent.mkdir(parents=True)
    omni_file.write_text("omni")
    assert resolve_path(value, repo, omni) == omni_file

    repo_file.parent.mkdir(parents=True)
    repo_file.write_text("repo")
    assert resolve_path(value, repo, omni) == repo_file


@pytest.mark.parametrize("resolve_registry", [True, False])
def test_installed_registry_paths_obey_resolution_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve_registry: bool
) -> None:
    registry = tmp_path / "registry"
    monkeypatch.setenv("UNIT_DRIFT_REGISTRY_ROOT", str(registry))
    entry = ModelUnitEntry(
        name="x.service",
        tracked="tracked/x.service",
        installed="{registry_root}/units/x.service",
        hosts=[],
    )
    expected = (
        registry / "units/x.service" if resolve_registry else Path(entry.installed)
    )
    assert (
        _installed_path(entry, tmp_path / "home", resolve_registry=resolve_registry)
        == expected
    )
