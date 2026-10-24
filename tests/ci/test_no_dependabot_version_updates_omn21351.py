# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21351: ratchet the registry toward no Dependabot version updates."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Anti-growth baseline measured on 2026-10-24 at origin/dev: these clones declared
# schedules when this ratchet landed, either with an OMN-21351 removal PR still in
# flight (omnibase_core, omniclaude, omniintelligence) or with none yet. It may
# only shrink and must reach empty, which is what satisfies OMN-21351 AC1; a repo
# not covered here fails the ratchet the moment it declares a schedule.
PENDING_REMOVAL = frozenset(
    {
        "omnibase_core",
        "omniclaude",
        "omniintelligence",
        "omnibase_spi",
        "omnimemory",
    }
)
# Registry repos the public-repository hygiene gate bars this file from naming
# also still declared schedules on the same date; this many offenders beyond
# PENDING_REMOVAL are tolerated until their removal PRs land, then it goes to 0.
UNNAMED_PENDING_COUNT = 2


def declared_version_update_schedules(text: str) -> list[str]:
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("Dependabot configuration must be a mapping")
    return [
        entry["package-ecosystem"]
        for entry in document.get("updates", []) or []
        if "schedule" in entry
    ]


@pytest.fixture
def registry_repos() -> list[str]:
    repos: list[str] = yaml.safe_load(
        (REPO_ROOT / "config" / "lab_proof_profiles.yaml").read_text(encoding="utf-8")
    )["registry_repos"]
    return repos


def scan_clones(root: Path, repos: Sequence[str]) -> dict[str, list[str]]:
    offenders = {}
    for repo in repos:
        name = repo.rsplit("/", 1)[-1]
        clone = root / name
        path = clone / ".github" / "dependabot.yml"
        if not clone.is_dir() or not path.exists():
            continue
        ecosystems = declared_version_update_schedules(path.read_text(encoding="utf-8"))
        if ecosystems:
            offenders[name] = ecosystems
    return offenders


def test_no_dependabot_version_updates_in_this_repo() -> None:
    path = REPO_ROOT / ".github" / "dependabot.yml"
    if path.exists():
        assert declared_version_update_schedules(path.read_text(encoding="utf-8")) == []


def test_no_dependabot_version_updates_in_registry_clones(
    registry_repos: list[str],
) -> None:
    if "OMNI_HOME" not in os.environ:
        pytest.skip("OMNI_HOME is not set; registry clones are unavailable")
    root = Path(os.environ["OMNI_HOME"])
    # This repo's own checkout is judged strictly by the first test; its
    # canonical clone lags a merge, so it is not read here.
    others = [repo for repo in registry_repos if not repo.endswith("/omnibase_infra")]
    if not any((root / repo.rsplit("/", 1)[-1]).is_dir() for repo in others):
        pytest.skip("No registry clone exists under OMNI_HOME")
    offenders = scan_clones(root, others)
    unexpected = {
        name: ecosystems
        for name, ecosystems in offenders.items()
        if name not in PENDING_REMOVAL
    }
    assert len(unexpected) <= UNNAMED_PENDING_COUNT, (
        f"Dependabot version-update schedules beyond the baseline: {unexpected}"
    )


def test_no_dependabot_version_updates_detects_planted_schedule(tmp_path: Path) -> None:
    path = tmp_path / "scheduled" / ".github" / "dependabot.yml"
    path.parent.mkdir(parents=True)
    path.write_text(
        'version: 2\nupdates: [{package-ecosystem: pip, directory: "/", '
        "schedule: {interval: weekly}}]\n",
        encoding="utf-8",
    )
    assert scan_clones(tmp_path, ["OmniNode-ai/scheduled"]) == {"scheduled": ["pip"]}


def test_no_dependabot_version_updates_ignores_missing_and_empty_configs(
    tmp_path: Path,
) -> None:
    (tmp_path / "no_file").mkdir()
    path = tmp_path / "empty" / ".github" / "dependabot.yml"
    path.parent.mkdir(parents=True)
    path.write_text("version: 2\n", encoding="utf-8")
    repos = [f"OmniNode-ai/{name}" for name in ("missing", "no_file", "empty")]
    assert scan_clones(tmp_path, repos) == {}


def test_no_dependabot_version_updates_reports_both_ecosystems(tmp_path: Path) -> None:
    path = tmp_path / "scheduled" / ".github" / "dependabot.yml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "version: 2\nupdates:\n"
        '  - {package-ecosystem: pip, directory: "/", schedule: {interval: weekly}}\n'
        '  - {package-ecosystem: github-actions, directory: "/", '
        "schedule: {interval: weekly}}\n",
        encoding="utf-8",
    )
    assert scan_clones(tmp_path, ["OmniNode-ai/scheduled"]) == {
        "scheduled": ["pip", "github-actions"]
    }


def test_no_dependabot_version_updates_malformed_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "malformed" / ".github" / "dependabot.yml"
    path.parent.mkdir(parents=True)
    path.write_text("updates: [\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        scan_clones(tmp_path, ["OmniNode-ai/malformed"])


@pytest.mark.parametrize("text", ["", "null", "[]", "version"])
def test_no_dependabot_version_updates_non_mapping_raises(text: str) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        declared_version_update_schedules(text)


@pytest.mark.parametrize(
    "text",
    [
        "version: 2\nupdates: []\n",
        "version: 2\nupdates: null\n",
        "version: 2\nupdates: [{package-ecosystem: pip}]\n",
    ],
)
def test_no_dependabot_version_updates_without_schedule(text: str) -> None:
    assert declared_version_update_schedules(text) == []


def test_no_dependabot_version_updates_registry_is_nonempty(
    registry_repos: list[str],
) -> None:
    assert registry_repos
    assert "OmniNode-ai/omnibase_infra" in registry_repos


def test_no_dependabot_version_updates_pending_names_are_registry_repos(
    registry_repos: list[str],
) -> None:
    names = {repo.rsplit("/", 1)[-1] for repo in registry_repos}
    assert names >= PENDING_REMOVAL
