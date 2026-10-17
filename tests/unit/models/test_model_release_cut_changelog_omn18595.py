# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The release-cut changelog rule binds only heads that change CHANGELOG.md (OMN-18595).

dev carried pyproject 0.38.75 under a top heading v0.38.74; a cascade bump citing
OMN-18595 leaves CHANGELOG.md alone and must pass, while a release cut must name
its version.
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.model_release_cut_changelog import (
    ModelReleaseCutChangelog,
)

_DEV_CHANGELOG = "# Changelog\n\n## v0.38.74\n\n- entry\n\n## v0.38.73\n"


def _decision(changelog: str, *, cut: bool) -> ModelReleaseCutChangelog:
    return ModelReleaseCutChangelog(
        pyproject_version="0.38.75", changelog_text=changelog, is_release_cut=cut
    )


@pytest.mark.unit
def test_a_head_that_leaves_the_changelog_alone_passes() -> None:
    decision = _decision(_DEV_CHANGELOG, cut=False)
    assert decision.violations == ()
    assert decision.exit_code == 0


@pytest.mark.unit
def test_a_release_cut_naming_its_version_passes() -> None:
    decision = _decision("# Changelog\n\n## v0.38.75\n\n" + _DEV_CHANGELOG, cut=True)
    assert decision.top_heading_version == "0.38.75"
    assert decision.exit_code == 0


@pytest.mark.unit
def test_a_release_cut_naming_another_version_fails() -> None:
    decision = _decision("# Changelog\n\n## v9.9.9\n\n" + _DEV_CHANGELOG, cut=True)
    assert decision.exit_code == 1
    assert "9.9.9" in decision.violations[0]


@pytest.mark.unit
def test_a_release_cut_without_a_heading_fails() -> None:
    decision = _decision("# Changelog\n", cut=True)
    assert decision.top_heading_version == ""
    assert decision.exit_code == 1


@pytest.mark.unit
def test_dev_read_as_a_release_cut_fails() -> None:
    # An unreadable merge base counts as a release cut; dev's state then fails.
    assert _decision(_DEV_CHANGELOG, cut=True).exit_code == 1
