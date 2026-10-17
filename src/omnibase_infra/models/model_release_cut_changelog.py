# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Release-cut changelog decision for contracts/OMN-18595.yaml (OMN-18595).

The dependency cascade carries OMN-18595 into downstream bump pull requests
that cut no release. A head is a release cut when it changes CHANGELOG.md
against its merge base with the base branch, or when that merge base cannot be
read (fail closed); only a release cut must carry a first ``## v<version>``
heading naming the pyproject.toml version. The contract's check collects the
facts; this model only decides. No I/O.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

_HEADING = re.compile(r"^## v(\S+)", re.MULTILINE)


class ModelReleaseCutChangelog(BaseModel):
    """Facts read at a head, and the release-cut rule over them."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pyproject_version: str = Field(
        ..., description="project.version in pyproject.toml."
    )
    changelog_text: str = Field(..., description="CHANGELOG.md contents at the head.")
    is_release_cut: bool = Field(
        ...,
        description=(
            "True when the head changes CHANGELOG.md against its merge base, "
            "or that merge base cannot be read."
        ),
    )

    @property
    def top_heading_version(self) -> str:
        """Version named by the first ``## v<version>`` heading, or ''."""
        match = _HEADING.search(self.changelog_text)
        return match.group(1) if match else ""

    @property
    def violations(self) -> tuple[str, ...]:
        """Why the head fails the rule; empty when it holds or is not a release cut."""
        if (
            not self.is_release_cut
            or self.top_heading_version == self.pyproject_version
        ):
            return ()
        heading = self.top_heading_version or "no version"
        return (
            f"release cut: top CHANGELOG.md heading names {heading!r}, "
            f"pyproject.toml version is {self.pyproject_version!r}",
        )

    @property
    def exit_code(self) -> int:
        """0 when the rule holds, 1 when it does not."""
        return 1 if self.violations else 0


__all__ = ["ModelReleaseCutChangelog"]
