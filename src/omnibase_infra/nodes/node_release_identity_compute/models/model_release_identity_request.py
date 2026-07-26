# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Release-identity request model — pre-collected inputs for the fitness gate.

Input model for the pure ``node_release_identity_compute`` COMPUTE node. All I/O
(reading ``pyproject.toml``, listing published git tags, diffing changed files) is
performed by the thin CLI collector/shim BEFORE the handler runs, and the results
are handed to the handler as this typed request. The handler is therefore pure and
deterministic: identical requests always produce identical decisions with no
subprocess or filesystem access.

Ticket: OMN-14471 (refactor of legacy scripts/check_release_identity.py, OMN-13412)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelReleaseIdentityRequest(BaseModel):
    """Pre-collected inputs for the release-identity fitness decision.

    Attributes:
        pyproject_version_raw: Raw ``project.version`` string read from
            ``pyproject.toml``. ``None`` when the key is absent (the collector
            passes through whatever ``data["project"]["version"]`` yielded, or
            ``None``). An empty or malformed value is a config error decided by
            the handler, not the collector.
        pyproject_path: Filesystem path to the ``pyproject.toml`` the version was
            read from. Used verbatim in the "no project.version" config-error
            message so the refactor preserves the legacy gate's output byte-for-byte.
        published_tags: Raw tag lines exactly as emitted by ``git tag --list``
            (order preserved, unparsed). Empty when the repository has no tags.
        changed_files: The changed-file set relative to the diff base, or ``None``
            when the source-change set could NOT be determined (no ``--base`` and
            no explicit ``--changed-file`` list). ``None`` means "cannot prove the
            diff is exempt" and the handler enforces the version-ahead invariant —
            mirroring the legacy gate's fail-safe branch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pyproject_version_raw: str | None = Field(
        default=None,
        description="Raw project.version string, or None if the key is absent.",
    )
    pyproject_path: str = Field(
        ...,
        min_length=1,
        description="Path to pyproject.toml (verbatim in config-error messages).",
    )
    published_tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Raw `git tag --list` lines; empty when there are no tags.",
    )
    changed_files: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Changed files vs the diff base, or None when undeterminable "
            "(no base + no explicit list) -> enforce the invariant."
        ),
    )


__all__: list[str] = ["ModelReleaseIdentityRequest"]
