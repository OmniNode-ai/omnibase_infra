# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a closed cascade bump DECLARES about itself (OMN-18233, clause 1).

Parsed out of the machine-readable provenance block the dependency-cascade
generator emits (OMN-16286). A pull request that declares none is not a cascade
bump as far as the supersession predicate is concerned, and keeps blocking —
which is the abandoned-work case the plan review named and refused to ignore.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCascadeProvenance(BaseModel):
    """What a cascade bump declares: a source package and a required version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_repo: str = Field(
        ...,
        min_length=3,
        description=(
            "`owner/name` of the repository whose release triggered this bump, "
            "verbatim from the provenance block's `Source repo` field."
        ),
    )
    required_version: str = Field(
        ...,
        min_length=1,
        description=(
            "The released version this bump was opened to deliver, verbatim "
            "from the provenance block's `Released version` field. PEP 440 "
            "parseable — a version that is not is refused at parse time, "
            "because clause 3 compares versions and cannot compare prose."
        ),
    )
    distribution: str = Field(
        ...,
        min_length=1,
        description=(
            "PEP 503 normalisation of the source repository's name — the "
            "spelling a `pyproject.toml` pin and a `uv.lock` entry actually "
            "carry (`omnibase_core` -> `omnibase-core`)."
        ),
    )


__all__ = ["ModelCascadeProvenance"]
