# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The whole proof-profile registry, config/lab_proof_profiles.yaml.

Ticket: OMN-19565
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)
from omnibase_infra.lab_proof.model_lab_proof_companion_rule import (
    ModelLabProofCompanionRule,
)
from omnibase_infra.lab_proof.model_lab_proof_excluded_repo import (
    ModelLabProofExcludedRepo,
)
from omnibase_infra.lab_proof.model_lab_proof_profile import ModelLabProofProfile


class ModelLabProofProfileRegistry(BaseModel):
    """Every registry repository, exactly one profile each.

    ``registry_repos`` is the repository list this registry answers for, copied
    from the registry table it cites in ``registry_source``. A repository that
    takes no pull requests is listed in ``excluded_repos`` with a reason, so a
    missing row is always an error and never a silent omission.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    registry_source: str = Field(min_length=1)
    registry_repos: tuple[str, ...] = Field(min_length=1)
    excluded_repos: tuple[ModelLabProofExcludedRepo, ...] = ()
    change_control_companion: ModelLabProofCompanionRule
    profiles: tuple[ModelLabProofProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_row_per_repo(self) -> ModelLabProofProfileRegistry:
        listed = list(self.registry_repos)
        if len(listed) != len(set(listed)):
            raise ValueError("registry_repos lists a repository twice")
        excluded = {entry.repo for entry in self.excluded_repos}
        both = sorted(excluded & set(listed))
        if both:
            raise ValueError(f"repositories both listed and excluded: {both}")
        rows: dict[str, int] = {}
        for profile in self.profiles:
            rows[profile.repo] = rows.get(profile.repo, 0) + 1
        missing = sorted(repo for repo in listed if repo not in rows)
        if missing:
            raise ValueError(
                "registry repositories with no proof profile row: " + ", ".join(missing)
            )
        extra = sorted(repo for repo in rows if repo not in listed)
        if extra:
            raise ValueError(
                "proof profile rows for repositories not in registry_repos: "
                + ", ".join(extra)
            )
        duplicated = sorted(repo for repo, count in rows.items() if count > 1)
        if duplicated:
            raise ValueError(
                "repositories with more than one profile row (use variants): "
                + ", ".join(duplicated)
            )
        ids = [profile.profile_key for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate profile_key")
        companion_repo = self.change_control_companion.repo
        for profile in self.profiles:
            if (
                EnumLabProofExemptClass.BOT_CHANGE_CONTROL_COMPANION
                in profile.exempt_classes
                and profile.repo != companion_repo
            ):
                raise ValueError(
                    f"profile {profile.profile_key}: bot_change_control_companion is "
                    f"only valid on {companion_repo}"
                )
        return self

    def profile_for(self, repo: str) -> ModelLabProofProfile:
        """Return the one profile row for ``repo``, or raise KeyError."""
        for profile in self.profiles:
            if profile.repo == repo:
                return profile
        raise KeyError(f"no lab proof profile for {repo}")


__all__ = ["ModelLabProofProfileRegistry"]
