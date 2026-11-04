# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One compose file and service the standalone recreate route may touch."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelStandaloneComposeRecreateTarget(BaseModel):
    """A declared target of ``node_standalone_compose_recreate_effect``.

    ``container_key`` is the compose service's fixed container name, the one container the
    recreate may change.
    ``env_file_variable`` names the environment variable, on the host that serves the target,
    that holds the path of the host's own compose env file. An agent whose environment lacks it
    does not serve the target.
    ``carried_files`` are the repo-relative files the compose service bind-mounts from the host's
    tree. The recreate writes each from the requested git ref into that tree before the service is
    recreated, so a host copy that no merge refreshes cannot go stale.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_file: str = Field(min_length=1)
    service: str = Field(min_length=1)
    container_key: str = Field(min_length=1)
    env_file_variable: str = Field(min_length=1)
    carried_files: tuple[str, ...] = ()

    @field_validator("carried_files")
    @classmethod
    def _carried_files_are_plain_repo_relative_paths(
        cls, paths: tuple[str, ...]
    ) -> tuple[str, ...]:
        for path in paths:
            parts = PurePosixPath(path).parts
            if (
                not parts
                or PurePosixPath(path).is_absolute()
                or ".." in parts
                or path.startswith("-")
                or "\\" in path
                or ":" in path
            ):
                raise ValueError(
                    f"carried file {path!r} is not a plain repo-relative path"
                )
        return paths
