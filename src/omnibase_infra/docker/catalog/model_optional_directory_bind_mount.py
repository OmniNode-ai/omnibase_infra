# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed declaration for an optional host-directory bind mount."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class ModelOptionalDirectoryBindMount:
    """A host directory bind mount whose source comes from an env var.

    The mount is OPTIONAL in that the repository commits no value for
    ``source_env``; it is not conditional in the render. The generator always
    emits it, with a directory-valued compose default for the unset case, so the
    rendered compose -- and in particular its required-var name set -- is a
    function of this declaration and not of the machine that ran the render
    (OMN-17291).

    The default is a DIRECTORY, never a file sentinel such as ``/dev/null``,
    which is what stops Docker being asked to bind a file onto a directory
    target. A configured source is validated as an existing absolute directory
    at render time.
    """

    source_env: str
    container_path: str
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.source_env:
            raise ValueError(
                "optional directory bind mount source_env must not be empty"
            )
        if not self.container_path.startswith("/"):
            raise ValueError(
                "optional directory bind mount container_path must be absolute"
            )
