# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed declaration for an optional host-directory bind mount."""

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

    ``absent_source`` is that default, declared per mount in the service
    manifest. It is a DIRECTORY, never a file sentinel such as ``/dev/null``,
    which is what stops Docker being asked to bind a file onto a directory
    target. A configured source is validated as an existing absolute directory
    at render time.
    """

    source_env: str
    container_path: str
    absent_source: str
    read_only: bool = True

    def __post_init__(self) -> None:
        strs = (self.source_env, self.container_path, self.absent_source)
        if not all(isinstance(v, str) for v in strs) or not isinstance(
            self.read_only, bool
        ):
            raise ValueError("source_env, paths must be strings; read_only a bool")
        if not self.source_env or not all(p.startswith("/") for p in strs[1:]):
            raise ValueError("source_env is required; paths must be absolute")
