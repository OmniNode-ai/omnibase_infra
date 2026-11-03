# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared ASGI mount served on the runtime's HTTP port (OMN-21288)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FACTORY_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_PREFIX_PATTERN = re.compile(r"^/[A-Za-z0-9._~\-]+(/[A-Za-z0-9._~\-]+)*$")


class ModelHttpMount(BaseModel):
    """An ASGI application a contract asks the runtime to serve.

    Declared in a node ``contract.yaml`` as the top-level ``http_mount`` block::

        http_mount:
          factory: "package.module:create_app"
          path_prefix: "/api/v1"

    The runtime calls ``factory`` with no arguments, expects an ASGI3 application
    back, runs its lifespan, and serves every request whose path is
    ``path_prefix`` or below it on the runtime's existing HTTP port. The
    application sees the request path unchanged and an empty ``root_path``, so
    an application that already owns absolute routes keeps them.

    A mount that cannot be built, or whose lifespan does not start, fails the
    runtime boot; it is never served partly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    factory: str = Field(
        ...,
        description=(
            "ASGI application factory as '<module>:<callable>', resolved by import "
            "and called with no arguments (the uvicorn --factory form)."
        ),
    )
    path_prefix: str = Field(
        ...,
        description=(
            "Absolute URL path prefix served by the mounted application. Starts "
            "with '/', is never the root, and has no trailing slash."
        ),
    )

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value: str) -> str:
        """Require the '<module>:<callable>' import form."""
        if not _FACTORY_PATTERN.match(value):
            raise ValueError(
                f"http_mount.factory must be '<module>:<callable>', got {value!r}"
            )
        return value

    @field_validator("path_prefix")
    @classmethod
    def validate_path_prefix(cls, value: str) -> str:
        """Require an absolute, non-root prefix with no trailing slash."""
        if not _PREFIX_PATTERN.match(value):
            raise ValueError(
                "http_mount.path_prefix must start with '/', name at least one "
                f"segment and carry no trailing slash, got {value!r}"
            )
        return value

    @classmethod
    def from_contract_block(cls, raw: object) -> ModelHttpMount | None:
        """Parse the contract's ``http_mount`` block; ``None`` when undeclared."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"http_mount must be a mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)

    def covers(self, path: str) -> bool:
        """True when ``path`` is the prefix itself or lies below it."""
        return path == self.path_prefix or path.startswith(f"{self.path_prefix}/")


__all__ = ["ModelHttpMount"]
