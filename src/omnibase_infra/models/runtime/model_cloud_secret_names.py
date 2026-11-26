# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cloud-mode secret names the runtime must be able to resolve (OMN-26712).

In cloud mode the gateway and the tenant identity writer connect to the cloud database through
the secret resolver. ``REQUIRED`` is the list the cloud owner stores secrets for; a rendered
resolver config that omits one refuses to boot, naming it.

Cloud mode is the explicit flag ``ONEX_CLOUD_MODE``, default OFF: a strict gate that needs the
cloud owner's secrets in place ships behind a flag and is flipped once they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)


class ModelCloudSecretNames(BaseModel):
    """The cloud-mode connection names and the boot refusals that enforce them."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    MODE_ENV_VAR: ClassVar[str] = "ONEX_CLOUD_MODE"
    RESOLVER_PATH_ENV_VAR: ClassVar[str] = "ONEX_SECRET_RESOLVER_CONFIG_PATH"
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "gateway.database.dsn",
        "database.tenant_identity_writer.dsn",
    )

    @classmethod
    def enabled(cls, environ: Mapping[str, str]) -> bool:
        """Is ``ONEX_CLOUD_MODE`` set to a true value?"""
        value = environ.get(cls.MODE_ENV_VAR, "").strip().lower()
        return value in {"1", "true", "yes"}

    @classmethod
    def render_required(cls, environ: Mapping[str, str]) -> bool:
        """Does boot render the resolver config: a declared target path, or cloud mode."""
        return bool(environ.get(cls.RESOLVER_PATH_ENV_VAR)) or cls.enabled(environ)

    @classmethod
    def refuse_disabled_render(cls, environ: Mapping[str, str]) -> Path | None:
        """Refuse a cloud-mode boot whose rendering is switched off; else return ``None``."""
        if cls.enabled(environ):
            raise ProtocolConfigurationError(
                f"CLOUD_SECRET_NAMES_MISSING: {cls.MODE_ENV_VAR} is set and secret "
                "resolver config rendering is disabled; the required connection names "
                f"{list(cls.REQUIRED)} cannot be resolved"
            )
        return None

    @classmethod
    def check(
        cls, config: ModelSecretResolverConfig, environ: Mapping[str, str]
    ) -> ModelSecretResolverConfig:
        """Return ``config``; in cloud mode raise when it omits a required name."""
        if cls.enabled(environ):
            declared = {mapping.logical_name for mapping in config.mappings}
            missing = [name for name in cls.REQUIRED if name not in declared]
            if missing:
                raise ProtocolConfigurationError(
                    f"CLOUD_SECRET_NAMES_MISSING: {cls.MODE_ENV_VAR} is set and the "
                    f"secret resolver config declares no mapping for: {missing}"
                )
        return config
