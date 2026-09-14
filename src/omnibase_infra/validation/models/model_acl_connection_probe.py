# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One post-apply connection assertion for a (database, principal) pair."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.enums.enum_acl_connection_probe_kind import (
    EnumAclConnectionProbeKind,
)

#: Environment-variable prefix each per-principal probe DSN is read from.
ROLE_DSN_ENV_PREFIX = "ACL_ROLE_DSN_"


def resolve_role_dsn_env_name(principal: str) -> str:
    """Name the environment variable a principal's probe DSN is read from."""
    return f"{ROLE_DSN_ENV_PREFIX}{principal.upper()}"


class ModelAclConnectionProbe(BaseModel):
    """One connection assertion, carrying no credential value.

    The probe names the environment variable its DSN is read from and never
    the value, so a report, a log line, or a traceback cannot leak a secret.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    database: str
    kind: EnumAclConnectionProbeKind
    principal: str | None = None

    def describe(self) -> str:
        """Render the probe for a report, naming no credential value."""
        if self.kind is EnumAclConnectionProbeKind.NEGATIVE_PUBLIC:
            return f"{self.kind.value} PUBLIC -> {self.database}"
        principal = self.principal or "PUBLIC"
        env_name = resolve_role_dsn_env_name(principal)
        return f"{self.kind.value} {principal} -> {self.database} (dsn from {env_name})"


__all__ = [
    "ROLE_DSN_ENV_PREFIX",
    "ModelAclConnectionProbe",
    "resolve_role_dsn_env_name",
]
