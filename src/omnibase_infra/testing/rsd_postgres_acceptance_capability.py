# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topology-neutral operator capability contract for RSD PostgreSQL tests.

This module deliberately contains no PostgreSQL client, endpoint, DSN, secret,
or environment lookup.  An operator supplies the capability implementation at
pytest configuration time; this module only validates its opaque reference and
the evidence accompanying it.
"""

from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.testing.protocol_rsd_postgres_acceptance_capability import (
    CapabilityResolver,
    PostgresLifecycleConnectionFactory,
    RsdPostgresAcceptanceCapability,
)

POSTGRES_ACCEPTANCE_CAPABILITY_REF = "capability://rsd/postgres/acceptance"
_CAPABILITY_REF_PATTERN = (
    r"^capability://rsd/postgres/"
    r"(?:acceptance|[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)


class ModelRsdPostgresAcceptanceEvidence(BaseModel):
    """Value-free proof supplied by the operator-owned capability.

    The three attestation refs identify separately governed proofs without
    carrying database, host, user, role, or credential values into the test
    runner.  ``authority_disposition`` records that these guarantees are a
    trusted operator boundary; it does not dynamically inspect a connection.
    ``Literal[True]`` makes the declared lifecycle guarantees fail closed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["rsd_postgres_acceptance_evidence.v1"]
    capability_ref: str = Field(pattern=_CAPABILITY_REF_PATTERN)
    target_identity_attestation_ref: str = Field(
        pattern=r"^attestation://rsd/postgres/target/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    session_identity_attestation_ref: str = Field(
        pattern=r"^attestation://rsd/postgres/session/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    role_identity_attestation_ref: str = Field(
        pattern=r"^attestation://rsd/postgres/role/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    authority_disposition: Literal["trusted_operator_factory_contract"]
    fresh_lease: Literal[True]
    exclusive_lease: Literal[True]
    transaction_idle: Literal[True]


class RsdPostgresAcceptanceResolutionError(RuntimeError):
    """Normalized, topology-safe capability resolution failure."""


def resolve_postgres_lifecycle_factory(
    resolver: CapabilityResolver,
    capability_ref: str,
) -> PostgresLifecycleConnectionFactory:
    """Resolve and validate the injected factory without performing I/O."""

    if not isinstance(capability_ref, str) or not re.fullmatch(
        _CAPABILITY_REF_PATTERN, capability_ref
    ):
        raise RsdPostgresAcceptanceResolutionError(
            "invalid RSD PostgreSQL acceptance capability reference"
        )

    try:
        capability = resolver(capability_ref)
    except RsdPostgresAcceptanceResolutionError:
        raise
    except Exception:  # noqa: BLE001 — normalize arbitrary provider failures
        raise RsdPostgresAcceptanceResolutionError(
            "operator capability lookup failed"
        ) from None
    try:
        factory = getattr(capability, "connection_factory", None)
        evidence = getattr(capability, "evidence", None)
        if not callable(factory):
            raise RsdPostgresAcceptanceResolutionError(
                "operator capability must provide a connection factory"
            )
        if not isinstance(evidence, ModelRsdPostgresAcceptanceEvidence):
            raise RsdPostgresAcceptanceResolutionError(
                "operator capability must provide typed identity evidence"
            )
        if evidence.capability_ref != capability_ref:
            raise RsdPostgresAcceptanceResolutionError(
                "operator capability evidence reference does not match"
            )
        if evidence.authority_disposition != "trusted_operator_factory_contract":
            raise RsdPostgresAcceptanceResolutionError(
                "operator capability authority disposition is not trusted"
            )
    except RsdPostgresAcceptanceResolutionError:
        raise
    except Exception:  # noqa: BLE001 — normalize arbitrary property failures
        raise RsdPostgresAcceptanceResolutionError(
            "operator capability inspection failed"
        ) from None
    return cast("PostgresLifecycleConnectionFactory", factory)


__all__ = [
    "CapabilityResolver",
    "ModelRsdPostgresAcceptanceEvidence",
    "POSTGRES_ACCEPTANCE_CAPABILITY_REF",
    "PostgresLifecycleConnectionFactory",
    "RsdPostgresAcceptanceCapability",
    "RsdPostgresAcceptanceResolutionError",
    "resolve_postgres_lifecycle_factory",
]
