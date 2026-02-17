# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical-required security policy model.

Defines the ``INFISICAL_REQUIRED`` policy for nodes and handlers that mandate
Infisical as their secret backend. When this policy is active, only Infisical
sources are permitted.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelInfisicalPolicy(BaseModel):
    """Policy requiring Infisical as the secret management backend.

    When attached to a handler or node, this policy enforces:
    - All secret references MUST use ``infisical:`` scheme
    - SecretResolver MUST have an Infisical handler configured

    Attributes:
        policy_name: Fixed to ``INFISICAL_REQUIRED``.
        enforce: Whether to enforce (True) or warn (False).
        allowed_source_types: Secret source types permitted under this policy.

    Example:
        >>> policy = ModelInfisicalPolicy()
        >>> policy.policy_name
        'INFISICAL_REQUIRED'
        >>> "infisical" in policy.allowed_source_types
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    policy_name: Literal["INFISICAL_REQUIRED"] = Field(
        default="INFISICAL_REQUIRED",
        description="Policy identifier.",
    )
    enforce: bool = Field(
        default=True,
        description="Whether to enforce the policy (True) or only warn (False).",
    )
    allowed_source_types: frozenset[str] = Field(
        default=frozenset({"infisical"}),
        description="Secret source types permitted under this policy.",
    )


__all__: list[str] = ["ModelInfisicalPolicy"]
