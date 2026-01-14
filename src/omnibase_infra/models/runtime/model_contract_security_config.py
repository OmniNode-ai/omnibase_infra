# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Security Configuration Model.

This module provides ModelContractSecurityConfig, a Pydantic model for
security configuration in handler contracts.

.. versionadded:: 0.7.0
    Created as part of OMN-1317 to fix one-model-per-file violation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractSecurityConfig(BaseModel):
    """Security configuration for handler contracts.

    Optional security metadata that can be specified in handler contracts
    for security-sensitive handlers.

    Attributes:
        trusted_namespace: Expected namespace prefix for handler class.
            If set, loader should verify handler_class starts with this prefix.
        audit_logging: Whether audit logging is enabled for this handler.
    """

    model_config = ConfigDict(extra="forbid")

    trusted_namespace: str | None = Field(
        default=None,
        description="Expected namespace prefix for handler class validation",
    )
    audit_logging: bool = Field(
        default=False,
        description="Whether audit logging is enabled for this handler",
    )


__all__ = ["ModelContractSecurityConfig"]
