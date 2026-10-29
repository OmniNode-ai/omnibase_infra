# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for ``broker.principal.list`` (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerPrincipalListRequest(BaseModel):
    """List the broker's principals that carry a declaration's prefix.

    A principal is named ``<prefix><login>-<device>`` (the plan node's
    ``principal_name``, as ``scripts/lab_principal_issuer.py`` names it), so no
    separator is added after the prefix: the declarations' ``dev-`` already ends
    in one, and a principal is listed when its name starts with the prefix as
    given. The prefix follows the declaration rule ``[a-z][a-z0-9-]{0,15}``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    admin_url: str = Field(..., min_length=1, description="The Redpanda Admin API URL.")
    prefix: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9-]{0,15}$",
        description="The declaration's principal_prefix, matched as a plain prefix.",
    )
