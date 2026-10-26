# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lease reply published on ``lease-expired`` (OMN-21424)."""

from __future__ import annotations

from typing import Literal

from omnibase_infra.models.lease.model_lease_reply import ModelLeaseReply


class ModelLeaseExpired(ModelLeaseReply):
    """A ``lease-expired`` reply; its class name routes it to its topic."""

    outcome: Literal["lease-expired"] = "lease-expired"


__all__ = ["ModelLeaseExpired"]
