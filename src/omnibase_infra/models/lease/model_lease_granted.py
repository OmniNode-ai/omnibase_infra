# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lease reply published on ``lease-granted`` (OMN-21424)."""

from __future__ import annotations

from typing import Literal

from omnibase_infra.models.lease.model_lease_reply import ModelLeaseReply


class ModelLeaseGranted(ModelLeaseReply):
    """A ``lease-granted`` reply; its class name routes it to its topic."""

    outcome: Literal["lease-granted"] = "lease-granted"


__all__ = ["ModelLeaseGranted"]
