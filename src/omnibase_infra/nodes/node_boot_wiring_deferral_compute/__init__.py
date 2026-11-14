# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Which contracts boot leaves unwired until their first request (OMN-21318)."""

from omnibase_infra.nodes.node_boot_wiring_deferral_compute.handler import (
    NodeBootWiringDeferralCompute,
)

__all__ = ["NodeBootWiringDeferralCompute"]
