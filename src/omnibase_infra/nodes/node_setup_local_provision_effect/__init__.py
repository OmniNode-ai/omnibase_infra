# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Setup local provision effect node — Docker Compose provisioning for local services.

Ticket: OMN-3493
"""

from omnibase_infra.nodes.node_setup_local_provision_effect.node import (
    NodeLocalProvisionEffect,
)

__all__: list[str] = ["NodeLocalProvisionEffect"]
