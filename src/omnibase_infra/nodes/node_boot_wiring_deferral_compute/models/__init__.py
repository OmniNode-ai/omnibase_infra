# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Models for the boot wiring deferral COMPUTE node."""

from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models.model_boot_wiring_candidate import (
    ModelBootWiringCandidate,
)
from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models.model_boot_wiring_deferral_request import (
    ModelBootWiringDeferralRequest,
)
from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models.model_boot_wiring_deferral_result import (
    ModelBootWiringDeferralResult,
)

__all__ = [
    "ModelBootWiringCandidate",
    "ModelBootWiringDeferralRequest",
    "ModelBootWiringDeferralResult",
]
