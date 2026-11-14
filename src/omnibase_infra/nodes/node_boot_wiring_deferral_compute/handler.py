# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure boot wiring deferral decision (OMN-21318).

A candidate none of whose subscribe topics is in use is deferred: its handlers
are neither imported nor registered at boot and are wired by its first request.
Every input is explicit; no broker, clock, file or environment is read here.
"""

from __future__ import annotations

from omnibase_infra.handlers.boot_wiring_deferral import (
    decide_deferred_contracts,
)
from omnibase_infra.models.contract_hotload.model_boot_wiring_deferral_request import (
    ModelBootWiringDeferralRequest,
)
from omnibase_infra.models.contract_hotload.model_boot_wiring_deferral_result import (
    ModelBootWiringDeferralResult,
)

__all__ = ["NodeBootWiringDeferralCompute"]


class NodeBootWiringDeferralCompute:
    """Definition-B handler: the broker's reading in, the deferred names out."""

    def handle(
        self, request: ModelBootWiringDeferralRequest
    ) -> ModelBootWiringDeferralResult:
        return decide_deferred_contracts(request)
