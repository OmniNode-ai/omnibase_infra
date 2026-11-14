# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure boot wiring deferral decision (OMN-21318).

A candidate none of whose subscribe topics is in use is deferred: its handlers
are neither imported nor registered at boot and are wired by its first request.
Every input is explicit; no broker, clock, file or environment is read here.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models import (
    ModelBootWiringDeferralRequest,
    ModelBootWiringDeferralResult,
)

__all__ = ["NodeBootWiringDeferralCompute", "decide_deferred_contracts"]


def decide_deferred_contracts(
    request: ModelBootWiringDeferralRequest,
) -> ModelBootWiringDeferralResult:
    """The candidates none of whose subscribe topics is in use, sorted by name."""
    return ModelBootWiringDeferralResult(
        deferred_contract_names=tuple(
            sorted(
                candidate.contract_name
                for candidate in request.candidates
                if not request.in_use_topics.intersection(candidate.subscribe_topics)
            )
        )
    )


class NodeBootWiringDeferralCompute:
    """Definition-B handler: the broker's reading in, the deferred names out."""

    def handle(
        self, request: ModelBootWiringDeferralRequest
    ) -> ModelBootWiringDeferralResult:
        return decide_deferred_contracts(request)
