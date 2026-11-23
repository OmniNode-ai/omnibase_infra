# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the wiring-report ratchet (OMN-26335)."""

from omnibase_infra.models.wiring_ratchet.model_wiring_counts import ModelWiringCounts
from omnibase_infra.models.wiring_ratchet.model_wiring_report_ratchet_request import (
    ModelWiringReportRatchetRequest,
)

__all__ = ["ModelWiringCounts", "ModelWiringReportRatchetRequest"]
