# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline event -- result published.

The pipeline's terminal event, published by the validation orchestrator and
projected into the validation event ledger. It is the adjudicator's verdict
under its own event class (the published-event routing keys on the class
name).

Ticket: OMN-21056
"""

from __future__ import annotations

from omnibase_infra.models.validation.model_validation_adjudication_completed import (
    ModelValidationAdjudicationCompleted,
)


class ModelValidationResultPublished(ModelValidationAdjudicationCompleted):
    """Final validation result for one pattern candidate."""


__all__: list[str] = ["ModelValidationResultPublished"]
