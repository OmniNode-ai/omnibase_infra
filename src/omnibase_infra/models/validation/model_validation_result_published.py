# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Validation result model — the pipeline's terminal event payload.

The adjudicator's ``ModelVerdict`` is already the ``adjudication-completed``
event. The orchestrator republishes it as ``validation-result-published`` under
this class, because the contract's ``published_events`` map routes a returned
model to its topic by class name.

Ticket: OMN-21056
"""

from __future__ import annotations

from omnibase_infra.models.validation.model_verdict import (
    ModelVerdict,
)


class ModelValidationResultPublished(ModelVerdict):
    """The final validation verdict for a pattern candidate, as published."""


__all__: list[str] = ["ModelValidationResultPublished"]
