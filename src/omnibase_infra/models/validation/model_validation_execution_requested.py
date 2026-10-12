# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline event -- execution requested.

Published by the validation orchestrator once a plan exists; consumed by the
validation executor. It carries the plan itself, so it is a distinct event
class (the published-event routing keys on the class name) over the plan's
own fields.

Ticket: OMN-21056
"""

from __future__ import annotations

from omnibase_infra.models.validation.model_validation_plan import ModelValidationPlan


class ModelValidationExecutionRequested(ModelValidationPlan):
    """Request to execute every enabled check of a validation plan."""


__all__: list[str] = ["ModelValidationExecutionRequested"]
