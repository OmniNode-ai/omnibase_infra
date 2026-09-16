# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract's own worked examples, which both consumers run."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.gate_binding.model_gate_binding_accepted_fixture import (
    ModelGateBindingAcceptedFixture,
)
from omnibase_infra.gate_binding.model_gate_binding_line_fixture import (
    ModelGateBindingLineFixture,
)
from omnibase_infra.gate_binding.model_gate_binding_rejected_fixture import (
    ModelGateBindingRejectedFixture,
)


class ModelGateBindingFixtures(BaseModel):
    """Examples a consumer runs against its own resolver.

    They live in the contract rather than in either repository's tests so a
    form added to the grammar turns BOTH suites red until both resolvers
    support it. That is the mechanism that keeps the two from drifting apart
    again.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    accepted: tuple[ModelGateBindingAcceptedFixture, ...] = Field(default=())
    rejected: tuple[ModelGateBindingRejectedFixture, ...] = Field(default=())
    line_pattern_superset_fixtures: tuple[ModelGateBindingLineFixture, ...] = Field(
        default=()
    )
