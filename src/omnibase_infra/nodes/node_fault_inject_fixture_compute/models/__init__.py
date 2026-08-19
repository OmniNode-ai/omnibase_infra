# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Models for the fault-injection fixture node.

Ticket: OMN-16265
"""

from omnibase_infra.nodes.node_fault_inject_fixture_compute.models.model_fault_inject_fixture_command import (
    MAX_INFLATE_RESULT_BYTES,
    ModelFaultInjectFixtureCommand,
)
from omnibase_infra.nodes.node_fault_inject_fixture_compute.models.model_fault_inject_fixture_result import (
    ModelFaultInjectFixtureResult,
)

__all__: list[str] = [
    "MAX_INFLATE_RESULT_BYTES",
    "ModelFaultInjectFixtureCommand",
    "ModelFaultInjectFixtureResult",
]
