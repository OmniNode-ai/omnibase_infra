# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Permanent fault-injection fixture node — DLQ offset-withholding proof.

OMN-14498 follow-on. Ticket: OMN-16265
"""

from omnibase_infra.nodes.node_fault_inject_fixture_compute.node import (
    NodeFaultInjectFixtureCompute,
)

__all__: list[str] = ["NodeFaultInjectFixtureCompute"]
