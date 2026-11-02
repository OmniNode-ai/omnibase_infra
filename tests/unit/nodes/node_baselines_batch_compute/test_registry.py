# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry discoverability tests for NodeBaselinesBatchCompute.

Mirrors tests/unit/test_baseline_comparison_compute.py pattern.

Ticket: OMN-3045
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


class TestHandlerImportable:
    """Prove handler module path in contract.yaml is resolvable."""

    def test_handler_importable(self) -> None:
        """Handler module path in contract is importable."""
        mod = importlib.import_module(
            "omnibase_infra.nodes.node_baselines_batch_compute.handlers.handler_baselines_batch_compute"
        )
        assert hasattr(mod, "HandlerBaselinesBatchCompute")

    def test_models_importable(self) -> None:
        """Models module path in contract is importable."""
        mod = importlib.import_module(
            "omnibase_infra.nodes.node_baselines_batch_compute.models"
        )
        assert hasattr(mod, "ModelBaselinesBatchComputeCommand")
        assert hasattr(mod, "ModelBaselinesBatchComputeOutput")

    def test_node_importable(self) -> None:
        """Node class is importable from package root."""
        from omnibase_infra.nodes.node_baselines_batch_compute.node import (
            NodeBaselinesBatchCompute,
        )

        assert NodeBaselinesBatchCompute is not None

    def test_all_exports_importable(self) -> None:
        """All public exports from package __init__ are importable."""
        from omnibase_infra.nodes.node_baselines_batch_compute import (
            HandlerBaselinesBatchCompute,
            ModelBaselinesBatchComputeCommand,
            ModelBaselinesBatchComputeOutput,
            NodeBaselinesBatchCompute,
        )

        assert NodeBaselinesBatchCompute is not None
        assert HandlerBaselinesBatchCompute is not None
        assert ModelBaselinesBatchComputeCommand is not None
        assert ModelBaselinesBatchComputeOutput is not None
