# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the LLM models public exports (migrated from nodes.effects.models).

Tests cover:
- Individual importability of each LLM model class from the new canonical location
- Individual importability of adapter functions (OMN-2318)
- __all__ completeness for model names and adapter functions
- All __all__ entries resolve to classes or callables

OMN-2103: Phase 3 shared LLM models - export verification
OMN-2318: Integrate SPI 0.9.0 LLM cost tracking contracts
OMN-3989: Migrated from omnibase_infra.nodes.effects.models -> omnibase_infra.models.llm
"""

from __future__ import annotations

import types

import pytest

# ============================================================================
# Individual Import Tests
# ============================================================================


class TestIndividualImports:
    """Verify each LLM model is importable from the canonical models.llm package."""

    def test_model_llm_function_call_importable(self) -> None:
        """ModelLlmFunctionCall is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmFunctionCall

        assert ModelLlmFunctionCall is not None

    def test_model_llm_function_def_importable(self) -> None:
        """ModelLlmFunctionDef is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmFunctionDef

        assert ModelLlmFunctionDef is not None

    def test_model_llm_tool_call_importable(self) -> None:
        """ModelLlmToolCall is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmToolCall

        assert ModelLlmToolCall is not None

    def test_model_llm_tool_choice_importable(self) -> None:
        """ModelLlmToolChoice is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmToolChoice

        assert ModelLlmToolChoice is not None

    def test_model_llm_tool_definition_importable(self) -> None:
        """ModelLlmToolDefinition is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmToolDefinition

        assert ModelLlmToolDefinition is not None

    def test_model_llm_usage_importable(self) -> None:
        """ModelLlmUsage is importable from models.llm package."""
        from omnibase_infra.models.llm import ModelLlmUsage

        assert ModelLlmUsage is not None


class TestAdapterImports:
    """Verify adapter functions are importable from models.llm package."""

    def test_to_call_metrics_importable(self) -> None:
        """to_call_metrics is importable from models.llm package."""
        from omnibase_infra.models.llm import to_call_metrics

        assert callable(to_call_metrics)

    def test_to_usage_normalized_importable(self) -> None:
        """to_usage_normalized is importable from models.llm package."""
        from omnibase_infra.models.llm import to_usage_normalized

        assert callable(to_usage_normalized)

    def test_to_usage_raw_importable(self) -> None:
        """to_usage_raw is importable from models.llm package."""
        from omnibase_infra.models.llm import to_usage_raw

        assert callable(to_usage_raw)


# ============================================================================
# __all__ Completeness Tests
# ============================================================================


class TestAllExports:
    """Verify __all__ contains all expected names and entries resolve correctly."""

    _EXPECTED_NEW_MODELS: tuple[str, ...] = (
        "ModelLlmFunctionCall",
        "ModelLlmFunctionDef",
        "ModelLlmToolCall",
        "ModelLlmToolChoice",
        "ModelLlmToolDefinition",
        "ModelLlmUsage",
    )

    _EXPECTED_ADAPTER_FUNCTIONS: tuple[str, ...] = (
        "to_call_metrics",
        "to_usage_normalized",
        "to_usage_raw",
    )

    def test_all_contains_all_new_models(self) -> None:
        """__all__ includes every LLM model class name in models.llm."""
        import omnibase_infra.models.llm as llm_models

        all_exports = set(llm_models.__all__)

        for name in self._EXPECTED_NEW_MODELS:
            assert name in all_exports, f"{name} missing from __all__"

    def test_all_contains_adapter_functions(self) -> None:
        """__all__ includes adapter function names (OMN-2318)."""
        import omnibase_infra.models.llm as llm_models

        all_exports = set(llm_models.__all__)

        for name in self._EXPECTED_ADAPTER_FUNCTIONS:
            assert name in all_exports, f"{name} missing from __all__"

    def test_all_exports_are_classes_or_callables(self) -> None:
        """Every name in __all__ resolves to a class or callable."""
        import omnibase_infra.models.llm as llm_models

        for name in llm_models.__all__:
            obj = getattr(llm_models, name)
            assert isinstance(obj, (type, types.FunctionType)), (
                f"__all__ entry {name!r} is {type(obj).__name__}, "
                f"expected a class or function"
            )
