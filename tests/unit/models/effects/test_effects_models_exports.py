# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Tests for the effects models __init__.py public exports.

Tests cover:
- Individual importability of each new LLM model class
- __all__ completeness for the six new model names
- All __all__ entries resolve to classes (types)

OMN-2103: Phase 3 shared LLM models - export verification
"""

from __future__ import annotations

import pytest

# ============================================================================
# Individual Import Tests
# ============================================================================


class TestIndividualImports:
    """Verify each new LLM model is importable from the effects models package."""

    def test_model_llm_function_call_importable(self) -> None:
        """ModelLlmFunctionCall is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmFunctionCall

        assert ModelLlmFunctionCall is not None

    def test_model_llm_function_def_importable(self) -> None:
        """ModelLlmFunctionDef is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmFunctionDef

        assert ModelLlmFunctionDef is not None

    def test_model_llm_tool_call_importable(self) -> None:
        """ModelLlmToolCall is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmToolCall

        assert ModelLlmToolCall is not None

    def test_model_llm_tool_choice_importable(self) -> None:
        """ModelLlmToolChoice is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmToolChoice

        assert ModelLlmToolChoice is not None

    def test_model_llm_tool_definition_importable(self) -> None:
        """ModelLlmToolDefinition is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmToolDefinition

        assert ModelLlmToolDefinition is not None

    def test_model_llm_usage_importable(self) -> None:
        """ModelLlmUsage is importable from effects models package."""
        from omnibase_infra.nodes.effects.models import ModelLlmUsage

        assert ModelLlmUsage is not None


# ============================================================================
# __all__ Completeness Tests
# ============================================================================


class TestAllExports:
    """Verify __all__ contains all new model names and all entries are classes."""

    _EXPECTED_NEW_MODELS: tuple[str, ...] = (
        "ModelLlmFunctionCall",
        "ModelLlmFunctionDef",
        "ModelLlmToolCall",
        "ModelLlmToolChoice",
        "ModelLlmToolDefinition",
        "ModelLlmUsage",
    )

    def test_all_contains_all_new_models(self) -> None:
        """__all__ includes every new LLM model class name."""
        import omnibase_infra.nodes.effects.models as effects_models

        all_exports = set(effects_models.__all__)

        for name in self._EXPECTED_NEW_MODELS:
            assert name in all_exports, f"{name} missing from __all__"

    def test_all_exports_are_classes(self) -> None:
        """Every name in __all__ resolves to a class (type)."""
        import omnibase_infra.nodes.effects.models as effects_models

        for name in effects_models.__all__:
            obj = getattr(effects_models, name)
            assert isinstance(obj, type), (
                f"__all__ entry {name!r} is {type(obj).__name__}, expected a class"
            )
