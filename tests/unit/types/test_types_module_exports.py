# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for types module exports.

This module verifies that all exports from omnibase_infra.types are importable
and accessible at runtime. This prevents regressions when adding or modifying
type aliases and type definitions.

Related:
    - OMN-1358: Reduce union complexity with type aliases
    - src/omnibase_infra/types/__init__.py
"""

from __future__ import annotations


class TestTypesModuleExports:
    """Tests for types module export availability."""

    def test_types_module_can_be_imported(self) -> None:
        """Verify types module can be imported without errors."""
        from omnibase_infra import types

        assert types is not None

    def test_all_exports_in_dunder_all(self) -> None:
        """Verify all items in __all__ can be imported."""
        from omnibase_infra import types

        assert hasattr(types, "__all__")
        assert isinstance(types.__all__, list)
        assert len(types.__all__) > 0

        # Verify each export in __all__ is accessible
        for export_name in types.__all__:
            assert hasattr(types, export_name), (
                f"Export '{export_name}' listed in __all__ but not accessible"
            )

    def test_type_aliases_are_exported(self) -> None:
        """Verify type alias exports are accessible."""
        from omnibase_infra.types import (
            ASTFunctionDef,
            MessageOutputCategory,
            PathInput,
            PolicyTypeInput,
        )

        # Type aliases should be defined (not None)
        assert ASTFunctionDef is not None
        assert MessageOutputCategory is not None
        assert PathInput is not None
        assert PolicyTypeInput is not None

    def test_model_exports_are_accessible(self) -> None:
        """Verify model and TypedDict exports are accessible."""
        from omnibase_infra.types import (
            ModelParsedDSN,
            TypeCacheInfo,
            TypedDictCapabilities,
        )

        assert ModelParsedDSN is not None
        assert TypeCacheInfo is not None
        assert TypedDictCapabilities is not None

    def test_model_parsed_dsn_is_instantiable(self) -> None:
        """Verify ModelParsedDSN can be instantiated."""
        from omnibase_infra.types import ModelParsedDSN

        # Should be a Pydantic model that can be instantiated
        dsn = ModelParsedDSN(
            scheme="postgresql",
            hostname="localhost",
            port=5432,
            username="user",
            password="pass",  # noqa: S106 - test fixture
            database="testdb",
        )
        assert dsn.scheme == "postgresql"
        assert dsn.hostname == "localhost"
        assert dsn.port == 5432

    def test_type_cache_info_is_namedtuple(self) -> None:
        """Verify TypeCacheInfo is a NamedTuple that can be instantiated."""
        from omnibase_infra.types import TypeCacheInfo

        # Should be a NamedTuple with 4 fields matching functools._CacheInfo
        cache_info = TypeCacheInfo(hits=10, misses=2, maxsize=128, currsize=5)
        assert cache_info.hits == 10
        assert cache_info.misses == 2
        assert cache_info.maxsize == 128
        assert cache_info.currsize == 5

    def test_typed_dict_capabilities_structure(self) -> None:
        """Verify TypedDictCapabilities has expected keys."""
        from omnibase_infra.types import TypedDictCapabilities

        # TypedDict should define expected keys
        assert hasattr(TypedDictCapabilities, "__annotations__")
        annotations = TypedDictCapabilities.__annotations__
        assert "operations" in annotations
        assert "protocols" in annotations
        assert "has_fsm" in annotations
        assert "method_signatures" in annotations


class TestTypesModuleSubmoduleAccess:
    """Tests for accessing types from submodules directly."""

    def test_type_aliases_direct_import(self) -> None:
        """Verify type aliases can be imported from submodule directly."""
        from omnibase_infra.types.type_infra_aliases import (
            ASTFunctionDef,
            MessageOutputCategory,
            PathInput,
            PolicyTypeInput,
        )

        assert ASTFunctionDef is not None
        assert MessageOutputCategory is not None
        assert PathInput is not None
        assert PolicyTypeInput is not None

    def test_type_cache_info_direct_import(self) -> None:
        """Verify TypeCacheInfo can be imported from submodule directly."""
        from omnibase_infra.types.type_cache_info import TypeCacheInfo

        assert TypeCacheInfo is not None

    def test_model_parsed_dsn_direct_import(self) -> None:
        """Verify ModelParsedDSN can be imported from submodule directly."""
        from omnibase_infra.types.type_dsn import ModelParsedDSN

        assert ModelParsedDSN is not None

    def test_typed_dict_capabilities_direct_import(self) -> None:
        """Verify TypedDictCapabilities can be imported from submodule directly."""
        from omnibase_infra.types.typed_dict_capabilities import TypedDictCapabilities

        assert TypedDictCapabilities is not None


class TestTypeAliasMyPyCompatibility:
    """Verify type aliases work with strict type checking.

    These tests ensure that the type aliases defined in omnibase_infra.types
    are compatible with mypy strict mode and can be used in function signatures
    without type errors.

    The tests verify:
    1. MessageOutputCategory accepts both EnumMessageCategory and EnumNodeOutputType
    2. Type aliases can be used in function parameters and return types
    3. All enum values of both types are accepted by the union type alias
    """

    def test_message_output_category_accepts_enum_message_category(self) -> None:
        """Verify MessageOutputCategory accepts EnumMessageCategory values."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.types import MessageOutputCategory

        def accepts_category(cat: MessageOutputCategory) -> str:
            """Function using MessageOutputCategory type alias."""
            return str(cat)

        # All EnumMessageCategory values should be accepted
        result_event = accepts_category(EnumMessageCategory.EVENT)
        result_command = accepts_category(EnumMessageCategory.COMMAND)
        result_intent = accepts_category(EnumMessageCategory.INTENT)

        assert "event" in result_event.lower()
        assert "command" in result_command.lower()
        assert "intent" in result_intent.lower()

    def test_message_output_category_accepts_enum_node_output_type(self) -> None:
        """Verify MessageOutputCategory accepts EnumNodeOutputType values."""
        from omnibase_infra.enums import EnumNodeOutputType
        from omnibase_infra.types import MessageOutputCategory

        def accepts_category(cat: MessageOutputCategory) -> str:
            """Function using MessageOutputCategory type alias."""
            return str(cat)

        # All EnumNodeOutputType values should be accepted
        result_event = accepts_category(EnumNodeOutputType.EVENT)
        result_command = accepts_category(EnumNodeOutputType.COMMAND)
        result_intent = accepts_category(EnumNodeOutputType.INTENT)
        result_projection = accepts_category(EnumNodeOutputType.PROJECTION)

        assert "event" in result_event.lower()
        assert "command" in result_command.lower()
        assert "intent" in result_intent.lower()
        assert "projection" in result_projection.lower()

    def test_message_output_category_in_function_signature(self) -> None:
        """Verify MessageOutputCategory works in function signatures."""
        from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType
        from omnibase_infra.types import MessageOutputCategory

        def process_category(category: MessageOutputCategory) -> MessageOutputCategory:
            """Function with MessageOutputCategory in both param and return type."""
            return category

        # Both enum types should work for input and output
        msg_cat = EnumMessageCategory.EVENT
        node_out = EnumNodeOutputType.PROJECTION

        result1 = process_category(msg_cat)
        result2 = process_category(node_out)

        assert result1 == msg_cat
        assert result2 == node_out

    def test_message_output_category_in_collection_types(self) -> None:
        """Verify MessageOutputCategory works in collection type hints."""
        from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType
        from omnibase_infra.types import MessageOutputCategory

        def collect_categories(
            categories: list[MessageOutputCategory],
        ) -> dict[str, MessageOutputCategory]:
            """Function using MessageOutputCategory in collection types."""
            return {cat.value: cat for cat in categories}

        mixed_categories: list[MessageOutputCategory] = [
            EnumMessageCategory.EVENT,
            EnumNodeOutputType.PROJECTION,
            EnumMessageCategory.COMMAND,
            EnumNodeOutputType.INTENT,
        ]

        result = collect_categories(mixed_categories)

        assert len(result) == 4
        assert "event" in result
        assert "projection" in result

    def test_message_output_category_type_identity(self) -> None:
        """Verify MessageOutputCategory preserves type identity of enum values."""
        from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType
        from omnibase_infra.types import MessageOutputCategory

        def get_category(cat: MessageOutputCategory) -> MessageOutputCategory:
            """Pass-through function to verify type is preserved."""
            return cat

        # Verify EnumMessageCategory identity is preserved
        msg_event = EnumMessageCategory.EVENT
        result_msg = get_category(msg_event)
        assert isinstance(result_msg, EnumMessageCategory)
        assert result_msg is msg_event

        # Verify EnumNodeOutputType identity is preserved
        node_projection = EnumNodeOutputType.PROJECTION
        result_node = get_category(node_projection)
        assert isinstance(result_node, EnumNodeOutputType)
        assert result_node is node_projection

    def test_path_input_accepts_path_and_str(self) -> None:
        """Verify PathInput accepts both Path and str values."""
        from pathlib import Path

        from omnibase_infra.types import PathInput

        def process_path(path: PathInput) -> str:
            """Function using PathInput type alias."""
            return str(path)

        # Both Path and str should be accepted
        path_obj = Path("/workspace/example/test.py")
        path_str = "/workspace/example/test.py"

        result1 = process_path(path_obj)
        result2 = process_path(path_str)

        assert result1 == "/workspace/example/test.py"
        assert result2 == "/workspace/example/test.py"

    def test_policy_type_input_accepts_enum_and_str(self) -> None:
        """Verify PolicyTypeInput accepts both EnumPolicyType and str values."""
        from omnibase_infra.enums import EnumPolicyType
        from omnibase_infra.types import PolicyTypeInput

        def process_policy(policy: PolicyTypeInput) -> str:
            """Function using PolicyTypeInput type alias."""
            return str(policy)

        # Both EnumPolicyType and str should be accepted
        result1 = process_policy(EnumPolicyType.ORCHESTRATOR)
        result2 = process_policy("custom_policy")

        assert "orchestrator" in result1.lower()
        assert result2 == "custom_policy"

    def test_ast_function_def_accepts_both_function_types(self) -> None:
        """Verify ASTFunctionDef accepts both ast function definition types."""
        import ast

        from omnibase_infra.types import ASTFunctionDef

        def get_function_name(func: ASTFunctionDef) -> str:
            """Function using ASTFunctionDef type alias."""
            return func.name

        # Parse sample code to get both function types
        sync_code = "def sync_func(): pass"
        async_code = "async def async_func(): pass"

        sync_tree = ast.parse(sync_code)
        async_tree = ast.parse(async_code)

        sync_func = sync_tree.body[0]
        async_func = async_tree.body[0]

        assert isinstance(sync_func, ast.FunctionDef)
        assert isinstance(async_func, ast.AsyncFunctionDef)

        # Both should be accepted by the type alias
        result1 = get_function_name(sync_func)
        result2 = get_function_name(async_func)

        assert result1 == "sync_func"
        assert result2 == "async_func"


class TestTypesModuleExportConsistency:
    """Tests verifying export consistency between top-level and submodules."""

    def test_exports_match_submodule_definitions(self) -> None:
        """Verify top-level exports match submodule definitions."""
        from omnibase_infra import types
        from omnibase_infra.types import (
            type_cache_info,
            type_dsn,
            type_infra_aliases,
            typed_dict_capabilities,
        )

        # Type aliases should be the same object
        assert types.ASTFunctionDef is type_infra_aliases.ASTFunctionDef
        assert types.MessageOutputCategory is type_infra_aliases.MessageOutputCategory
        assert types.PathInput is type_infra_aliases.PathInput
        assert types.PolicyTypeInput is type_infra_aliases.PolicyTypeInput

        # Models and TypedDicts should be the same object
        assert types.ModelParsedDSN is type_dsn.ModelParsedDSN
        assert types.TypeCacheInfo is type_cache_info.TypeCacheInfo
        assert (
            types.TypedDictCapabilities is typed_dict_capabilities.TypedDictCapabilities
        )

    def test_all_exports_count_matches_expected(self) -> None:
        """Verify the number of exports matches expected count."""
        from omnibase_infra import types

        # 4 type aliases + 3 models/TypedDicts = 7 exports
        expected_exports = {
            "ASTFunctionDef",
            "MessageOutputCategory",
            "PathInput",
            "PolicyTypeInput",
            "ModelParsedDSN",
            "TypeCacheInfo",
            "TypedDictCapabilities",
        }
        assert set(types.__all__) == expected_exports
