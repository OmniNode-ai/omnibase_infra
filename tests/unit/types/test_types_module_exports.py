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
            VersionInput,
        )

        # Type aliases should be defined (not None)
        assert ASTFunctionDef is not None
        assert MessageOutputCategory is not None
        assert PathInput is not None
        assert PolicyTypeInput is not None
        assert VersionInput is not None

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
            VersionInput,
        )

        assert ASTFunctionDef is not None
        assert MessageOutputCategory is not None
        assert PathInput is not None
        assert PolicyTypeInput is not None
        assert VersionInput is not None

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
        assert types.VersionInput is type_infra_aliases.VersionInput

        # Models and TypedDicts should be the same object
        assert types.ModelParsedDSN is type_dsn.ModelParsedDSN
        assert types.TypeCacheInfo is type_cache_info.TypeCacheInfo
        assert (
            types.TypedDictCapabilities is typed_dict_capabilities.TypedDictCapabilities
        )

    def test_all_exports_count_matches_expected(self) -> None:
        """Verify the number of exports matches expected count."""
        from omnibase_infra import types

        # 5 type aliases + 3 models/TypedDicts = 8 exports
        expected_exports = {
            "ASTFunctionDef",
            "MessageOutputCategory",
            "PathInput",
            "PolicyTypeInput",
            "VersionInput",
            "ModelParsedDSN",
            "TypeCacheInfo",
            "TypedDictCapabilities",
        }
        assert set(types.__all__) == expected_exports
