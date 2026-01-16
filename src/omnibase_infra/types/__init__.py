# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Types module for omnibase_infra.

This module re-exports commonly used types for external consumption.
Types are organized into categories:

Top-level exports (this module):
    - ModelParsedDSN: Pydantic model for parsed DSN components
    - TypeCacheInfo: NamedTuple for cache hit/miss statistics
    - TypedDictCapabilities: TypedDict for node capability introspection

Type aliases (from type_infra_aliases):
    - ASTFunctionDef: ast.AsyncFunctionDef | ast.FunctionDef
    - MessageOutputCategory: EnumMessageCategory | EnumNodeOutputType
    - PathInput: Path | str (named to avoid collision with stdlib os.PathLike)
    - PolicyTypeInput: EnumPolicyType | str

Note on cache-related TypedDicts:
    TypedDictIntrospectionCache and TypedDictPerformanceMetricsCache are NOT
    re-exported here because they are internal implementation details of the
    MixinNodeIntrospection mixin. These types are tightly coupled to the mixin's
    caching implementation and should not be used directly by external consumers.
    If needed for testing or advanced use cases, import from the typed_dict submodule:
        from omnibase_infra.types.typed_dict import TypedDictIntrospectionCache
"""

from omnibase_infra.types.type_cache_info import TypeCacheInfo
from omnibase_infra.types.type_dsn import ModelParsedDSN
from omnibase_infra.types.type_infra_aliases import (
    ASTFunctionDef,
    MessageOutputCategory,
    PathInput,
    PolicyTypeInput,
)
from omnibase_infra.types.typed_dict_capabilities import TypedDictCapabilities

__all__: list[str] = [
    # Type aliases
    "ASTFunctionDef",
    "MessageOutputCategory",
    "PathInput",
    "PolicyTypeInput",
    # Models and TypedDicts
    "ModelParsedDSN",
    "TypeCacheInfo",
    "TypedDictCapabilities",
]
