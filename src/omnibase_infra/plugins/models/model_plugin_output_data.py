# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for plugin output data.

This module provides the ModelPluginOutputData Pydantic BaseModel that replaces
the former PluginOutputData TypedDict definition.

Design Notes:
    - Uses ConfigDict(extra="allow") to support arbitrary fields
    - Supports dict-like access via __getitem__ for flexible API usage
    - Can be instantiated from dicts using model_validate()
    - Follows ONEX naming convention: Model<Name>
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.mixins import MixinDictLikeAccessors


class ModelPluginOutputData(MixinDictLikeAccessors, BaseModel):
    """Base Pydantic model for plugin output data.

    This model replaces PluginOutputData TypedDict and allows arbitrary
    fields to support flexible plugin output structures.

    Common Fields:
        result: Primary computation result
        metadata: Output metadata (execution time, version, etc.)
        errors: List of validation or computation errors
        warnings: List of non-fatal warnings

    Configuration:
        - extra="allow": Accepts arbitrary additional fields
        - frozen=False: Allows mutation
        - populate_by_name=True: Allows field access by alias

    Example:
        ```python
        output = ModelPluginOutputData(
            result={"normalized": [0.1, 0.5, 1.0]},
            metadata={"execution_time_ms": 15},
            errors=[],
            warnings=[],
        )
        ```

    Note:
        This is a base type hint. Concrete plugins should define their
        own output structure for stronger type safety.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
        populate_by_name=True,
        from_attributes=True,  # pytest-xdist compatibility
    )


__all__: list[str] = ["ModelPluginOutputData"]
