# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for plugin input data.

This module provides the ModelPluginInputData Pydantic BaseModel that replaces
the former PluginInputData TypedDict definition.

Design Notes:
    - Uses ConfigDict(extra="allow") to support arbitrary fields
    - Supports dict-like access via __getitem__ for flexible API usage
    - Can be instantiated from dicts using model_validate()
    - Follows ONEX naming convention: Model<Name>
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.mixins import MixinDictLikeAccessors


class ModelPluginInputData(MixinDictLikeAccessors, BaseModel):
    """Base Pydantic model for plugin input data.

    This model replaces PluginInputData TypedDict and allows arbitrary fields
    to support flexible plugin input structures.

    Attributes:
        All fields are optional. Concrete plugins should document their
        required fields and use validate_input() for runtime validation.

    Configuration:
        - extra="allow": Accepts arbitrary additional fields
        - frozen=False: Allows mutation (though plugins should not mutate)
        - populate_by_name=True: Allows field access by alias

    Example:
        ```python
        # Create from dict
        input_data = ModelPluginInputData.model_validate({"values": [1, 2, 3]})

        # Access with get (dict-like)
        values = input_data.get("values", [])

        # Access as attribute (if field exists)
        values = getattr(input_data, "values", [])
        ```
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
        populate_by_name=True,
        from_attributes=True,  # pytest-xdist compatibility
    )


__all__: list[str] = ["ModelPluginInputData"]
