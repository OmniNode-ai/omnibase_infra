# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Models for the entry method placement compute node.

Provides the placement enum, the classification request input,
and the placement decision output model.

Ticket: OMN-24492
"""

from omnibase_infra.enums.enum_entry_method_placement import (
    EnumEntryMethodPlacement,
)
from omnibase_infra.models.entry_method_placement.model_entry_method_placement import (
    ModelEntryMethodPlacement,
)
from omnibase_infra.models.entry_method_placement.model_entry_method_placement_request import (
    ModelEntryMethodPlacementRequest,
)

__all__: list[str] = [
    "EnumEntryMethodPlacement",
    "ModelEntryMethodPlacement",
    "ModelEntryMethodPlacementRequest",
]
