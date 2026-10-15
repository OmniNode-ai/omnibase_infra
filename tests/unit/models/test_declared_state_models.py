# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An unobservable diff cannot hold operations."""

import pytest

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind,
)
from omnibase_infra.models.declared_state.model_declared_state_diff import (
    ModelDeclaredStateDiff,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)

pytestmark = pytest.mark.unit


def test_unobservable_diff_with_operations_is_refused() -> None:
    operation = ModelDeclaredStateOperation(
        kind=EnumDeclaredStateOperationKind.CREATE,
        surface_kind=EnumDeclaredStateSurfaceKind.BROKER_ACL,
        surface_key="scratch",
        item_key="a",
        digest="1",
    )
    with pytest.raises(ValueError, match="unobservable"):
        ModelDeclaredStateDiff(
            surface_kind=EnumDeclaredStateSurfaceKind.BROKER_ACL,
            surface_key="scratch",
            generation="g1",
            operations=(operation,),
            unobservable_reason="adapter returned nothing",
        )
