# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node-inventory image-label errors (OMN-18708).

Raised when the inventory an image is stamped with cannot be built, cannot be
parsed, or does not describe the image that carries it.

Error Hierarchy:
    RuntimeHostError
    └── NodeInventoryError -- the inventory is empty, malformed, carries a
        contract with no content hash, or disagrees with the image's own
        discovery pass

Why every one of those is an error rather than a degraded value. The label
exists so a promotion gate with no cluster access can name what an image
contains. A gate reading an empty label learns "this image contains nothing",
which is never true and is indistinguishable from "nobody stamped this image".
An absent label is a visible gap; an empty one is a false fact.

Related:
    - OMN-18708: the image label this error guards.
    - OMN-18709: the contract content hash the label's third field carries.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

__all__ = ["NodeInventoryError"]


class NodeInventoryError(RuntimeHostError):
    """Raised when a node inventory cannot be built, parsed, or trusted.

    Attributes:
        image_ref: The image the inventory was read from, when the failure
            happened on a readback rather than on a build.
        contract_name: The contract that made the inventory unusable, when one
            entry is responsible.
    """

    def __init__(
        self,
        message: str,
        *,
        image_ref: str = "",
        contract_name: str = "",
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.image_ref = image_ref
        self.contract_name = contract_name

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="node_inventory",
                correlation_id=correlation_id,
            )

        ctx = dict(extra_context)
        ctx.setdefault("image_ref", image_ref)
        ctx.setdefault("contract_name", contract_name)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **ctx,
        )
