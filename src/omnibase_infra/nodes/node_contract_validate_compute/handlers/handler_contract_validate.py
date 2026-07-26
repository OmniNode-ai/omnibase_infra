# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime handler for contract validation compute (canonical definition B)."""

from __future__ import annotations

from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_core.services.service_contract_validator import ServiceContractValidator
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)


class HandlerContractValidate:
    """Canonical def-B handler for deterministic contract validation."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelContractValidateInput
    ) -> ModelContractValidationResult:
        """Validate a contract through the runtime compute-node boundary."""
        validator = ServiceContractValidator()
        if request.model_code is not None:
            assert request.contract_content is not None
            return validator.validate_model_compliance(
                request.model_code,
                request.contract_content,
            )
        if request.file_path is not None:
            return validator.validate_contract_file(
                request.file_path,
                request.contract_type,
                request.base_dir,
            )
        assert request.contract_content is not None
        return validator.validate_contract_yaml(
            request.contract_content,
            request.contract_type,
        )


__all__ = ["HandlerContractValidate"]
