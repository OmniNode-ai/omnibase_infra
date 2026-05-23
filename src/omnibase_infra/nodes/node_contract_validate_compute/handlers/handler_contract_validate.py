# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime handler for contract validation compute."""

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
    """Handler descriptor for deterministic contract validation operations."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE


async def handle_contract_validate(
    input_data: ModelContractValidateInput,
) -> ModelContractValidationResult:
    """Validate a contract through the runtime compute-node boundary."""
    validator = ServiceContractValidator()
    if input_data.model_code is not None:
        assert input_data.contract_content is not None
        return validator.validate_model_compliance(
            input_data.model_code,
            input_data.contract_content,
        )
    if input_data.file_path is not None:
        return validator.validate_contract_file(
            input_data.file_path,
            input_data.contract_type,
            input_data.base_dir,
        )
    assert input_data.contract_content is not None
    return validator.validate_contract_yaml(
        input_data.contract_content,
        input_data.contract_type,
    )


__all__ = ["HandlerContractValidate", "handle_contract_validate"]
