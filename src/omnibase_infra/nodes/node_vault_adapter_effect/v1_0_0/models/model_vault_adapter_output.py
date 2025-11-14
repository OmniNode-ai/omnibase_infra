#!/usr/bin/env python3

from pydantic import BaseModel, Field

from omnibase_infra.models.vault.model_vault_secret_response import (
    ModelVaultSecretResponse,
)


class ModelVaultAdapterOutput(BaseModel):
    """Output model for Vault adapter operation results.

    Node-specific model for returning Vault operation results through effect outputs.
    """

    vault_operation_result: (
        ModelVaultSecretResponse
        | dict[str, str | int | bool | list | None]
        | str
        | bool
    ) = Field(description="Result of Vault operation")

    success: bool = Field(description="Whether the operation succeeded")
    operation_type: str = Field(description="Type of Vault operation performed")
    correlation_id: str = Field(description="Correlation ID from request")
