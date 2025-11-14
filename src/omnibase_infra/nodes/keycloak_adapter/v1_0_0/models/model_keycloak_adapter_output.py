#!/usr/bin/env python3

from pydantic import BaseModel, Field

from omnibase_infra.models.keycloak.model_keycloak_auth_response import (
    ModelKeycloakAuthResponse,
)


class ModelKeycloakAdapterOutput(BaseModel):
    """Output model for Keycloak adapter operation results.

    Node-specific model for returning Keycloak operation results through effect outputs.
    """

    keycloak_operation_result: (
        ModelKeycloakAuthResponse
        | dict[str, str | int | bool | list | dict | None]
        | str
        | bool
    ) = Field(description="Result of Keycloak operation")

    success: bool = Field(description="Whether the operation succeeded")
    operation_type: str = Field(description="Type of Keycloak operation performed")
    correlation_id: str = Field(description="Correlation ID from request")
