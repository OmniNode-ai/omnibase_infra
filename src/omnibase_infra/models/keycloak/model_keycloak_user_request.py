"""Keycloak user management request model."""

from pydantic import BaseModel, Field


class ModelKeycloakUserRequest(BaseModel):
    """Request model for Keycloak user management operations."""

    operation: str = Field(
        description="User operation: 'create', 'update', 'delete', 'get', 'list', 'reset_password'"
    )

    user_id: str | None = Field(
        default=None,
        description="User ID for update/delete/get operations"
    )

    username: str | None = Field(
        default=None,
        description="Username for create/update operations"
    )

    email: str | None = Field(
        default=None,
        description="Email for create/update operations"
    )

    first_name: str | None = Field(
        default=None,
        description="First name for create/update operations"
    )

    last_name: str | None = Field(
        default=None,
        description="Last name for create/update operations"
    )

    enabled: bool = Field(
        default=True,
        description="Whether user account is enabled"
    )

    password: str | None = Field(
        default=None,
        description="Password for create/reset_password operations"
    )

    roles: list[str] | None = Field(
        default=None,
        description="Roles to assign to user"
    )

    groups: list[str] | None = Field(
        default=None,
        description="Groups to assign to user"
    )

    attributes: dict | None = Field(
        default=None,
        description="Custom user attributes"
    )

    realm: str = Field(
        default="master",
        description="Keycloak realm"
    )

    correlation_id: str = Field(
        description="Correlation ID for request tracking"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "operation": "create",
                "username": "john.doe",
                "email": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "enabled": True,
                "roles": ["user"],
                "realm": "production",
                "correlation_id": "req_123456"
            }
        }
