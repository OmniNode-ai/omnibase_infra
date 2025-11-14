"""Keycloak authentication request model."""

from pydantic import BaseModel, Field


class ModelKeycloakAuthRequest(BaseModel):
    """Request model for Keycloak authentication operations."""

    operation: str = Field(
        description="Auth operation: 'login', 'logout', 'refresh_token', 'verify_token'"
    )

    username: str | None = Field(
        default=None,
        description="Username for login operation"
    )

    password: str | None = Field(
        default=None,
        description="Password for login operation"
    )

    token: str | None = Field(
        default=None,
        description="Access token for verify/logout operations"
    )

    refresh_token: str | None = Field(
        default=None,
        description="Refresh token for token refresh operation"
    )

    client_id: str = Field(
        description="Keycloak client ID"
    )

    client_secret: str | None = Field(
        default=None,
        description="Keycloak client secret (for confidential clients)"
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
                "operation": "login",
                "username": "user@example.com",
                "password": "***",
                "client_id": "my-app",
                "realm": "production",
                "correlation_id": "req_123456"
            }
        }
