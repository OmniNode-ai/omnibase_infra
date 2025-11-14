"""Keycloak authentication response model."""

from pydantic import BaseModel, Field


class ModelKeycloakAuthResponse(BaseModel):
    """Response model for Keycloak authentication operations."""

    success: bool = Field(
        description="Whether the operation was successful"
    )

    access_token: str | None = Field(
        default=None,
        description="JWT access token"
    )

    refresh_token: str | None = Field(
        default=None,
        description="Refresh token for obtaining new access tokens"
    )

    expires_in: int | None = Field(
        default=None,
        description="Token expiration time in seconds"
    )

    refresh_expires_in: int | None = Field(
        default=None,
        description="Refresh token expiration time in seconds"
    )

    token_type: str | None = Field(
        default="Bearer",
        description="Token type (typically 'Bearer')"
    )

    session_state: str | None = Field(
        default=None,
        description="Keycloak session state"
    )

    scope: str | None = Field(
        default=None,
        description="Token scope"
    )

    user_id: str | None = Field(
        default=None,
        description="User ID from token"
    )

    username: str | None = Field(
        default=None,
        description="Username from token"
    )

    roles: list[str] | None = Field(
        default=None,
        description="User roles from token"
    )

    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )

    correlation_id: str = Field(
        description="Correlation ID from request"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "expires_in": 300,
                "refresh_expires_in": 1800,
                "token_type": "Bearer",
                "user_id": "user-123",
                "username": "user@example.com",
                "roles": ["user", "admin"],
                "correlation_id": "req_123456"
            }
        }
