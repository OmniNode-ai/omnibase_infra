#!/usr/bin/env python3

from typing import Literal

from pydantic import BaseModel, Field


class ModelKeycloakAdapterInput(BaseModel):
    """Input model for Keycloak adapter operations from event envelopes.

    Node-specific model for processing event envelope payloads into Keycloak operations.
    """

    action: Literal[
        "keycloak_login",
        "keycloak_logout",
        "keycloak_refresh_token",
        "keycloak_verify_token",
        "keycloak_create_user",
        "keycloak_get_user",
        "keycloak_update_user",
        "keycloak_delete_user",
        "keycloak_assign_roles",
        "keycloak_health_check",
    ] = Field(description="Keycloak operation to perform")

    # Authentication parameters
    username: str | None = Field(default=None, description="Username for login")
    password: str | None = Field(default=None, description="Password for login")
    token: str | None = Field(default=None, description="Access token for verify/logout")
    refresh_token: str | None = Field(default=None, description="Refresh token")

    # User management parameters
    user_id: str | None = Field(default=None, description="User ID for operations")
    email: str | None = Field(default=None, description="User email")
    first_name: str | None = Field(default=None, description="User first name")
    last_name: str | None = Field(default=None, description="User last name")
    enabled: bool = Field(default=True, description="User enabled status")
    email_verified: bool = Field(default=False, description="Email verification status")
    attributes: dict | None = Field(default=None, description="User custom attributes")

    # Role management parameters
    roles: list[str] | None = Field(default=None, description="User roles")

    # Common fields
    client_id: str = Field(description="Keycloak client ID")
    realm: str = Field(default="master", description="Keycloak realm")
    correlation_id: str = Field(description="Correlation ID for request tracking")
