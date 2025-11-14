#!/usr/bin/env python3

from typing import Literal

from pydantic import BaseModel, Field


class ModelVaultAdapterInput(BaseModel):
    """Input model for Vault adapter operations from event envelopes.

    Node-specific model for processing event envelope payloads into Vault operations.
    """

    action: Literal[
        "vault_get_secret",
        "vault_set_secret",
        "vault_delete_secret",
        "vault_list_secrets",
        "vault_create_token",
        "vault_renew_token",
        "vault_revoke_token",
        "vault_health_check",
    ] = Field(description="Vault operation to perform")

    # Secret operation parameters
    path: str | None = Field(default=None, description="Secret path in Vault")
    mount_path: str = Field(default="secret", description="Vault mount path")
    secret_data: dict | None = Field(default=None, description="Secret data for write operations")
    version: int | None = Field(default=None, description="Secret version to retrieve")

    # Token operation parameters
    token: str | None = Field(default=None, description="Token for renew/revoke operations")
    policies: list[str] | None = Field(default=None, description="Policies for token creation")
    ttl: str | None = Field(default=None, description="Token TTL (e.g., '768h')")
    renewable: bool = Field(default=True, description="Whether token is renewable")
    increment: int | None = Field(default=None, description="Increment for token renewal")
    metadata: dict | None = Field(default=None, description="Token metadata")

    # Common fields
    correlation_id: str = Field(description="Correlation ID for request tracking")
