# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for gateway integration tests.

This module provides fixtures for testing envelope signing, validation, and
policy enforcement. All fixtures use in-memory Ed25519 keys to avoid filesystem
dependencies in tests.

Fixtures:
    ed25519_keypair: Generate a fresh Ed25519 keypair for testing.
    ed25519_private_key: Extract private key from keypair.
    ed25519_public_key: Extract public key from keypair.
    gateway_config: Create a test gateway configuration.
    envelope_signer: Create a test envelope signer.
    envelope_validator: Create a test envelope validator.
    policy_engine: Create a test policy engine.
    sample_payload: Create a sample Pydantic payload for testing.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.gateway import (
    ModelGatewayConfig,
    ServiceEnvelopeSigner,
    ServiceEnvelopeValidator,
    ServicePolicyEngine,
)

# =============================================================================
# Test Models
# =============================================================================


class ModelTestPayload(BaseModel):
    """Sample payload model for testing envelope signing.

    A minimal Pydantic model that can be serialized and hashed for
    envelope signing operations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: str = Field(..., description="Action type")
    resource_id: str = Field(..., description="Resource identifier")
    data: dict[str, str] = Field(default_factory=dict, description="Additional data")


# =============================================================================
# Keypair Fixtures
# =============================================================================


class Ed25519Keypair(NamedTuple):
    """Named tuple containing Ed25519 key pair."""

    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey


@pytest.fixture
def ed25519_keypair() -> Ed25519Keypair:
    """Generate a fresh Ed25519 keypair for testing.

    Each test invocation gets a unique keypair, ensuring test isolation.
    Uses cryptography library directly for key generation.

    Returns:
        Ed25519Keypair containing private and public keys.
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return Ed25519Keypair(private_key=private_key, public_key=public_key)


@pytest.fixture
def ed25519_private_key(ed25519_keypair: Ed25519Keypair) -> Ed25519PrivateKey:
    """Extract private key from keypair fixture.

    Args:
        ed25519_keypair: The keypair fixture.

    Returns:
        Ed25519PrivateKey for signing operations.
    """
    return ed25519_keypair.private_key


@pytest.fixture
def ed25519_public_key(ed25519_keypair: Ed25519Keypair) -> Ed25519PublicKey:
    """Extract public key from keypair fixture.

    Args:
        ed25519_keypair: The keypair fixture.

    Returns:
        Ed25519PublicKey for verification operations.
    """
    return ed25519_keypair.public_key


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def gateway_config() -> ModelGatewayConfig:
    """Create a test gateway configuration.

    Creates a minimal gateway configuration suitable for testing.
    Key paths are not set since tests use in-memory keys.

    Returns:
        ModelGatewayConfig with test realm and runtime_id.
    """
    return ModelGatewayConfig(
        realm="test",
        runtime_id="test-runtime-001",
        enabled=True,
        allowed_topics=("events.*", "commands.*"),
        reject_unsigned=True,
    )


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def envelope_signer(
    ed25519_private_key: Ed25519PrivateKey,
    gateway_config: ModelGatewayConfig,
) -> ServiceEnvelopeSigner:
    """Create a test envelope signer.

    Args:
        ed25519_private_key: Private key for signing.
        gateway_config: Gateway configuration.

    Returns:
        ServiceEnvelopeSigner configured with test keys.
    """
    return ServiceEnvelopeSigner(
        realm=gateway_config.realm,
        runtime_id=gateway_config.runtime_id,
        private_key=ed25519_private_key,
    )


@pytest.fixture
def envelope_validator(
    ed25519_public_key: Ed25519PublicKey,
    gateway_config: ModelGatewayConfig,
) -> ServiceEnvelopeValidator:
    """Create a test envelope validator.

    Args:
        ed25519_public_key: Public key for verification.
        gateway_config: Gateway configuration.

    Returns:
        ServiceEnvelopeValidator configured with test keys.
    """
    return ServiceEnvelopeValidator(
        expected_realm=gateway_config.realm,
        public_keys={gateway_config.runtime_id: ed25519_public_key},
        reject_unsigned=gateway_config.reject_unsigned,
    )


@pytest.fixture
def policy_engine(gateway_config: ModelGatewayConfig) -> ServicePolicyEngine:
    """Create a test policy engine.

    Args:
        gateway_config: Gateway configuration.

    Returns:
        ServicePolicyEngine configured with test topic allowlist.
    """
    return ServicePolicyEngine(
        allowed_topics=list(gateway_config.allowed_topics),
        expected_realm=gateway_config.realm,
        log_rejections=False,  # Suppress logging in tests
    )


# =============================================================================
# Payload Fixtures
# =============================================================================


@pytest.fixture
def sample_payload() -> ModelTestPayload:
    """Create a sample payload for testing.

    Returns:
        ModelTestPayload with test data.
    """
    return ModelTestPayload(
        action="created",
        resource_id="resource-123",
        data={"key1": "value1", "key2": "value2"},
    )


@pytest.fixture
def sample_payload_dict() -> dict[str, object]:
    """Create a sample payload as dictionary.

    Returns:
        Dictionary representation of test payload.
    """
    return {
        "action": "created",
        "resource_id": "resource-123",
        "data": {"key1": "value1", "key2": "value2"},
    }
