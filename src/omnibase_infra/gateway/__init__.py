# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gateway Module.

This module provides gateway functionality for envelope signing, realm enforcement,
and policy-based message filtering. The gateway acts as the security boundary for
the runtime, ensuring all messages are properly signed, validated, and filtered
before processing.

Components:
    - Envelope Signer: Ed25519 signing for outbound envelopes
    - Envelope Validator: Signature and realm validation for inbound envelopes
    - Policy Engine: Topic allowlist and realm boundary enforcement
    - Models: Gateway configuration and validation models
    - Enums: Gateway-specific enumerations
    - Utils: Key loading utilities for Ed25519 keys

Exports:
    EnumPolicyDecision: Policy evaluation decision (ALLOW, DENY)
    EnumValidationErrorCode: Error codes for envelope validation failures
    ModelGatewayConfig: Configuration model for gateway signing and validation
    PolicyDecision: Result of policy evaluation with reason
    ServiceEnvelopeSigner: Service for signing outbound envelopes with Ed25519
    ServiceEnvelopeValidator: Service for validating inbound envelope signatures and realm
    ServicePolicyEngine: Policy engine for message filtering
    ValidationResult: Result of envelope validation with error details
    load_private_key_from_pem: Load Ed25519 private key from PEM file
    load_public_key_from_pem: Load Ed25519 public key from PEM file
"""

from omnibase_infra.gateway.enums import EnumValidationErrorCode
from omnibase_infra.gateway.models import ModelGatewayConfig
from omnibase_infra.gateway.services import (
    EnumPolicyDecision,
    PolicyDecision,
    ServiceEnvelopeSigner,
    ServiceEnvelopeValidator,
    ServicePolicyEngine,
    ValidationResult,
)
from omnibase_infra.gateway.utils import (
    load_private_key_from_pem,
    load_public_key_from_pem,
)

__all__: list[str] = [
    "EnumPolicyDecision",
    "EnumValidationErrorCode",
    "ModelGatewayConfig",
    "PolicyDecision",
    "ServiceEnvelopeSigner",
    "ServiceEnvelopeValidator",
    "ServicePolicyEngine",
    "ValidationResult",
    "load_private_key_from_pem",
    "load_public_key_from_pem",
]
