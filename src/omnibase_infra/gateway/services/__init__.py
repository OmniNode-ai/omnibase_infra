# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gateway services for runtime envelope processing.

This module provides services for the runtime gateway, including:

- ServiceEnvelopeSigner: Service for signing outbound envelopes with Ed25519
- ServiceEnvelopeValidator: Service for validating inbound envelope signatures and realm
- ServicePolicyEngine: Policy evaluation for topic allowlist and realm enforcement

Exports:
    EnumPolicyDecision: Policy evaluation decision (ALLOW, DENY)
    PolicyDecision: Frozen dataclass with decision result and reason
    ServiceEnvelopeSigner: Service for signing outbound envelopes with Ed25519.
    ServiceEnvelopeValidator: Service for validating inbound envelope signatures and realm.
    ServicePolicyEngine: Policy engine for message filtering
    ValidationResult: Result of envelope validation with error details.
"""

from omnibase_infra.gateway.services.service_envelope_signer import (
    ServiceEnvelopeSigner,
)
from omnibase_infra.gateway.services.service_envelope_validator import (
    ServiceEnvelopeValidator,
    ValidationResult,
)
from omnibase_infra.gateway.services.service_policy_engine import (
    EnumPolicyDecision,
    PolicyDecision,
    ServicePolicyEngine,
)

__all__: list[str] = [
    "EnumPolicyDecision",
    "PolicyDecision",
    "ServiceEnvelopeSigner",
    "ServiceEnvelopeValidator",
    "ServicePolicyEngine",
    "ValidationResult",
]
