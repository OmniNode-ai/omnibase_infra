# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for gateway envelope signing, validation, and policy enforcement.

This package contains integration tests for the runtime gateway components:
    - ServiceEnvelopeSigner: Ed25519 envelope signing
    - ServiceEnvelopeValidator: Signature and realm validation
    - ServicePolicyEngine: Topic allowlist and realm enforcement
    - ModelGatewayConfig: Gateway configuration model

Test Structure:
    - test_envelope_signer.py: Signing functionality tests
    - test_envelope_validator.py: Validation functionality tests
    - test_policy_engine.py: Policy engine tests
    - test_gateway_config.py: Configuration model tests

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""
