# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Layer - Service integration and policy runtime.

This package provides infrastructure adapters, error handling, and policy
registry for ONEX services including:

- Service adapters: PostgreSQL, Kafka, Consul, Vault
- Error context and transport-aware error handling
- PolicyRegistry: Plugin system for pure decision logic
- Runtime protocols and configuration validation

Key Components:
    - PolicyRegistry: SINGLE SOURCE OF TRUTH for policy plugin registration
    - Infrastructure adapters for external services
    - Transport-aware error handling with ModelInfraErrorContext
    - Runtime host process and kernel for ONEX node execution
"""

__all__: list[str] = []
