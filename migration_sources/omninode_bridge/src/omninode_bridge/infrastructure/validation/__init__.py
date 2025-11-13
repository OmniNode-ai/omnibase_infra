#!/usr/bin/env python3
"""
Infrastructure Validation Package.

Provides validation utilities for infrastructure components including:
- JSONB field validation for PostgreSQL entity models
- Database type annotation enforcement
- Pydantic model compliance checks

ONEX v2.0 Compliance:
- Type-safe validation utilities
- Clear error messaging
- Comprehensive documentation
"""

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    JsonbField,
    JsonbValidatedModel,
    validate_jsonb_fields,
)

__all__ = [
    "JsonbField",
    "JsonbValidatedModel",
    "validate_jsonb_fields",
]
