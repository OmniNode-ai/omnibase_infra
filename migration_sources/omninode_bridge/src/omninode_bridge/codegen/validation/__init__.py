#!/usr/bin/env python3
"""
Node Validation Pipeline for ONEX v2.0 Mixin-Enhanced Nodes.

Provides comprehensive validation of generated nodes through multiple stages:
- Syntax validation (compile check)
- AST validation (structure and methods)
- Import resolution
- Type checking (optional mypy integration)
- ONEX compliance (mixin verification, patterns)
- Security scanning (dangerous patterns)

Example:
    >>> from omninode_bridge.codegen.validation import NodeValidator
    >>> validator = NodeValidator(
    ...     enable_type_checking=False,
    ...     enable_security_scan=True
    ... )
    >>> results = await validator.validate_generated_node(
    ...     node_file_content=generated_code,
    ...     contract=enhanced_contract
    ... )
    >>> if all(r.passed for r in results):
    ...     print("✅ All validation stages passed")
"""

from .models import EnumValidationStage, ModelValidationResult
from .validator import NodeValidator

__all__ = [
    "EnumValidationStage",
    "ModelValidationResult",
    "NodeValidator",
]
