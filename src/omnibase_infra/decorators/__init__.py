# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Decorators.

This module provides decorators for ONEX infrastructure patterns.

Decorators
----------
allow_any
    Exempt functions/classes from Any type validation. Used as a marker
    for the AST-based validator in ``omnibase_infra.validation.any_type_validator``.

Example Usage
-------------
    >>> from omnibase_infra.decorators import allow_any
    >>>
    >>> @allow_any("Required for legacy API - see OMN-1234")
    ... def legacy_handler(data: Any) -> Any:
    ...     return process_legacy(data)

See Also
--------
- ``omnibase_infra.validation.any_type_validator``: Recognizes ``@allow_any``
- CLAUDE.md: Any type policy documentation
"""

from omnibase_infra.decorators.allow_any import allow_any

__all__: list[str] = ["allow_any"]
