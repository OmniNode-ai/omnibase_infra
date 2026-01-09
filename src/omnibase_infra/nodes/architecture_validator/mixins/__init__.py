# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixins for architecture validator rules.

This module provides reusable mixins for architecture validation rules,
enabling code reuse across multiple rule implementations.

Available Mixins:
    MixinFilePathRule: Extracts file paths from targets with graceful fallback.
"""

from omnibase_infra.nodes.architecture_validator.mixins.mixin_file_path_rule import (
    MixinFilePathRule,
)

__all__ = ["MixinFilePathRule"]
