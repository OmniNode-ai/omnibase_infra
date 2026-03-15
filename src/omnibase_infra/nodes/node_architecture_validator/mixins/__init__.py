# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mixins for architecture validator rules.

Reusable mixins for architecture validation rules,
enabling code reuse across multiple rule implementations.

Available Mixins:
    MixinFilePathRule: Extracts file paths from targets with graceful fallback.
"""

from omnibase_infra.nodes.node_architecture_validator.mixins.mixin_file_path_rule import (
    MixinFilePathRule,
)

__all__ = ["MixinFilePathRule"]
