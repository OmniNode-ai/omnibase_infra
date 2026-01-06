# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""REMOVED: This module has been removed.

Use ``omnibase_core.models.primitives`` instead.

Migration guide:
    - Replace ``from omnibase_infra.models.model_semver import ModelSemVer``
      with ``from omnibase_core.models.primitives import ModelSemVer``
    - Replace ``ModelSemVer.from_string(...)`` with ``ModelSemVer.parse(...)``
    - Replace ``str(semver)`` with ``semver.to_string()`` (both work, but to_string() is preferred)
    - Create SEMVER_DEFAULT inline: ``SEMVER_DEFAULT = ModelSemVer.parse("1.0.0")``
"""

# Fail hard on import
raise ImportError(
    "omnibase_infra.models.model_semver has been removed. "
    "Use omnibase_core.models.primitives.model_semver instead. "
    "Replace from_string() with parse() and str() with to_string()."
)
