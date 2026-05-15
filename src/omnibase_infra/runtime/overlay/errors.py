# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay-specific error classes for OverlayFileLoader and related components."""

from __future__ import annotations

from omnibase_infra.errors import RuntimeHostError


class OverlayNotFoundError(RuntimeHostError):
    """Overlay file is missing; user must run onboarding to generate one."""


class OverlaySchemaInvalidError(RuntimeHostError):
    """Overlay file is malformed YAML or fails Pydantic schema validation."""


class UnsupportedOverlayVersionError(RuntimeHostError):
    """Overlay file declares a version not in the supported set."""


class OverlayPermissionError(RuntimeHostError):
    """Overlay file permissions are too open for a secret-containing file."""


class OverlayMergeConflictError(RuntimeHostError):
    """Irreconcilable conflict detected in the overlay stack."""


class RequiredConfigMissingError(RuntimeHostError):
    """A contract-required config key has no value in the overlay."""
