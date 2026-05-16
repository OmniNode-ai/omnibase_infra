# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay-specific error types for runtime boot."""

from __future__ import annotations

from omnibase_infra.errors.error_infra import RuntimeHostError


class OverlayNotFoundError(RuntimeHostError):
    """Raised when the overlay file is missing and no fallback is available."""


class OverlaySchemaInvalidError(RuntimeHostError):
    """Raised when the overlay file fails YAML parsing or Pydantic validation."""


class OverlayPermissionError(RuntimeHostError):
    """Overlay file permissions too open for a secret-containing file."""


class RequiredConfigMissingError(RuntimeHostError):
    """One or more required config keys declared in contracts are absent from the overlay."""
