# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations


class OverlayNotFoundError(FileNotFoundError):
    """Overlay YAML file not found at the specified path."""


class OverlaySchemaInvalidError(ValueError):
    """Overlay YAML does not conform to ModelOverlayFile schema."""


class OverlayPermissionError(PermissionError):
    """Overlay file has overly permissive filesystem permissions."""


class RequiredConfigMissingError(ValueError):
    """Contract requires config keys that neither overlay nor environment provides."""


class ContractParseError(ValueError):
    """A contract file under contracts_dir is unreadable, unparseable, or malformed.

    Resolver fails closed on this error rather than silently skipping bad contracts,
    which would under-report required_keys and let boot succeed with missing config.
    """
