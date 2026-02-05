# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Ed25519 Key Loading Utilities.

This module provides utility functions for loading Ed25519 keys from PEM files.
These utilities support the gateway's envelope signing and verification operations.

Security Considerations:
    - Private key files should have restricted permissions (0600)
    - Key paths should be validated before loading
    - Key contents are never logged or exposed in error messages

Exports:
    load_private_key_from_pem: Load Ed25519 private key from PEM file
    load_public_key_from_pem: Load Ed25519 public key from PEM file

Example:
    >>> from pathlib import Path
    >>> private_key = load_private_key_from_pem(Path("/etc/onex/keys/private.pem"))
    >>> public_key = load_public_key_from_pem(Path("/etc/onex/keys/public.pem"))

"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)

logger = logging.getLogger(__name__)


def load_private_key_from_pem(path: Path) -> Ed25519PrivateKey:
    """Load Ed25519 private key from PEM file.

    Reads a PEM-encoded Ed25519 private key from the filesystem. The key
    must be in standard PEM format without password protection.

    Args:
        path: Path to the PEM file containing the private key. Must be an
            existing file with read permissions.

    Returns:
        Ed25519PrivateKey instance ready for signing operations.

    Raises:
        ProtocolConfigurationError: If the file does not exist, cannot be read,
            or does not contain a valid Ed25519 private key. The error message
            does not include key contents for security reasons.

    Security:
        - File contents are read into memory only during loading
        - Key bytes are not logged or included in error messages
        - Caller should ensure file has appropriate permissions (0600)

    Example:
        >>> from pathlib import Path
        >>> key = load_private_key_from_pem(Path("/etc/onex/keys/private.pem"))
        >>> # Key is ready for signing operations

    """
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="load_private_key_from_pem",
        target_name=str(path),
    )

    if not path.exists():
        raise ProtocolConfigurationError(
            f"Private key file does not exist: {path}",
            context=context,
        )

    try:
        key_bytes = path.read_bytes()
    except OSError as e:
        logger.exception(
            "Failed to read private key file (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Failed to read private key file: {path}",
            context=context,
        ) from e

    try:
        private_key = load_pem_private_key(key_bytes, password=None)
    except UnsupportedAlgorithm as e:
        logger.exception(
            "Unsupported key algorithm in private key file (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Unsupported key algorithm in private key file: {path}",
            context=context,
        ) from e
    except (ValueError, TypeError) as e:
        logger.exception(
            "Failed to parse PEM private key (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Invalid PEM format for private key: {path}",
            context=context,
        ) from e

    if not isinstance(private_key, Ed25519PrivateKey):
        raise ProtocolConfigurationError(
            f"Key is not Ed25519: {path} (got {type(private_key).__name__})",
            context=context,
        )

    logger.debug(
        "Loaded Ed25519 private key (correlation_id=%s)",
        context.correlation_id,
        extra={"path": str(path)},
    )

    return private_key


def load_public_key_from_pem(path: Path) -> Ed25519PublicKey:
    """Load Ed25519 public key from PEM file.

    Reads a PEM-encoded Ed25519 public key from the filesystem. The key
    must be in standard PEM format.

    Args:
        path: Path to the PEM file containing the public key. Must be an
            existing file with read permissions.

    Returns:
        Ed25519PublicKey instance ready for verification operations.

    Raises:
        ProtocolConfigurationError: If the file does not exist, cannot be read,
            or does not contain a valid Ed25519 public key.

    Example:
        >>> from pathlib import Path
        >>> key = load_public_key_from_pem(Path("/etc/onex/keys/public.pem"))
        >>> # Key is ready for signature verification

    """
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="load_public_key_from_pem",
        target_name=str(path),
    )

    if not path.exists():
        raise ProtocolConfigurationError(
            f"Public key file does not exist: {path}",
            context=context,
        )

    try:
        key_bytes = path.read_bytes()
    except OSError as e:
        logger.exception(
            "Failed to read public key file (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Failed to read public key file: {path}",
            context=context,
        ) from e

    try:
        public_key = load_pem_public_key(key_bytes)
    except UnsupportedAlgorithm as e:
        logger.exception(
            "Unsupported key algorithm in public key file (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Unsupported key algorithm in public key file: {path}",
            context=context,
        ) from e
    except (ValueError, TypeError) as e:
        logger.exception(
            "Failed to parse PEM public key (correlation_id=%s)",
            context.correlation_id,
            extra={"path": str(path)},
        )
        raise ProtocolConfigurationError(
            f"Invalid PEM format for public key: {path}",
            context=context,
        ) from e

    if not isinstance(public_key, Ed25519PublicKey):
        raise ProtocolConfigurationError(
            f"Key is not Ed25519: {path} (got {type(public_key).__name__})",
            context=context,
        )

    logger.debug(
        "Loaded Ed25519 public key (correlation_id=%s)",
        context.correlation_id,
        extra={"path": str(path)},
    )

    return public_key


__all__: list[str] = [
    "load_private_key_from_pem",
    "load_public_key_from_pem",
]
