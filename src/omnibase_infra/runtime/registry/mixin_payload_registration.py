# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Payload Registration Mixin.

Provides registration methods for the RegistryPayload class.
This mixin implements the freeze-after-init pattern for thread-safe
concurrent access.

Design Principles:
    - Thread-safe registration via lock
    - Freeze to lock and validate
    - Thread-safe queries after freeze

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
    - RegistryPayload: Main class that uses this mixin

.. versionadded:: 0.7.0
"""

from __future__ import annotations

__all__ = [
    "MixinPayloadRegistration",
]

import logging
import threading

from pydantic import BaseModel

from omnibase_infra.errors.error_payload_registry import PayloadRegistryError

logger = logging.getLogger(__name__)


class MixinPayloadRegistration:
    """
    Mixin providing registration methods for payload registry.

    This mixin implements:
        - Payload type registration with version keying
        - Freeze mechanism for thread-safe access
        - State initialization

    Requires the following attributes to be defined by the host class:
        - _entries: dict[tuple[str, str], type[BaseModel]]
        - _lock: threading.Lock
        - _frozen: bool

    .. versionadded:: 0.7.0
    """

    _entries: dict[tuple[str, str], type[BaseModel]]
    _lock: threading.Lock
    _frozen: bool

    def _init_payload_state(self) -> None:
        """Initialize registration state.

        Must be called from the host class __init__.
        """
        self._entries = {}
        self._lock = threading.Lock()
        self._frozen = False

    def register(
        self,
        payload_type: str,
        version: str,
        model_class: type[BaseModel],
    ) -> None:
        """
        Register a payload model class for a (payload_type, version) tuple.

        Args:
            payload_type: The payload type identifier (e.g., "ModelClaudeHookEvent").
                Must be a non-empty string.
            version: The semantic version string (e.g., "1.0.0").
                Must be a non-empty string.
            model_class: The Pydantic BaseModel subclass to register.
                Must be a concrete subclass (not BaseModel itself).

        Raises:
            PayloadRegistryError: If registry is frozen
            PayloadRegistryError: If (payload_type, version) is already registered
            PayloadRegistryError: If model_class is not a BaseModel subclass
            PayloadRegistryError: If payload_type or version is empty

        Thread Safety:
            Thread-safe via lock during registration phase.

        .. versionadded:: 0.7.0
        """
        # Validate inputs before acquiring lock
        if not payload_type or not payload_type.strip():
            raise PayloadRegistryError(
                "payload_type must be a non-empty string",
                payload_type=payload_type,
                version=version,
            )

        if not version or not version.strip():
            raise PayloadRegistryError(
                "version must be a non-empty string",
                payload_type=payload_type,
                version=version,
            )

        if not isinstance(model_class, type) or not issubclass(model_class, BaseModel):
            raise PayloadRegistryError(
                f"model_class must be a Pydantic BaseModel subclass, "
                f"got {type(model_class).__name__}: {model_class!r}",
                payload_type=payload_type,
                version=version,
            )

        if model_class is BaseModel:
            raise PayloadRegistryError(
                "model_class must be a concrete BaseModel subclass, "
                "not BaseModel itself",
                payload_type=payload_type,
                version=version,
            )

        key = (payload_type, version)

        # Thread-safe registration with atomic check-and-set
        with self._lock:
            if self._frozen:
                raise PayloadRegistryError(
                    f"Cannot register ({payload_type!r}, {version!r}): "
                    f"registry is frozen",
                    payload_type=payload_type,
                    version=version,
                )

            if key in self._entries:
                existing = self._entries[key]
                raise PayloadRegistryError(
                    f"({payload_type!r}, {version!r}) is already registered "
                    f"to {existing.__name__}. "
                    f"Cannot register {model_class.__name__}.",
                    payload_type=payload_type,
                    version=version,
                )

            self._entries[key] = model_class
            logger.debug(
                "Registered payload type (%s, %s) -> %s",
                payload_type,
                version,
                model_class.__name__,
            )

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, register() will raise PayloadRegistryError.
        Idempotent: calling freeze() multiple times has no additional effect.

        Thread Safety:
            Thread-safe via lock.

        .. versionadded:: 0.7.0
        """
        with self._lock:
            if self._frozen:
                return  # Idempotent
            self._frozen = True
            logger.info(
                "Payload registry frozen with %d entries",
                len(self._entries),
            )

    def _require_frozen(self, method_name: str) -> None:
        """Raise if registry is not frozen.

        Args:
            method_name: Name of the calling method for error message.

        Raises:
            PayloadRegistryError: If registry is not frozen.
        """
        if not self._frozen:
            raise PayloadRegistryError(
                f"Cannot call {method_name}(): registry is not frozen. "
                f"Call freeze() first.",
            )
