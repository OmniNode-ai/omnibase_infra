# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Type Registration Mixin.

Provides registration methods for the RegistryMessageType class.
This mixin implements the freeze-after-init pattern for thread-safe
concurrent access.

Design Principles:
    - Single-threaded registration phase
    - Freeze to lock and validate
    - Thread-safe queries after freeze

Related:
    - OMN-937: Central Message Type Registry implementation
    - RegistryMessageType: Main class that uses this mixin

.. versionadded:: 0.6.0
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import UTC, datetime

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.errors import MessageTypeRegistryError
from omnibase_infra.models.errors import ModelMessageTypeRegistryErrorContext
from omnibase_infra.models.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)

__all__ = [
    "MixinMessageTypeRegistration",
]

logger = logging.getLogger(__name__)


class MixinMessageTypeRegistration:
    """
    Mixin providing registration methods for message type registry.

    This mixin implements:
        - Message type registration with handler mappings
        - Simplified registration convenience method
        - Freeze mechanism for thread-safe access

    Requires the following attributes to be defined by the host class:
        - _entries: dict[str, ModelMessageTypeEntry]
        - _domains: set[str]
        - _handler_references: set[str]
        - _category_index: dict[EnumMessageCategory, list[str]]
        - _domain_index: dict[str, list[str]]
        - _frozen: bool
        - _lock: threading.Lock
        - _logger: logging.Logger

    .. versionadded:: 0.6.0
    """

    # Type hints for mixin attributes (defined by host class)
    _entries: dict[str, ModelMessageTypeEntry]
    _domains: set[str]
    _handler_references: set[str]
    _category_index: dict[EnumMessageCategory, list[str]]
    _domain_index: dict[str, list[str]]
    _frozen: bool
    _lock: threading.Lock
    _logger: logging.Logger

    def _init_registration_state(
        self,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """
        Initialize registration state for the mixin.

        Should be called from the host class __init__.

        Args:
            logger_instance: Optional custom logger for structured logging.
                If not provided, uses module-level logger.
        """
        self._logger = logger_instance if logger_instance is not None else logger

        # Primary storage: message_type -> entry
        self._entries = {}

        # Domain tracking
        self._domains = set()

        # Handler reference tracking (for validation)
        self._handler_references = set()

        # Indexes for efficient queries
        self._category_index = defaultdict(list)
        self._domain_index = defaultdict(list)

        # Freeze state
        self._frozen = False
        self._lock = threading.Lock()

    def register_message_type(
        self,
        entry: ModelMessageTypeEntry,
    ) -> None:
        """
        Register a message type with its handler mappings.

        Associates a message type with handler(s) and defines constraints.
        If the message type is already registered, handlers are merged
        (fan-out pattern).

        Args:
            entry: The message type entry containing handler mappings.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE)
            ModelOnexError: If entry is None (INVALID_PARAMETER)
            MessageTypeRegistryError: If entry validation fails

        Example:
            >>> registry.register_message_type(ModelMessageTypeEntry(
            ...     message_type="OrderCreated",
            ...     handler_ids=("order-handler",),
            ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            ...     domain_constraint=ModelDomainConstraint(owning_domain="order"),
            ... ))

        .. versionadded:: 0.5.0
        """
        if entry is None:
            raise ModelOnexError(
                message="Cannot register None entry. ModelMessageTypeEntry is required.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        with self._lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot register message type: RegistryMessageType is frozen. "
                    "Registration is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            message_type = entry.message_type

            # If already registered, merge handlers (fan-out support)
            if message_type in self._entries:
                existing = self._entries[message_type]
                # Validate constraints match
                if existing.allowed_categories != entry.allowed_categories:
                    raise MessageTypeRegistryError(
                        f"Category constraint mismatch for message type "
                        f"'{message_type}': existing={existing.allowed_categories}, "
                        f"new={entry.allowed_categories}. "
                        f"All registrations for a message type must have the same "
                        f"allowed categories.",
                        registry_context=ModelMessageTypeRegistryErrorContext(
                            message_type=message_type,
                        ),
                    )
                if existing.domain_constraint != entry.domain_constraint:
                    # Build detailed mismatch description
                    differences: list[str] = []
                    if (
                        existing.domain_constraint.owning_domain
                        != entry.domain_constraint.owning_domain
                    ):
                        differences.append(
                            f"owning_domain: existing="
                            f"'{existing.domain_constraint.owning_domain}', "
                            f"new='{entry.domain_constraint.owning_domain}'"
                        )
                    if (
                        existing.domain_constraint.allowed_cross_domains
                        != entry.domain_constraint.allowed_cross_domains
                    ):
                        differences.append(
                            f"allowed_cross_domains: existing="
                            f"{set(existing.domain_constraint.allowed_cross_domains)}, "
                            f"new={set(entry.domain_constraint.allowed_cross_domains)}"
                        )
                    if (
                        existing.domain_constraint.allow_all_domains
                        != entry.domain_constraint.allow_all_domains
                    ):
                        differences.append(
                            f"allow_all_domains: existing="
                            f"{existing.domain_constraint.allow_all_domains}, "
                            f"new={entry.domain_constraint.allow_all_domains}"
                        )
                    if (
                        existing.domain_constraint.require_explicit_opt_in
                        != entry.domain_constraint.require_explicit_opt_in
                    ):
                        differences.append(
                            f"require_explicit_opt_in: existing="
                            f"{existing.domain_constraint.require_explicit_opt_in}, "
                            f"new={entry.domain_constraint.require_explicit_opt_in}"
                        )

                    raise MessageTypeRegistryError(
                        f"Domain constraint mismatch for message type "
                        f"'{message_type}': {'; '.join(differences)}. "
                        f"All registrations for a message type must have the same "
                        f"domain constraint configuration.",
                        registry_context=ModelMessageTypeRegistryErrorContext(
                            message_type=message_type,
                            domain=entry.domain_constraint.owning_domain,
                        ),
                    )

                # Merge handlers
                for handler_id in entry.handler_ids:
                    if handler_id not in existing.handler_ids:
                        self._entries[message_type] = existing.with_additional_handler(
                            handler_id
                        )
                        self._handler_references.add(handler_id)
                        existing = self._entries[message_type]

                self._logger.debug(
                    "Merged handlers for message type '%s': %s",
                    message_type,
                    self._entries[message_type].handler_ids,
                )
            else:
                # New registration
                self._entries[message_type] = entry

                # Track domain - validate full domain constraint, not just owning_domain
                domain = entry.domain_constraint.owning_domain
                self._domains.add(domain)

                # Also track allowed_cross_domains for comprehensive domain awareness
                # This enables domain coverage reporting and validation
                for cross_domain in entry.domain_constraint.allowed_cross_domains:
                    if cross_domain:  # Validate non-empty cross-domain names
                        self._domains.add(cross_domain)

                # Track handler references
                for handler_id in entry.handler_ids:
                    self._handler_references.add(handler_id)

                # Update indexes
                for category in entry.allowed_categories:
                    self._category_index[category].append(message_type)
                self._domain_index[domain].append(message_type)

                self._logger.debug(
                    "Registered message type '%s' with handlers %s "
                    "(domain=%s, categories=%s)",
                    message_type,
                    entry.handler_ids,
                    domain,
                    [c.value for c in entry.allowed_categories],
                )

    def register_simple(
        self,
        message_type: str,
        handler_id: str,
        category: EnumMessageCategory,
        domain: str,
        *,
        description: str | None = None,
        allow_cross_domains: frozenset[str] | None = None,
    ) -> None:
        """
        Convenience method to register a message type with minimal parameters.

        Creates a ModelMessageTypeEntry internally with sensible defaults.

        Args:
            message_type: The message type name (e.g., "UserCreated").
            handler_id: The handler ID to process this type.
            category: The allowed message category.
            domain: The owning domain.
            description: Optional description of the message type.
            allow_cross_domains: Optional set of additional domains to allow.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE)
            MessageTypeRegistryError: If validation fails

        Example:
            >>> registry.register_simple(
            ...     message_type="UserCreated",
            ...     handler_id="user-handler",
            ...     category=EnumMessageCategory.EVENT,
            ...     domain="user",
            ...     description="User creation event",
            ... )

        .. versionadded:: 0.5.0
        """
        constraint = ModelDomainConstraint(
            owning_domain=domain,
            allowed_cross_domains=allow_cross_domains or frozenset(),
        )

        entry = ModelMessageTypeEntry(
            message_type=message_type,
            handler_ids=(handler_id,),
            allowed_categories=frozenset([category]),
            domain_constraint=constraint,
            description=description,
            registered_at=datetime.now(UTC),
        )

        self.register_message_type(entry)

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, registration methods will raise ModelOnexError with
        INVALID_STATE. This enables thread-safe concurrent access during
        the query phase.

        Idempotent: Calling freeze() multiple times has no additional effect.

        Example:
            >>> registry.register_message_type(entry)
            >>> registry.freeze()
            >>> assert registry.is_frozen

        Note:
            This is a one-way operation. There is no unfreeze() method
            by design, as unfreezing would defeat thread-safety guarantees.

        .. versionadded:: 0.5.0
        """
        with self._lock:
            if self._frozen:
                # Idempotent - already frozen
                return

            self._frozen = True
            self._logger.info(
                "RegistryMessageType frozen with %d message types across %d domains",
                len(self._entries),
                len(self._domains),
            )
