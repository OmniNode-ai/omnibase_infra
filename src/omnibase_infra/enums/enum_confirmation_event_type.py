# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Confirmation event type enumeration.

This module defines the event type enumeration for registration confirmation
events emitted by infrastructure backends.
"""

from enum import Enum


class EnumConfirmationEventType(str, Enum):
    """Registration confirmation event types.

    Identifies the type of confirmation event emitted when a backend
    successfully processes a registration operation.

    Attributes:
        CONSUL_REGISTERED: Service successfully registered with Consul
        POSTGRES_REGISTRATION_UPSERTED: Registration record upserted in PostgreSQL
    """

    CONSUL_REGISTERED = "consul.registered"
    POSTGRES_REGISTRATION_UPSERTED = "postgres.registration_upserted"


__all__: list[str] = ["EnumConfirmationEventType"]
