# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Message Category Enumeration for ONEX Execution Shape Validation.

Defines the canonical message categories for ONEX event-driven architecture.
Each category corresponds to a specific Kafka topic naming convention and
has semantic meaning for handler type constraints.

Topic Naming Conventions:
    - EVENT: Read from `*.events` topics (e.g., `order.events`, `user.events`)
    - COMMAND: Read from `*.commands` topics (e.g., `order.commands`)
    - INTENT: Read from `*.intents` topics (e.g., `checkout.intents`)
    - PROJECTION: State projections for read models

Message Category Semantics:
    - EVENTs: Immutable facts about what has happened (past tense)
    - COMMANDs: Requests for action (imperative mood)
    - INTENTs: User intentions requiring validation (future-oriented)
    - PROJECTIONs: Derived state for query optimization
"""

from enum import Enum


class EnumMessageCategory(str, Enum):
    """Message categories for ONEX event-driven architecture.

    These represent the canonical message types that flow through
    the ONEX Kafka-based event bus. Each category has specific
    topic naming conventions and handler constraints.

    Attributes:
        EVENT: Domain events representing facts about what happened.
            Read from `*.events` topics. Immutable, past-tense naming.
            Example: OrderCreated, PaymentReceived, UserRegistered
        COMMAND: Commands requesting an action to be performed.
            Read from `*.commands` topics. Imperative mood naming.
            Example: CreateOrder, ProcessPayment, SendNotification
        INTENT: User intents requiring validation before processing.
            Read from `*.intents` topics. Future-oriented naming.
            Example: CheckoutIntent, SubscriptionIntent, TransferIntent
        PROJECTION: State projections for optimized read models.
            Used by reducers for state consolidation.
            Example: OrderSummaryProjection, UserProfileProjection
    """

    EVENT = "event"
    COMMAND = "command"
    INTENT = "intent"
    PROJECTION = "projection"


__all__ = ["EnumMessageCategory"]
