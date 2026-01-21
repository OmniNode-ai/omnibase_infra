# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Notification Constants Module.

Provides constants for notification publishing, particularly for state
transition notifications used by projectors and orchestrators.

Sentinel Values
---------------
- **FROM_STATE_INITIAL**: Sentinel value used when publishing a state transition
  notification for a NEW entity (one that didn't exist before the projection).
  This is used instead of an empty string to provide clear semantic meaning.

Usage:
    >>> from omnibase_infra.runtime.constants_notification import FROM_STATE_INITIAL
    >>>
    >>> # In notification publishing code:
    >>> effective_from_state = from_state if from_state is not None else FROM_STATE_INITIAL
    >>>
    >>> # In notification consumers:
    >>> if notification.from_state == FROM_STATE_INITIAL:
    ...     # This is a new entity, not a state transition from existing state
    ...     handle_new_entity(notification)

Why Not Empty String?
    Using an empty string ("") for new entities is semantically ambiguous:
    - An empty string could be confused with an actual empty state value
    - It doesn't clearly communicate "this entity was just created"
    - Consumers cannot distinguish between "no previous state" and "empty state"

    The "__INITIAL__" sentinel provides:
    - Clear semantic meaning (obviously not a real FSM state name)
    - Easy detection in consumer code
    - Explicit intent communication

Related:
    - ModelStateTransitionNotification: The notification model that uses this value
    - MixinProjectorNotificationPublishing: The mixin that sets this value
    - TransitionNotificationPublisher: The publisher that emits these notifications

Related Tickets:
    - OMN-1139: Integrate TransitionNotificationPublisher with ProjectorShell

.. versionadded:: 0.8.0
    Created as part of OMN-1139 notification integration.
"""

from typing import Final

# Sentinel value for from_state when publishing notifications for NEW entities.
# This indicates the entity was just created and has no previous state.
# Consumers can check: if notification.from_state == FROM_STATE_INITIAL
FROM_STATE_INITIAL: Final[str] = "__INITIAL__"
"""Sentinel value indicating a new entity with no previous state.

This value is used as the ``from_state`` field in ModelStateTransitionNotification
when publishing a notification for a newly created entity. It replaces the
semantically ambiguous empty string ("") with a clear, distinguishable marker.

Example:
    >>> notification = ModelStateTransitionNotification(
    ...     aggregate_type="registration",
    ...     aggregate_id=uuid4(),
    ...     from_state=FROM_STATE_INITIAL,  # New entity
    ...     to_state="pending",
    ...     # ... other fields
    ... )

Note:
    The double underscore prefix/suffix convention (``__INITIAL__``) follows
    Python's pattern for special/dunder values, making it obviously not a
    real FSM state name that would be used in domain logic.
"""

__all__: list[str] = ["FROM_STATE_INITIAL"]
