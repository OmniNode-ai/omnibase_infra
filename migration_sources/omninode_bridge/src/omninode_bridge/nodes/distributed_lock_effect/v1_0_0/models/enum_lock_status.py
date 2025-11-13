#!/usr/bin/env python3
"""
Lock Status Enumeration for NodeDistributedLockEffect.

Defines distributed lock status states for tracking lock lifecycle.
Part of the O.N.E. v0.1 compliant bridge architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumLockStatus
- String-based enum for JSON serialization
- Integration with ModelLockInfo
"""

from enum import Enum


class EnumLockStatus(str, Enum):
    """
    Distributed lock status states.

    Status Lifecycle:
    - AVAILABLE → ACQUIRED: Lock successfully acquired
    - ACQUIRED → EXPIRED: Lock lease expired without renewal
    - ACQUIRED → RELEASED: Lock explicitly released by owner
    - EXPIRED/RELEASED: Can transition back to AVAILABLE after cleanup

    Usage:
        Used by NodeDistributedLockEffect to track lock status in
        ModelLockInfo responses and database persistence.
    """

    AVAILABLE = "available"
    """
    Lock is available for acquisition.

    Initial state when lock is not held by any owner or after cleanup
    of expired/released locks.

    Transitions:
        - AVAILABLE → ACQUIRED: When lock is successfully acquired
    """

    ACQUIRED = "acquired"
    """
    Lock is currently held by an owner.

    Active state during lock ownership with valid lease.

    Transitions:
        - ACQUIRED → RELEASED: When owner explicitly releases lock
        - ACQUIRED → EXPIRED: When lease duration expires without renewal

    Lock Metadata:
        - owner_id: Current lock owner
        - acquired_at: Timestamp when lock was acquired
        - expires_at: Timestamp when lock lease expires
        - lease_duration: Lock lease duration in seconds
    """

    EXPIRED = "expired"
    """
    Lock lease has expired without renewal.

    Terminal state indicating lock owner failed to renew lease before
    expiration. Lock is eligible for cleanup.

    Transitions:
        - EXPIRED → AVAILABLE: After cleanup task removes expired lock

    Cleanup:
        Background cleanup task removes expired locks based on
        expiration timestamp and max_age_seconds threshold.
    """

    RELEASED = "released"
    """
    Lock was explicitly released by owner.

    Terminal state indicating normal lock release. Lock is eligible
    for cleanup or immediate reacquisition.

    Transitions:
        - RELEASED → AVAILABLE: After cleanup or immediate reuse

    Cleanup:
        Released locks may be immediately available for reacquisition
        or removed by background cleanup task.
    """

    def is_available_for_acquisition(self) -> bool:
        """
        Check if lock in this status can be acquired.

        Returns:
            True for AVAILABLE status.
            False for ACQUIRED, EXPIRED, RELEASED statuses.

        Usage:
            Used during lock acquisition to determine if lock can be acquired.
        """
        return self == EnumLockStatus.AVAILABLE

    def is_active(self) -> bool:
        """
        Check if lock is actively held.

        Returns:
            True for ACQUIRED status (lock is held and lease is valid).
            False for AVAILABLE, EXPIRED, RELEASED statuses.

        Usage:
            Useful for monitoring active lock count and detecting
            lock contention issues.
        """
        return self == EnumLockStatus.ACQUIRED

    def is_terminal(self) -> bool:
        """
        Check if this is a terminal status (eligible for cleanup).

        Returns:
            True for EXPIRED, RELEASED statuses.
            False for AVAILABLE, ACQUIRED statuses.

        Usage:
            Used by cleanup task to identify locks eligible for removal
            from the database.
        """
        return self in (EnumLockStatus.EXPIRED, EnumLockStatus.RELEASED)

    def can_transition_to(self, target: "EnumLockStatus") -> bool:
        """
        Validate if transition to target status is allowed.

        Args:
            target: Target status to transition to

        Returns:
            True if transition is valid, False otherwise

        Valid Transitions:
            - AVAILABLE → ACQUIRED
            - ACQUIRED → RELEASED
            - ACQUIRED → EXPIRED
            - EXPIRED → AVAILABLE (after cleanup)
            - RELEASED → AVAILABLE (after cleanup)
        """
        valid_transitions = {
            EnumLockStatus.AVAILABLE: {EnumLockStatus.ACQUIRED},
            EnumLockStatus.ACQUIRED: {
                EnumLockStatus.RELEASED,
                EnumLockStatus.EXPIRED,
            },
            EnumLockStatus.EXPIRED: {EnumLockStatus.AVAILABLE},
            EnumLockStatus.RELEASED: {EnumLockStatus.AVAILABLE},
        }

        return target in valid_transitions.get(self, set())


__all__ = ["EnumLockStatus"]
