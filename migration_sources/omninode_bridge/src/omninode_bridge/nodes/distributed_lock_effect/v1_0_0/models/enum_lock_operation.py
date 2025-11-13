#!/usr/bin/env python3
"""
Lock Operation Type Enumeration for NodeDistributedLockEffect.

Defines distributed lock operation types for PostgreSQL-backed distributed locking.
Part of the O.N.E. v0.1 compliant bridge architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumLockOperation
- String-based enum for JSON serialization
- Integration with ModelDistributedLockRequest
"""

from enum import Enum


class EnumLockOperation(str, Enum):
    """
    Distributed lock operation types.

    Operations:
    - ACQUIRE: Acquire a distributed lock with lease duration
    - RELEASE: Release a held lock
    - EXTEND: Extend lock lease duration before expiration
    - QUERY: Query lock status and metadata
    - CLEANUP: Clean up expired locks (background task)

    Usage:
        Used by NodeDistributedLockEffect to handle distributed locking
        operations across multiple instances with PostgreSQL backend.
    """

    ACQUIRE = "acquire"
    """
    Acquire a distributed lock with lease duration.

    Requires:
        - lock_name: Unique lock identifier
        - owner_id: Lock owner identifier
        - lease_duration: Lock lease duration in seconds (default: 30.0)

    Returns:
        - success: True if lock acquired
        - lock_info: Lock metadata (acquired_at, expires_at, owner_id)

    Error Cases:
        - Lock already held by another owner
        - Maximum acquisition attempts exceeded
        - Database connection failure
    """

    RELEASE = "release"
    """
    Release a held distributed lock.

    Requires:
        - lock_name: Unique lock identifier
        - owner_id: Lock owner identifier (must match current owner)

    Returns:
        - success: True if lock released
        - lock_info: Lock metadata after release

    Error Cases:
        - Lock not held by this owner
        - Lock already released
        - Database connection failure
    """

    EXTEND = "extend"
    """
    Extend lock lease duration before expiration.

    Requires:
        - lock_name: Unique lock identifier
        - owner_id: Lock owner identifier (must match current owner)
        - extension_duration: Additional duration in seconds

    Returns:
        - success: True if lease extended
        - lock_info: Updated lock metadata with new expiration

    Error Cases:
        - Lock not held by this owner
        - Lock already expired
        - Database connection failure
    """

    QUERY = "query"
    """
    Query lock status and metadata.

    Requires:
        - lock_name: Unique lock identifier

    Optional:
        - owner_id: Filter by owner

    Returns:
        - lock_info: Lock metadata (status, owner, expiration)

    Error Cases:
        - Lock not found
        - Database connection failure
    """

    CLEANUP = "cleanup"
    """
    Clean up expired locks (background task).

    Requires:
        - None (cleanup all expired locks)

    Optional:
        - max_age_seconds: Maximum lock age for cleanup

    Returns:
        - cleaned_count: Number of expired locks removed
        - lock_names: List of cleaned lock names

    Error Cases:
        - Database connection failure

    Usage:
        Typically called by background cleanup task every 60 seconds
        to prevent lock table growth and remove stale locks.
    """

    def is_mutating(self) -> bool:
        """
        Check if this operation modifies lock state.

        Returns:
            True for ACQUIRE, RELEASE, EXTEND, CLEANUP operations.
            False for QUERY operation.

        Usage:
            Useful for determining transaction isolation levels and
            read replica routing (QUERY can use read replicas).
        """
        return self in (
            EnumLockOperation.ACQUIRE,
            EnumLockOperation.RELEASE,
            EnumLockOperation.EXTEND,
            EnumLockOperation.CLEANUP,
        )

    def requires_owner_id(self) -> bool:
        """
        Check if this operation requires owner_id field.

        Returns:
            True for ACQUIRE, RELEASE, EXTEND operations.
            False for QUERY, CLEANUP operations.

        Usage:
            Used for input validation to ensure required fields are present.
        """
        return self in (
            EnumLockOperation.ACQUIRE,
            EnumLockOperation.RELEASE,
            EnumLockOperation.EXTEND,
        )

    def is_background_operation(self) -> bool:
        """
        Check if this operation is typically called by background tasks.

        Returns:
            True for CLEANUP operation.
            False for all other operations.

        Usage:
            Useful for monitoring and metrics collection to distinguish
            user-initiated operations from background maintenance tasks.
        """
        return self == EnumLockOperation.CLEANUP


__all__ = ["EnumLockOperation"]
