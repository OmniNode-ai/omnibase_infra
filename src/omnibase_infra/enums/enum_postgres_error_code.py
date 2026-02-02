# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Error Code Enumeration.

Defines structured error codes for PostgreSQL persistence operations. These codes
enable precise error classification, debugging, and programmatic error handling
for contract registry persistence via NodeContractPersistenceEffect.

Error Code Categories:
    - Connection errors: Connection, timeout, authentication failures
    - Operation errors: Specific operation failures (upsert, topic, etc.)
    - Unknown errors: Catch-all for unclassified failures

Usage:
    >>> from omnibase_infra.enums import EnumPostgresErrorCode
    >>> error_code = EnumPostgresErrorCode.CONNECTION_ERROR
    >>> print(f"Error: {error_code.value}")
    Error: POSTGRES_CONNECTION_ERROR

    >>> # Check if error is retriable
    >>> if error_code.is_retriable:
    ...     print("Will retry operation")

    >>> # Categorize error type
    >>> if error_code.is_connection_error:
    ...     print("Connection-level failure")

See Also:
    - NodeContractPersistenceEffect: Effect node using these error codes
    - ContractRegistryReducer: Source of intents that may trigger these errors
    - contract.yaml: Error code definitions in error_handling.error_codes
"""

from enum import Enum


class EnumPostgresErrorCode(str, Enum):
    """Error codes for PostgreSQL persistence operations.

    These codes provide structured classification for failures during
    contract registry persistence operations. Each code maps to a specific
    failure scenario as defined in the contract.yaml error_codes section.

    Connection Errors (retriable):
        CONNECTION_ERROR: Connection to PostgreSQL server failed.
            The database server is unreachable or connection was refused.
            Verify PostgreSQL server is running and network is accessible.

        TIMEOUT_ERROR: PostgreSQL operation exceeded timeout.
            The operation took longer than the configured timeout threshold.
            Check database load and query performance.

    Authentication Errors (non-retriable):
        AUTH_ERROR: Authentication with PostgreSQL server failed.
            Invalid credentials or insufficient privileges.
            Verify POSTGRES_USER and POSTGRES_PASSWORD in .env.

    Operation Errors (non-retriable):
        UPSERT_ERROR: PostgreSQL upsert operation failed.
            Insert/update of contract record failed due to constraint
            violation, invalid data, or schema mismatch.

        TOPIC_UPDATE_ERROR: PostgreSQL topic update failed.
            Failed to update topic routing table. Check JSONB array
            operations and topic table schema.

        MARK_STALE_ERROR: PostgreSQL mark stale operation failed.
            Batch staleness marking failed. Check is_stale column
            and last_seen_at timestamp handling.

        HEARTBEAT_ERROR: PostgreSQL heartbeat update failed.
            Heartbeat timestamp update failed. Verify contract_id
            exists and last_seen_at column is writable.

        DEACTIVATE_ERROR: PostgreSQL deactivation failed.
            Soft delete (is_active=false) failed. Check contract_id
            validity and is_active column constraints.

        CLEANUP_ERROR: PostgreSQL topic cleanup failed.
            Failed to remove contract from topic arrays. Check JSONB
            array manipulation and referential integrity.

    Unknown Errors (non-retriable):
        UNKNOWN_ERROR: Unknown error during PostgreSQL operation.
            Catch-all for unclassified PostgreSQL failures.
            Check logs for underlying exception details.
    """

    # Connection errors
    CONNECTION_ERROR = "POSTGRES_CONNECTION_ERROR"
    TIMEOUT_ERROR = "POSTGRES_TIMEOUT_ERROR"
    AUTH_ERROR = "POSTGRES_AUTH_ERROR"

    # Operation errors
    UPSERT_ERROR = "POSTGRES_UPSERT_ERROR"
    TOPIC_UPDATE_ERROR = "POSTGRES_TOPIC_UPDATE_ERROR"
    MARK_STALE_ERROR = "POSTGRES_MARK_STALE_ERROR"
    HEARTBEAT_ERROR = "POSTGRES_HEARTBEAT_ERROR"
    DEACTIVATE_ERROR = "POSTGRES_DEACTIVATE_ERROR"
    CLEANUP_ERROR = "POSTGRES_CLEANUP_ERROR"

    # Unknown errors
    UNKNOWN_ERROR = "POSTGRES_UNKNOWN_ERROR"

    @property
    def is_retriable(self) -> bool:
        """Check if this error is retriable.

        Retriable errors indicate transient failures that may succeed
        on retry, such as connection issues or timeouts. Non-retriable
        errors indicate permanent failures requiring intervention.

        Returns:
            True if the error is retriable, False otherwise.
        """
        return self in {
            EnumPostgresErrorCode.CONNECTION_ERROR,
            EnumPostgresErrorCode.TIMEOUT_ERROR,
        }

    @property
    def is_connection_error(self) -> bool:
        """Check if this is a connection-level error.

        Connection errors indicate infrastructure-level failures
        rather than operation-specific issues.

        Returns:
            True if this is a connection-level error.
        """
        return self in {
            EnumPostgresErrorCode.CONNECTION_ERROR,
            EnumPostgresErrorCode.TIMEOUT_ERROR,
            EnumPostgresErrorCode.AUTH_ERROR,
        }

    @property
    def is_operation_error(self) -> bool:
        """Check if this is an operation-specific error.

        Operation errors indicate failures in specific database
        operations rather than infrastructure issues.

        Returns:
            True if this is an operation-specific error.
        """
        return self in {
            EnumPostgresErrorCode.UPSERT_ERROR,
            EnumPostgresErrorCode.TOPIC_UPDATE_ERROR,
            EnumPostgresErrorCode.MARK_STALE_ERROR,
            EnumPostgresErrorCode.HEARTBEAT_ERROR,
            EnumPostgresErrorCode.DEACTIVATE_ERROR,
            EnumPostgresErrorCode.CLEANUP_ERROR,
        }

    @property
    def description(self) -> str:
        """Get human-readable description of the error code.

        Returns:
            Description string for the error code.
        """
        descriptions = {
            EnumPostgresErrorCode.CONNECTION_ERROR: (
                "Connection to PostgreSQL server failed"
            ),
            EnumPostgresErrorCode.TIMEOUT_ERROR: (
                "PostgreSQL operation exceeded timeout"
            ),
            EnumPostgresErrorCode.AUTH_ERROR: (
                "Authentication with PostgreSQL server failed"
            ),
            EnumPostgresErrorCode.UPSERT_ERROR: "PostgreSQL upsert operation failed",
            EnumPostgresErrorCode.TOPIC_UPDATE_ERROR: "PostgreSQL topic update failed",
            EnumPostgresErrorCode.MARK_STALE_ERROR: (
                "PostgreSQL mark stale operation failed"
            ),
            EnumPostgresErrorCode.HEARTBEAT_ERROR: "PostgreSQL heartbeat update failed",
            EnumPostgresErrorCode.DEACTIVATE_ERROR: "PostgreSQL deactivation failed",
            EnumPostgresErrorCode.CLEANUP_ERROR: "PostgreSQL topic cleanup failed",
            EnumPostgresErrorCode.UNKNOWN_ERROR: (
                "Unknown error during PostgreSQL operation"
            ),
        }
        return descriptions.get(self, "Unknown error")


__all__ = ["EnumPostgresErrorCode"]
