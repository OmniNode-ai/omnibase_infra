"""
Repository for node registration operations.

This module provides a high-performance repository pattern for managing
node registrations in the PostgreSQL database, with support for:
- CRUD operations with prepared statement caching
- Batch operations for efficiency
- Heartbeat tracking for health monitoring
- Query filtering by node type and capabilities
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from ..models.node_registration import (
    ModelNodeRegistration,
    ModelNodeRegistrationCreate,
    ModelNodeRegistrationUpdate,
)
from .postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class NodeRegistrationRepository:
    """
    Repository for managing node registrations in PostgreSQL.

    Provides high-performance CRUD operations for node registration data,
    leveraging the PostgresClient's prepared statement caching and connection
    pooling for optimal performance.

    Example:
        ```python
        # Initialize repository
        postgres_client = PostgresClient()
        await postgres_client.connect()
        repository = NodeRegistrationRepository(postgres_client)

        # Create registration
        registration = await repository.create_registration(
            ModelNodeRegistrationCreate(
                node_id="metadata-stamping-v1",
                node_type="effect",
                capabilities={"operations": ["stamp", "validate"]},
                endpoints={"stamp": "http://service:8053/api/v1/stamp"}
            )
        )

        # Get registration
        node = await repository.get_registration("metadata-stamping-v1")

        # Update heartbeat
        await repository.update_heartbeat("metadata-stamping-v1")

        # List all nodes
        nodes = await repository.list_all_registrations()
        ```
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Initialize repository with PostgreSQL client.

        Args:
            postgres_client: Connected PostgresClient instance
        """
        self.client = postgres_client

        # Field whitelist for application logic (prevent updating immutable fields)
        # Note: SQL injection is prevented by asyncpg's parameterized queries
        self._allowed_update_fields = {
            "capabilities",
            "endpoints",
            "metadata",
            "health_endpoint",
            "last_heartbeat",
        }

    def _validate_update_fields(self, update_data: dict) -> None:
        """
        Validate that all fields in update data are in the allowed whitelist.

        This is for application logic (preventing updates to immutable fields like
        'id', 'node_id', 'registered_at'), NOT for SQL injection prevention.
        SQL injection is prevented by asyncpg's parameterized queries ($1, $2, etc.).

        Args:
            update_data: Dictionary of fields to update

        Raises:
            ValueError: If any field is not in the allowed whitelist
        """
        if not isinstance(update_data, dict):
            raise ValueError("Update data must be a dictionary")

        provided_fields = set(update_data.keys())
        invalid_fields = provided_fields - self._allowed_update_fields

        if invalid_fields:
            raise ValueError(
                f"Invalid update fields: {invalid_fields}. "
                f"Allowed fields: {self._allowed_update_fields}"
            )

    async def create_registration(
        self, registration: ModelNodeRegistrationCreate
    ) -> ModelNodeRegistration:
        """
        Create a new node registration or update existing one (UPSERT).

        If a node with the same node_id already exists, this method will update
        the existing registration with the new values, preserving the original
        registered_at timestamp.

        Args:
            registration: Node registration creation model

        Returns:
            Created or updated node registration with generated ID and timestamps

        Raises:
            Exception: If database operation fails
        """
        # Get node_version from registration or use default
        node_version = getattr(registration, "node_version", "1.0.0")

        # UPSERT query: INSERT with ON CONFLICT UPDATE for re-registration support
        # Preserves original registered_at timestamp but updates all other fields
        query = """
            INSERT INTO node_registrations (
                node_id, node_type, node_version, capabilities, endpoints, metadata, health_endpoint
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (node_id)
            DO UPDATE SET
                node_type = EXCLUDED.node_type,
                node_version = EXCLUDED.node_version,
                capabilities = EXCLUDED.capabilities,
                endpoints = EXCLUDED.endpoints,
                metadata = EXCLUDED.metadata,
                health_endpoint = EXCLUDED.health_endpoint,
                last_heartbeat = NOW(),
                updated_at = NOW()
            RETURNING node_id, node_type, capabilities, endpoints, metadata,
                      health_endpoint, last_heartbeat, registered_at, updated_at
        """

        try:
            # Convert JSONB fields to JSON strings for asyncpg prepared statements
            capabilities_json = json.dumps(registration.capabilities)
            endpoints_json = json.dumps(registration.endpoints)
            metadata_json = json.dumps(registration.metadata)

            row = await self.client.fetch_one(
                query,
                registration.node_id,
                registration.node_type,
                node_version,
                capabilities_json,
                endpoints_json,
                metadata_json,
                registration.health_endpoint,
            )

            if not row:
                raise RuntimeError("Failed to create node registration")

            # Parse JSON strings back to dicts for the model
            return ModelNodeRegistration(
                node_id=row["node_id"],
                node_type=row["node_type"],
                capabilities=(
                    json.loads(row["capabilities"])
                    if isinstance(row["capabilities"], str)
                    else row["capabilities"]
                ),
                endpoints=(
                    json.loads(row["endpoints"])
                    if isinstance(row["endpoints"], str)
                    else row["endpoints"]
                ),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                ),
                health_endpoint=row["health_endpoint"],
                last_heartbeat=row["last_heartbeat"],
                registered_at=row["registered_at"],
                updated_at=row["updated_at"],
            )

        except Exception as e:
            logger.error(f"Failed to create node registration: {e}")
            raise

    async def get_registration(self, node_id: str) -> Optional[ModelNodeRegistration]:
        """
        Get a node registration by node_id.

        Args:
            node_id: Unique node identifier

        Returns:
            Node registration if found, None otherwise
        """
        query = """
            SELECT node_id, node_type, capabilities, endpoints, metadata,
                   health_endpoint, last_heartbeat, registered_at, updated_at
            FROM node_registrations
            WHERE node_id = $1
        """

        try:
            row = await self.client.fetch_one(query, node_id)

            if not row:
                return None

            # Parse JSON strings back to dicts for the model
            return ModelNodeRegistration(
                node_id=row["node_id"],
                node_type=row["node_type"],
                capabilities=(
                    json.loads(row["capabilities"])
                    if isinstance(row["capabilities"], str)
                    else row["capabilities"]
                ),
                endpoints=(
                    json.loads(row["endpoints"])
                    if isinstance(row["endpoints"], str)
                    else row["endpoints"]
                ),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                ),
                health_endpoint=row["health_endpoint"],
                last_heartbeat=row["last_heartbeat"],
                registered_at=row["registered_at"],
                updated_at=row["updated_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get node registration for {node_id}: {e}")
            return None

    async def update_registration(
        self, node_id: str, update: ModelNodeRegistrationUpdate
    ) -> Optional[ModelNodeRegistration]:
        """
        Update a node registration using parameterized queries.

        SQL injection is prevented by asyncpg's parameterized queries ($1, $2, etc.),
        which treat all parameters as data values, not SQL code.

        Args:
            node_id: Unique node identifier
            update: Fields to update

        Returns:
            Updated node registration if found, None otherwise
        """
        # Convert update model to dict and validate fields
        update_dict = update.model_dump(exclude_unset=True)

        # Validate that all provided fields are in the whitelist (application logic)
        self._validate_update_fields(update_dict)

        if not update_dict:
            # No fields to update
            return await self.get_registration(node_id)

        # Build parameterized update query
        # Field names are safe because they're validated against the whitelist
        # Values are safe because they're passed as parameters ($1, $2, etc.)
        set_clauses = []
        params = []
        param_index = 1

        for field, value in update_dict.items():
            if value is not None:
                set_clauses.append(f"{field} = ${param_index}")
                params.append(value)
                param_index += 1

        if not set_clauses:
            # No valid fields to update
            return await self.get_registration(node_id)

        # Always update updated_at timestamp
        set_clauses.append("updated_at = NOW()")

        # Add node_id as last parameter
        params.append(node_id)

        # Construct query with validated field names and parameterized values
        set_clause_str = ", ".join(set_clauses)
        query = f"""
            UPDATE node_registrations
            SET {set_clause_str}
            WHERE node_id = ${param_index}
            RETURNING node_id, node_type, capabilities, endpoints, metadata,
                      health_endpoint, last_heartbeat, registered_at, updated_at
        """

        try:
            row = await self.client.fetch_one(query, *params)

            if not row:
                return None

            # Parse JSON strings back to dicts for the model
            return ModelNodeRegistration(
                node_id=row["node_id"],
                node_type=row["node_type"],
                capabilities=(
                    json.loads(row["capabilities"])
                    if isinstance(row["capabilities"], str)
                    else row["capabilities"]
                ),
                endpoints=(
                    json.loads(row["endpoints"])
                    if isinstance(row["endpoints"], str)
                    else row["endpoints"]
                ),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                ),
                health_endpoint=row["health_endpoint"],
                last_heartbeat=row["last_heartbeat"],
                registered_at=row["registered_at"],
                updated_at=row["updated_at"],
            )

        except Exception as e:
            logger.error(f"Failed to update node registration for {node_id}: {e}")
            return None

    async def update_heartbeat(
        self, node_id: str, heartbeat_time: Optional[datetime] = None
    ) -> bool:
        """
        Update the last heartbeat timestamp for a node.

        Args:
            node_id: Unique node identifier
            heartbeat_time: Optional specific heartbeat time (defaults to NOW())

        Returns:
            True if update succeeded, False otherwise
        """
        if heartbeat_time:
            query = """
                UPDATE node_registrations
                SET last_heartbeat = $1, updated_at = NOW()
                WHERE node_id = $2
            """
            params = (heartbeat_time, node_id)
        else:
            query = """
                UPDATE node_registrations
                SET last_heartbeat = NOW(), updated_at = NOW()
                WHERE node_id = $1
            """
            params = (node_id,)

        try:
            result = await self.client.execute_query(query, *params)
            # Result format is "UPDATE N" where N is number of rows affected
            return result == "UPDATE 1"

        except Exception as e:
            logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False

    async def list_all_registrations(
        self, node_type: Optional[str] = None
    ) -> list[ModelNodeRegistration]:
        """
        List all node registrations, optionally filtered by node type.

        Args:
            node_type: Optional filter by ONEX node type

        Returns:
            List of node registrations
        """
        if node_type:
            query = """
                SELECT node_id, node_type, capabilities, endpoints, metadata,
                       health_endpoint, last_heartbeat, registered_at, updated_at
                FROM node_registrations
                WHERE node_type = $1
                ORDER BY registered_at DESC
            """
            params = (node_type,)
        else:
            query = """
                SELECT node_id, node_type, capabilities, endpoints, metadata,
                       health_endpoint, last_heartbeat, registered_at, updated_at
                FROM node_registrations
                ORDER BY registered_at DESC
            """
            params = ()

        try:
            rows = await self.client.fetch_all(query, *params)

            # Parse JSON strings back to dicts for the model
            return [
                ModelNodeRegistration(
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    capabilities=(
                        json.loads(row["capabilities"])
                        if isinstance(row["capabilities"], str)
                        else row["capabilities"]
                    ),
                    endpoints=(
                        json.loads(row["endpoints"])
                        if isinstance(row["endpoints"], str)
                        else row["endpoints"]
                    ),
                    metadata=(
                        json.loads(row["metadata"])
                        if isinstance(row["metadata"], str)
                        else row["metadata"]
                    ),
                    health_endpoint=row["health_endpoint"],
                    last_heartbeat=row["last_heartbeat"],
                    registered_at=row["registered_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to list node registrations: {e}")
            return []

    async def delete_registration(self, node_id: str) -> bool:
        """
        Delete a node registration.

        Args:
            node_id: Unique node identifier

        Returns:
            True if deletion succeeded, False otherwise
        """
        query = """
            DELETE FROM node_registrations
            WHERE node_id = $1
        """

        try:
            result = await self.client.execute_query(query, node_id)
            # Result format is "DELETE N" where N is number of rows affected
            return result == "DELETE 1"

        except Exception as e:
            logger.error(f"Failed to delete node registration for {node_id}: {e}")
            return False

    async def delete_node_registration(self, node_id: str) -> bool:
        """
        Delete a node registration (alias for delete_registration).

        This method is an alias for delete_registration() to support
        rollback operations in the registry node.

        Args:
            node_id: Unique node identifier

        Returns:
            True if deletion succeeded, False otherwise
        """
        return await self.delete_registration(node_id)

    async def find_by_capability(
        self, capability_key: str, capability_value: Any
    ) -> list[ModelNodeRegistration]:
        """
        Find nodes by a specific capability.

        Uses JSONB querying to find nodes with a specific capability key-value pair.

        Args:
            capability_key: Capability key to search for
            capability_value: Expected value for the capability

        Returns:
            List of matching node registrations
        """
        query = """
            SELECT node_id, node_type, capabilities, endpoints, metadata,
                   health_endpoint, last_heartbeat, registered_at, updated_at
            FROM node_registrations
            WHERE capabilities @> $1::jsonb
            ORDER BY registered_at DESC
        """

        # Create JSONB query for the capability
        capability_query = {capability_key: capability_value}

        try:
            rows = await self.client.fetch_all(query, capability_query)

            # Parse JSON strings back to dicts for the model
            return [
                ModelNodeRegistration(
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    capabilities=(
                        json.loads(row["capabilities"])
                        if isinstance(row["capabilities"], str)
                        else row["capabilities"]
                    ),
                    endpoints=(
                        json.loads(row["endpoints"])
                        if isinstance(row["endpoints"], str)
                        else row["endpoints"]
                    ),
                    metadata=(
                        json.loads(row["metadata"])
                        if isinstance(row["metadata"], str)
                        else row["metadata"]
                    ),
                    health_endpoint=row["health_endpoint"],
                    last_heartbeat=row["last_heartbeat"],
                    registered_at=row["registered_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(
                f"Failed to find nodes by capability {capability_key}={capability_value}: {e}"
            )
            return []

    async def get_healthy_nodes(
        self, max_age_seconds: int = 300
    ) -> list[ModelNodeRegistration]:
        """
        Get nodes with recent heartbeats (healthy nodes).

        Args:
            max_age_seconds: Maximum age of heartbeat in seconds (default: 5 minutes)

        Returns:
            List of nodes with recent heartbeats
        """
        query = """
            SELECT node_id, node_type, capabilities, endpoints, metadata,
                   health_endpoint, last_heartbeat, registered_at, updated_at
            FROM node_registrations
            WHERE last_heartbeat >= NOW() - INTERVAL '1 second' * $1
            ORDER BY last_heartbeat DESC
        """

        try:
            rows = await self.client.fetch_all(query, max_age_seconds)

            # Parse JSON strings back to dicts for the model
            return [
                ModelNodeRegistration(
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    capabilities=(
                        json.loads(row["capabilities"])
                        if isinstance(row["capabilities"], str)
                        else row["capabilities"]
                    ),
                    endpoints=(
                        json.loads(row["endpoints"])
                        if isinstance(row["endpoints"], str)
                        else row["endpoints"]
                    ),
                    metadata=(
                        json.loads(row["metadata"])
                        if isinstance(row["metadata"], str)
                        else row["metadata"]
                    ),
                    health_endpoint=row["health_endpoint"],
                    last_heartbeat=row["last_heartbeat"],
                    registered_at=row["registered_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get healthy nodes: {e}")
            return []
