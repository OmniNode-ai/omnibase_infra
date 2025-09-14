"""PostgreSQL connection manager EFFECT node implementation."""

import os
import stat
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import asyncpg
from asyncpg import Connection, Pool, Record

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.nodes.base.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer

from omnibase_infra.models.postgres.model_postgres_connection_config import ModelPostgresConnectionConfig
from omnibase_infra.models.postgres.model_postgres_health_data import ModelPostgresHealthData

from .models import (
    ModelPostgresConnectionManagerInput,
    ModelPostgresConnectionManagerOutput,
    ModelExecuteQueryInput,
    ModelExecuteQueryOutput,
    ModelFetchOneInput,
    ModelFetchOneOutput,
    ModelFetchValueInput,
    ModelFetchValueOutput,
    ModelGetHealthInput,
    ModelGetHealthOutput,
)


"""PostgreSQL connection manager EFFECT node implementation."""

import os
import stat
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import asyncpg
from asyncpg import Connection, Pool, Record




class NodePostgresConnectionManagerEffect(NodeEffectService):
    """PostgreSQL connection manager EFFECT node providing enterprise-grade connection pooling and database operations."""

    def __init__(self, container: ModelONEXContainer):
        """Initialize PostgreSQL connection manager EFFECT node.
        
        Args:
            container: ONEX container for dependency injection
        """
        super().__init__(container)
        self.pool: Optional[Pool] = None
        self.config: Optional[ModelPostgresConnectionConfig] = None
        self.is_initialized = False
        
        # Statistics and monitoring
        self.connection_stats = {
            "total_connections_created": 0,
            "total_queries_executed": 0,
            "total_errors": 0,
            "last_health_check": None,
        }

    async def _setup(self) -> None:
        """Set up the PostgreSQL connection pool."""
        try:
            # Get configuration from container
            self.config = await self._get_config_from_container()
            
            # Validate SSL file permissions if SSL files are specified
            if self.config.ssl_cert_file:
                self._validate_ssl_file_permissions(self.config.ssl_cert_file)
            if self.config.ssl_key_file:
                self._validate_ssl_file_permissions(self.config.ssl_key_file)
            if self.config.ssl_ca_file:
                self._validate_ssl_file_permissions(self.config.ssl_ca_file)

            # Build connection parameters
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.user,
                "password": self.config.password,
                "min_size": self.config.min_connections,
                "max_size": self.config.max_connections,
                "max_inactive_connection_lifetime": self.config.max_inactive_connection_lifetime,
                "max_queries": self.config.max_queries,
                "command_timeout": self.config.command_timeout,
                "server_settings": self.config.server_settings or {},
            }

            # Add SSL configuration if specified
            if self.config.ssl_mode != "disable":
                connection_params["ssl"] = self.config.ssl_mode
                if self.config.ssl_cert_file:
                    connection_params["ssl_cert"] = self.config.ssl_cert_file
                if self.config.ssl_key_file:
                    connection_params["ssl_key"] = self.config.ssl_key_file
                if self.config.ssl_ca_file:
                    connection_params["ssl_ca"] = self.config.ssl_ca_file

            # Create connection pool
            self.pool = await asyncpg.create_pool(**connection_params)

            # Test connection and set search path
            async with self.pool.acquire() as conn:
                await conn.execute(f"SET search_path TO {self.config.schema}, public")
                result = await conn.fetchval("SELECT current_schema()")
                if result != self.config.schema:
                    raise OnexError(
                        code=CoreErrorCode.DATABASE_CONNECTION_ERROR,
                        message=f"Failed to set schema to {self.config.schema}, got {result}",
                    )

            self.is_initialized = True
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.NODE_SETUP_ERROR,
                message="Failed to initialize PostgreSQL connection manager EFFECT node",
                details={"error": str(e)},
            ) from e

    async def _teardown(self) -> None:
        """Clean up the PostgreSQL connection pool."""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
            self.is_initialized = False
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.NODE_TEARDOWN_ERROR,
                message="Failed to teardown PostgreSQL connection manager EFFECT node",
                details={"error": str(e)},
            ) from e

    async def _get_config_from_container(self) -> ModelPostgresConnectionConfig:
        """Get PostgreSQL configuration from container."""
        try:
            # Try to get from container first, fallback to environment
            config = self.container.get_service("ModelPostgresConnectionConfig", required=False)
            if config is None:
                config = ModelPostgresConnectionConfig.from_environment()
            return config
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message="Failed to get PostgreSQL configuration",
                details={"error": str(e)},
            ) from e

    def _validate_ssl_file_permissions(self, file_path: str) -> None:
        """Validate SSL file permissions for security."""
        try:
            if not os.path.exists(file_path):
                raise OnexError(
                    code=CoreErrorCode.CONFIGURATION_ERROR,
                    message=f"SSL file not found: {file_path}",
                )

            # Check file permissions
            file_stat = os.stat(file_path)
            file_mode = stat.filemode(file_stat.st_mode)
            
            # SSL files should not be world-readable
            if file_stat.st_mode & stat.S_IROTH:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_ERROR,
                    message=f"SSL file {file_path} is world-readable (permissions: {file_mode})",
                )

        except OnexError:
            raise
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Failed to validate SSL file permissions for {file_path}: {str(e)}",
            ) from e

    # io_operations implementation

    async def execute_query(self, input_data: ModelExecuteQueryInput) -> ModelExecuteQueryOutput:
        """Execute a PostgreSQL query that may or may not return records."""
        if not self.is_initialized or not self.pool:
            raise OnexError(
                code=CoreErrorCode.NODE_NOT_INITIALIZED,
                message="PostgreSQL connection manager not initialized",
            )

        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                if input_data.parameters:
                    records = await conn.fetch(input_data.query, *input_data.parameters.values())
                else:
                    records = await conn.fetch(input_data.query)
                
                # Convert records to dictionaries
                result_records = [dict(record) for record in records]
                execution_time_ms = (time.time() - start_time) * 1000
                
                self.connection_stats["total_queries_executed"] += 1
                
                return ModelExecuteQueryOutput(
                    success=True,
                    records=result_records,
                    affected_rows=len(result_records),
                    execution_time_ms=execution_time_ms,
                )

        except Exception as e:
            self.connection_stats["total_errors"] += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to execute query: {str(e)}",
                details={"query": input_data.query},
            ) from e

    async def fetch_one(self, input_data: ModelFetchOneInput) -> ModelFetchOneOutput:
        """Fetch a single record from a PostgreSQL query."""
        if not self.is_initialized or not self.pool:
            raise OnexError(
                code=CoreErrorCode.NODE_NOT_INITIALIZED,
                message="PostgreSQL connection manager not initialized",
            )

        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                if input_data.parameters:
                    record = await conn.fetchrow(input_data.query, *input_data.parameters.values())
                else:
                    record = await conn.fetchrow(input_data.query)
                
                execution_time_ms = (time.time() - start_time) * 1000
                result_record = dict(record) if record else None
                
                self.connection_stats["total_queries_executed"] += 1
                
                return ModelFetchOneOutput(
                    success=True,
                    record=result_record,
                    execution_time_ms=execution_time_ms,
                )

        except Exception as e:
            self.connection_stats["total_errors"] += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to fetch one record: {str(e)}",
                details={"query": input_data.query},
            ) from e

    async def fetch_value(self, input_data: ModelFetchValueInput) -> ModelFetchValueOutput:
        """Fetch a single scalar value from a PostgreSQL query."""
        if not self.is_initialized or not self.pool:
            raise OnexError(
                code=CoreErrorCode.NODE_NOT_INITIALIZED,
                message="PostgreSQL connection manager not initialized",
            )

        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                if input_data.parameters:
                    value = await conn.fetchval(input_data.query, *input_data.parameters.values())
                else:
                    value = await conn.fetchval(input_data.query)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                self.connection_stats["total_queries_executed"] += 1
                
                return ModelFetchValueOutput(
                    success=True,
                    value=value,
                    execution_time_ms=execution_time_ms,
                )

        except Exception as e:
            self.connection_stats["total_errors"] += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to fetch value: {str(e)}",
                details={"query": input_data.query},
            ) from e

    async def get_health(self, input_data: ModelGetHealthInput) -> ModelGetHealthOutput:
        """Get comprehensive health status of the PostgreSQL connection pool."""
        try:
            if not self.is_initialized or not self.pool:
                health_data = ModelPostgresHealthData(
                    is_connected=False,
                    connection_pool_size=0,
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    database_version="Unknown",
                    schema=self.config.schema if self.config else "Unknown",
                    last_error="Connection pool not initialized",
                )
                
                return ModelGetHealthOutput(
                    success=False,
                    health_data=health_data,
                    timestamp=datetime.now(),
                )

            # Test connection and gather health data
            start_time = time.time()
            
            async with self.pool.acquire() as conn:
                # Basic connectivity test
                await conn.fetchval("SELECT 1")
                
                # Get database version
                db_version = await conn.fetchval("SELECT version()")
                
                # Get current schema
                current_schema = await conn.fetchval("SELECT current_schema()")
                
            response_time_ms = (time.time() - start_time) * 1000
            
            health_data = ModelPostgresHealthData(
                is_connected=True,
                connection_pool_size=self.pool.get_size(),
                active_connections=self.pool.get_size() - self.pool.get_idle_size(),
                idle_connections=self.pool.get_idle_size(),
                total_connections=self.pool.get_size(),
                database_version=db_version,
                schema=current_schema,
                response_time_ms=response_time_ms,
            )
            
            self.connection_stats["last_health_check"] = datetime.now()
            
            return ModelGetHealthOutput(
                success=True,
                health_data=health_data,
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.connection_stats["total_errors"] += 1
            
            health_data = ModelPostgresHealthData(
                is_connected=False,
                connection_pool_size=0,
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                database_version="Unknown",
                schema=self.config.schema if self.config else "Unknown",
                last_error=str(e),
            )
            
            return ModelGetHealthOutput(
                success=False,
                health_data=health_data,
                timestamp=datetime.now(),
            )

    @asynccontextmanager
    async def transaction(self):
        """Provide a transactional connection context (internal use)."""
        if not self.is_initialized or not self.pool:
            raise OnexError(
                code=CoreErrorCode.NODE_NOT_INITIALIZED,
                message="PostgreSQL connection manager not initialized",
            )
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn