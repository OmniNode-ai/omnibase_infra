"""
mysql - Create a MySQL database adapter Effect node with the following features:
    - Connection pooling (10-100 connections)
    - Automatic retry logic with exponential backoff (max 3 retries)
    - Circuit breaker pattern for resilience
    - Full CRUD operations: Create, Read, Update, Delete, List, BulkInsert
    - Transaction support with rollback capability
    - Prepared statements for SQL injection prevention
    - Connection health monitoring
    - Structured logging with query metrics
    - Async/await support for all operations

Generated: 2025-10-30T11:00:41.855167+00:00
ONEX v2.0 Compliant
"""

from .node import NodeMysqlEffect

__all__ = ["NodeMysqlEffect"]
