# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for Registry Effect Node dependencies.

These protocols define the interfaces for handler and event bus dependencies
using duck typing with Python's Protocol class.

Type Aliases:
    JsonPrimitive: Basic JSON-serializable types (str, int, float, bool, None)
    JsonValue: Recursive JSON value type supporting nested structures
    EnvelopeDict: Dictionary type for operation envelopes
    ResultDict: Dictionary type for operation results
    HandlerResponse: Union type for handler response types (Pydantic models or dict)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse

# JSON-serializable value type for strong typing (per ONEX guidelines: never use Any/object).
#
# DESIGN NOTE: Why we use `dict | list` instead of recursive `dict[str, JsonValue]`:
# 1. Recursive type aliases cause infinite recursion in mypy/pyright during type inference
# 2. Pydantic v2 cannot generate JSON schema for recursively-defined type aliases
# 3. The runtime behavior is identical - Python's dict/list accept any JSON-serializable values
#
# This is a documented ONEX exception for JSON-value containers. The trade-off is reduced
# static type safety for nested structures in exchange for practical tooling compatibility.
# For strongly-typed nested data, prefer explicit Pydantic models over JsonValue.
JsonPrimitive = str | int | float | bool | None
JsonValue = str | int | float | bool | None | dict | list

# Envelope dictionary types for protocol methods.
# These types are intentionally broad to support various payload structures
# while still being more specific than 'Any' or 'object'.
# The 'payload' value can be a dict with various JSON-serializable values,
# including typed lists like list[str] for SQL parameters.
EnvelopeDict = dict[str, str | dict[str, JsonValue | list[str]] | list[str] | UUID]
ResultDict = dict[
    str, str | dict[str, JsonValue | list[dict]] | list[dict] | int | bool
]


@runtime_checkable
class ProtocolConsulExecutor(Protocol):
    """Protocol for Consul handler executor objects.

    Consul handlers must implement an async execute method that accepts an envelope
    dictionary and returns a ModelConsulHandlerResponse.
    """

    async def execute(self, envelope: EnvelopeDict) -> ModelConsulHandlerResponse:
        """Execute a Consul operation based on the envelope contents.

        Args:
            envelope: Dictionary containing operation details with keys:
                - operation: The Consul operation (e.g., "consul.register", "consul.deregister")
                - payload: Operation-specific data
                - correlation_id: UUID for distributed tracing

        Returns:
            ModelConsulHandlerResponse with:
                - status: "success" or "error"
                - payload: ModelConsulHandlerPayload with operation-specific data
                - correlation_id: UUID for request/response correlation
        """
        ...


@runtime_checkable
class ProtocolDbExecutor(Protocol):
    """Protocol for Database handler executor objects.

    Database handlers must implement an async execute method that accepts an envelope
    dictionary and returns a ModelDbQueryResponse.
    """

    async def execute(self, envelope: EnvelopeDict) -> ModelDbQueryResponse:
        """Execute a database operation based on the envelope contents.

        Args:
            envelope: Dictionary containing operation details with keys:
                - operation: The database operation (e.g., "db.query", "db.execute")
                - payload: Operation-specific data with "sql" and optional "params"
                - correlation_id: UUID for distributed tracing

        Returns:
            ModelDbQueryResponse with:
                - status: "success" or "error"
                - payload: ModelDbQueryPayload with rows and row_count
                - correlation_id: UUID for request/response correlation
        """
        ...


@runtime_checkable
class ProtocolEventBus(Protocol):
    """Protocol for event bus objects.

    Event bus must implement an async publish method for sending messages
    to topics.
    """

    async def publish(self, topic: str, key: bytes, value: bytes) -> None:
        """Publish a message to a topic.

        Args:
            topic: The topic name to publish to
            key: Message key as bytes
            value: Message value as bytes (typically JSON-encoded)
        """
        ...


__all__ = [
    "EnvelopeDict",
    "JsonPrimitive",
    "JsonValue",
    "ProtocolConsulExecutor",
    "ProtocolDbExecutor",
    "ProtocolEventBus",
    "ResultDict",
]
