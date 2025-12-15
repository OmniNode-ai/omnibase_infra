# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Models Module.

This module exports Pydantic models for handler request/response structures.
All models are strongly typed to eliminate Any usage.

Exports:
    ModelDbQueryPayload: Payload containing database query results
    ModelDbQueryResponse: Full database query response envelope
    ModelDbHealthResponse: Database adapter health check response
    ModelDbDescribeResponse: Database adapter metadata and capabilities
"""

from omnibase_infra.handlers.models.model_db_describe_response import (
    ModelDbDescribeResponse,
)
from omnibase_infra.handlers.models.model_db_health_response import (
    ModelDbHealthResponse,
)
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse

__all__: list[str] = [
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbHealthResponse",
    "ModelDbDescribeResponse",
]
