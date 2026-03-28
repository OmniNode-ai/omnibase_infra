# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session graph schema for Memgraph projection.

Defines the Cypher DDL for the session coordination graph and re-exports
node label and relationship type enums for convenience.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 8).
"""

from __future__ import annotations

from omnibase_infra.services.session_registry.enum_node_label import (
    EnumNodeLabel,
)
from omnibase_infra.services.session_registry.enum_relationship_type import (
    EnumRelationshipType,
)

__all__ = [
    "EnumNodeLabel",
    "EnumRelationshipType",
    "SESSION_GRAPH_SCHEMA",
]

SESSION_GRAPH_SCHEMA: str = """\
CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;
CREATE CONSTRAINT ON (t:Task) ASSERT t.task_id IS UNIQUE;
CREATE CONSTRAINT ON (f:File) ASSERT f.path IS UNIQUE;
CREATE CONSTRAINT ON (p:PullRequest) ASSERT p.pr_id IS UNIQUE;
CREATE CONSTRAINT ON (r:Repository) ASSERT r.name IS UNIQUE;
CREATE INDEX ON :Task(status);
CREATE INDEX ON :Session(last_activity);
"""
