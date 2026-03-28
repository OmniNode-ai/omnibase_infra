# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cypher query builders for session graph analysis via Memgraph.

Provides parameterized Cypher queries for conflict detection, dependency
chain traversal, and lineage tree analysis. Queries are built as strings
with parameter placeholders ($task_id) for safe execution via the
Memgraph driver.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 10).

Queries:
    - file_conflict: Find other active tasks touching overlapping files.
    - dependency_chain: Traverse DEPENDS_ON chain (variable-length *1..5).
    - lineage_tree: Full task -> sessions -> files -> PRs tree.
"""

from __future__ import annotations


def build_file_conflict_query(task_id: str) -> str:
    """Build a Cypher query to find tasks with overlapping file touches.

    Finds all other active tasks that touch at least one file in common
    with the given task. Returns each conflicting task_id and the list
    of shared file paths.

    Args:
        task_id: The Linear ticket ID (e.g., "OMN-1234").

    Returns:
        A Cypher query string with the task_id literal interpolated.
    """
    return (
        f"MATCH (t1:Task {{task_id: '{task_id}'}})"
        f"-[:TOUCHES]->(f:File)<-[:TOUCHES]-(t2:Task) "
        f"WHERE t1 <> t2 AND t2.status = 'active' "
        f"RETURN t2.task_id AS task_id, collect(f.path) AS shared_files"
    )


def build_dependency_chain_query(task_id: str) -> str:
    """Build a Cypher query to traverse the DEPENDS_ON chain.

    Follows DEPENDS_ON edges up to 5 hops deep from the given task,
    returning the ordered chain of task_ids in each dependency path.

    Args:
        task_id: The Linear ticket ID (e.g., "OMN-1234").

    Returns:
        A Cypher query string with the task_id literal interpolated.
    """
    return (
        f"MATCH path = (t:Task {{task_id: '{task_id}'}})"
        f"-[:DEPENDS_ON*1..5]->(dep:Task) "
        f"RETURN [n IN nodes(path) | n.task_id] AS chain"
    )


def build_lineage_tree_query(task_id: str) -> str:
    """Build a Cypher query for full task lineage tree.

    Returns the task node along with all connected sessions, files,
    and pull requests via optional matches. Useful for displaying a
    complete picture of a task's footprint across the system.

    Args:
        task_id: The Linear ticket ID (e.g., "OMN-1234").

    Returns:
        A Cypher query string with the task_id literal interpolated.
    """
    return (
        f"MATCH (t:Task {{task_id: '{task_id}'}}) "
        f"OPTIONAL MATCH (t)<-[:WORKS_ON]-(s:Session) "
        f"OPTIONAL MATCH (t)-[:TOUCHES]->(f:File) "
        f"OPTIONAL MATCH (t)-[:PRODUCED]->(p:PullRequest) "
        f"RETURN t, collect(DISTINCT s) AS sessions, "
        f"collect(DISTINCT f) AS files, collect(DISTINCT p) AS prs"
    )
