"""
OnexTree Standalone Agent Intelligence System.

Provides fast, in-memory project structure intelligence for Claude Code agents
with sub-5ms lookup performance and automatic filesystem watching.

Core Components:
- ModelOnextreeRoot: Pydantic models for .onextree format
- OnexTreeGenerator: Filesystem tree generation
- OnexTreeQueryEngine: Fast in-memory query engine
- OnexTreeWatcher: Automatic filesystem updates
- OnexTreeMCPServer: MCP server with 5 core tools

Performance Targets:
- Lookup: < 5ms
- Tree generation: < 100ms for 10K files
- Watcher latency: < 1s
- Memory: < 20MB for typical projects
"""

from .models import ModelOnextreeRoot, OnexTreeNode, ProjectStatistics

__all__ = [
    "OnexTreeNode",
    "ProjectStatistics",
    "ModelOnextreeRoot",
]
