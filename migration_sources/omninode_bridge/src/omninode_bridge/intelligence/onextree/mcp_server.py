"""
MCP server for OnexTree agent intelligence tools.

Exposes 5 core tools for fast project structure intelligence:
1. lookup_file - Fast exact path lookup
2. check_exists - Existence check with duplicate detection
3. get_structure - Hierarchical structure query
4. safe_create_file - Duplicate prevention check
5. get_file_context - Semantic context for decisions
"""

import json
from pathlib import Path
from typing import Any

import structlog
from mcp.server import Server
from mcp.types import TextContent, Tool

logger = structlog.get_logger(__name__)

from .filesystem_watcher import OnexTreeWatcher
from .generator import OnexTreeGenerator
from .parser import ToolOnextreeProcessor
from .query_engine import OnexTreeQueryEngine


class OnexTreeMCPServer:
    """
    MCP server exposing OnexTree intelligence tools.

    Provides 5 core tools for agent intelligence with sub-5ms performance.
    """

    def __init__(self, project_root: Path):
        """
        Initialize MCP server.

        Args:
            project_root: Root directory of project to index
        """
        self.project_root = Path(project_root).resolve()
        self.query_engine = OnexTreeQueryEngine()
        self.generator = OnexTreeGenerator(self.project_root)
        self.parser = ToolOnextreeProcessor()
        self.watcher = OnexTreeWatcher(
            self.project_root, on_change_callback=self._on_filesystem_change
        )

        self.server = Server("onextree-intelligence")
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all MCP tools with the server."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="mcp__onextree__lookup_file",
                    description="Fast exact path lookup in project tree (< 5ms). Returns file metadata if found.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Relative path from project root",
                            }
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="mcp__onextree__check_exists",
                    description="Check if file exists with duplicate detection. Optionally finds similar files by name.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Relative path from project root",
                            },
                            "check_similar": {
                                "type": "boolean",
                                "default": True,
                                "description": "Also check for similar files by name",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="mcp__onextree__get_structure",
                    description="Get hierarchical project structure. Returns directory tree with statistics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "root_path": {
                                "type": "string",
                                "default": "",
                                "description": "Root directory for query (empty for project root)",
                            },
                            "depth": {
                                "type": "integer",
                                "default": 3,
                                "description": "Maximum traversal depth (1=immediate children only)",
                            },
                            "file_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "File extensions to include (e.g. ['py', 'js'])",
                            },
                        },
                    },
                ),
                Tool(
                    name="mcp__onextree__safe_create_file",
                    description="Check before creating file to prevent duplicates. Returns recommendation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "proposed_path": {
                                "type": "string",
                                "description": "Proposed file path relative to project root",
                            },
                            "file_type": {
                                "type": "string",
                                "description": "File type/extension",
                            },
                            "purpose": {
                                "type": "string",
                                "description": "Intended purpose of the file",
                            },
                        },
                        "required": ["proposed_path"],
                    },
                ),
                Tool(
                    name="mcp__onextree__get_context",
                    description="Get semantic context for file location. Returns parent directory and sibling files.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "File path relative to project root",
                            },
                            "context_radius": {
                                "type": "integer",
                                "default": 2,
                                "description": "Levels of context to include (not yet implemented)",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "mcp__onextree__lookup_file":
                return await self._handle_lookup_file(arguments)
            elif name == "mcp__onextree__check_exists":
                return await self._handle_check_exists(arguments)
            elif name == "mcp__onextree__get_structure":
                return await self._handle_get_structure(arguments)
            elif name == "mcp__onextree__safe_create_file":
                return await self._handle_safe_create(arguments)
            elif name == "mcp__onextree__get_context":
                return await self._handle_get_context(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_lookup_file(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle lookup_file tool."""
        file_path = args["file_path"]
        node = await self.query_engine.lookup_file(file_path)

        if node:
            result = {
                "found": True,
                "path": node.path,
                "type": node.type,
                "name": node.name,
                "size": node.size,
                "extension": node.extension,
                "last_modified": (
                    node.last_modified.isoformat() if node.last_modified else None
                ),
            }
        else:
            result = {"found": False, "path": file_path}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_check_exists(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle check_exists tool."""
        file_path = args["file_path"]
        check_similar = args.get("check_similar", True)

        exists = await self.query_engine.check_file_exists(file_path)
        result = {"exists": exists, "path": file_path}

        # Check for similar files if requested and file doesn't exist
        if check_similar and not exists:
            file_name = Path(file_path).name
            similar_nodes = await self.query_engine.find_by_name(file_name, limit=5)

            result["similar_files"] = [
                {"path": node.path, "type": node.type, "reason": "same_name"}
                for node in similar_nodes
            ]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_get_structure(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle get_structure tool."""
        root_path = args.get("root_path", "")
        depth = args.get("depth", 3)
        file_types = args.get("file_types", [])

        # Build hierarchical structure with depth limiting
        structure = await self._build_structure_recursive(
            root_path, depth, file_types, current_depth=0
        )

        # Add statistics if available
        stats = await self.query_engine.get_statistics()
        if stats:
            structure["statistics"] = stats

        return [TextContent(type="text", text=json.dumps(structure, indent=2))]

    async def _build_structure_recursive(
        self,
        path: str,
        max_depth: int,
        file_types: list[str],
        current_depth: int = 0,
    ) -> dict[str, Any]:
        """
        Recursively build directory structure with depth limiting.

        Args:
            path: Current directory path
            max_depth: Maximum depth to traverse
            file_types: File extensions to include (empty = all)
            current_depth: Current recursion depth

        Returns:
            Dictionary with path, children, and metadata
        """
        # Get immediate children
        children = await self.query_engine.get_directory_children(path)

        # Filter by file types if specified
        if file_types:
            children = [
                child
                for child in children
                if child.type == "directory"
                or (child.extension and child.extension in file_types)
            ]

        # Build child structures
        child_structures = []
        for child in children[:50]:  # Limit to 50 items per level
            child_data = {
                "path": child.path,
                "name": child.name,
                "type": child.type,
                "extension": child.extension,
                "size": child.size,
            }

            # Recurse into directories if we haven't reached max depth
            if child.type == "directory" and current_depth < max_depth:
                subdirectory = await self._build_structure_recursive(
                    child.path, max_depth, file_types, current_depth + 1
                )
                child_data["children"] = subdirectory.get("children", [])
                child_data["child_count"] = subdirectory.get("child_count", 0)

            child_structures.append(child_data)

        return {
            "path": path or "(project root)",
            "children": child_structures,
            "child_count": len(children),
            "showing": min(len(children), 50),
            "depth": current_depth,
            "max_depth": max_depth,
        }

    async def _handle_safe_create(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle safe_create_file tool."""
        proposed_path = args["proposed_path"]
        file_type = args.get("file_type")
        purpose = args.get("purpose")

        # Check if exact path exists
        exists = await self.query_engine.check_file_exists(proposed_path)

        result = {
            "proposed_path": proposed_path,
            "can_create": not exists,
            "collision": exists,
        }

        if exists:
            result["recommendation"] = "file_exists_choose_different_path"
            result["message"] = f"File already exists at {proposed_path}"
        else:
            # Check for similar files
            file_name = Path(proposed_path).name
            similar = await self.query_engine.find_by_name(file_name, limit=3)

            if similar:
                result["recommendation"] = "similar_files_found_verify_intent"
                result["similar_files"] = [
                    {"path": node.path, "type": node.type} for node in similar
                ]
                result["message"] = (
                    f"Found {len(similar)} similar file(s), verify this is intentional"
                )
            else:
                result["recommendation"] = "proceed"
                result["message"] = "No conflicts detected, safe to create"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_get_context(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle get_file_context tool."""
        file_path = args["file_path"]

        # Check if file exists
        node = await self.query_engine.lookup_file(file_path)

        if not node:
            result = {
                "file_path": file_path,
                "exists": False,
                "message": "File not found in tree",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Get parent directory
        parent_path = str(Path(file_path).parent)
        if parent_path == ".":
            parent_path = ""

        siblings = await self.query_engine.get_directory_children(parent_path)

        # Build context
        context = {
            "file_path": file_path,
            "exists": True,
            "file_info": {
                "name": node.name,
                "type": node.type,
                "extension": node.extension,
                "size": node.size,
            },
            "parent_directory": parent_path or "(project root)",
            "siblings": [
                {"path": s.path, "name": s.name, "type": s.type} for s in siblings[:20]
            ],
            "sibling_count": len(siblings),
        }

        # Add architectural hints based on path
        hints = []
        path_lower = file_path.lower()
        if "service" in path_lower:
            hints.append("Service pattern detected")
        if "model" in path_lower:
            hints.append("Model/data layer pattern detected")
        if "test" in path_lower:
            hints.append("Test file detected")
        if "controller" in path_lower:
            hints.append("Controller pattern detected")

        if hints:
            context["architectural_hints"] = hints

        return [TextContent(type="text", text=json.dumps(context, indent=2))]

    async def _on_filesystem_change(self) -> None:
        """Handle filesystem change event."""
        logger.info("Filesystem changed, regenerating tree")
        try:
            tree_root = await self.generator.generate_tree()
            await self.query_engine.load_tree(tree_root)

            # Save to .onextree file
            onextree_path = self.project_root / ".onextree"
            self.parser.write_onextree_file(tree_root, onextree_path)
            logger.info("Tree regenerated and saved")

        except Exception as e:
            logger.error("Error regenerating tree", error=str(e), exc_info=True)

    async def start(self) -> None:
        """
        Start MCP server.

        Generates initial tree, loads into query engine, and starts
        filesystem watcher.
        """
        logger.info("Generating initial tree")
        tree_root = await self.generator.generate_tree()
        await self.query_engine.load_tree(tree_root)

        # Save to .onextree file
        onextree_path = self.project_root / ".onextree"
        self.parser.write_onextree_file(tree_root, onextree_path)
        logger.info("Initial tree saved", onextree_path=str(onextree_path))

        # Start filesystem watcher
        self.watcher.start()

        logger.info(
            "OnexTree MCP Server started",
            project_root=str(self.project_root),
            total_files=tree_root.statistics.total_files,
            total_directories=tree_root.statistics.total_directories,
        )

    async def stop(self) -> None:
        """Stop MCP server and cleanup."""
        logger.info("Stopping OnexTree MCP Server")
        self.watcher.stop()
        logger.info("Server stopped")
