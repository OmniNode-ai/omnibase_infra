"""
In-memory query engine with fast indexes.

Provides sub-5ms lookups with O(1) hash-based indexing.
"""

import asyncio
from typing import Optional

import structlog

from .models import ModelOnextreeRoot, OnexTreeNode

logger = structlog.get_logger(__name__)


class OnexTreeQueryEngine:
    """
    Fast in-memory query engine for OnexTree.

    Performance targets:
    - Lookup: < 5ms
    - Index rebuild: < 100ms for 10K files
    - Memory: < 20MB for typical project

    Uses three primary indexes:
    1. exact_path_index: O(1) lookup by full path
    2. extension_index: O(1) lookup by file extension
    3. directory_index: O(1) lookup for directory children
    """

    def __init__(self):
        """Initialize query engine with empty indexes."""
        # Primary indexes
        self.exact_path_index: dict[str, OnexTreeNode] = {}
        self.extension_index: dict[str, list[OnexTreeNode]] = {}
        self.directory_index: dict[str, list[OnexTreeNode]] = {}
        self.name_index: dict[str, list[OnexTreeNode]] = {}  # For similarity search

        # Current tree root
        self.tree_root: Optional[ModelOnextreeRoot] = None
        self._index_lock = asyncio.Lock()

    async def load_tree(self, tree_root: ModelOnextreeRoot) -> None:
        """
        Load tree and rebuild indexes.

        Performance target: < 100ms for 10K files

        Args:
            tree_root: Tree root to load
        """
        async with self._index_lock:
            self.tree_root = tree_root

            # Clear existing indexes
            self.exact_path_index.clear()
            self.extension_index.clear()
            self.directory_index.clear()
            self.name_index.clear()

            # Rebuild indexes from tree
            await self._rebuild_indexes(tree_root.tree)

    async def _rebuild_indexes(self, node: OnexTreeNode, parent_path: str = "") -> None:
        """
        Recursively rebuild all indexes from tree.

        Args:
            node: Current node being indexed
            parent_path: Path of parent directory
        """
        current_path = node.path

        # Build exact path index
        self.exact_path_index[current_path] = node

        # Build name index for similarity search
        if node.name not in self.name_index:
            self.name_index[node.name] = []
        self.name_index[node.name].append(node)

        # Build extension index for files
        if node.type == "file" and node.extension:
            if node.extension not in self.extension_index:
                self.extension_index[node.extension] = []
            self.extension_index[node.extension].append(node)

        # Build directory index
        if node.type == "directory":
            if current_path not in self.directory_index:
                self.directory_index[current_path] = []

            # Recurse to children and add them to this directory's index
            if node.children:
                for child in node.children:
                    await self._rebuild_indexes(child, current_path)
                    self.directory_index[current_path].append(child)

    async def lookup_file(self, file_path: str) -> Optional[OnexTreeNode]:
        """
        Fast exact path lookup.

        Performance: O(1), < 1ms

        Args:
            file_path: Relative path from project root

        Returns:
            OnexTreeNode if found, None otherwise
        """
        return self.exact_path_index.get(file_path)

    async def check_file_exists(self, file_path: str) -> bool:
        """
        Fast existence check.

        Performance: O(1), < 1ms

        Args:
            file_path: Relative path from project root

        Returns:
            True if file exists in tree
        """
        return file_path in self.exact_path_index

    async def find_by_extension(
        self, extension: str, limit: int = 100
    ) -> list[OnexTreeNode]:
        """
        Find all files with given extension.

        Performance: O(1) lookup + O(n) filtering, < 5ms

        Args:
            extension: File extension without dot (e.g., 'py', 'js')
            limit: Maximum results to return

        Returns:
            List of matching OnexTreeNode objects
        """
        results = self.extension_index.get(extension, [])
        return results[:limit]

    async def find_by_name(self, name: str, limit: int = 100) -> list[OnexTreeNode]:
        """
        Find all files/directories with exact name match.

        Performance: O(1), < 2ms

        Args:
            name: File or directory name
            limit: Maximum results to return

        Returns:
            List of matching OnexTreeNode objects
        """
        results = self.name_index.get(name, [])
        return results[:limit]

    async def find_similar_names(
        self, name: str, limit: int = 10
    ) -> list[OnexTreeNode]:
        """
        Find files with similar names (simple substring match).

        Performance: O(n) over name index, < 10ms

        Args:
            name: Name to search for
            limit: Maximum results to return

        Returns:
            List of OnexTreeNode objects with similar names
        """
        results = []
        name_lower = name.lower()

        for indexed_name, nodes in self.name_index.items():
            if name_lower in indexed_name.lower():
                results.extend(nodes)
                if len(results) >= limit:
                    break

        return results[:limit]

    async def get_directory_children(self, dir_path: str) -> list[OnexTreeNode]:
        """
        Get all children of a directory.

        Performance: O(1), < 2ms

        Args:
            dir_path: Directory path relative to project root

        Returns:
            List of child OnexTreeNode objects
        """
        return self.directory_index.get(dir_path, [])

    async def get_statistics(self) -> Optional[dict[str, any]]:
        """
        Get current tree statistics.

        Returns:
            Statistics dictionary or None if no tree loaded
        """
        if not self.tree_root:
            return None

        return {
            "total_files": self.tree_root.statistics.total_files,
            "total_directories": self.tree_root.statistics.total_directories,
            "file_type_distribution": self.tree_root.statistics.file_type_distribution,
            "total_size_bytes": self.tree_root.statistics.total_size_bytes,
            "last_updated": self.tree_root.statistics.last_updated.isoformat(),
            "index_sizes": {
                "exact_path_entries": len(self.exact_path_index),
                "extension_types": len(self.extension_index),
                "directories_indexed": len(self.directory_index),
                "unique_names": len(self.name_index),
            },
        }

    async def search_by_path_pattern(
        self, pattern: str, limit: int = 50
    ) -> list[OnexTreeNode]:
        """
        Search for files matching path pattern (substring match).

        Performance: O(n) over all paths, < 10ms for 10K files

        Args:
            pattern: Pattern to match in path
            limit: Maximum results to return

        Returns:
            List of matching OnexTreeNode objects
        """
        results = []
        pattern_lower = pattern.lower()

        for path, node in self.exact_path_index.items():
            if pattern_lower in path.lower():
                results.append(node)
                if len(results) >= limit:
                    break

        return results
