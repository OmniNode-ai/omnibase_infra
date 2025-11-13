"""
Filesystem tree generation logic.

Efficiently scans filesystem and builds OnexTree structure with statistics.
"""

from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import structlog

from .config import DEFAULT_EXCLUDE_PATTERNS
from .models import ModelOnextreeRoot, OnexTreeNode, ProjectStatistics

logger = structlog.get_logger(__name__)


class OnexTreeGenerator:
    """
    Generates OnexTree from filesystem.

    Performance target: < 100ms for 10K files
    """

    def __init__(
        self,
        project_root: Path,
        exclude_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize tree generator.

        Args:
            project_root: Root directory to scan
            exclude_patterns: Patterns to exclude (uses defaults if None)
        """
        self.project_root = Path(project_root).resolve()
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.statistics = ProjectStatistics()

    async def generate_tree(self) -> ModelOnextreeRoot:
        """
        Generate complete tree from filesystem.

        Performance target: < 100ms for 10K files

        Returns:
            ModelOnextreeRoot with complete tree and statistics
        """
        # Reset statistics for fresh generation
        self.statistics = ProjectStatistics()

        # Scan root directory recursively
        root_node = await self._scan_directory(self.project_root)

        # Update statistics timestamp
        self.statistics.last_updated = datetime.now()

        return ModelOnextreeRoot(
            project_root=str(self.project_root),
            generated_at=datetime.now(),
            tree=root_node,
            statistics=self.statistics,
        )

    async def _scan_directory(self, dir_path: Path) -> OnexTreeNode:
        """
        Recursively scan directory and build tree node.

        Args:
            dir_path: Directory to scan

        Returns:
            OnexTreeNode representing directory with children
        """
        children = []

        try:
            # Sort entries for consistent ordering
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))

            for entry in entries:
                # Skip excluded paths
                if self._should_exclude(entry):
                    continue

                try:
                    if entry.is_dir():
                        # Recursively scan subdirectory
                        self.statistics.total_directories += 1
                        child_node = await self._scan_directory(entry)
                        children.append(child_node)
                    elif entry.is_file():
                        # Create file node
                        self.statistics.total_files += 1
                        child_node = self._create_file_node(entry)
                        children.append(child_node)
                except (PermissionError, OSError) as e:
                    # Skip files/dirs we can't access
                    logger.debug(
                        "Skipping inaccessible path", path=str(entry), error=str(e)
                    )
                    continue

        except PermissionError as e:
            logger.warning(
                "Permission denied accessing directory",
                directory=str(dir_path),
                error=str(e),
            )

        # Create directory node
        relative_path = str(dir_path.relative_to(self.project_root))
        if relative_path == ".":
            relative_path = ""

        return OnexTreeNode(
            path=relative_path,
            name=(
                dir_path.name
                if dir_path != self.project_root
                else self.project_root.name
            ),
            type="directory",
            children=children,
        )

    def _create_file_node(self, file_path: Path) -> OnexTreeNode:
        """
        Create node for single file.

        Args:
            file_path: Path to file

        Returns:
            OnexTreeNode representing the file
        """
        try:
            stat = file_path.stat()
            extension = file_path.suffix[1:] if file_path.suffix else None

            # Update statistics
            if extension:
                self.statistics.file_type_distribution[extension] = (
                    self.statistics.file_type_distribution.get(extension, 0) + 1
                )
            self.statistics.total_size_bytes += stat.st_size

            # Create file node
            relative_path = str(file_path.relative_to(self.project_root))

            return OnexTreeNode(
                path=relative_path,
                name=file_path.name,
                type="file",
                size=stat.st_size,
                extension=extension,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
            )

        except (OSError, PermissionError) as e:
            logger.debug(
                "Error reading file metadata", file_path=str(file_path), error=str(e)
            )
            # Return minimal node if we can't stat the file
            return OnexTreeNode(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                type="file",
            )

    def _should_exclude(self, path: Path) -> bool:
        """
        Check if path matches any exclusion patterns.

        Args:
            path: Path to check

        Returns:
            True if path should be excluded
        """
        path_str = path.name

        # Check against each exclusion pattern
        return any(fnmatch(path_str, pattern) for pattern in self.exclude_patterns)

    def _infer_purpose(self, file_path: Path) -> Optional[str]:
        """
        Infer file purpose from path and name (future enhancement).

        Args:
            file_path: Path to analyze

        Returns:
            Inferred purpose string or None
        """
        # Basic inference rules (can be enhanced)
        path_str = str(file_path).lower()
        name = file_path.name.lower()

        if "test" in path_str or name.startswith("test_"):
            return "test"
        elif "service" in path_str:
            return "service"
        elif "model" in path_str:
            return "model"
        elif "controller" in path_str:
            return "controller"
        elif "repository" in path_str or "repo" in path_str:
            return "repository"
        elif name in ("__init__.py", "__main__.py"):
            return "module_init"
        elif name in ("readme.md", "readme.txt"):
            return "documentation"

        return None
