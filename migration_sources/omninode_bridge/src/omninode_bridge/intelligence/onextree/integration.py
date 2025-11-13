"""
Integration layer for OnexTree with metadata stamping service.

Provides adapter classes for integrating OnexTree with:
- Unified file processor (context enrichment)
- Batch processing system (file list provision)
- Event-driven architecture (Kafka event publishing)

This layer enables OnexTree to enhance metadata stamping operations without
requiring changes to the metadata stamping service itself.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import ModelOnextreeRoot
from .query_engine import OnexTreeQueryEngine


class OnexTreeUnifiedFileProcessorIntegration:
    """
    Wrapper integrating OnexTree with UnifiedFileProcessor.

    **Reference:** METADATA_STAMPING_SERVICE_IMPLEMENTATION_PLAN.md, Phase 3
    **Component:** src/omninode_bridge/services/metadata_stamping (stamping_engine.py)
    **Integration Point:** Hook into file processing pipeline for context enrichment
    **Timeline:** Ready for use when UnifiedFileProcessor is implemented

    Provides rich context to file processing operations including:
    - Project structure awareness
    - Related files and dependencies
    - Architectural pattern detection
    - Semantic purpose inference
    """

    def __init__(self, query_engine: OnexTreeQueryEngine):
        """
        Initialize integration with query engine.

        Args:
            query_engine: OnexTree query engine instance
        """
        self.query_engine = query_engine

    async def enrich_file_context(
        self, file_path: str, base_context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Provide OnexTree context to file processor.

        Called by file processor before stamping to add structural and
        architectural context that enhances metadata quality.

        Args:
            file_path: Relative path from project root
            base_context: Optional existing context to augment

        Returns:
            Dictionary with OnexTree context enrichment
        """
        context = base_context or {}

        # Lookup file in tree
        node = await self.query_engine.lookup_file(file_path)

        if not node:
            context["onextree_context"] = {
                "exists_in_tree": False,
                "status": "file_not_found",
                "recommendation": "verify_path_or_regenerate_tree",
            }
            return context

        # Get parent directory context
        parent_path = str(Path(file_path).parent)
        if parent_path == ".":
            parent_path = ""

        siblings = await self.query_engine.get_directory_children(parent_path)

        # Get statistics
        stats = await self.query_engine.get_statistics()

        # Build enriched context
        context["onextree_context"] = {
            "exists_in_tree": True,
            "file_metadata": {
                "path": node.path,
                "name": node.name,
                "type": node.type,
                "size": node.size,
                "extension": node.extension,
                "last_modified": (
                    node.last_modified.isoformat() if node.last_modified else None
                ),
            },
            "structural_context": {
                "parent_directory": parent_path or "(project root)",
                "sibling_count": len(siblings),
                "siblings": [
                    {"path": s.path, "name": s.name, "type": s.type}
                    for s in siblings[:10]  # Limit to 10 for performance
                ],
            },
            "semantic_context": {
                "architectural_pattern": node.architectural_pattern,
                "inferred_purpose": node.inferred_purpose,
                "related_files": node.related_files or [],
            },
            "project_context": {
                "total_files": stats.get("total_files") if stats else None,
                "total_directories": stats.get("total_directories") if stats else None,
            },
        }

        return context

    async def validate_file_path(self, file_path: str) -> dict[str, Any]:
        """
        Validate file path before processing.

        Checks for duplicates and similar files to prevent issues.

        Args:
            file_path: Proposed file path

        Returns:
            Validation result with recommendations
        """
        exists = await self.query_engine.check_file_exists(file_path)

        result = {
            "path": file_path,
            "exists": exists,
            "is_valid": True,
        }

        if exists:
            result["is_valid"] = False
            result["warning"] = "file_already_exists"
            result["recommendation"] = "use_existing_or_choose_different_path"
        else:
            # Check for similar files
            file_name = Path(file_path).name
            similar = await self.query_engine.find_by_name(file_name, limit=5)

            if similar:
                result["similar_files"] = [
                    {"path": s.path, "type": s.type, "extension": s.extension}
                    for s in similar
                ]
                result["info"] = "similar_files_found"

        return result


class OnexTreeBatchProcessingIntegration:
    """
    Integration with batch processing system.

    **Reference:** METADATA_STAMPING_SERVICE_IMPLEMENTATION_PLAN.md, Phase 2, Week 3
    **Component:** src/metadata_stamping (future batch_processor.py)
    **Integration Point:** Provide file lists for batch operations
    **Timeline:** Ready for use when batch processing is implemented

    Eliminates need for filesystem traversal during batch operations by
    providing pre-indexed, pre-filtered file lists from OnexTree.
    """

    def __init__(self, query_engine: OnexTreeQueryEngine):
        """
        Initialize integration with query engine.

        Args:
            query_engine: OnexTree query engine instance
        """
        self.query_engine = query_engine

    async def get_files_for_batch_processing(
        self,
        file_types: Optional[list[str]] = None,
        max_files: int = 1000,
        directory_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Provide file lists to batch processor.

        Benefits:
        - No filesystem traversal needed (instant)
        - Pre-filtered by file type
        - Can filter by directory
        - Returns file metadata for optimization

        Args:
            file_types: List of file extensions to include (e.g., ['py', 'js'])
            max_files: Maximum files to return
            directory_filter: Optional directory path prefix to filter by

        Returns:
            List of file info dictionaries
        """
        if not file_types:
            # Return all files
            all_paths = list(self.query_engine.exact_path_index.keys())[:max_files]
            nodes = [self.query_engine.exact_path_index[path] for path in all_paths]
        else:
            # Filter by file types
            nodes = []
            for ext in file_types:
                ext_nodes = await self.query_engine.find_by_extension(
                    ext, limit=max_files
                )
                nodes.extend(ext_nodes)

            # Trim to max_files
            nodes = nodes[:max_files]

        # Apply directory filter if specified
        if directory_filter:
            nodes = [node for node in nodes if node.path.startswith(directory_filter)]

        # Return enriched file info
        return [
            {
                "path": node.path,
                "name": node.name,
                "extension": node.extension,
                "size": node.size,
                "last_modified": (
                    node.last_modified.isoformat() if node.last_modified else None
                ),
            }
            for node in nodes
        ]

    async def get_batch_statistics(
        self, file_types: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Get statistics for batch processing planning.

        Helps batch processor estimate resources and time.

        Args:
            file_types: Optional file types to get stats for

        Returns:
            Statistics dictionary
        """
        stats = await self.query_engine.get_statistics()

        if not stats:
            return {"available": False}

        result = {
            "available": True,
            "total_files": stats.get("total_files", 0),
            "total_directories": stats.get("total_directories", 0),
            "file_type_distribution": stats.get("file_type_distribution", {}),
        }

        # Add filtered stats if file_types specified
        if file_types:
            filtered_count = sum(
                result["file_type_distribution"].get(ext, 0) for ext in file_types
            )
            result["filtered_count"] = filtered_count
            result["filter_applied"] = file_types

        return result


class OnexTreeEventPublisher:
    """
    Event publisher for OnexTree tree update events.

    **Reference:** METADATA_STAMPING_SERVICE_IMPLEMENTATION_PLAN.md, Phase 4, Week 6
    **Component:** Kafka event publishing (from PR #7)
    **Integration Point:** OnexTree publishes tree update events
    **Timeline:** Ready for use with existing EventPublisher

    Publishes events when tree is updated for:
    - Distributed cache invalidation
    - Triggering downstream processes
    - Audit logging and observability
    """

    def __init__(self, event_publisher=None):
        """
        Initialize event publisher.

        Args:
            event_publisher: Optional EventPublisher instance from metadata stamping
                           If None, events will be logged but not published
        """
        self.event_publisher = event_publisher
        self.enabled = event_publisher is not None

    async def publish_tree_generated_event(
        self, tree_root: ModelOnextreeRoot, generation_time_ms: float
    ) -> bool:
        """
        Publish event when tree is initially generated.

        Args:
            tree_root: Generated tree root
            generation_time_ms: Time taken to generate tree

        Returns:
            True if published successfully
        """
        if not self.enabled:
            if tree_root:
                print(
                    f"[OnexTree] Tree generated: {tree_root.statistics.total_files} files in {generation_time_ms:.2f}ms"
                )
            else:
                print(f"[OnexTree] Tree generated in {generation_time_ms:.2f}ms")
            return False

        event_data = {
            "event_type": "onextree.tree.generated",
            "timestamp": datetime.now().isoformat(),
            "project_root": tree_root.project_root,
            "statistics": {
                "total_files": tree_root.statistics.total_files,
                "total_directories": tree_root.statistics.total_directories,
                "file_type_distribution": tree_root.statistics.file_type_distribution,
                "total_size_bytes": tree_root.statistics.total_size_bytes,
            },
            "performance": {"generation_time_ms": generation_time_ms},
        }

        try:
            # Use event publisher's publish method
            await self.event_publisher.publish("onextree.tree.generated", event_data)
            return True
        except Exception as e:
            print(f"[OnexTree] Failed to publish tree generated event: {e}")
            return False

    async def publish_tree_updated_event(
        self,
        tree_root: ModelOnextreeRoot,
        trigger: str,
        files_changed: Optional[list[str]] = None,
    ) -> bool:
        """
        Publish event when tree is updated due to filesystem changes.

        Args:
            tree_root: Updated tree root
            trigger: What triggered the update (e.g., "filesystem_change")
            files_changed: Optional list of changed file paths

        Returns:
            True if published successfully
        """
        if not self.enabled:
            print(f"[OnexTree] Tree updated: trigger={trigger}")
            return False

        event_data = {
            "event_type": "onextree.tree.updated",
            "timestamp": datetime.now().isoformat(),
            "project_root": tree_root.project_root,
            "trigger": trigger,
            "files_changed": files_changed or [],
            "statistics": {
                "total_files": tree_root.statistics.total_files,
                "total_directories": tree_root.statistics.total_directories,
            },
        }

        try:
            await self.event_publisher.publish("onextree.tree.updated", event_data)
            return True
        except Exception as e:
            print(f"[OnexTree] Failed to publish tree updated event: {e}")
            return False

    async def publish_query_metrics_event(
        self, operation: str, execution_time_ms: float, result_count: int
    ) -> bool:
        """
        Publish event for query performance metrics.

        Args:
            operation: Query operation type
            execution_time_ms: Execution time in milliseconds
            result_count: Number of results returned

        Returns:
            True if published successfully
        """
        if not self.enabled:
            return False

        event_data = {
            "event_type": "onextree.query.metrics",
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "execution_time_ms": execution_time_ms,
            "result_count": result_count,
        }

        try:
            await self.event_publisher.publish("onextree.query.metrics", event_data)
            return True
        except Exception as e:
            print(f"[OnexTree] Failed to publish query metrics event: {e}")
            return False


class OnexTreeIntegrationManager:
    """
    Central manager for all OnexTree integrations.

    Simplifies setup and provides unified interface for all integration
    components with feature flag support.
    """

    def __init__(
        self,
        query_engine: OnexTreeQueryEngine,
        enable_file_processor: bool = False,
        enable_batch_processing: bool = False,
        enable_events: bool = False,
        event_publisher=None,
    ):
        """
        Initialize integration manager.

        Args:
            query_engine: OnexTree query engine instance
            enable_file_processor: Enable file processor integration
            enable_batch_processing: Enable batch processing integration
            enable_events: Enable event publishing
            event_publisher: Optional event publisher instance
        """
        self.query_engine = query_engine

        # Initialize integrations based on feature flags
        self.file_processor = (
            OnexTreeUnifiedFileProcessorIntegration(query_engine)
            if enable_file_processor
            else None
        )

        self.batch_processing = (
            OnexTreeBatchProcessingIntegration(query_engine)
            if enable_batch_processing
            else None
        )

        self.event_publisher = (
            OnexTreeEventPublisher(event_publisher) if enable_events else None
        )

    def is_file_processor_enabled(self) -> bool:
        """Check if file processor integration is enabled."""
        return self.file_processor is not None

    def is_batch_processing_enabled(self) -> bool:
        """Check if batch processing integration is enabled."""
        return self.batch_processing is not None

    def is_events_enabled(self) -> bool:
        """Check if event publishing is enabled."""
        return self.event_publisher is not None

    async def get_integration_status(self) -> dict[str, Any]:
        """
        Get status of all integrations.

        Returns:
            Status dictionary with integration availability
        """
        stats = await self.query_engine.get_statistics()

        return {
            "integrations": {
                "file_processor": {
                    "enabled": self.is_file_processor_enabled(),
                    "status": (
                        "ready" if self.is_file_processor_enabled() else "disabled"
                    ),
                },
                "batch_processing": {
                    "enabled": self.is_batch_processing_enabled(),
                    "status": (
                        "ready" if self.is_batch_processing_enabled() else "disabled"
                    ),
                },
                "events": {
                    "enabled": self.is_events_enabled(),
                    "status": "ready" if self.is_events_enabled() else "disabled",
                },
            },
            "query_engine": {
                "loaded": stats is not None,
                "total_files": stats.get("total_files") if stats else 0,
                "total_directories": stats.get("total_directories") if stats else 0,
            },
        }
