"""
Template Manager for code generation workflows.

Provides high-level template management with:
- LRU-cached template loading
- Jinja2 template rendering
- Metrics integration
- Support for multiple node types

Performance Targets:
- Template lookup (cached): <1ms
- Template lookup (uncached): 10-50ms
- Template rendering: <10ms per template
- Cache hit rate: 85-95%
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    TemplateNotFound,
    select_autoescape,
)

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.template_cache import TemplateLRUCache
from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateCacheStats,
    TemplateMetadata,
    TemplateRenderContext,
    TemplateType,
)

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    High-level template management with LRU caching and Jinja2 rendering.

    Design:
    - LRU cache for template storage
    - Jinja2 environment for rendering
    - MetricsCollector integration for performance tracking
    - Support for file system and string-based templates

    Performance:
    - Cached template load: <1ms
    - Uncached template load: 10-50ms (depends on disk I/O)
    - Template rendering: <10ms per template
    - Cache hit rate target: 85-95%

    Example:
        ```python
        # Initialize
        manager = TemplateManager(
            template_dir="/path/to/templates",
            cache=TemplateLRUCache(max_size=100),
            metrics_collector=MetricsCollector()
        )
        await manager.start()

        # Load template (with caching)
        template = await manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT
        )

        # Render template
        rendered = await manager.render_template(
            template_id="node_effect_v1",
            context={"node_name": "MyEffect", "version": "1.0.0"}
        )

        # Get cache statistics
        stats = manager.get_cache_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        ```
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        cache: Optional[TemplateLRUCache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        cache_size: int = 100,
        cache_memory_mb: float = 100.0,
        enable_autoescape: bool = False,
    ) -> None:
        """
        Initialize TemplateManager.

        Args:
            template_dir: Directory containing template files (optional)
            cache: LRU cache instance (creates default if None)
            metrics_collector: MetricsCollector instance (optional)
            cache_size: Cache size if creating default cache (default: 100)
            cache_memory_mb: Max cache memory in MB if creating default cache (default: 100.0)
            enable_autoescape: Enable Jinja2 autoescaping (default: False)
        """
        self._template_dir = Path(template_dir) if template_dir else None
        self._cache = cache or TemplateLRUCache(
            max_size=cache_size, max_memory_mb=cache_memory_mb
        )
        self._metrics = metrics_collector
        self._enable_autoescape = enable_autoescape

        # Jinja2 environment
        self._jinja_env: Optional[Environment] = None

        # Runtime state
        self._started = False

        logger.info(
            f"TemplateManager initialized: "
            f"template_dir={self._template_dir}, "
            f"cache_size={self._cache.max_size()}, "
            f"cache_memory={cache_memory_mb:.1f}MB"
        )

    async def start(self) -> None:
        """
        Start template manager and initialize Jinja2 environment.

        Example:
            ```python
            await manager.start()
            ```
        """
        # Initialize Jinja2 environment
        if self._template_dir and self._template_dir.exists():
            loader: BaseLoader = FileSystemLoader(str(self._template_dir))
            logger.info(f"Using FileSystemLoader for directory: {self._template_dir}")
        else:
            # Use BaseLoader for string-based templates
            loader = BaseLoader()
            logger.info("Using BaseLoader for string-based templates")

        self._jinja_env = Environment(
            loader=loader,
            autoescape=select_autoescape() if self._enable_autoescape else False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._started = True
        logger.info("TemplateManager started")

    async def stop(self) -> None:
        """
        Stop template manager and clear cache.

        Example:
            ```python
            await manager.stop()
            ```
        """
        self._cache.clear()
        self._started = False
        logger.info("TemplateManager stopped")

    async def load_template(
        self,
        template_id: str,
        template_type: TemplateType,
        force_reload: bool = False,
    ) -> Template:
        """
        Load template with caching.

        Performance:
        - Cached: <1ms
        - Uncached: 10-50ms (depends on disk I/O)

        Args:
            template_id: Template identifier (e.g., "node_effect_v1")
            template_type: Template type (effect, compute, reducer, etc.)
            force_reload: Force reload from disk (bypass cache)

        Returns:
            Template instance

        Raises:
            FileNotFoundError: If template file not found
            ValueError: If template manager not started

        Example:
            ```python
            template = await manager.load_template(
                template_id="node_effect_v1",
                template_type=TemplateType.EFFECT
            )
            ```
        """
        if not self._started:
            raise ValueError("TemplateManager not started. Call start() first.")

        start_time = time.perf_counter()

        # Check cache first (unless force_reload)
        if not force_reload:
            cached_template = self._cache.get(template_id)
            if cached_template:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Record metrics
                if self._metrics:
                    await self._metrics.record_timing(
                        "template_load_cached_ms",
                        elapsed_ms,
                        tags={
                            "template_id": template_id,
                            "template_type": str(template_type),
                        },
                    )

                logger.debug(f"Cache hit: {template_id} ({elapsed_ms:.2f}ms)")
                return cached_template

        # Cache miss - load from disk
        template = await self._load_from_disk(template_id, template_type)

        # Cache the loaded template
        self._cache.put(template)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        if self._metrics:
            await self._metrics.record_timing(
                "template_load_uncached_ms",
                elapsed_ms,
                tags={"template_id": template_id, "template_type": str(template_type)},
            )

        logger.debug(f"Cache miss: {template_id} ({elapsed_ms:.2f}ms)")
        return template

    async def _load_from_disk(
        self, template_id: str, template_type: TemplateType
    ) -> Template:
        """
        Load template from disk (internal).

        Args:
            template_id: Template identifier
            template_type: Template type

        Returns:
            Template instance

        Raises:
            FileNotFoundError: If template file not found
        """
        # Build template file path
        if self._template_dir:
            # Try multiple possible filenames
            possible_paths = [
                self._template_dir / f"{template_id}.jinja2",
                self._template_dir / f"{template_id}.j2",
                self._template_dir / f"{template_id}.txt",
                self._template_dir / template_type.value / f"{template_id}.jinja2",
                self._template_dir / template_type.value / f"{template_id}.j2",
            ]

            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break

            if not file_path:
                raise FileNotFoundError(
                    f"Template not found: {template_id} "
                    f"(searched: {[str(p) for p in possible_paths]})"
                )

            # Load template content
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")

            # Create template metadata
            metadata = TemplateMetadata(
                description=f"Template {template_id}",
                node_type_category=template_type.value,
            )

            # Create template instance
            template = Template(
                template_id=template_id,
                template_type=template_type,
                content=content,
                metadata=metadata,
                file_path=str(file_path),
            )

            logger.debug(
                f"Loaded template from disk: {template_id} "
                f"(size: {template.size_bytes}B, path: {file_path})"
            )

            return template
        else:
            raise ValueError(
                "No template directory configured. " "Cannot load templates from disk."
            )

    async def render_template(
        self,
        template_id: str,
        context: Optional[dict[str, Any]] = None,
        render_context: Optional[TemplateRenderContext] = None,
    ) -> str:
        """
        Render template with Jinja2.

        Performance Target: <10ms per template

        Args:
            template_id: Template identifier
            context: Variables to inject into template (simple dict)
            render_context: Full render context with filters, globals, etc.

        Returns:
            Rendered template as string

        Raises:
            ValueError: If template manager not started
            TemplateNotFound: If template not found in cache or disk

        Example:
            ```python
            rendered = await manager.render_template(
                template_id="node_effect_v1",
                context={
                    "node_name": "MyEffect",
                    "version": "1.0.0",
                    "description": "My effect node"
                }
            )
            ```
        """
        if not self._started or not self._jinja_env:
            raise ValueError("TemplateManager not started. Call start() first.")

        start_time = time.perf_counter()

        # Get template from cache (or load it)
        template = self._cache.get(template_id)
        if not template:
            raise TemplateNotFound(
                f"Template not found in cache: {template_id}. "
                f"Call load_template() first."
            )

        # Prepare render context
        variables = context or {}
        if render_context:
            variables.update(render_context.variables)

            # Add custom filters and globals if provided
            if render_context.filters:
                self._jinja_env.filters.update(render_context.filters)
            if render_context.globals:
                self._jinja_env.globals.update(render_context.globals)

        # Render template
        jinja_template = self._jinja_env.from_string(template.content)
        rendered = await asyncio.to_thread(jinja_template.render, **variables)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        if self._metrics:
            await self._metrics.record_timing(
                "template_render_ms",
                elapsed_ms,
                tags={
                    "template_id": template_id,
                    "template_type": str(template.template_type),
                },
            )

        logger.debug(
            f"Rendered template: {template_id} "
            f"(time: {elapsed_ms:.2f}ms, output: {len(rendered)} chars)"
        )

        return rendered

    async def load_and_render(
        self,
        template_id: str,
        template_type: TemplateType,
        context: dict[str, Any],
        render_context: Optional[TemplateRenderContext] = None,
    ) -> str:
        """
        Load and render template in one call (convenience method).

        Args:
            template_id: Template identifier
            template_type: Template type
            context: Variables to inject into template
            render_context: Optional full render context

        Returns:
            Rendered template as string

        Example:
            ```python
            rendered = await manager.load_and_render(
                template_id="node_effect_v1",
                template_type=TemplateType.EFFECT,
                context={"node_name": "MyEffect"}
            )
            ```
        """
        # Load template (with caching)
        await self.load_template(template_id, template_type)

        # Render template
        return await self.render_template(
            template_id, context=context, render_context=render_context
        )

    def get_cache_stats(self) -> TemplateCacheStats:
        """
        Get cache statistics.

        Returns:
            TemplateCacheStats with hit rate and metrics

        Example:
            ```python
            stats = manager.get_cache_stats()
            print(f"Hit rate: {stats.hit_rate:.2%}")
            print(f"Cache size: {stats.current_size}/{stats.max_size}")
            ```
        """
        return self._cache.get_stats()

    def get_timing_stats(self) -> dict[str, float]:
        """
        Get timing statistics for cache operations.

        Returns:
            Dictionary with timing metrics in milliseconds

        Example:
            ```python
            timing = manager.get_timing_stats()
            print(f"Avg cached load: {timing['get_avg_ms']:.2f}ms")
            print(f"P99 cached load: {timing['get_p99_ms']:.2f}ms")
            ```
        """
        return self._cache.get_timing_stats()

    async def invalidate_template(self, template_id: str) -> bool:
        """
        Invalidate (remove) template from cache.

        Args:
            template_id: Template identifier

        Returns:
            True if template was found and removed, False otherwise

        Example:
            ```python
            if await manager.invalidate_template("node_effect_v1"):
                print("Template invalidated")
            ```
        """
        invalidated = self._cache.invalidate(template_id)

        if invalidated and self._metrics:
            await self._metrics.record_counter(
                "template_invalidations",
                tags={"template_id": template_id},
            )

        return invalidated

    async def preload_templates(
        self, template_specs: list[tuple[str, TemplateType]]
    ) -> int:
        """
        Preload multiple templates into cache.

        Useful for warming up the cache at startup.

        Args:
            template_specs: List of (template_id, template_type) tuples

        Returns:
            Number of templates successfully loaded

        Example:
            ```python
            preloaded = await manager.preload_templates([
                ("node_effect_v1", TemplateType.EFFECT),
                ("node_compute_v1", TemplateType.COMPUTE),
                ("node_reducer_v1", TemplateType.REDUCER),
            ])
            print(f"Preloaded {preloaded} templates")
            ```
        """
        loaded_count = 0

        for template_id, template_type in template_specs:
            try:
                await self.load_template(template_id, template_type)
                loaded_count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to preload template {template_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Preloaded {loaded_count}/{len(template_specs)} templates into cache"
        )
        return loaded_count

    def has_template(self, template_id: str) -> bool:
        """
        Check if template exists in cache.

        Args:
            template_id: Template identifier

        Returns:
            True if template is cached, False otherwise

        Example:
            ```python
            if manager.has_template("node_effect_v1"):
                print("Template is cached")
            ```
        """
        return self._cache.has(template_id)

    async def clear_cache(self) -> None:
        """
        Clear all cached templates.

        Example:
            ```python
            await manager.clear_cache()
            print("Cache cleared")
            ```
        """
        self._cache.clear()

        if self._metrics:
            await self._metrics.record_counter("template_cache_clears")

        logger.info("Template cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_cache_stats()
        return (
            f"TemplateManager(template_dir={self._template_dir}, "
            f"cache_size={stats.current_size}/{stats.max_size}, "
            f"hit_rate={stats.hit_rate:.2%})"
        )
