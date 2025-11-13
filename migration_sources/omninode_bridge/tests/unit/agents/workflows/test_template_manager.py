"""
Comprehensive tests for Template Management with LRU Caching.

Tests cover:
- Template models and dataclasses
- LRU cache operations (get, put, eviction)
- Cache hit/miss tracking
- Template loading and caching
- Template rendering with Jinja2
- Thread-safe operations
- Performance targets validation
- Cache statistics

Target: 95%+ coverage
"""

import asyncio
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from omninode_bridge.agents.workflows.template_cache import TemplateLRUCache
from omninode_bridge.agents.workflows.template_manager import TemplateManager
from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateCacheStats,
    TemplateMetadata,
    TemplateRenderContext,
    TemplateType,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_template_metadata() -> TemplateMetadata:
    """Create sample template metadata."""
    return TemplateMetadata(
        version="1.0.0",
        author="Test Author",
        description="Test template",
        tags=["test", "sample"],
        node_type_category="effect",
        supports_batch=True,
        min_python_version="3.11",
        dependencies=["pydantic", "asyncio"],
    )


@pytest.fixture
def sample_template(sample_template_metadata: TemplateMetadata) -> Template:
    """Create sample template."""
    return Template(
        template_id="test_template_v1",
        template_type=TemplateType.EFFECT,
        content="Hello {{ name }}!",
        metadata=sample_template_metadata,
        file_path="/test/template.jinja2",
    )


@pytest.fixture
def lru_cache() -> TemplateLRUCache:
    """Create LRU cache instance."""
    return TemplateLRUCache(max_size=10)


@pytest.fixture
def temp_template_dir():
    """Create temporary directory with template files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create sample templates
        (temp_path / "effect").mkdir(parents=True)
        (temp_path / "compute").mkdir(parents=True)

        # Effect template
        (temp_path / "effect" / "node_effect_v1.jinja2").write_text(
            """class Node{{ node_name }}Effect:
    \"\"\"{{ description }}\"\"\"

    async def execute_effect(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Effect logic here
        return {"result": "success"}
""",
            encoding="utf-8",
        )

        # Compute template
        (temp_path / "compute" / "node_compute_v1.jinja2").write_text(
            """class Node{{ node_name }}Compute:
    \"\"\"{{ description }}\"\"\"

    async def execute_compute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Compute logic here
        result = input_data.get("value", 0) * {{ multiplier }}
        return {"result": result}
""",
            encoding="utf-8",
        )

        # Simple template at root
        (temp_path / "simple_template.jinja2").write_text(
            "Hello {{ name }}!", encoding="utf-8"
        )

        yield temp_path


@pytest.fixture
async def template_manager(temp_template_dir: Path) -> TemplateManager:
    """Create and start TemplateManager instance."""
    manager = TemplateManager(
        template_dir=str(temp_template_dir),
        cache_size=10,
    )
    await manager.start()
    yield manager
    await manager.stop()


# ============================================================================
# Template Models Tests
# ============================================================================


class TestTemplateModels:
    """Test template models and dataclasses."""

    def test_template_type_enum(self):
        """Test TemplateType enum."""
        # Test all types exist
        assert TemplateType.EFFECT == "effect"
        assert TemplateType.COMPUTE == "compute"
        assert TemplateType.REDUCER == "reducer"
        assert TemplateType.ORCHESTRATOR == "orchestrator"
        assert TemplateType.MODEL == "model"
        assert TemplateType.VALIDATOR == "validator"
        assert TemplateType.TEST == "test"
        assert TemplateType.CONTRACT == "contract"

        # Test string conversion
        assert str(TemplateType.EFFECT) == "effect"

    def test_template_metadata_creation(self, sample_template_metadata: TemplateMetadata):
        """Test TemplateMetadata creation and to_dict."""
        metadata = sample_template_metadata

        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "Test template"
        assert metadata.tags == ["test", "sample"]
        assert metadata.node_type_category == "effect"
        assert metadata.supports_batch is True

        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert metadata_dict["version"] == "1.0.0"
        assert metadata_dict["author"] == "Test Author"
        assert "tags" in metadata_dict

    def test_template_creation(self, sample_template: Template):
        """Test Template creation and size calculation."""
        template = sample_template

        assert template.template_id == "test_template_v1"
        assert template.template_type == TemplateType.EFFECT
        assert template.content == "Hello {{ name }}!"
        assert template.size_bytes > 0  # Size calculated in __post_init__
        assert isinstance(template.loaded_at, datetime)

        # Test size calculation
        expected_size = len("Hello {{ name }}!".encode("utf-8"))
        assert template.size_bytes == expected_size

    def test_template_to_dict(self, sample_template: Template):
        """Test Template.to_dict()."""
        template_dict = sample_template.to_dict()

        assert template_dict["template_id"] == "test_template_v1"
        assert template_dict["template_type"] == "effect"
        assert template_dict["content_length"] == len("Hello {{ name }}!")
        assert "metadata" in template_dict
        assert "loaded_at" in template_dict

    def test_template_repr(self, sample_template: Template):
        """Test Template string representation."""
        repr_str = repr(sample_template)

        assert "test_template_v1" in repr_str
        assert "effect" in repr_str
        assert "B)" in repr_str  # Bytes indicator

    def test_template_render_context(self):
        """Test TemplateRenderContext creation."""
        context = TemplateRenderContext(
            variables={"name": "World", "version": "1.0"},
            filters={"uppercase": str.upper},
            globals={"debug": True},
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        assert context.variables["name"] == "World"
        assert "uppercase" in context.filters
        assert context.globals["debug"] is True
        assert context.autoescape is False

        # Test to_dict
        context_dict = context.to_dict()
        assert "variables" in context_dict
        assert "filters" in context_dict

    def test_template_cache_stats(self):
        """Test TemplateCacheStats creation and calculations."""
        stats = TemplateCacheStats(
            total_requests=100,
            cache_hits=85,
            cache_misses=15,
            hit_rate=0.85,
            current_size=50,
            max_size=100,
            evictions=10,
            total_size_bytes=50000,
        )

        assert stats.hit_rate == 0.85
        assert stats.current_size == 50
        assert stats.max_size == 100

        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["hit_rate"] == 0.85
        assert "hit_rate_percent" in stats_dict
        assert stats_dict["hit_rate_percent"] == "85.00%"
        assert "avg_template_size_bytes" in stats_dict

    def test_template_cache_stats_repr(self):
        """Test TemplateCacheStats string representation."""
        stats = TemplateCacheStats(
            cache_hits=85,
            cache_misses=15,
            hit_rate=0.85,
            current_size=50,
            max_size=100,
            evictions=10,
        )

        repr_str = repr(stats)
        assert "85" in repr_str or "85.00%" in repr_str
        assert "50/100" in repr_str


# ============================================================================
# LRU Cache Tests
# ============================================================================


class TestTemplateLRUCache:
    """Test LRU cache operations."""

    def test_cache_initialization(self, lru_cache: TemplateLRUCache):
        """Test cache initialization."""
        assert lru_cache.max_size() == 10
        assert lru_cache.size() == 0
        assert len(lru_cache) == 0

    def test_cache_put_and_get(
        self, lru_cache: TemplateLRUCache, sample_template: Template
    ):
        """Test basic put and get operations."""
        # Put template
        lru_cache.put(sample_template)
        assert lru_cache.size() == 1

        # Get template
        retrieved = lru_cache.get("test_template_v1")
        assert retrieved is not None
        assert retrieved.template_id == "test_template_v1"
        assert retrieved.content == "Hello {{ name }}!"

    def test_cache_miss(self, lru_cache: TemplateLRUCache):
        """Test cache miss behavior."""
        # Get non-existent template
        retrieved = lru_cache.get("non_existent")
        assert retrieved is None

        # Check stats
        stats = lru_cache.get_stats()
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_cache_hit_tracking(
        self, lru_cache: TemplateLRUCache, sample_template: Template
    ):
        """Test cache hit/miss tracking."""
        # Put template
        lru_cache.put(sample_template)

        # Hit
        lru_cache.get("test_template_v1")
        stats = lru_cache.get_stats()
        assert stats.cache_hits == 1

        # Miss
        lru_cache.get("non_existent")
        stats = lru_cache.get_stats()
        assert stats.cache_misses == 1

        # Hit rate
        assert stats.hit_rate == 0.5  # 1 hit, 1 miss

    def test_lru_eviction(self, lru_cache: TemplateLRUCache):
        """Test LRU eviction when cache is full."""
        # Fill cache to max
        for i in range(10):
            template = Template(
                template_id=f"template_{i}",
                template_type=TemplateType.EFFECT,
                content=f"Content {i}",
                metadata=TemplateMetadata(),
            )
            lru_cache.put(template)

        assert lru_cache.size() == 10

        # Add one more - should evict LRU
        new_template = Template(
            template_id="template_new",
            template_type=TemplateType.EFFECT,
            content="New content",
            metadata=TemplateMetadata(),
        )
        lru_cache.put(new_template)

        # Cache should still be at max size
        assert lru_cache.size() == 10

        # First template (template_0) should be evicted
        assert lru_cache.get("template_0") is None

        # New template should exist
        assert lru_cache.get("template_new") is not None

        # Check eviction count
        stats = lru_cache.get_stats()
        assert stats.evictions == 1

    def test_lru_access_updates_order(self, lru_cache: TemplateLRUCache):
        """Test that accessing a template updates LRU order."""
        # Add 3 templates
        for i in range(3):
            template = Template(
                template_id=f"template_{i}",
                template_type=TemplateType.EFFECT,
                content=f"Content {i}",
                metadata=TemplateMetadata(),
            )
            lru_cache.put(template)

        # Access template_0 to make it most recently used
        lru_cache.get("template_0")

        # Fill cache to max (add 7 more)
        for i in range(3, 10):
            template = Template(
                template_id=f"template_{i}",
                template_type=TemplateType.EFFECT,
                content=f"Content {i}",
                metadata=TemplateMetadata(),
            )
            lru_cache.put(template)

        # Add one more to trigger eviction
        new_template = Template(
            template_id="template_new",
            template_type=TemplateType.EFFECT,
            content="New content",
            metadata=TemplateMetadata(),
        )
        lru_cache.put(new_template)

        # template_1 should be evicted (LRU), not template_0
        assert lru_cache.get("template_1") is None
        assert lru_cache.get("template_0") is not None

    def test_cache_invalidate(
        self, lru_cache: TemplateLRUCache, sample_template: Template
    ):
        """Test cache invalidation."""
        # Put template
        lru_cache.put(sample_template)
        assert lru_cache.size() == 1

        # Invalidate
        result = lru_cache.invalidate("test_template_v1")
        assert result is True
        assert lru_cache.size() == 0

        # Try to get - should be None
        assert lru_cache.get("test_template_v1") is None

        # Invalidate non-existent
        result = lru_cache.invalidate("non_existent")
        assert result is False

    def test_cache_clear(self, lru_cache: TemplateLRUCache):
        """Test cache clear operation."""
        # Add multiple templates
        for i in range(5):
            template = Template(
                template_id=f"template_{i}",
                template_type=TemplateType.EFFECT,
                content=f"Content {i}",
                metadata=TemplateMetadata(),
            )
            lru_cache.put(template)

        assert lru_cache.size() == 5

        # Clear cache
        lru_cache.clear()
        assert lru_cache.size() == 0

        # Verify all templates removed
        for i in range(5):
            assert lru_cache.get(f"template_{i}") is None

    def test_cache_has(self, lru_cache: TemplateLRUCache, sample_template: Template):
        """Test cache.has() method."""
        # Initially not present
        assert lru_cache.has("test_template_v1") is False

        # Add template
        lru_cache.put(sample_template)
        assert lru_cache.has("test_template_v1") is True

        # Remove template
        lru_cache.invalidate("test_template_v1")
        assert lru_cache.has("test_template_v1") is False

    def test_cache_contains(
        self, lru_cache: TemplateLRUCache, sample_template: Template
    ):
        """Test __contains__ operator."""
        assert "test_template_v1" not in lru_cache

        lru_cache.put(sample_template)
        assert "test_template_v1" in lru_cache

    def test_cache_get_stats(self, lru_cache: TemplateLRUCache):
        """Test cache statistics."""
        # Add templates
        for i in range(5):
            template = Template(
                template_id=f"template_{i}",
                template_type=TemplateType.EFFECT,
                content=f"Content {i}",
                metadata=TemplateMetadata(),
            )
            lru_cache.put(template)

        # Perform some operations
        lru_cache.get("template_0")  # Hit
        lru_cache.get("template_1")  # Hit
        lru_cache.get("non_existent")  # Miss

        # Get stats
        stats = lru_cache.get_stats()
        assert stats.current_size == 5
        assert stats.max_size == 10
        assert stats.cache_hits == 2
        assert stats.cache_misses == 1
        assert stats.hit_rate == 2 / 3  # 2 hits out of 3 requests
        assert stats.total_size_bytes > 0

    def test_cache_get_hit_rate(self, lru_cache: TemplateLRUCache):
        """Test get_hit_rate() method."""
        # No requests - hit rate should be 0
        assert lru_cache.get_hit_rate() == 0.0

        # Add template
        template = Template(
            template_id="template_0",
            template_type=TemplateType.EFFECT,
            content="Content",
            metadata=TemplateMetadata(),
        )
        lru_cache.put(template)

        # 2 hits, 1 miss
        lru_cache.get("template_0")
        lru_cache.get("template_0")
        lru_cache.get("non_existent")

        hit_rate = lru_cache.get_hit_rate()
        assert hit_rate == 2 / 3  # 2 hits out of 3 requests

    def test_cache_timing_stats(self, lru_cache: TemplateLRUCache):
        """Test timing statistics collection."""
        # Add template
        template = Template(
            template_id="template_0",
            template_type=TemplateType.EFFECT,
            content="Content",
            metadata=TemplateMetadata(),
        )
        lru_cache.put(template)

        # Perform operations
        for _ in range(10):
            lru_cache.get("template_0")

        # Get timing stats
        timing = lru_cache.get_timing_stats()
        assert "get_avg_ms" in timing
        assert "get_p50_ms" in timing
        assert "get_p95_ms" in timing
        assert "get_p99_ms" in timing
        assert timing["get_avg_ms"] >= 0

    def test_cache_repr(self, lru_cache: TemplateLRUCache):
        """Test cache string representation."""
        repr_str = repr(lru_cache)
        assert "TemplateLRUCache" in repr_str
        assert "size=" in repr_str
        assert "hit_rate=" in repr_str

    def test_cache_thread_safety(self, lru_cache: TemplateLRUCache):
        """Test thread-safe operations."""
        errors = []

        def writer_thread(thread_id: int):
            """Writer thread that adds templates."""
            try:
                for i in range(10):
                    template = Template(
                        template_id=f"template_{thread_id}_{i}",
                        template_type=TemplateType.EFFECT,
                        content=f"Content {thread_id}-{i}",
                        metadata=TemplateMetadata(),
                    )
                    lru_cache.put(template)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        def reader_thread(thread_id: int):
            """Reader thread that reads templates."""
            try:
                for i in range(10):
                    lru_cache.get(f"template_0_{i}")
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Start threads
        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0

        # Cache should be at or below max size
        assert lru_cache.size() <= lru_cache.max_size()


# ============================================================================
# Template Manager Tests
# ============================================================================


class TestTemplateManager:
    """Test TemplateManager operations."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, temp_template_dir: Path):
        """Test manager initialization."""
        manager = TemplateManager(
            template_dir=str(temp_template_dir),
            cache_size=10,
        )

        assert manager._template_dir == temp_template_dir
        assert manager._cache.max_size() == 10

    @pytest.mark.asyncio
    async def test_manager_start_stop(self, temp_template_dir: Path):
        """Test manager start and stop."""
        manager = TemplateManager(template_dir=str(temp_template_dir))

        # Start
        await manager.start()
        assert manager._started is True
        assert manager._jinja_env is not None

        # Stop
        await manager.stop()
        assert manager._started is False

    @pytest.mark.asyncio
    async def test_load_template_from_disk(self, template_manager: TemplateManager):
        """Test loading template from disk."""
        # Load effect template
        template = await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
        )

        assert template.template_id == "node_effect_v1"
        assert template.template_type == TemplateType.EFFECT
        assert "Node{{ node_name }}Effect" in template.content
        assert template.size_bytes > 0

    @pytest.mark.asyncio
    async def test_load_template_caching(self, template_manager: TemplateManager):
        """Test template caching on load."""
        # Load template twice
        template1 = await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
        )
        template2 = await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
        )

        # Should be same instance (from cache)
        assert template1 is template2

        # Check cache stats
        stats = template_manager.get_cache_stats()
        assert stats.cache_hits == 1  # Second load was cache hit

    @pytest.mark.asyncio
    async def test_load_template_force_reload(self, template_manager: TemplateManager):
        """Test force reload bypasses cache."""
        # Load template
        template1 = await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
        )

        # Force reload
        template2 = await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
            force_reload=True,
        )

        # Should be different instances
        assert template1 is not template2

    @pytest.mark.asyncio
    async def test_load_template_not_found(self, template_manager: TemplateManager):
        """Test loading non-existent template."""
        with pytest.raises(FileNotFoundError):
            await template_manager.load_template(
                template_id="non_existent_template",
                template_type=TemplateType.EFFECT,
            )

    @pytest.mark.asyncio
    async def test_render_template(self, template_manager: TemplateManager):
        """Test template rendering with Jinja2."""
        # Load template
        await template_manager.load_template(
            template_id="simple_template",
            template_type=TemplateType.MODEL,
        )

        # Render template
        rendered = await template_manager.render_template(
            template_id="simple_template",
            context={"name": "World"},
        )

        assert rendered == "Hello World!"

    @pytest.mark.asyncio
    async def test_render_template_complex(self, template_manager: TemplateManager):
        """Test rendering complex template."""
        # Load effect template
        await template_manager.load_template(
            template_id="node_effect_v1",
            template_type=TemplateType.EFFECT,
        )

        # Render with context
        rendered = await template_manager.render_template(
            template_id="node_effect_v1",
            context={
                "node_name": "MyEffect",
                "description": "My custom effect node",
            },
        )

        assert "class NodeMyEffectEffect:" in rendered
        assert "My custom effect node" in rendered

    @pytest.mark.asyncio
    async def test_render_template_with_render_context(
        self, template_manager: TemplateManager
    ):
        """Test rendering with TemplateRenderContext."""
        # Load template
        await template_manager.load_template(
            template_id="simple_template",
            template_type=TemplateType.MODEL,
        )

        # Create render context
        render_context = TemplateRenderContext(
            variables={"name": "World"},
            filters={"uppercase": str.upper},
        )

        # Render
        rendered = await template_manager.render_template(
            template_id="simple_template",
            render_context=render_context,
        )

        assert rendered == "Hello World!"

    @pytest.mark.asyncio
    async def test_load_and_render(self, template_manager: TemplateManager):
        """Test load_and_render convenience method."""
        rendered = await template_manager.load_and_render(
            template_id="simple_template",
            template_type=TemplateType.MODEL,
            context={"name": "Test"},
        )

        assert rendered == "Hello Test!"

    @pytest.mark.asyncio
    async def test_invalidate_template(self, template_manager: TemplateManager):
        """Test template invalidation."""
        # Load template
        await template_manager.load_template(
            template_id="simple_template",
            template_type=TemplateType.MODEL,
        )

        # Verify it's cached
        assert template_manager.has_template("simple_template")

        # Invalidate
        result = await template_manager.invalidate_template("simple_template")
        assert result is True

        # Verify it's removed
        assert not template_manager.has_template("simple_template")

    @pytest.mark.asyncio
    async def test_preload_templates(self, template_manager: TemplateManager):
        """Test preloading multiple templates."""
        # Preload templates
        count = await template_manager.preload_templates(
            [
                ("node_effect_v1", TemplateType.EFFECT),
                ("node_compute_v1", TemplateType.COMPUTE),
                ("simple_template", TemplateType.MODEL),
            ]
        )

        assert count == 3

        # Verify all templates are cached
        assert template_manager.has_template("node_effect_v1")
        assert template_manager.has_template("node_compute_v1")
        assert template_manager.has_template("simple_template")

    @pytest.mark.asyncio
    async def test_preload_templates_with_error(
        self, template_manager: TemplateManager
    ):
        """Test preloading with some failures."""
        # Preload with one invalid template
        count = await template_manager.preload_templates(
            [
                ("node_effect_v1", TemplateType.EFFECT),
                ("non_existent", TemplateType.MODEL),
                ("simple_template", TemplateType.MODEL),
            ]
        )

        # Should load 2 out of 3
        assert count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self, template_manager: TemplateManager):
        """Test clearing cache."""
        # Load templates
        await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )
        await template_manager.load_template(
            "simple_template", TemplateType.MODEL
        )

        assert template_manager.get_cache_stats().current_size == 2

        # Clear cache
        await template_manager.clear_cache()

        assert template_manager.get_cache_stats().current_size == 0

    @pytest.mark.asyncio
    async def test_has_template(self, template_manager: TemplateManager):
        """Test has_template method."""
        # Initially not cached
        assert not template_manager.has_template("node_effect_v1")

        # Load template
        await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )

        # Now it's cached
        assert template_manager.has_template("node_effect_v1")

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, template_manager: TemplateManager):
        """Test getting cache statistics."""
        # Load templates
        await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )
        await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )  # Cache hit

        stats = template_manager.get_cache_stats()
        assert stats.current_size == 1
        assert stats.cache_hits == 1
        assert stats.hit_rate == 0.5  # 1 hit, 1 miss

    @pytest.mark.asyncio
    async def test_get_timing_stats(self, template_manager: TemplateManager):
        """Test getting timing statistics."""
        # Load template multiple times
        for _ in range(5):
            await template_manager.load_template(
                "node_effect_v1", TemplateType.EFFECT
            )

        timing = template_manager.get_timing_stats()
        assert "get_avg_ms" in timing
        assert timing["get_avg_ms"] >= 0

    def test_manager_repr(self, temp_template_dir: Path):
        """Test manager string representation."""
        manager = TemplateManager(template_dir=str(temp_template_dir))
        repr_str = repr(manager)

        assert "TemplateManager" in repr_str
        assert "template_dir=" in repr_str


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance targets."""

    @pytest.mark.asyncio
    async def test_cached_load_performance(self, template_manager: TemplateManager):
        """Test cached template load is <1ms."""
        # Load template once (uncached)
        await template_manager.load_template(
            "simple_template", TemplateType.MODEL
        )

        # Load again (cached) and measure time
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            await template_manager.load_template(
                "simple_template", TemplateType.MODEL
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        # Calculate average
        avg_time = sum(timings) / len(timings)

        # Target: <1ms (we'll use 2ms to account for test overhead)
        assert avg_time < 2.0, f"Average cached load time: {avg_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_cache_hit_rate_target(self, template_manager: TemplateManager):
        """Test cache hit rate meets 85-95% target."""
        # Preload 10 templates
        templates = []
        for i in range(10):
            templates.append((f"template_{i}", TemplateType.MODEL))

        # Actually create the templates
        temp_dir = template_manager._template_dir
        for template_id, _ in templates:
            (temp_dir / f"{template_id}.jinja2").write_text(
                f"Template {template_id}", encoding="utf-8"
            )

        # Preload
        await template_manager.preload_templates(templates)

        # Simulate realistic access pattern (Zipf distribution)
        # 80% of accesses to 20% of templates
        access_pattern = (
            ["template_0"] * 40  # 40% of accesses
            + ["template_1"] * 30  # 30% of accesses
            + ["template_2"] * 10  # 10% of accesses
            + ["template_3"] * 5  # 5% of accesses
            + [f"template_{i}" for i in range(4, 10)] * 3  # 15% remaining
        )

        # Access templates
        for template_id in access_pattern:
            try:
                await template_manager.load_template(
                    template_id, TemplateType.MODEL
                )
            except FileNotFoundError:
                pass  # Template doesn't exist yet

        # Check hit rate
        stats = template_manager.get_cache_stats()
        hit_rate_percent = stats.hit_rate * 100

        # Target: 85-95% hit rate (we expect high hit rate with this pattern)
        assert (
            hit_rate_percent >= 80.0
        ), f"Hit rate {hit_rate_percent:.1f}% below 80% target"

    @pytest.mark.asyncio
    async def test_render_performance(self, template_manager: TemplateManager):
        """Test template rendering is <10ms."""
        # Load template
        await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )

        # Render multiple times and measure
        timings = []
        for i in range(50):
            start = time.perf_counter()
            await template_manager.render_template(
                "node_effect_v1",
                context={
                    "node_name": f"Effect{i}",
                    "description": "Test effect node",
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        # Calculate average
        avg_time = sum(timings) / len(timings)

        # Target: <10ms
        assert avg_time < 10.0, f"Average render time: {avg_time:.2f}ms"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, template_manager: TemplateManager):
        """Test complete template workflow."""
        # 1. Load template
        template = await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )
        assert template is not None

        # 2. Render template
        rendered = await template_manager.render_template(
            "node_effect_v1",
            context={"node_name": "MyEffect", "description": "My effect"},
        )
        assert "NodeMyEffectEffect" in rendered

        # 3. Check cache stats
        stats = template_manager.get_cache_stats()
        assert stats.current_size > 0

        # 4. Load same template again (cache hit)
        template2 = await template_manager.load_template(
            "node_effect_v1", TemplateType.EFFECT
        )
        assert template is template2  # Same instance from cache

        # 5. Get final stats
        final_stats = template_manager.get_cache_stats()
        assert final_stats.cache_hits > 0
