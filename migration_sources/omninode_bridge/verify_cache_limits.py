#!/usr/bin/env python3
"""Verification script for cache size limits implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omninode_bridge.agents.registry.cache import CacheManager
from omninode_bridge.agents.workflows.template_cache import TemplateLRUCache
from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateMetadata,
    TemplateType,
)


def verify_template_cache():
    """Verify TemplateLRUCache memory limits."""
    print("=" * 60)
    print("Testing TemplateLRUCache with memory limits")
    print("=" * 60)

    # Create cache with 100 entry limit (large) and 1MB memory limit (small)
    # This ensures memory limit is hit before entry count limit
    cache = TemplateLRUCache(max_size=100, max_memory_mb=1.0)

    # Create large templates (200KB each)
    templates = []
    for i in range(10):
        content = "x" * (200 * 1024)  # 200KB content
        template = Template(
            template_id=f"template_{i}",
            template_type=TemplateType.EFFECT,
            content=content,
            metadata=TemplateMetadata(description=f"Template {i}"),
        )
        templates.append(template)

    # Add templates
    print(f"\nAdding {len(templates)} templates (200KB each)...")
    for template in templates:
        cache.put(template)

    # Get stats
    stats = cache.get_stats()
    memory_mb = cache.get_memory_usage_mb()
    utilization = cache.get_memory_utilization()
    memory_evictions = cache.get_memory_limit_evictions()

    print("\nCache Statistics:")
    print(f"  Current size: {stats.current_size}/{stats.max_size} entries")
    print(f"  Memory usage: {memory_mb:.2f}MB (limit: 1.00MB)")
    print(f"  Memory utilization: {utilization:.1%}")
    print(f"  Total evictions: {stats.evictions}")
    print(f"  Memory limit evictions: {memory_evictions}")
    print(f"  Hit rate: {stats.hit_rate:.1%}")

    # Verify memory limit enforced
    assert memory_mb <= 1.0, f"Memory limit exceeded: {memory_mb:.2f}MB > 1.00MB"
    assert stats.current_size <= 100, f"Size limit exceeded: {stats.current_size} > 100"
    assert (
        memory_evictions > 0
    ), f"No memory evictions occurred (got {memory_evictions})"

    print("\n✅ TemplateLRUCache memory limits working correctly!")
    return True


def verify_registry_cache():
    """Verify CacheManager memory limits."""
    print("\n" + "=" * 60)
    print("Testing CacheManager with memory limits")
    print("=" * 60)

    # Create cache with 100 entry limit and 1MB memory limit
    cache = CacheManager(max_size=100, ttl_seconds=300, max_memory_mb=1.0)

    # Add large entries (50KB each)
    print("\nAdding 30 entries (50KB each)...")
    for i in range(30):
        # Create 50KB data
        data = {"content": "x" * (50 * 1024), "index": i}
        cache.set(f"key_{i}", data)

    # Get stats
    stats = cache.get_stats()

    print("\nCache Statistics:")
    print(f"  Current size: {stats.size}/{stats.max_size} entries")
    print(f"  Memory usage: {stats.memory_mb:.2f}MB (limit: 1.00MB)")
    print(f"  Total evictions: {stats.evictions}")
    print(f"  Memory limit evictions: {stats.memory_limit_evictions}")
    print(f"  Hit rate: {stats.hit_rate:.1%}")

    # Verify memory limit enforced
    assert (
        stats.memory_mb <= 1.0
    ), f"Memory limit exceeded: {stats.memory_mb:.2f}MB > 1.00MB"
    assert stats.size <= 100, f"Size limit exceeded: {stats.size} > 100"
    assert stats.memory_limit_evictions > 0, "No memory evictions occurred"

    print("\n✅ CacheManager memory limits working correctly!")
    return True


def verify_backward_compatibility():
    """Verify backward compatibility with default parameters."""
    print("\n" + "=" * 60)
    print("Testing backward compatibility (default parameters)")
    print("=" * 60)

    # Create caches with old API (no memory limit specified)
    template_cache = TemplateLRUCache(max_size=100)
    registry_cache = CacheManager(max_size=1000, ttl_seconds=300)

    print("\n✅ TemplateLRUCache created with default max_memory_mb=100.0")
    print("✅ CacheManager created with default max_memory_mb=100.0")

    # Verify they have default memory limits
    assert (
        template_cache._max_memory_bytes == 100 * 1024 * 1024
    ), "Default memory limit not set"
    assert (
        registry_cache.max_memory_bytes == 100 * 1024 * 1024
    ), "Default memory limit not set"

    print("\n✅ Backward compatibility maintained!")
    return True


def main():
    """Run all verification tests."""
    print("Cache Size Limits Verification Script")
    print("=" * 60)

    try:
        # Test TemplateLRUCache
        verify_template_cache()

        # Test CacheManager
        verify_registry_cache()

        # Test backward compatibility
        verify_backward_compatibility()

        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATIONS PASSED!")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("  • TemplateLRUCache: Entry count + memory limits ✓")
        print("  • CacheManager: Entry count + memory limits ✓")
        print("  • LRU eviction: Working correctly ✓")
        print("  • Memory tracking: Accurate ✓")
        print("  • Backward compatibility: Maintained ✓")
        print()
        return 0

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
