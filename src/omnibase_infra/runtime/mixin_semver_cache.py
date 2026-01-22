# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semver Cache Mixin.

This module provides semantic version parsing and caching functionality,
extracted from RegistryPolicy to reduce class method count.

The mixin provides:
- LRU-cached semver parsing
- Configurable cache size
- Thread-safe cache initialization
- Cache statistics for testing

This mixin is designed to be used with RegistryPolicy and follows the
ONEX naming convention: mixin_<name>.py -> Mixin<Name>.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.runtime.util_version import normalize_version


class MixinSemverCache:
    """Mixin providing semantic version caching functionality.

    This mixin extracts semver cache methods from RegistryPolicy to reduce
    the class method count and provide better separation of concerns.

    The cache is implemented using functools.lru_cache with configurable
    size. Version strings are normalized before caching to ensure equivalent
    versions share the same cache entry.

    Class Attributes:
        SEMVER_CACHE_SIZE: Default cache size (128 entries)
        _semver_cache: Cached parser function (wrapper)
        _semver_cache_inner: Inner LRU-cached function
        _semver_cache_lock: Thread lock for initialization

    Methods:
        configure_semver_cache: Configure cache size before first use
        _reset_semver_cache: Reset cache for testing
        _get_semver_parser: Get or create cached parser
        _parse_semver: Parse version string to ModelSemVer
        _get_semver_cache_info: Get cache statistics
    """

    # Semver cache size - can be overridden via class attribute before first parse
    # or via configure_semver_cache() method
    SEMVER_CACHE_SIZE: int = 128

    # Cached semver parser function (lazily initialized)
    # This is the outer wrapper function that normalizes before calling the cached function
    _semver_cache: Callable[[str], ModelSemVer] | None = None

    # Inner LRU-cached function (stores reference for cache_clear() access)
    # This is the actual @lru_cache decorated function
    _semver_cache_inner: Callable[[str], ModelSemVer] | None = None

    # Lock for thread-safe cache initialization
    _semver_cache_lock: threading.Lock = threading.Lock()

    @classmethod
    def configure_semver_cache(cls, maxsize: int) -> None:
        """Configure semver cache size. Must be called before first parse.

        This method allows configuring the LRU cache size for semver parsing
        in large deployments with many policy versions. For most deployments,
        the default of 128 entries is sufficient.

        When to Increase Cache Size:
            - Very large deployments with > 100 unique policy versions
            - High-frequency lookups across many version combinations
            - Observed cache eviction causing performance regression

        Args:
            maxsize: Maximum cache entries (default: 128).
                     Recommended range: 64-512 for most deployments.
                     Each entry uses ~100 bytes.

        Raises:
            ProtocolConfigurationError: If cache already initialized (first parse already occurred)

        Example:
            >>> # Configure before any registry operations
            >>> MixinSemverCache.configure_semver_cache(maxsize=256)
            >>> registry = RegistryPolicy()

        Note:
            For testing purposes, use _reset_semver_cache() to clear the cache
            and allow reconfiguration.
        """
        with cls._semver_cache_lock:
            if cls._semver_cache is not None:
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="configure_semver_cache",
                )
                raise ProtocolConfigurationError(
                    "Cannot reconfigure semver cache after first use. "
                    "Set SEMVER_CACHE_SIZE before creating any "
                    "registry instances, or use _reset_semver_cache() for testing.",
                    context=context,
                )
            cls.SEMVER_CACHE_SIZE = maxsize

    @classmethod
    def _reset_semver_cache(cls) -> None:
        """Reset semver cache. For testing only.

        Clears the cached semver parser, allowing reconfiguration of cache size.
        This should only be used in test fixtures to ensure test isolation.

        Thread Safety:
            This method is thread-safe and uses the class-level lock. The reset
            operation is atomic - either the cache is fully reset or not at all.

            In-flight Operations:
                If other threads have already obtained a reference to the cache
                via _get_semver_parser(), they will continue using the old cache
                until they complete. This is safe because the old cache remains
                a valid callable until garbage collected. New operations after
                reset will get the new cache instance when created.

            Memory Reclamation:
                The old cache's internal LRU entries are explicitly cleared via
                cache_clear() before the reference is released. This ensures
                prompt memory reclamation rather than waiting for garbage
                collection.

            Concurrent Reset:
                Multiple concurrent reset calls are safe. Each reset will clear
                the current cache (if any) and set the reference to None. The
                lock ensures only one reset executes at a time.

        Example:
            >>> # In test fixture
            >>> MixinSemverCache._reset_semver_cache()
            >>> MixinSemverCache.SEMVER_CACHE_SIZE = 64
            >>> # Now cache will be initialized with size 64 on next use
        """
        with cls._semver_cache_lock:
            # Clear the inner LRU-cached function (has the actual cache)
            inner_cache = cls._semver_cache_inner
            if inner_cache is not None:
                # Clear internal LRU cache entries before releasing reference.
                # This ensures prompt memory reclamation rather than waiting
                # for garbage collection of the orphaned function object.
                # NOTE: cache_clear() is added by @lru_cache decorator but not
                # reflected in Callable type annotation. This is a known mypy
                # limitation with lru_cache wrappers.
                inner_cache.cache_clear()  # type: ignore[attr-defined]  # NOTE: lru_cache dynamic method
            cls._semver_cache = None
            cls._semver_cache_inner = None

    @classmethod
    def _get_semver_parser(cls) -> Callable[[str], ModelSemVer]:
        """Get or create the semver parser with configured cache size.

        This method implements lazy initialization of the LRU-cached semver parser.
        The cache size is determined by SEMVER_CACHE_SIZE at initialization time.

        Thread Safety:
            Uses double-checked locking pattern for thread-safe lazy initialization.
            The fast path stores the cache reference in a local variable to prevent
            TOCTOU (time-of-check-time-of-use) race conditions where another thread
            could call _reset_semver_cache() between the None check and the return.

        Cache Key Normalization:
            Version strings are normalized BEFORE being used as cache keys to ensure
            that equivalent versions (e.g., "1.0" and "1.0.0") share the same cache
            entry. This prevents cache fragmentation and improves hit rates.

        Returns:
            Cached semver parsing function that returns ModelSemVer instances.

        Performance:
            - First call: Creates LRU-cached function (one-time cost)
            - Subsequent calls: Returns cached function reference (O(1))
            - Cache hit rate improved by normalizing keys before lookup
        """
        # Fast path: cache already initialized
        # CRITICAL: Store in local variable to prevent TOCTOU race condition.
        # Without this, another thread could call _reset_semver_cache() between
        # the None check and the return, causing this method to return None.
        cache = cls._semver_cache
        if cache is not None:
            return cache

        # Slow path: initialize with lock
        with cls._semver_cache_lock:
            # Double-check after acquiring lock
            if cls._semver_cache is not None:
                return cls._semver_cache

            # Create LRU-cached parser with configured size
            # The cache key is the NORMALIZED version string to prevent
            # fragmentation (e.g., "1.0" and "1.0.0" share the same entry)
            @functools.lru_cache(maxsize=cls.SEMVER_CACHE_SIZE)
            def _parse_semver_cached(normalized_version: str) -> ModelSemVer:
                """Parse normalized semantic version string into ModelSemVer.

                This function receives ALREADY NORMALIZED version strings.
                The normalization is done by the wrapper function before
                caching to ensure equivalent versions share cache entries.

                Args:
                    normalized_version: Pre-normalized version in "x.y.z" or
                        "x.y.z-prerelease" format

                Returns:
                    ModelSemVer instance for comparison

                Raises:
                    ProtocolConfigurationError: If version format is invalid
                """
                # ModelOnexError is imported at module level
                try:
                    return ModelSemVer.parse(normalized_version)
                except ModelOnexError as e:
                    context = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="parse_semver",
                    )
                    raise ProtocolConfigurationError(
                        str(e),
                        version=normalized_version,
                        context=context,
                    ) from e
                except ValueError as e:
                    context = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="parse_semver",
                    )
                    raise ProtocolConfigurationError(
                        str(e),
                        version=normalized_version,
                        context=context,
                    ) from e

            def _parse_semver_impl(version: str) -> ModelSemVer:
                """Parse semantic version string into ModelSemVer.

                Implementation moved here to support configurable cache size.
                See _parse_semver docstring for full documentation.

                IMPORTANT: This wrapper normalizes version strings BEFORE
                passing to the LRU-cached parsing function. This ensures that
                equivalent versions (e.g., "1.0" and "1.0.0", "v1.0.0" and "1.0.0")
                share the same cache entry, improving cache hit rates.

                All validation (empty strings, prerelease suffix, format) is
                delegated to normalize_version to eliminate code duplication.
                """
                # Delegate all validation to normalize_version (single source of truth)
                # This eliminates duplicated validation logic (empty check, prerelease suffix)
                try:
                    normalized = normalize_version(version)
                except ValueError as e:
                    context = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="normalize_version",
                    )
                    raise ProtocolConfigurationError(
                        str(e),
                        version=version,
                        context=context,
                    ) from e

                # Now call the cached function with the NORMALIZED version
                # This ensures "1.0", "1.0.0", "v1.0.0" all use the same cache entry
                return _parse_semver_cached(normalized)

            # Store both the outer wrapper and inner cached function
            # The wrapper is what callers use (_semver_cache)
            # The inner function is needed for cache_clear() access (_semver_cache_inner)
            cls._semver_cache = _parse_semver_impl
            cls._semver_cache_inner = _parse_semver_cached
            return cls._semver_cache

    @classmethod
    def _parse_semver(cls, version: str) -> ModelSemVer:
        """Parse semantic version string into ModelSemVer for comparison.

        This method implements SEMANTIC VERSION SORTING, not lexicographic sorting.
        This is critical for correct "latest version" selection.

        Why This Matters (PR #36 feedback):
            Lexicographic sorting (string comparison):
                "1.10.0" < "1.9.0" WRONG (because '1' < '9' in strings)
                "10.0.0" < "2.0.0" WRONG (because '1' < '2' in strings)

            Semantic version sorting (integer comparison):
                1.10.0 > 1.9.0 CORRECT (because 10 > 9 as integers)
                10.0.0 > 2.0.0 CORRECT (because 10 > 2 as integers)

        Implementation:
            - Returns ModelSemVer instance with integer major, minor, patch
            - ModelSemVer implements comparison operators for correct ordering
            - Prerelease is parsed but NOT used in comparisons (major.minor.patch only)
            - "1.0.0-alpha" and "1.0.0" compare as EQUAL (same major.minor.patch)

        Supported Formats:
            - Full: "1.2.3", "1.2.3-beta"
            - Partial: "1" -> (1, 0, 0), "1.2" -> (1, 2, 0)
            - Prerelease: "1.0.0-alpha", "2.1.0-rc.1"

        Validation:
            - Rejects empty strings
            - Rejects non-numeric components
            - Rejects negative numbers
            - Rejects >3 version parts (e.g., "1.2.3.4")

        Performance:
            This method uses an LRU cache with configurable size (default: 128)
            to avoid re-parsing the same version strings repeatedly, improving
            performance for lookups that compare multiple versions.

            Cache Size Configuration:
                For large deployments, configure before first use:
                    MixinSemverCache.configure_semver_cache(maxsize=256)

            Cache Size Rationale (default 128):
                - Typical registry: 10-50 unique policy versions
                - Peak scenarios: 50-100 versions across multiple policy types
                - Each cache entry: ~200 bytes (string key + ModelSemVer instance)
                - Total memory: ~25.6KB worst case (negligible overhead)
                - Hit rate: >95% for repeated get() calls with version comparisons
                - Eviction: Rare in practice, LRU ensures least-used versions purged

        Args:
            version: Semantic version string (e.g., "1.2.3" or "1.0.0-beta")

        Returns:
            ModelSemVer instance for comparison.
            Components are INTEGERS (not strings) for correct semantic sorting.
            Prerelease is parsed and stored but ignored in version comparisons.

        Raises:
            ProtocolConfigurationError: If version format is invalid

        Examples:
            >>> MixinSemverCache._parse_semver("1.9.0")
            ModelSemVer(major=1, minor=9, patch=0, prerelease='')
            >>> MixinSemverCache._parse_semver("1.10.0")
            ModelSemVer(major=1, minor=10, patch=0, prerelease='')
            >>> MixinSemverCache._parse_semver("1.10.0") > MixinSemverCache._parse_semver("1.9.0")
            True
            >>> MixinSemverCache._parse_semver("10.0.0") > MixinSemverCache._parse_semver("2.0.0")
            True
            >>> MixinSemverCache._parse_semver("1.0.0-alpha")
            ModelSemVer(major=1, minor=0, patch=0, prerelease='alpha')
            >>> # Prerelease is parsed but NOT used in comparisons:
            >>> MixinSemverCache._parse_semver("1.0.0-alpha") == MixinSemverCache._parse_semver("1.0.0")
            True  # Same major.minor.patch, prerelease ignored
        """
        parser = cls._get_semver_parser()
        return parser(version)

    @classmethod
    def _get_semver_cache_info(cls) -> functools._CacheInfo | None:
        """Get cache statistics for the semver parser. For testing only.

        Returns the cache_info() from the inner LRU-cached function.
        This allows tests to verify cache behavior without accessing
        internal implementation details.

        Returns:
            functools._CacheInfo with hits, misses, maxsize, currsize,
            or None if cache not yet initialized.

        Example:
            >>> MixinSemverCache._reset_semver_cache()
            >>> MixinSemverCache._parse_semver("1.0.0")
            >>> info = MixinSemverCache._get_semver_cache_info()
            >>> info.misses  # First call is a miss
            1
            >>> MixinSemverCache._parse_semver("1.0.0")
            >>> info = MixinSemverCache._get_semver_cache_info()
            >>> info.hits  # Second call is a hit
            1
        """
        if cls._semver_cache_inner is None:
            return None
        # NOTE: cache_info() is dynamically added by @lru_cache decorator but not
        # reflected in Callable type annotation. This is a known mypy limitation.
        # The return type is functools._CacheInfo.
        result: functools._CacheInfo = cls._semver_cache_inner.cache_info()  # type: ignore[attr-defined]  # NOTE: lru_cache dynamic method
        return result


__all__: list[str] = ["MixinSemverCache"]
