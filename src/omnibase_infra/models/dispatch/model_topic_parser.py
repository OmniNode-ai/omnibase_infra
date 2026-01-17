# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Topic Parser for ONEX Deterministic Routing.

Provides structured parsing of ONEX topic names to support deterministic routing
based on topic category. Handles both ONEX Kafka format (onex.<domain>.<type>)
and Environment-Aware format (<env>.<domain>.<category>.<version>).

Design Pattern:
    ModelTopicParser is a stateless utility class that provides topic parsing
    and pattern matching capabilities. It extracts structured information from
    topic strings including:
    - Topic standard detection (ONEX Kafka vs Environment-Aware)
    - Domain extraction
    - Message category inference (EVENT, COMMAND, INTENT)
    - Topic type (events, commands, intents, snapshots)
    - Environment and version (for Environment-Aware format)

    This enables deterministic routing decisions based on topic structure
    without requiring dispatcher registration lookups.

Thread Safety:
    ModelTopicParser is safe for concurrent use across threads.

    - **Module-level parse cache** (``@lru_cache`` on ``_parse_topic_cached``):
      Thread-safe. Python's ``functools.lru_cache`` uses internal locking to
      ensure atomic cache updates, making concurrent parsing of topics safe.

    - **Instance-level pattern cache** (``_pattern_cache`` dict in ``__init__``):
      Safe but not synchronized. Concurrent calls to ``matches_pattern()`` on
      the same parser instance may compile the same regex pattern multiple times
      (duplicate work), but dict assignment in CPython is atomic so there's no
      risk of data corruption. For single-threaded use or when patterns are
      pre-warmed, this is optimal. For high-concurrency pattern matching on
      shared instances, consider using separate parser instances per thread or
      pre-warming patterns during initialization.

Example:
    >>> from omnibase_infra.models.dispatch import ModelTopicParser, ModelParsedTopic
    >>>
    >>> parser = ModelTopicParser()
    >>>
    >>> # Parse ONEX Kafka format
    >>> result = parser.parse("onex.registration.events")
    >>> result.domain
    'registration'
    >>> result.category
    <EnumMessageCategory.EVENT: 'event'>
    >>>
    >>> # Parse Environment-Aware format
    >>> result = parser.parse("dev.user.events.v1")
    >>> result.environment
    'dev'
    >>> result.version
    'v1'
    >>>
    >>> # Get category for routing
    >>> parser.get_category("onex.registration.commands")
    <EnumMessageCategory.COMMAND: 'command'>
    >>>
    >>> # Pattern matching
    >>> parser.matches_pattern("onex.*.events", "onex.registration.events")
    True

See Also:
    omnibase_infra.enums.EnumMessageCategory: Message category classification
        (EVENT, COMMAND, INTENT) for routing decisions.
    omnibase_infra.enums.EnumTopicType: Topic type suffix enumeration
        (events, commands, intents, snapshots).
    omnibase_infra.enums.EnumTopicStandard: Topic naming standard detection
        (ONEX_KAFKA, ENVIRONMENT_AWARE, UNKNOWN).
    omnibase_infra.models.dispatch.ModelParsedTopic: Structured parse result
        with extracted components and validation status.
    omnibase_infra.runtime.MessageDispatchEngine: Uses topic parsing for
        deterministic message routing to registered dispatchers.
    CLAUDE.md "Enum Usage: Message Routing vs Node Validation": Project-level
        documentation explaining when to use EnumMessageCategory (for message
        routing, topic parsing, dispatcher selection) vs EnumNodeOutputType
        (for execution shape validation, handler return type validation).

Topic Taxonomy Reference:
    The ONEX topic naming taxonomy defines standardized patterns for message
    routing:

    **ONEX Kafka Format** (canonical):
        Pattern: ``onex.<domain>.<type>``
        - Prefix: Always ``onex`` (fixed namespace identifier)
        - Domain: Bounded context name (e.g., registration, discovery, order)
        - Type: Message category suffix (events, commands, intents, snapshots)
        - Examples:
            - ``onex.registration.events`` - Domain events from registration
            - ``onex.discovery.commands`` - Commands for discovery service
            - ``onex.checkout.intents`` - User intents for checkout workflow

    **Environment-Aware Format** (deployment-specific):
        Pattern: ``<env>.<domain>.<category>.<version>``
        - Environment: Deployment target (dev, staging, prod, test, local)
        - Domain: Bounded context name
        - Category: Message category suffix (events, commands, intents)
        - Version: API version identifier (v1, v2, etc.)
        - Examples:
            - ``dev.user.events.v1`` - Development user events, version 1
            - ``prod.order.commands.v2`` - Production order commands, version 2
            - ``staging.payment.intents.v1`` - Staging payment intents

    **Domain Naming Rules**:
        - Lowercase alphanumeric characters with hyphens
        - Must start with a letter
        - Single letter domains are allowed (e.g., ``onex.a.events``)
        - Multi-part domains use hyphens (e.g., ``order-fulfillment``)

    **Category-Based Routing**:
        Topic category determines the message processing pattern:
        - ``events``: Immutable facts processed by reducers and projections
        - ``commands``: Action instructions processed by command handlers
        - ``intents``: User intentions processed by orchestrators
        - ``snapshots``: State snapshots for materialized views (no category mapping)
"""

import re
from functools import lru_cache

from omnibase_core.enums import EnumTopicType
from omnibase_infra.enums import EnumMessageCategory, EnumTopicStandard
from omnibase_infra.models.dispatch.model_parsed_topic import ModelParsedTopic
from omnibase_infra.types import TypeCacheInfo

# Module-level LRU cache for topic parsing performance.
# Since ModelTopicParser is stateless and all class-level attributes are constants,
# we can safely cache parse results at the module level. This provides significant
# performance benefits for repeated topic parsing (common in production).
#
# Cache size of 1024 is chosen to balance memory usage with hit rate:
# - Typical production environments have 10-100 unique topics
# - Cache can hold results for multiple environments/deployments
# - LRU eviction ensures frequently-used topics stay cached
_TOPIC_PARSE_CACHE_SIZE = 1024


@lru_cache(maxsize=_TOPIC_PARSE_CACHE_SIZE)
def _parse_topic_cached(topic: str) -> ModelParsedTopic:
    """
    Module-level cached topic parsing implementation.

    This function contains the actual parsing logic and is decorated with
    @lru_cache to provide automatic caching with LRU eviction.

    Args:
        topic: The topic string to parse (must be non-empty, stripped)

    Returns:
        ModelParsedTopic with extracted components and validation status

    Note:
        This function is internal to the module. Use ModelTopicParser.parse()
        for the public API, which handles empty/whitespace topics before
        delegating to this cached implementation.
    """
    # Try ONEX Kafka format first (canonical)
    onex_match = ModelTopicParser._ONEX_KAFKA_PATTERN.match(topic)
    if onex_match:
        domain = onex_match.group("domain").lower()
        type_str = onex_match.group("type").lower()

        topic_type = ModelTopicParser._TOPIC_TYPE_MAP.get(type_str)
        category = ModelTopicParser._CATEGORY_MAP.get(type_str)

        return ModelParsedTopic(
            raw_topic=topic,
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain=domain,
            category=category,
            topic_type=topic_type,
            is_valid=True,
        )

    # Try Environment-Aware format
    env_match = ModelTopicParser._ENV_AWARE_PATTERN.match(topic)
    if env_match:
        environment = env_match.group("env").lower()
        domain = env_match.group("domain").lower()
        category_str = env_match.group("category").lower()
        version = env_match.group("version").lower()

        topic_type = ModelTopicParser._TOPIC_TYPE_MAP.get(category_str)
        category = ModelTopicParser._CATEGORY_MAP.get(category_str)

        return ModelParsedTopic(
            raw_topic=topic,
            standard=EnumTopicStandard.ENVIRONMENT_AWARE,
            domain=domain,
            category=category,
            topic_type=topic_type,
            environment=environment,
            version=version,
            is_valid=True,
        )

    # -------------------------------------------------------------------------
    # FALLBACK PARSING LOGIC
    # -------------------------------------------------------------------------
    # Why this fallback exists:
    # The primary patterns (ONEX Kafka and Environment-Aware) are strict and
    # require exact format matches. However, in practice we may encounter:
    #
    # 1. **Legacy topic formats** from older systems that predate ONEX standards
    #    (e.g., "myapp.orders.events" without the "onex." prefix)
    #
    # 2. **Partial matches** where the topic contains a valid category suffix
    #    (events, commands, intents) but doesn't fully conform to either standard
    #    (e.g., "custom-prefix.domain.events.extra-suffix")
    #
    # 3. **Custom deployments** with non-standard environment prefixes or
    #    versioning schemes that still use ONEX-style category suffixes
    #
    # The fallback uses EnumMessageCategory.from_topic() which performs a more
    # lenient suffix-based search to find category indicators anywhere in the
    # topic string.
    # -------------------------------------------------------------------------
    category = EnumMessageCategory.from_topic(topic)
    if category is not None:
        # We found a valid category suffix in the topic. Now attempt to extract
        # the domain by locating the category suffix position and taking the
        # segment immediately before it.
        #
        # Domain extraction logic:
        # Given topic "some.prefix.mydomain.events.extra", we:
        # 1. Find ".events" at position X
        # 2. Take everything before that: "some.prefix.mydomain"
        # 3. Split by "." and take the last segment: "mydomain"
        # 4. If there are multiple prefix segments, assume first is environment
        topic_lower = topic.lower()
        category_suffix = f".{category.topic_suffix}"
        if category_suffix in topic_lower:
            # Find the domain: everything before the category suffix
            suffix_idx = topic_lower.find(category_suffix)
            prefix = topic[:suffix_idx]
            # Remove environment prefix if present
            parts = prefix.split(".")
            domain = parts[-1] if parts else None
            env = parts[0] if len(parts) > 1 else None

            # -------------------------------------------------------------------------
            # is_valid=True semantics for UNKNOWN-standard topics:
            # -------------------------------------------------------------------------
            # A topic parsed with EnumTopicStandard.UNKNOWN and is_valid=True means:
            # - The topic does NOT conform to any recognized naming standard
            # - BUT we successfully extracted a message category (EVENT/COMMAND/INTENT)
            # - This makes the topic "partially valid" for routing purposes
            #
            # The dispatch engine can still route these messages because the category
            # is the primary routing key. However, consumers should be aware that:
            # - Domain extraction is best-effort and may be incorrect
            # - Environment/version fields may be absent or inferred incorrectly
            # - The topic naming doesn't follow ONEX best practices
            #
            # In contrast, is_valid=False (below) means we couldn't extract ANY
            # useful routing information from the topic.
            # -------------------------------------------------------------------------
            return ModelParsedTopic(
                raw_topic=topic,
                standard=EnumTopicStandard.UNKNOWN,
                domain=domain,
                category=category,
                environment=env,
                is_valid=True,  # Partially valid - category extracted
            )

    # Unknown format
    return ModelParsedTopic(
        raw_topic=topic,
        standard=EnumTopicStandard.UNKNOWN,
        is_valid=False,
        validation_error=(
            f"Topic '{topic}' does not match any known format. "
            "Expected: onex.<domain>.<type> or <env>.<domain>.<category>.<version>"
        ),
    )


def get_topic_parse_cache_info() -> TypeCacheInfo:
    """
    Get cache statistics for topic parsing.

    Returns:
        TypeCacheInfo: A named tuple with hits, misses, maxsize, and currsize.

    Example:
        >>> from omnibase_infra.models.dispatch import get_topic_parse_cache_info
        >>> info = get_topic_parse_cache_info()
        >>> print(f"Cache hit rate: {info.hits / (info.hits + info.misses):.2%}")
        Cache hit rate: 95.00%
    """
    # Convert from functools._CacheInfo to our typed TypeCacheInfo
    info = _parse_topic_cached.cache_info()
    return TypeCacheInfo(
        hits=info.hits,
        misses=info.misses,
        maxsize=info.maxsize,
        currsize=info.currsize,
    )


def clear_topic_parse_cache() -> None:
    """
    Clear the topic parsing cache.

    This is useful for testing or when topic patterns change dynamically.
    In production, this should rarely be needed as the LRU eviction
    handles cache management automatically.
    """
    _parse_topic_cached.cache_clear()


class ModelTopicParser:
    """
    Parser for ONEX topic names supporting multiple format standards.

    Provides structured parsing of topic strings to extract routing-relevant
    information. Supports both ONEX Kafka format (onex.<domain>.<type>) and
    Environment-Aware format (<env>.<domain>.<category>.<version>).

    The parser is stateless and all methods are pure functions, making it
    safe for concurrent use across threads.

    Patterns:
        - ONEX Kafka: onex.<domain>.<type>
          Examples: onex.registration.events, onex.discovery.commands
        - Environment-Aware: <env>.<domain>.<category>.<version>
          Examples: dev.user.events.v1, prod.order.commands.v2

    Example:
        >>> parser = ModelTopicParser()
        >>>
        >>> # Parse and extract category for routing
        >>> category = parser.get_category("onex.registration.events")
        >>> category
        <EnumMessageCategory.EVENT: 'event'>
        >>>
        >>> # Check pattern matching
        >>> parser.matches_pattern("onex.*.events", "onex.registration.events")
        True
        >>> parser.matches_pattern("**.events", "dev.user.events.v1")
        True

    Attributes:
        _ONEX_KAFKA_PATTERN: Compiled regex for ONEX Kafka format validation.
        _ENV_AWARE_PATTERN: Compiled regex for Environment-Aware format validation.
        _DOMAIN_PATTERN: Compiled regex for domain name validation.
        _VALID_TOPIC_TYPES: Frozenset of valid topic type suffixes.
        _TOPIC_TYPE_MAP: Mapping from suffix to EnumTopicType enum values.
        _CATEGORY_MAP: Mapping from suffix to EnumMessageCategory enum values.

    See Also:
        omnibase_infra.models.dispatch.ModelParsedTopic: The structured result
            returned by parse(), containing all extracted topic components.
        omnibase_infra.enums.EnumMessageCategory: Message category enum used
            for deterministic routing (EVENT, COMMAND, INTENT).
        omnibase_infra.enums.EnumTopicType: Topic type enum representing the
            valid suffixes (events, commands, intents, snapshots).
        omnibase_infra.enums.EnumTopicStandard: Enum for topic naming standard
            detection (ONEX_KAFKA, ENVIRONMENT_AWARE, UNKNOWN).
        omnibase_infra.runtime.MessageDispatchEngine: The dispatch engine that
            uses this parser for topic-based routing decisions.
        get_topic_parse_cache_info: Function to retrieve LRU cache statistics.
        clear_topic_parse_cache: Function to clear the topic parse cache.

    Topic Taxonomy:
        See module-level docstring for complete topic taxonomy documentation,
        including format specifications, domain naming rules, and category-based
        routing semantics.

        External Documentation:
            - ONEX Topic Taxonomy: docs/architecture/TOPIC_TAXONOMY.md (TODO(OMN-981): create)
            - Environment-Aware Topics: docs/patterns/ENVIRONMENT_TOPICS.md (TODO(OMN-982): create)
            - Message Categories: EnumMessageCategory in omnibase_infra.enums
    """

    # ONEX Kafka format: onex.<domain>.<type>
    # Domain: lowercase alphanumeric with hyphens, starting with letter
    # Type: one of commands, events, intents, snapshots
    _ONEX_KAFKA_PATTERN = re.compile(
        r"^onex\.(?P<domain>[a-z][a-z0-9-]*[a-z0-9]|[a-z])\."
        r"(?P<type>commands|events|intents|snapshots)$",
        re.IGNORECASE,
    )

    # Environment-Aware format: <env>.<domain>.<category>.<version>
    # Env: dev, prod, staging, test, local (case-insensitive)
    # Domain: alphanumeric with hyphens
    # Category: events, commands, intents (plural form)
    # Version: v followed by digits
    _ENV_AWARE_PATTERN = re.compile(
        r"^(?P<env>dev|prod|staging|test|local)\."
        r"(?P<domain>[a-z][a-z0-9-]*[a-z0-9]|[a-z])\."
        r"(?P<category>commands|events|intents)\."
        r"(?P<version>v\d+)$",
        re.IGNORECASE,
    )

    # Domain validation pattern (reused from constants_topic_taxonomy)
    _DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z]$")

    # Valid topic types for validation
    _VALID_TOPIC_TYPES = frozenset({"commands", "events", "intents", "snapshots"})

    # Mapping from topic type suffix to EnumTopicType
    _TOPIC_TYPE_MAP: dict[str, EnumTopicType] = {
        "commands": EnumTopicType.COMMANDS,
        "events": EnumTopicType.EVENTS,
        "intents": EnumTopicType.INTENTS,
        "snapshots": EnumTopicType.SNAPSHOTS,
    }

    # Mapping from topic type suffix to EnumMessageCategory
    # Note: snapshots don't have a direct category mapping
    _CATEGORY_MAP: dict[str, EnumMessageCategory] = {
        "commands": EnumMessageCategory.COMMAND,
        "events": EnumMessageCategory.EVENT,
        "intents": EnumMessageCategory.INTENT,
    }

    def parse(self, topic: str) -> ModelParsedTopic:
        """
        Parse a topic string and extract structured information.

        Attempts to parse the topic against known formats (ONEX Kafka and
        Environment-Aware) and returns a structured result with all extracted
        components.

        Caching:
            Results are cached with LRU eviction (maxsize=1024) at the module
            level. This provides significant performance benefits for repeated
            topic parsing, which is common in production message dispatch.
            Cache statistics can be monitored via get_topic_parse_cache_info().

        Args:
            topic: The topic string to parse

        Returns:
            ModelParsedTopic with extracted components and validation status

        Example:
            >>> parser = ModelTopicParser()
            >>> result = parser.parse("onex.registration.events")
            >>> result.standard
            <EnumTopicStandard.ONEX_KAFKA: 'onex_kafka'>
            >>> result.domain
            'registration'
            >>> result.category
            <EnumMessageCategory.EVENT: 'event'>

        See Also:
            get_topic_parse_cache_info: Get cache statistics (hits, misses, size)
            clear_topic_parse_cache: Clear the parse cache if needed
        """
        # Handle empty/whitespace topics (not cached - edge case)
        if not topic or not topic.strip():
            return ModelParsedTopic(
                raw_topic="<empty>",  # Use placeholder to satisfy min_length constraint
                standard=EnumTopicStandard.UNKNOWN,
                is_valid=False,
                validation_error="Topic cannot be empty or whitespace",
            )

        # Delegate to cached implementation for actual parsing
        return _parse_topic_cached(topic.strip())

    def get_category(self, topic: str) -> EnumMessageCategory | None:
        """
        Extract the message category from a topic for routing.

        This is a convenience method that parses the topic and returns
        just the category, which is the primary input for deterministic
        routing decisions.

        Args:
            topic: The topic string to analyze

        Returns:
            EnumMessageCategory if detected, None otherwise

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.get_category("onex.registration.events")
            <EnumMessageCategory.EVENT: 'event'>
            >>> parser.get_category("dev.user.commands.v1")
            <EnumMessageCategory.COMMAND: 'command'>
            >>> parser.get_category("invalid.topic")
            None

        See Also:
            EnumMessageCategory: The enum type returned, representing message
                categories (EVENT, COMMAND, INTENT) for routing decisions.
            EnumMessageCategory.topic_suffix: Property that returns the plural
                suffix (events, commands, intents) for topic construction.
            MessageDispatchEngine: Uses category for dispatcher selection.
        """
        parsed = self.parse(topic)
        return parsed.category

    def matches_pattern(self, pattern: str, topic: str) -> bool:
        """
        Check if a topic matches a glob-style pattern.

        Supports the following wildcards:
        - '*' (single asterisk): Matches any single segment (no dots)
        - '**' (double asterisk): Matches any number of segments (including dots)

        Pattern matching is case-insensitive.

        Args:
            pattern: The glob pattern to match against
            topic: The topic to check

        Returns:
            True if the topic matches the pattern, False otherwise

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.matches_pattern("onex.*.events", "onex.registration.events")
            True
            >>> parser.matches_pattern("onex.*.events", "onex.discovery.events")
            True
            >>> parser.matches_pattern("onex.*.events", "onex.discovery.commands")
            False
            >>> parser.matches_pattern("**.events", "dev.user.events.v1")
            False  # ** matches segments but v1 is after .events
            >>> parser.matches_pattern("**.events.*", "dev.user.events.v1")
            True
            >>> parser.matches_pattern("dev.**", "dev.user.events.v1")
            True
        """
        if not pattern or not topic:
            return False

        # Compile pattern to regex
        regex_pattern = self._pattern_to_regex(pattern)
        return bool(regex_pattern.match(topic))

    def __init__(self) -> None:
        """Initialize the topic parser with an empty pattern cache."""
        self._pattern_cache: dict[str, re.Pattern[str]] = {}

    def _pattern_to_regex(self, pattern: str) -> re.Pattern[str]:
        """Convert a glob-style pattern to a compiled regex.

        Handles:
        - '*' -> matches any single segment (no dots)
        - '**' -> matches any number of segments (including empty)

        Uses an instance-level cache to avoid recompiling frequently used patterns.
        """
        # Check cache first
        if pattern in self._pattern_cache:
            return self._pattern_cache[pattern]

        # Handle ** first (must be done before single *)
        # Use a placeholder to avoid double-processing
        escaped = pattern.replace("**", "__DOUBLE_STAR__")

        # Escape special regex characters except *
        escaped = re.escape(escaped)

        # Convert back ** placeholder to multi-segment match
        # ** matches zero or more segments: (?:[^.]+(?:\.[^.]+)*)?
        # This matches: nothing, or one segment, or multiple segments separated by dots
        escaped = escaped.replace("__DOUBLE_STAR__", "(?:[^.]+(?:\\.[^.]+)*)?")

        # Convert single * to single-segment match (no dots)
        escaped = escaped.replace(r"\*", "[^.]+")

        # Compile and cache
        compiled = re.compile(f"^{escaped}$", re.IGNORECASE)
        self._pattern_cache[pattern] = compiled
        return compiled

    def validate_topic(
        self, topic: str, strict: bool = False
    ) -> tuple[bool, str | None]:
        """
        Validate a topic string against ONEX standards.

        Args:
            topic: The topic string to validate
            strict: If True, requires exact match to ONEX Kafka format.
                   If False, accepts any format that yields a valid category.

        Returns:
            Tuple of (is_valid, error_message).
            - (True, None) if valid
            - (False, error_message) if invalid

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.validate_topic("onex.registration.events")
            (True, None)
            >>> parser.validate_topic("onex.registration.events", strict=True)
            (True, None)
            >>> parser.validate_topic("dev.user.events.v1")
            (True, None)
            >>> parser.validate_topic("dev.user.events.v1", strict=True)
            (False, "Topic 'dev.user.events.v1' does not match ONEX Kafka format...")
            >>> parser.validate_topic("invalid")
            (False, "Topic 'invalid' does not match any known format...")

        See Also:
            EnumTopicStandard: Enum used to classify topic naming standards.
                ONEX_KAFKA is the canonical format; ENVIRONMENT_AWARE is for
                deployment-specific topics; UNKNOWN indicates unrecognized format.
            ModelParsedTopic.is_valid: Boolean indicating parse success.
            ModelParsedTopic.validation_error: Error message on parse failure.
        """
        parsed = self.parse(topic)

        if strict:
            if parsed.standard != EnumTopicStandard.ONEX_KAFKA:
                return (
                    False,
                    f"Topic '{topic}' does not match ONEX Kafka format "
                    f"(onex.<domain>.<type>). Detected standard: {parsed.standard.value}",
                )
            return (True, None)

        # Non-strict: accept any valid parsed topic
        if parsed.is_valid:
            return (True, None)

        return (False, parsed.validation_error)

    def is_onex_kafka_format(self, topic: str) -> bool:
        """
        Check if a topic follows the ONEX Kafka naming standard.

        Args:
            topic: The topic string to check

        Returns:
            True if the topic matches onex.<domain>.<type> format

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.is_onex_kafka_format("onex.registration.events")
            True
            >>> parser.is_onex_kafka_format("dev.user.events.v1")
            False

        See Also:
            EnumTopicStandard.ONEX_KAFKA: The enum value for this format.
            is_environment_aware_format: Check for the alternate format.
            validate_topic: Validation with strict mode for ONEX Kafka only.
        """
        return bool(self._ONEX_KAFKA_PATTERN.match(topic.strip()))

    def is_environment_aware_format(self, topic: str) -> bool:
        """
        Check if a topic follows the Environment-Aware naming standard.

        Args:
            topic: The topic string to check

        Returns:
            True if the topic matches <env>.<domain>.<category>.<version> format

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.is_environment_aware_format("dev.user.events.v1")
            True
            >>> parser.is_environment_aware_format("onex.registration.events")
            False

        Note:
            Environment-Aware format includes additional metadata:
            - Environment prefix (dev, staging, prod, test, local)
            - Version suffix (v1, v2, etc.)

            These components are extracted via parse() and available in
            ModelParsedTopic.environment and ModelParsedTopic.version.

        See Also:
            EnumTopicStandard.ENVIRONMENT_AWARE: The enum value for this format.
            is_onex_kafka_format: Check for the canonical ONEX format.
            ModelParsedTopic.environment: Environment extracted from topic.
            ModelParsedTopic.version: Version extracted from topic.
        """
        return bool(self._ENV_AWARE_PATTERN.match(topic.strip()))

    def extract_domain(self, topic: str) -> str | None:
        """
        Extract the domain from a topic string.

        Args:
            topic: The topic string to analyze

        Returns:
            The domain name if extractable, None otherwise

        Example:
            >>> parser = ModelTopicParser()
            >>> parser.extract_domain("onex.registration.events")
            'registration'
            >>> parser.extract_domain("dev.user.events.v1")
            'user'

        Note:
            Domain naming follows these rules:
            - Lowercase alphanumeric characters with hyphens
            - Must start with a letter
            - Single letter domains are valid (e.g., 'a', 'x')
            - Multi-part domains use hyphens (e.g., 'order-fulfillment')

        See Also:
            ModelParsedTopic.domain: The domain field in parse results.
            _DOMAIN_PATTERN: Class attribute with domain validation regex.
        """
        parsed = self.parse(topic)
        return parsed.domain


__all__ = [
    "ModelTopicParser",
    "clear_topic_parse_cache",
    "get_topic_parse_cache_info",
]
