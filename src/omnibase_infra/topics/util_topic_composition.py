"""Topic composition utilities for ONEX infrastructure.

IMPORTANT: build_full_topic() is the ONLY supported way to compose
Kafka topics in omnibase_infra. Direct string concatenation is prohibited.
"""

from omnibase_core.errors import OnexError
from omnibase_core.validation import validate_topic_suffix
from omnibase_core.validation.validator_topic_suffix import ENV_PREFIXES

MAX_NAMESPACE_LENGTH = 100
"""Maximum allowed namespace length.

Kafka topics have a 249 character limit. With env prefix (~10 chars),
separators (2 dots), and suffix (~50 chars), namespace should be limited
to ensure total topic length stays within bounds.
"""


class TopicCompositionError(OnexError):
    """Raised when topic composition fails due to invalid components.

    Extends OnexError to follow ONEX error handling conventions.
    """


def build_full_topic(env: str, namespace: str, suffix: str) -> str:
    """Build full topic from components with validation.

    Args:
        env: Environment prefix (e.g., "dev", "staging", "prod", "test", "local")
        namespace: Namespace/tenant identifier (e.g., "omnibase", "myapp", "tenant-123")
        suffix: Validated topic suffix (e.g., "onex.evt.platform.node-introspection.v1")

    Returns:
        Full topic string: {env}.{namespace}.{suffix}

    Raises:
        TopicCompositionError: If env is not a valid environment prefix
        TopicCompositionError: If namespace is empty or contains invalid characters
        TopicCompositionError: If suffix doesn't match ONEX topic format

    Namespace Validation Rules:
        - Must not be empty
        - Allowed characters: alphanumeric (a-z, A-Z, 0-9), hyphens (-), underscores (_)
        - Numeric-only namespaces ARE valid (e.g., "12345") to support numeric tenant IDs
        - Invalid: spaces, dots, special characters

    Example:
        >>> build_full_topic("dev", "omnibase", "onex.evt.platform.node-introspection.v1")
        'dev.omnibase.onex.evt.platform.node-introspection.v1'

        >>> build_full_topic("prod", "myapp", "onex.cmd.platform.request-introspection.v1")
        'prod.myapp.onex.cmd.platform.request-introspection.v1'
    """
    # Validate environment prefix
    if env not in ENV_PREFIXES:
        raise TopicCompositionError(
            f"Invalid environment prefix '{env}'. "
            f"Must be one of: {', '.join(sorted(ENV_PREFIXES))}"
        )

    # Validate namespace
    # Allowed characters: alphanumeric (a-z, A-Z, 0-9), hyphens, underscores
    # Note: Numeric-only namespaces (e.g., "12345") ARE valid because isalnum()
    # returns True for digit-only strings. This is intentional to support
    # numeric tenant/organization IDs as namespaces.
    if not namespace:
        raise TopicCompositionError("Namespace cannot be empty")
    if len(namespace) > MAX_NAMESPACE_LENGTH:
        raise TopicCompositionError(
            f"Namespace exceeds maximum length of {MAX_NAMESPACE_LENGTH} characters. "
            f"Got {len(namespace)} characters."
        )
    if not namespace.replace("-", "").replace("_", "").isalnum():
        raise TopicCompositionError(
            f"Invalid namespace '{namespace}'. "
            "Must contain only alphanumeric characters, hyphens, and underscores"
        )

    # Enforce lowercase to ensure composed topics are valid
    # (ONEX topic suffixes are lowercase, so namespaces should match)
    if namespace != namespace.lower():
        raise TopicCompositionError(
            f"Namespace must be lowercase: '{namespace}'. "
            "Use lowercase to ensure consistent topic naming."
        )

    # Validate suffix using omnibase_core validation
    result = validate_topic_suffix(suffix)
    if not result.is_valid:
        raise TopicCompositionError(f"Invalid topic suffix '{suffix}': {result.error}")

    # Compose full topic
    return f"{env}.{namespace}.{suffix}"
