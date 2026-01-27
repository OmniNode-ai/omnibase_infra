"""Topic composition utilities for ONEX infrastructure.

IMPORTANT: build_full_topic() is the ONLY supported way to compose
Kafka topics in omnibase_infra. Direct string concatenation is prohibited.
"""

from omnibase_core.validation import validate_topic_suffix
from omnibase_core.validation.validator_topic_suffix import ENV_PREFIXES


class TopicCompositionError(ValueError):
    """Raised when topic composition fails due to invalid components."""


def build_full_topic(env: str, namespace: str, suffix: str) -> str:
    """Build full topic from components with validation.

    Args:
        env: Environment prefix (e.g., "dev", "staging", "prod", "test", "local")
        namespace: Namespace/tenant identifier (e.g., "omnibase", "myapp")
        suffix: Validated topic suffix (e.g., "onex.evt.platform.node-introspection.v1")

    Returns:
        Full topic string: {env}.{namespace}.{suffix}

    Raises:
        TopicCompositionError: If env is not a valid environment prefix
        TopicCompositionError: If namespace is empty or contains invalid characters
        TopicCompositionError: If suffix doesn't match ONEX topic format

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
    if not namespace:
        raise TopicCompositionError("Namespace cannot be empty")
    if not namespace.replace("-", "").replace("_", "").isalnum():
        raise TopicCompositionError(
            f"Invalid namespace '{namespace}'. "
            "Must contain only alphanumeric characters, hyphens, and underscores"
        )

    # Validate suffix using omnibase_core validation
    result = validate_topic_suffix(suffix)
    if not result.is_valid:
        raise TopicCompositionError(f"Invalid topic suffix '{suffix}': {result.error}")

    # Compose full topic
    return f"{env}.{namespace}.{suffix}"
