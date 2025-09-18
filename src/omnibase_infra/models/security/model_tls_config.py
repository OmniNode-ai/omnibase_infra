"""TLS Configuration Models.

DEPRECATED: This file contains multiple models and violates ONEX one-model-per-file rule.
Use the following files instead:
- model_kafka_producer_config.py
- model_security_policy.py
- model_credential_cache_entry.py

Maintaining imports for backward compatibility only.
"""

# Import from new single-model files for backward compatibility
from .model_kafka_producer_config import ModelKafkaProducerConfig
from .model_security_policy import ModelSecurityPolicy
from .model_credential_cache_entry import ModelCredentialCacheEntry

# Re-export for backward compatibility
__all__ = [
    "ModelKafkaProducerConfig",
    "ModelSecurityPolicy",
    "ModelCredentialCacheEntry"
]

# Legacy class definitions removed - use imports above
