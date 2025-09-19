"""TLS Configuration Models - Backward Compatibility Bridge.

DEPRECATED: This file provides backward compatibility imports only.
Use the individual model files directly:
- ModelKafkaProducerConfig: model_kafka_producer_config.py
- ModelSecurityPolicy: model_security_policy.py
- ModelCredentialCacheEntry: model_credential_cache_entry.py

This file will be removed in a future version.
"""

# Backward compatibility imports
from .model_credential_cache_entry import ModelCredentialCacheEntry
from .model_kafka_producer_config import ModelKafkaProducerConfig
from .model_security_policy import ModelSecurityPolicy

__all__ = [
    "ModelCredentialCacheEntry",
    "ModelKafkaProducerConfig",
    "ModelSecurityPolicy",
]
