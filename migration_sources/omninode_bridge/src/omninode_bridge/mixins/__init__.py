"""
ONEX node capability mixins (local implementation).

This module contains mixins that will be contributed to omnibase_core.
They are implemented locally first to unblock development, then copied
to omnibase_core for framework-wide availability.

Mixins:
    MixinIntentPublisher: Provides intent publishing capability for coordination I/O
"""

from omninode_bridge.mixins.mixin_intent_publisher import MixinIntentPublisher
from omninode_bridge.models.model_intent_publish_result import ModelIntentPublishResult

__all__ = [
    "MixinIntentPublisher",
    "ModelIntentPublishResult",
]
