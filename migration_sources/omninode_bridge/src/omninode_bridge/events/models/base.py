"""
Base event model for all OmniNode Bridge events.

This module provides the EventBase class with strict validation
configured via Pydantic ConfigDict.

ONEX v2.0 Compliance:
- Suffix-based naming: EventBase
- Strong typing with Pydantic v2
- Strict validation with extra="forbid"
"""

from pydantic import BaseModel, ConfigDict


class EventBase(BaseModel):
    """
    Base event model with strict validation for all OmniNode Bridge events.

    Configuration:
        - extra="forbid": Reject unexpected fields to catch typos and schema mismatches early
        - frozen=False: Allow field mutation after creation (default)

    Usage:
        Inherit from this class for all event models to ensure consistent
        strict validation across the codebase:

        ```python
        class MyEvent(EventBase):
            event_type: str
            timestamp: datetime
            payload: dict[str, Any]
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
    )
