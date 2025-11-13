"""Intent model for pure reducer operations.

Intents represent side effects that should be performed by
effect nodes or services, following ONEX v2.0 pure function architecture.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .enum_intent_type import EnumIntentType


class ModelIntent(BaseModel):
    """
    Intent model for side effects in pure reducer.

    Intents are returned by the pure reducer to indicate
    what side effects (database writes, event publishing)
    should be performed by other nodes in the workflow.

    ONEX v2.0 Intent Publisher Pattern:
    - COMPUTE nodes emit intents (no I/O)
    - EFFECT nodes consume intents and perform I/O
    - Orchestrators route intents to appropriate EFFECT nodes
    """

    intent_type: str = Field(
        ...,
        description="Type of intent (use EnumIntentType values)",
    )

    target: str = Field(
        ...,
        description="Target service or node for this intent (store_effect, event_bus, etc.)",
    )

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Intent payload data",
    )

    priority: int = Field(
        default=0,
        description="Execution priority (higher = more urgent)",
        ge=0,
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "intent_type": EnumIntentType.PERSIST_STATE.value,
                "target": "store_effect",
                "payload": {
                    "aggregated_data": {
                        "omninode.services.metadata": {
                            "total_stamps": 100,
                            "total_size_bytes": 1024000,
                        }
                    },
                    "fsm_states": {
                        "workflow-123": "COMPLETED",
                    },
                },
                "priority": 1,
            }
        }
