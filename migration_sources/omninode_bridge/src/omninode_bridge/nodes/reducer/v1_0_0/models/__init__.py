#!/usr/bin/env python3
"""
NodeBridgeReducer Models - Pydantic v2 Models.

Provides strongly-typed Pydantic models for the NodeBridgeReducer:
- EnumAggregationType: Aggregation strategy types
- EnumReducerEvent: Kafka event types for reducer
- ModelReducerInputState: Streaming item input state
- ModelReducerOutputState: Aggregated result output state
- ModelBridgeState: PostgreSQL-persisted bridge state
- ModelIntent: Intent-based side effect specification

ONEX v2.0 Compliance:
- Suffix-based naming conventions
- O.N.E. v0.1 protocol compliance
- Strong typing with Pydantic v2
"""

from .enum_aggregation_type import EnumAggregationType
from .enum_intent_type import EnumIntentType
from .enum_reducer_event import EnumReducerEvent
from .model_bridge_state import ModelBridgeState
from .model_input_state import ModelReducerInputState
from .model_intent import ModelIntent
from .model_output_state import ModelReducerOutputState

__all__ = [
    "EnumAggregationType",
    "EnumIntentType",
    "EnumReducerEvent",
    "ModelBridgeState",
    "ModelReducerInputState",
    "ModelIntent",
    "ModelReducerOutputState",
]
