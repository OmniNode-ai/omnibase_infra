# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test helpers for omnibase_infra unit tests.

This module provides deterministic utilities for testing infrastructure
handlers and services, enabling predictable and reproducible test behavior.

Available Utilities:
    Deterministic:
        - DeterministicClock: Fixed clock for reproducible time-based tests
        - DeterministicIdGenerator: Fixed ID generator for reproducible tests

    Log Helpers:
        - filter_handler_warnings: Filter warning messages from handlers
        - get_warning_messages: Extract warning messages from log records

    AST Analysis:
        - get_imported_root_modules: Extract root module names from imports
        - find_datetime_now_calls: Find datetime.now()/utcnow() calls
        - find_time_module_calls: Find time.time()/monotonic() calls
        - find_io_method_calls: Find I/O method calls matching patterns
        - is_docstring: Check if a statement is a docstring

    Chaos Testing (OMN-955):
        - ChaosChainConfig: Configuration for chaos injection in chain tests
        - ChainedMessage: Message model with correlation/causation tracking
        - ChainBuilder: Builder for message chains with chaos injection
        - create_envelope_from_chained_message: Convert ChainedMessage to envelope

    Replay Testing (OMN-955):
        - compare_outputs: Compare reducer outputs for determinism verification
        - OrderingViolation: Model for ordering violations in event sequences
        - detect_timestamp_order_violations: Detect timestamp ordering issues
        - detect_sequence_number_violations: Detect sequence number gaps
        - EventSequenceEntry: Entry in an event sequence log
        - EventSequenceLog: Log of events for replay testing
        - EventFactory: Factory for deterministic event creation
        - create_introspection_event: Helper for creating introspection events

    Statistics (OMN-955):
        - PerformanceStats: Comprehensive statistics for timing samples
        - MemoryTracker: tracemalloc-based memory tracking
        - MemorySnapshot: Memory snapshot at a point in time
        - BinomialConfidenceInterval: Confidence interval for proportions
        - calculate_binomial_confidence_interval: Wilson score interval
        - minimum_sample_size_for_tolerance: Calculate required sample size
        - run_with_warmup: Async operation timing with warmup
        - run_with_warmup_sync: Sync operation timing with warmup
"""

from tests.helpers.ast_analysis import (
    find_datetime_now_calls,
    find_io_method_calls,
    find_time_module_calls,
    get_imported_root_modules,
    is_docstring,
)
from tests.helpers.chaos_utils import (
    ChainBuilder,
    ChainedMessage,
    ChaosChainConfig,
    create_envelope_from_chained_message,
)
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator
from tests.helpers.log_helpers import filter_handler_warnings, get_warning_messages
from tests.helpers.replay_utils import (
    EventFactory,
    EventSequenceEntry,
    EventSequenceEntryDict,
    EventSequenceLog,
    EventSequenceLogDict,
    ModelOutputComparison,
    NodeType,
    OrderingViolation,
    compare_outputs,
    create_introspection_event,
    detect_sequence_number_violations,
    detect_timestamp_order_violations,
)
from tests.helpers.statistics_utils import (
    BinomialConfidenceInterval,
    MemorySnapshot,
    MemoryTracker,
    PerformanceStats,
    calculate_binomial_confidence_interval,
    minimum_sample_size_for_tolerance,
    run_with_warmup,
    run_with_warmup_sync,
)

__all__ = [
    # Deterministic utilities
    "DeterministicClock",
    "DeterministicIdGenerator",
    # Log helpers
    "filter_handler_warnings",
    "get_warning_messages",
    # AST analysis
    "find_datetime_now_calls",
    "find_io_method_calls",
    "find_time_module_calls",
    "get_imported_root_modules",
    "is_docstring",
    # Chaos testing utilities
    "ChaosChainConfig",
    "ChainedMessage",
    "ChainBuilder",
    "create_envelope_from_chained_message",
    # Replay testing utilities
    "ModelOutputComparison",
    "compare_outputs",
    "OrderingViolation",
    "detect_timestamp_order_violations",
    "detect_sequence_number_violations",
    "EventSequenceEntryDict",
    "EventSequenceLogDict",
    "EventSequenceEntry",
    "EventSequenceLog",
    "EventFactory",
    "NodeType",
    "create_introspection_event",
    # Statistics utilities
    "PerformanceStats",
    "MemoryTracker",
    "MemorySnapshot",
    "BinomialConfidenceInterval",
    "calculate_binomial_confidence_interval",
    "minimum_sample_size_for_tolerance",
    "run_with_warmup",
    "run_with_warmup_sync",
]
