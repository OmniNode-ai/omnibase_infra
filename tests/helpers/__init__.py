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
"""

from tests.helpers.ast_analysis import (
    find_datetime_now_calls,
    find_io_method_calls,
    find_time_module_calls,
    get_imported_root_modules,
    is_docstring,
)
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator
from tests.helpers.log_helpers import filter_handler_warnings, get_warning_messages

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
]
