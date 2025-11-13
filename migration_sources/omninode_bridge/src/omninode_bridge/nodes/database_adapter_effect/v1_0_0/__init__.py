"""
Database Adapter Effect Node v1.0.0.

This package provides database adapter effect node components for omninode_bridge,
including circuit breaker patterns for resilient database connectivity, structured
logging with correlation ID tracking, and comprehensive security validation.

Components:
    - DatabaseCircuitBreaker: Circuit breaker for database operation protection
    - CircuitBreakerState: State enumeration (CLOSED, OPEN, HALF_OPEN)
    - CircuitBreakerOpenError: Exception raised when circuit is open
    - DatabaseStructuredLogger: Structured logging with correlation ID tracking
    - DatabaseOperationType: Operation type enumeration for logging categorization
    - PerformanceCategory: Performance categorization (FAST, SLOW, VERY_SLOW)
    - get_database_logger: Factory function for logger instances
    - DatabaseSecurityValidator: Comprehensive security validation for database operations
    - ValidationResult: Validation result dataclass with errors and warnings

Author: OmniNode Bridge Team
Created: October 7, 2025
Version: 1.0.0
"""

from .circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitBreakerState,
    DatabaseCircuitBreaker,
)
from .security_validator import DatabaseSecurityValidator, ValidationResult
from .structured_logger import (
    DatabaseOperationType,
    DatabaseStructuredLogger,
    PerformanceCategory,
    get_database_logger,
)

__all__ = [
    "DatabaseCircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",
    "DatabaseStructuredLogger",
    "DatabaseOperationType",
    "PerformanceCategory",
    "get_database_logger",
    "DatabaseSecurityValidator",
    "ValidationResult",
]
