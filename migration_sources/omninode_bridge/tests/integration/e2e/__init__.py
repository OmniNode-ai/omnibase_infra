"""
End-to-End Integration Tests for Event Bus Infrastructure.

This test suite validates complete event flows across all services:
- Orchestrator → Reducer workflows
- Metadata Stamping → Database persistence
- Cross-service coordination (OnexTree intelligence)
- Event bus resilience (circuit breaker, graceful degradation)
- Database Adapter event consumption
- Performance validation against thresholds

Test Infrastructure:
- Testcontainers for Kafka, PostgreSQL, Consul
- CI-ready (can run in GitHub Actions)
- Async fixtures with proper cleanup
- Performance benchmarking
"""

__all__ = []
