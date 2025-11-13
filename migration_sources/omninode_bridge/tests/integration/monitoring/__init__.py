"""Integration tests for monitoring components.

This package contains integration tests for:
- CodegenDLQMonitor: Dead Letter Queue monitoring and alerting
- Performance monitoring and metrics
- Health check integrations

All tests require real infrastructure (Kafka, etc.) and are marked with
@pytest.mark.requires_infrastructure.
"""
