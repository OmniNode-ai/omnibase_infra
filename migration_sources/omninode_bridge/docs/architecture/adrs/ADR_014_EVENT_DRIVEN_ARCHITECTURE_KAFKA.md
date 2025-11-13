# ADR-014: Event-Driven Architecture with Kafka

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of event-driven communication between services

## Context

The multi-service architecture requires reliable, scalable communication between services. The system needs to handle:

- Asynchronous event processing between services
- High throughput event ingestion from webhooks
- Event ordering and delivery guarantees
- Service decoupling and resilience
- Event replay and audit capabilities

Traditional synchronous HTTP communication creates tight coupling and can lead to cascading failures in distributed systems. We need a solution that provides loose coupling, durability, and scalability.

## Decision

We adopt Apache Kafka as the primary event streaming platform for inter-service communication with the following architecture:

### Event Flow Architecture
1. **HookReceiver** converts incoming webhooks to internal events and publishes to Kafka
2. **Event Topics** organize events by type and processing requirements:
   - `service-lifecycle-events`: Service startup, shutdown, health events
   - `tool-execution-events`: AI tool execution and results
   - `workflow-events`: Workflow state changes and task updates
   - `audit-events`: Security and compliance events

3. **Service Consumers** subscribe to relevant topic partitions based on their responsibilities

### Event Schema Design
- **Envelope Pattern**: All events wrapped in standard envelope with correlation IDs
- **Event Versioning**: Schema evolution support for backward compatibility
- **Event Correlation**: Distributed tracing through correlation IDs
- **Event Metadata**: Timestamps, source service, event type classification

### Kafka Configuration
- **Partitioning**: By correlation ID for event ordering within workflows
- **Replication**: Factor of 3 for production reliability
- **Retention**: 30 days for event replay and audit
- **Compaction**: Log compaction for state events

## Consequences

### Positive Consequences

- **Service Decoupling**: Services communicate without direct dependencies
- **Scalability**: Each service scales independently based on event processing capacity
- **Durability**: Events persisted until consumed, preventing data loss
- **Event Replay**: Ability to replay events for debugging or data recovery
- **Audit Trail**: Complete event history for compliance and debugging
- **Fault Tolerance**: Service failures don't block event publishing
- **Parallel Processing**: Multiple consumers can process events in parallel

### Negative Consequences

- **Eventual Consistency**: Distributed state updates are eventually consistent
- **Operational Complexity**: Kafka cluster management and monitoring required
- **Network Overhead**: Additional network hops for event processing
- **Debugging Complexity**: Distributed tracing required for request correlation
- **Event Schema Management**: Schema evolution and compatibility challenges
- **Local Development**: Kafka infrastructure required for local testing

## Implementation Details

### Event Publishing (Producer)
```python
# HookReceiver service publishes events to Kafka
await kafka_client.publish_event(
    topic="service-lifecycle-events",
    key=correlation_id,  # For partitioning
    event=ServiceLifecycleEvent(
        event=ServiceEventType.STARTUP,
        service=service_name,
        correlation_id=correlation_id,
        metadata=event_metadata
    )
)
```

### Event Consumption (Consumer)
```python
# WorkflowCoordinator consumes relevant events
async for event in kafka_client.consume_events("workflow-events"):
    await workflow_processor.handle_event(event)
```

### Circuit Breaker Protection
- Kafka publishing operations protected by circuit breakers
- Graceful degradation when Kafka is unavailable
- Local event buffering with overflow handling

### Error Handling Strategy
- **Dead Letter Queues**: Failed events routed to DLQ for investigation
- **Retry Logic**: Exponential backoff for transient failures
- **Error Monitoring**: Metrics and alerting for event processing failures

## Compliance

This architecture aligns with ONEX standards by:

- **Event Sourcing**: Complete audit trail of system events
- **Loose Coupling**: Services communicate via events, not direct calls
- **Resilience**: Circuit breakers and graceful degradation patterns
- **Observability**: Correlation IDs for distributed tracing
- **Scalability**: Horizontal scaling through event partitioning

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-012: Event Sourcing Audit Strategy

## References

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Building Event-Driven Microservices](https://www.oreilly.com/library/view/building-event-driven-microservices/9781492057888/)
- [Event Sourcing Pattern](https://microservices.io/patterns/data/event-sourcing.html)
- [Apache Kafka Best Practices](https://kafka.apache.org/documentation/#bestpractices)
- [aiokafka Client Library](https://aiokafka.readthedocs.io/)
