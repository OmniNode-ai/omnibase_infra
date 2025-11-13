# Integration Patterns

**Category**: External System Connectivity
**Pattern Count**: 0 (will be populated in Task F2)

## Overview

Integration patterns facilitate communication with external systems (databases, message queues, APIs).

## Expected Patterns

1. **Event Publishing** - Publish events to Kafka
2. **Kafka Consumer** - Consume events from Kafka topics
3. **API Client** - HTTP client with retries and circuit breakers
4. **Database Adapter** - Database connection and query patterns
5. **Message Queue** - Asynchronous message processing

## Integration Principles

- **Idempotency**: Handle duplicate messages gracefully
- **Resilience**: Handle external failures gracefully
- **Loose Coupling**: Minimize dependencies on external systems
- **Versioning**: Support API/schema evolution

## Next Steps

Patterns will be extracted from production nodes during Task F2.
