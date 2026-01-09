# Kernel.py Event Processing Pipeline - Logging Enhancements

**Date**: 2025-12-27
**File**: `src/omnibase_infra/runtime/kernel.py`
**Purpose**: Add comprehensive structured logging to diagnose event processing pipeline failures

---

## Changes Summary

### 1. Dispatcher Creation Logging (Lines 519-546)

**Added**:
- Log when resolving `HandlerNodeIntrospected` from container
- Log successful handler resolution with class name
- Enhanced dispatcher creation log to INFO level with structured context

**Logging Points**:
```python
logger.debug("Resolving HandlerNodeIntrospected from container (correlation_id=%s)", ...)
logger.debug("HandlerNodeIntrospected resolved successfully (correlation_id=%s)", ...)
logger.info("Introspection dispatcher created and wired (correlation_id=%s)", ...)
```

**Context Captured**:
- `handler_class`: Handler class name
- `dispatcher_class`: Dispatcher class name
- `correlation_id`: Bootstrap correlation ID

---

### 2. Kafka Consumer Subscription Logging (Lines 902-930)

**Added**:
- Log when initiating subscription (before `subscribe()` call)
- Log subscription success with timing information
- Include topic, consumer group, and event bus type in context

**Logging Points**:
```python
logger.info("Subscribing to introspection events on Kafka (correlation_id=%s)", ...)
logger.info("Introspection event consumer started successfully in %.3fs (correlation_id=%s)", ...)
```

**Context Captured**:
- `topic`: Input topic name
- `consumer_group`: Consumer group ID
- `event_bus_type`: "kafka"
- `subscribe_duration_seconds`: Subscription setup time

---

### 3. Message Callback Comprehensive Logging (Lines 715-900)

**Major Enhancements**:

#### Entry Point (Lines 725-737)
```python
logger.debug("Introspection message callback invoked (correlation_id=%s)", ...)
```
- Logs every message received from Kafka
- Captures message offset, partition, topic
- Generates unique callback correlation ID for tracing

#### Message Parsing (Lines 741-775)
```python
logger.debug("Parsing message value as bytes (correlation_id=%s)", ...)
logger.debug("Message value already dict (correlation_id=%s)", ...)
logger.debug("Unexpected message value type: %s (correlation_id=%s)", ...)
```
- Logs message value type detection
- Logs value length for bytes/string
- Logs unexpected types

#### Event Validation (Lines 778-793)
```python
logger.debug("Validating payload as ModelNodeIntrospectionEvent (correlation_id=%s)", ...)
logger.info("Introspection event parsed successfully (correlation_id=%s)", ...)
```
- Logs validation attempt
- Logs successful parsing with node_id, node_type, version

#### Envelope Creation (Lines 810-817)
```python
logger.debug("Event envelope created (correlation_id=%s)", ...)
```
- Logs envelope creation
- Captures envelope correlation_id and timestamp

#### Dispatcher Routing (Lines 820-856)
```python
logger.info("Routing to introspection dispatcher (correlation_id=%s)", ...)
logger.info("Introspection event processed successfully: node_id=%s in %.3fs (correlation_id=%s)", ...)
logger.warning("Introspection event processing failed: %s (correlation_id=%s)", ...)
```
- Logs before dispatcher.handle() call
- Logs processing success/failure with timing
- Captures dispatcher duration, node_id, node_type

#### Exception Handling (Lines 858-889)
```python
# ValidationError - not an introspection event
logger.debug("Message is not a valid introspection event, skipping (correlation_id=%s)", ...)

# JSON decode errors
logger.warning("Failed to decode JSON from message: %s (correlation_id=%s)", ...)

# Generic exceptions
logger.error("Failed to process introspection message: %s (correlation_id=%s)", ..., exc_info=True)
```
- ValidationError: Debug level (expected for non-introspection messages)
- JSONDecodeError: Warning level with error position
- Generic exceptions: Error level with full stack trace

#### Callback Completion (Lines 891-900)
```python
logger.debug("Introspection message callback completed in %.3fs (correlation_id=%s)", ...)
```
- Always logs callback duration (in finally block)
- Provides total callback execution time

---

## Correlation ID Strategy

**Two-Level Correlation**:

1. **Bootstrap Correlation ID**: Generated at kernel startup, used for all bootstrap operations
   - Dispatcher creation
   - Consumer subscription
   - Infrastructure setup

2. **Callback Correlation ID**: Generated per message in callback
   - Unique per Kafka message processed
   - Tracks message from receipt → parsing → dispatch → completion
   - Allows tracing individual message flow

3. **Envelope Correlation ID**: Generated for dispatcher (currently uses uuid4())
   - Future enhancement: Extract from Kafka message headers

---

## Logging Levels Used

| Level | Usage |
|-------|-------|
| **DEBUG** | Message parsing, envelope creation, callback entry/exit, validation errors (expected) |
| **INFO** | Dispatcher creation, subscription, event parsing success, routing, processing success |
| **WARNING** | Processing failures, JSON decode errors |
| **ERROR** | Unexpected exceptions (with full stack trace) |

---

## Structured Context Fields

**All log messages include**:
- `correlation_id`: Bootstrap or callback correlation ID
- `extra={}` dict with additional context

**Common context fields**:
- `handler_class`, `dispatcher_class`: Component identification
- `topic`, `consumer_group`: Kafka configuration
- `node_id`, `node_type`, `event_version`: Event payload details
- `envelope_correlation_id`: Dispatcher envelope tracking
- `*_duration_seconds`: Timing metrics
- `error_type`, `error_message`: Error context

---

## Debugging Workflow

With these logs, you can now trace:

1. **Dispatcher Initialization**:
   ```
   [DEBUG] Resolving HandlerNodeIntrospected from container
   [DEBUG] HandlerNodeIntrospected resolved successfully (handler_class=...)
   [INFO] Introspection dispatcher created and wired (dispatcher_class=...)
   ```

2. **Consumer Subscription**:
   ```
   [INFO] Subscribing to introspection events on Kafka (topic=..., consumer_group=...)
   [INFO] Introspection event consumer started successfully in 0.123s
   ```

3. **Message Processing**:
   ```
   [DEBUG] Introspection message callback invoked (message_offset=X, partition=Y)
   [DEBUG] Parsing message value as bytes (value_length=...)
   [DEBUG] Validating payload as ModelNodeIntrospectionEvent
   [INFO] Introspection event parsed successfully (node_id=..., node_type=...)
   [DEBUG] Event envelope created (envelope_correlation_id=...)
   [INFO] Routing to introspection dispatcher (node_id=...)
   [INFO] Introspection event processed successfully: node_id=X in 0.045s
   [DEBUG] Introspection message callback completed in 0.050s
   ```

4. **Pipeline Breakage Detection**:
   - **If no callback invocations**: Consumer not receiving messages → check Kafka topic/partition
   - **If parsing fails**: Log shows JSON decode error → check message format
   - **If validation fails**: Log shows ValidationError → check event schema
   - **If dispatcher fails**: Log shows processing failure → check handler logic

---

## Example Log Output (Expected)

```
2025-12-27 10:30:45 [INFO] omnibase_infra.runtime.kernel: Introspection dispatcher created and wired (correlation_id=abc-123, dispatcher_class=DispatcherNodeIntrospected, handler_class=HandlerNodeIntrospected)

2025-12-27 10:30:46 [INFO] omnibase_infra.runtime.kernel: Subscribing to introspection events on Kafka (correlation_id=abc-123, topic=requests, consumer_group=onex-runtime-introspection, event_bus_type=kafka)

2025-12-27 10:30:46 [INFO] omnibase_infra.runtime.kernel: Introspection event consumer started successfully in 0.123s (correlation_id=abc-123, topic=requests, consumer_group=onex-runtime-introspection, subscribe_duration_seconds=0.123)

2025-12-27 10:30:50 [DEBUG] omnibase_infra.runtime.kernel: Introspection message callback invoked (correlation_id=def-456, message_offset=42, message_partition=0, message_topic=requests)

2025-12-27 10:30:50 [DEBUG] omnibase_infra.runtime.kernel: Parsing message value as bytes (correlation_id=def-456, value_length=512)

2025-12-27 10:30:50 [DEBUG] omnibase_infra.runtime.kernel: Validating payload as ModelNodeIntrospectionEvent (correlation_id=def-456)

2025-12-27 10:30:50 [INFO] omnibase_infra.runtime.kernel: Introspection event parsed successfully (correlation_id=def-456, node_id=node-abc-123, node_type=ORCHESTRATOR, event_version=1.0.0)

2025-12-27 10:30:50 [DEBUG] omnibase_infra.runtime.kernel: Event envelope created (correlation_id=def-456, envelope_correlation_id=ghi-789, envelope_timestamp=2025-12-27T10:30:50.123456+00:00)

2025-12-27 10:30:50 [INFO] omnibase_infra.runtime.kernel: Routing to introspection dispatcher (correlation_id=def-456, envelope_correlation_id=ghi-789, node_id=node-abc-123)

2025-12-27 10:30:50 [INFO] omnibase_infra.runtime.kernel: Introspection event processed successfully: node_id=node-abc-123 in 0.045s (correlation_id=def-456, envelope_correlation_id=ghi-789, dispatcher_duration_seconds=0.045, node_id=node-abc-123, node_type=ORCHESTRATOR)

2025-12-27 10:30:50 [DEBUG] omnibase_infra.runtime.kernel: Introspection message callback completed in 0.050s (correlation_id=def-456, callback_duration_seconds=0.050)
```

---

## Next Steps

1. **Run E2E tests** with `ONEX_LOG_LEVEL=DEBUG` to see full pipeline trace
2. **Search logs** for correlation IDs to trace individual messages
3. **Identify breakage point** by finding last successful log message in pipeline
4. **Fix underlying issue** based on which stage failed

---

## Files Modified

- `src/omnibase_infra/runtime/kernel.py`: Added comprehensive structured logging

## Success Criteria Met

✅ Logging added to all critical points in event processing pipeline
✅ Correlation IDs propagated through all log messages
✅ Structured format using `extra={}` dict
✅ Clear indication of where events enter and exit each stage
✅ Exception logging includes full context
✅ No syntax errors or type issues introduced
✅ Follows ONEX infrastructure patterns
