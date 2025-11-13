# Event Schemas and Message Formats

## Overview

This document defines all event schemas, message formats, and data structures used in the OmniNode Bridge ecosystem. The bridge uses a strongly-typed event system with JSON Schema validation to ensure reliable communication between services.

## Schema Conventions

### Base Event Structure

All events in the OmniNode Bridge follow a standard envelope format:

```json
{
  "event_id": "uuid4",
  "event_type": "string",
  "timestamp": "ISO8601",
  "version": "semver",
  "source": {
    "service_name": "string",
    "service_version": "string",
    "instance_id": "string"
  },
  "correlation_id": "uuid4",
  "trace_id": "string",
  "metadata": {},
  "payload": {}
}
```

### Schema Validation

```python
# schemas/base_schema.py
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import UUID

class EventSource(BaseModel):
    service_name: str = Field(..., description="Name of the originating service")
    service_version: str = Field(..., description="Version of the originating service")
    instance_id: str = Field(..., description="Unique instance identifier")

class BaseEvent(BaseModel):
    event_id: UUID = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event creation timestamp")
    version: str = Field("1.0.0", description="Event schema version")
    source: EventSource = Field(..., description="Event source information")
    correlation_id: Optional[UUID] = Field(None, description="Request correlation ID")
    trace_id: Optional[str] = Field(None, description="Distributed trace ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    payload: Dict[str, Any] = Field(..., description="Event-specific payload")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
```

## Hook Event Schemas

### Service Lifecycle Events

#### Service Started Event

```json
{
  "event_type": "service.lifecycle.started",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "version": "1.2.3",
      "type": "agent_service",
      "instance_id": "omniagent-prod-001",
      "host": "10.0.1.15",
      "port": 8000,
      "health_check_url": "http://10.0.1.15:8000/health",
      "capabilities": [
        "code_generation",
        "documentation",
        "testing"
      ],
      "dependencies": [
        {
          "service": "omnimcp",
          "type": "required",
          "version": ">=1.0.0"
        }
      ],
      "resources": {
        "cpu_limit": "2000m",
        "memory_limit": "4Gi",
        "disk_usage": "10Gi"
      }
    },
    "startup_info": {
      "startup_time_ms": 3500,
      "initialization_steps": [
        {
          "step": "load_configuration",
          "duration_ms": 150,
          "status": "success"
        },
        {
          "step": "connect_database",
          "duration_ms": 250,
          "status": "success"
        },
        {
          "step": "register_with_consul",
          "duration_ms": 100,
          "status": "success"
        }
      ]
    }
  }
}
```

**Schema Definition**:
```python
class ServiceInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+')
    type: str = Field(..., description="Service type classification")
    instance_id: str = Field(..., description="Unique instance identifier")
    host: str = Field(..., description="Service host address")
    port: int = Field(..., ge=1, le=65535)
    health_check_url: str = Field(..., description="Health check endpoint URL")
    capabilities: List[str] = Field(default_factory=list)
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, str] = Field(default_factory=dict)

class StartupStep(BaseModel):
    step: str
    duration_ms: int = Field(..., ge=0)
    status: str = Field(..., regex=r'^(success|failed|warning)$')
    details: Optional[Dict[str, Any]] = None

class StartupInfo(BaseModel):
    startup_time_ms: int = Field(..., ge=0)
    initialization_steps: List[StartupStep] = Field(default_factory=list)

class ServiceStartedPayload(BaseModel):
    service_info: ServiceInfo
    startup_info: StartupInfo
```

#### Service Stopped Event

```json
{
  "event_type": "service.lifecycle.stopped",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "instance_id": "omniagent-prod-001"
    },
    "shutdown_info": {
      "shutdown_reason": "graceful_shutdown",
      "uptime_seconds": 86400,
      "final_metrics": {
        "requests_processed": 15420,
        "errors_encountered": 23,
        "avg_response_time_ms": 150
      },
      "cleanup_steps": [
        {
          "step": "drain_connections",
          "duration_ms": 2000,
          "status": "success"
        },
        {
          "step": "save_state",
          "duration_ms": 500,
          "status": "success"
        }
      ]
    }
  }
}
```

#### Service Health Changed Event

```json
{
  "event_type": "service.lifecycle.health_changed",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "instance_id": "omniagent-prod-001"
    },
    "health_change": {
      "previous_status": "healthy",
      "current_status": "degraded",
      "change_reason": "high_error_rate",
      "health_details": {
        "database_connection": "healthy",
        "external_apis": "degraded",
        "memory_usage": "warning",
        "cpu_usage": "healthy"
      },
      "recovery_actions": [
        "reduce_traffic",
        "restart_external_connections"
      ]
    }
  }
}
```

### Tool Registration Events

#### Tool Discovered Event

```json
{
  "event_type": "tool.registration.discovered",
  "payload": {
    "tool_info": {
      "name": "code_analyzer",
      "version": "2.1.0",
      "service_source": "omniagent",
      "category": "code_analysis",
      "description": "Analyze code quality and patterns",
      "input_schema": {
        "type": "object",
        "properties": {
          "code": {"type": "string"},
          "language": {"type": "string"},
          "analysis_type": {"type": "string", "enum": ["quality", "security", "performance"]}
        },
        "required": ["code", "language"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "score": {"type": "number", "minimum": 0, "maximum": 100},
          "issues": {"type": "array"},
          "recommendations": {"type": "array"}
        }
      },
      "execution_requirements": {
        "max_execution_time_ms": 30000,
        "memory_limit_mb": 512,
        "requires_filesystem": true,
        "requires_network": false
      }
    },
    "discovery_context": {
      "discovery_method": "service_registration",
      "discovery_timestamp": "2024-01-15T10:30:00Z",
      "registration_source": "automatic"
    }
  }
}
```

#### Tool Executed Event

```json
{
  "event_type": "tool.execution.completed",
  "payload": {
    "execution_info": {
      "tool_name": "code_analyzer",
      "execution_id": "exec_550e8400-e29b-41d4-a716-446655440000",
      "requester_service": "omniplan",
      "target_service": "omniagent"
    },
    "execution_details": {
      "start_time": "2024-01-15T10:35:00Z",
      "end_time": "2024-01-15T10:35:02.5Z",
      "duration_ms": 2500,
      "status": "success",
      "input_size_bytes": 15420,
      "output_size_bytes": 3240
    },
    "performance_metrics": {
      "cpu_usage_percent": 45.2,
      "memory_peak_mb": 256,
      "io_operations": 127,
      "network_calls": 0
    },
    "intelligence_data": {
      "patterns_identified": [
        {
          "pattern_type": "code_quality",
          "confidence": 0.92,
          "description": "High complexity function detected"
        }
      ],
      "recommendations": [
        {
          "type": "refactoring",
          "priority": "medium",
          "description": "Consider breaking down large function"
        }
      ]
    }
  }
}
```

## Proxy Event Schemas

### Request Routing Events

#### Request Routed Event

```json
{
  "event_type": "proxy.request.routed",
  "payload": {
    "request_info": {
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "source_service": "omniplan",
      "target_service": "omniagent",
      "tool_name": "generate_code",
      "method": "POST",
      "path": "/tools/generate_code"
    },
    "routing_decision": {
      "routing_strategy": "load_balanced",
      "selected_instance": "omniagent-prod-002",
      "routing_factors": {
        "cpu_utilization": 0.45,
        "current_load": 23,
        "response_time_avg": 120,
        "health_score": 0.95
      },
      "alternative_instances": [
        "omniagent-prod-001",
        "omniagent-prod-003"
      ]
    },
    "caching_info": {
      "cache_key": "hash_of_request_content",
      "cache_hit": false,
      "cache_ttl_seconds": 300,
      "cacheable": true
    }
  }
}
```

#### Cache Event

```json
{
  "event_type": "proxy.cache.hit",
  "payload": {
    "cache_info": {
      "cache_key": "hash_of_request_content",
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "cached_at": "2024-01-15T10:30:00Z",
      "ttl_remaining_seconds": 180,
      "cache_size_bytes": 5240
    },
    "performance_impact": {
      "response_time_saved_ms": 1800,
      "bandwidth_saved_bytes": 5240,
      "cpu_saved_percent": 25.0
    },
    "adaptive_caching": {
      "hit_rate_trend": 0.78,
      "ttl_adjustment": "extend",
      "confidence_score": 0.85
    }
  }
}
```

## Intelligence Event Schemas

### Pattern Discovery Events

#### Intelligence Pattern Discovered Event

```json
{
  "event_type": "intelligence.pattern.discovered",
  "payload": {
    "pattern_info": {
      "pattern_id": "pattern_550e8400-e29b-41d4-a716-446655440000",
      "pattern_type": "service_communication",
      "confidence_score": 0.87,
      "discovery_method": "statistical_analysis",
      "pattern_data": {
        "source_services": ["omniplan", "omnimemory"],
        "target_service": "omniagent",
        "frequency": "high",
        "success_rate": 0.94,
        "avg_response_time": 145,
        "peak_usage_hours": ["09:00-11:00", "14:00-16:00"]
      }
    },
    "analysis_context": {
      "analysis_window": {
        "start_time": "2024-01-14T00:00:00Z",
        "end_time": "2024-01-15T00:00:00Z",
        "sample_size": 1547
      },
      "data_sources": [
        "hook_events",
        "proxy_logs",
        "performance_metrics"
      ]
    },
    "actionable_insights": {
      "optimization_opportunities": [
        {
          "type": "caching",
          "potential_improvement": "25% faster response",
          "implementation_effort": "low"
        },
        {
          "type": "load_balancing",
          "potential_improvement": "improved resource utilization",
          "implementation_effort": "medium"
        }
      ],
      "risk_indicators": [
        {
          "type": "single_point_of_failure",
          "severity": "medium",
          "mitigation": "add service redundancy"
        }
      ]
    }
  }
}
```

### Performance Analytics Events

#### Performance Anomaly Detected Event

```json
{
  "event_type": "intelligence.performance.anomaly_detected",
  "payload": {
    "anomaly_info": {
      "anomaly_id": "anomaly_550e8400-e29b-41d4-a716-446655440000",
      "detection_time": "2024-01-15T10:45:00Z",
      "anomaly_type": "response_time_spike",
      "severity": "high",
      "affected_services": ["omniagent"],
      "affected_tools": ["code_analyzer", "documentation_generator"]
    },
    "metrics_data": {
      "baseline_metrics": {
        "avg_response_time_ms": 150,
        "p95_response_time_ms": 300,
        "error_rate": 0.02,
        "throughput_rps": 45
      },
      "anomaly_metrics": {
        "avg_response_time_ms": 1200,
        "p95_response_time_ms": 2500,
        "error_rate": 0.15,
        "throughput_rps": 12
      },
      "degradation_factor": 8.0
    },
    "root_cause_analysis": {
      "probable_causes": [
        {
          "cause": "database_connection_pool_exhaustion",
          "confidence": 0.85,
          "evidence": [
            "high_connection_wait_times",
            "database_error_spike"
          ]
        },
        {
          "cause": "memory_pressure",
          "confidence": 0.65,
          "evidence": [
            "gc_pressure_increase",
            "memory_usage_spike"
          ]
        }
      ],
      "recommended_actions": [
        {
          "action": "increase_database_connection_pool",
          "priority": "high",
          "estimated_impact": "70% improvement"
        },
        {
          "action": "restart_service_instances",
          "priority": "medium",
          "estimated_impact": "temporary_relief"
        }
      ]
    }
  }
}
```

## Configuration Event Schemas

### Configuration Change Events

#### Configuration Updated Event

```json
{
  "event_type": "configuration.updated",
  "payload": {
    "change_info": {
      "configuration_key": "service.omniagent.worker_pool_size",
      "previous_value": "4",
      "new_value": "8",
      "change_source": "consul_kv_update",
      "changed_by": "system_administrator",
      "change_reason": "performance_optimization"
    },
    "change_impact": {
      "affected_services": ["omniagent"],
      "restart_required": true,
      "estimated_downtime_seconds": 30,
      "backward_compatible": true
    },
    "rollback_info": {
      "rollback_available": true,
      "rollback_window_hours": 24,
      "rollback_steps": [
        "revert_consul_key",
        "restart_affected_services"
      ]
    }
  }
}
```

## Error Event Schemas

### System Error Events

#### Service Error Event

```json
{
  "event_type": "error.service.critical",
  "payload": {
    "error_info": {
      "error_id": "error_550e8400-e29b-41d4-a716-446655440000",
      "service_name": "omniagent",
      "instance_id": "omniagent-prod-001",
      "error_type": "database_connection_failure",
      "severity": "critical",
      "message": "Failed to connect to PostgreSQL database after 3 retries",
      "stack_trace": "Traceback...",
      "first_occurrence": "2024-01-15T10:50:00Z",
      "occurrence_count": 5
    },
    "context_info": {
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user_12345",
      "operation": "tool_execution",
      "tool_name": "code_analyzer",
      "environment": "production"
    },
    "impact_assessment": {
      "affected_users": 25,
      "affected_operations": ["code_analysis", "documentation_generation"],
      "estimated_recovery_time_minutes": 15,
      "business_impact": "medium"
    },
    "recovery_actions": {
      "automatic_recovery_attempted": true,
      "recovery_steps": [
        "reconnect_database",
        "fallback_to_readonly_replica"
      ],
      "manual_intervention_required": false
    }
  }
}
```

## Event Validation and Processing

### Event Processor Implementation

```python
# event_processing/processor.py
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError
import logging

class EventProcessor:
    def __init__(self):
        self.schema_registry: Dict[str, Type[BaseModel]] = {}
        self.event_handlers: Dict[str, list] = {}
        self.logger = logging.getLogger(__name__)

    def register_schema(self, event_type: str, schema_class: Type[BaseModel]):
        """Register event schema for validation"""
        self.schema_registry[event_type] = schema_class

    def register_handler(self, event_type: str, handler_func):
        """Register event handler function"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler_func)

    async def process_event(self, raw_event: Dict[str, Any]) -> bool:
        """Process and validate incoming event"""
        try:
            # Validate base event structure
            base_event = BaseEvent(**raw_event)

            # Validate event-specific payload
            event_type = base_event.event_type
            if event_type in self.schema_registry:
                schema_class = self.schema_registry[event_type]
                validated_payload = schema_class(**base_event.payload)
                base_event.payload = validated_payload.dict()

            # Process with registered handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(base_event)

            self.logger.info(f"Successfully processed event {base_event.event_id}")
            return True

        except ValidationError as e:
            self.logger.error(f"Event validation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Event processing failed: {e}")
            return False

# Schema registration
processor = EventProcessor()

# Register service lifecycle schemas
processor.register_schema("service.lifecycle.started", ServiceStartedPayload)
processor.register_schema("service.lifecycle.stopped", ServiceStoppedPayload)
processor.register_schema("service.lifecycle.health_changed", ServiceHealthChangedPayload)

# Register tool schemas
processor.register_schema("tool.registration.discovered", ToolDiscoveredPayload)
processor.register_schema("tool.execution.completed", ToolExecutedPayload)

# Register proxy schemas
processor.register_schema("proxy.request.routed", RequestRoutedPayload)
processor.register_schema("proxy.cache.hit", CacheHitPayload)

# Register intelligence schemas
processor.register_schema("intelligence.pattern.discovered", PatternDiscoveredPayload)
processor.register_schema("intelligence.performance.anomaly_detected", PerformanceAnomalyPayload)
```

### Event Publishing

```python
# event_processing/publisher.py
import json
import asyncio
from aiokafka import AIOKafkaProducer
from typing import Dict, Any
import uuid
from datetime import datetime

class EventPublisher:
    def __init__(self, bootstrap_servers: str):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    async def start(self):
        await self.producer.start()

    async def stop(self):
        await self.producer.stop()

    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source_service: str,
        source_version: str,
        instance_id: str,
        correlation_id: str = None,
        trace_id: str = None
    ):
        """Publish event to Kafka topic"""

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "source": {
                "service_name": source_service,
                "service_version": source_version,
                "instance_id": instance_id
            },
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "metadata": {},
            "payload": payload
        }

        # Determine topic based on event type
        topic = self._get_topic_for_event(event_type)

        try:
            await self.producer.send_and_wait(topic, event)
            return event["event_id"]
        except Exception as e:
            raise Exception(f"Failed to publish event: {e}")

    def _get_topic_for_event(self, event_type: str) -> str:
        """Map event type to Kafka topic"""
        topic_mapping = {
            "service.lifecycle": "hooks.service_lifecycle",
            "tool.registration": "hooks.tool_registration",
            "tool.execution": "hooks.tool_execution",
            "proxy.request": "proxy.requests",
            "proxy.cache": "proxy.cache",
            "intelligence.pattern": "intelligence.patterns",
            "intelligence.performance": "intelligence.performance",
            "configuration": "configuration.changes",
            "error": "system.errors"
        }

        for prefix, topic in topic_mapping.items():
            if event_type.startswith(prefix):
                return topic

        return "events.general"
```

This comprehensive event schema documentation provides a complete reference for all message formats used in the OmniNode Bridge, ensuring consistent, validated, and well-structured communication between services.
