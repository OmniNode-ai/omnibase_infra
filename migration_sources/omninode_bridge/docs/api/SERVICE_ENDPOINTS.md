# Service API Endpoints

## Overview

This document provides comprehensive documentation for all REST API endpoints exposed by OmniNode Bridge services. Each service exposes a set of standardized endpoints for health checks, metrics, and service-specific functionality.

## API Standards

### Common Headers

All API requests should include these headers:

```http
Content-Type: application/json
Accept: application/json
X-Request-ID: uuid4
X-Correlation-ID: uuid4
X-Service-Auth: service_auth_token  # If authentication enabled
Authorization: Bearer jwt_token     # For user authentication
```

### Standard Response Format

```json
{
  "success": true,
  "data": {},
  "error": null,
  "metadata": {
    "request_id": "uuid4",
    "timestamp": "ISO8601",
    "processing_time_ms": 150,
    "version": "1.0.0"
  }
}
```

### Error Response Format

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {},
    "trace_id": "uuid4"
  },
  "metadata": {
    "request_id": "uuid4",
    "timestamp": "ISO8601",
    "processing_time_ms": 50,
    "version": "1.0.0"
  }
}
```

## HookReceiver Service API

**Base URL**: `http://hook-receiver:8000`
**Purpose**: Receives and processes service lifecycle hooks

### Health and Status Endpoints

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "uptime_seconds": 86400,
    "version": "1.0.0",
    "checks": {
      "database": {
        "status": "healthy",
        "response_time_ms": 15
      },
      "kafka": {
        "status": "healthy",
        "response_time_ms": 8
      },
      "redis": {
        "status": "healthy",
        "response_time_ms": 3
      }
    }
  }
}
```

#### Readiness Check
```http
GET /ready
```

**Response**:
```json
{
  "success": true,
  "data": {
    "ready": true,
    "critical_services": {
      "database": "healthy",
      "kafka": "healthy"
    }
  }
}
```

#### Metrics
```http
GET /metrics
```

**Response**: Prometheus metrics format
```
# HELP hooks_processed_total Total hooks processed
# TYPE hooks_processed_total counter
hooks_processed_total{hook_type="service_started",status="success"} 1547
hooks_processed_total{hook_type="service_stopped",status="success"} 1543
```

### Hook Processing Endpoints

#### Process Service Lifecycle Hook
```http
POST /hooks/service/lifecycle
```

**Request Body**:
```json
{
  "hook_type": "service_started",
  "service_info": {
    "name": "omniagent",
    "version": "1.2.3",
    "instance_id": "omniagent-prod-001",
    "host": "10.0.1.15",
    "port": 8000,
    "capabilities": ["code_generation", "documentation"],
    "dependencies": [
      {
        "service": "omnimcp",
        "type": "required",
        "version": ">=1.0.0"
      }
    ]
  },
  "startup_info": {
    "startup_time_ms": 3500,
    "initialization_steps": [
      {
        "step": "load_configuration",
        "duration_ms": 150,
        "status": "success"
      }
    ]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "hook_id": "hook_550e8400-e29b-41d4-a716-446655440000",
    "processing_status": "completed",
    "intelligence_analysis": {
      "patterns_discovered": [
        {
          "pattern_type": "startup_performance",
          "confidence": 0.87,
          "description": "Fast startup pattern identified"
        }
      ],
      "recommendations": [
        {
          "type": "optimization",
          "description": "Consider pre-loading configuration for faster startup"
        }
      ]
    },
    "event_published": true,
    "processing_time_ms": 245
  }
}
```

#### Process Tool Registration Hook
```http
POST /hooks/tool/registration
```

**Request Body**:
```json
{
  "hook_type": "tool_discovered",
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
        "language": {"type": "string"}
      }
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "score": {"type": "number"},
        "issues": {"type": "array"}
      }
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "tool_registration_id": "tool_550e8400-e29b-41d4-a716-446655440000",
    "registration_status": "completed",
    "tool_indexed": true,
    "capabilities_analyzed": true,
    "event_published": true
  }
}
```

### Intelligence Endpoints

#### Get Intelligence Patterns
```http
GET /intelligence/patterns?service={service}&time_window={hours}&pattern_type={type}
```

**Query Parameters**:
- `service` (optional): Filter by service name
- `time_window` (optional): Time window in hours (default: 24)
- `pattern_type` (optional): Filter by pattern type

**Response**:
```json
{
  "success": true,
  "data": {
    "patterns": [
      {
        "pattern_id": "pattern_550e8400-e29b-41d4-a716-446655440000",
        "pattern_type": "service_communication",
        "confidence_score": 0.87,
        "discovery_time": "2024-01-15T10:30:00Z",
        "affected_services": ["omniplan", "omniagent"],
        "description": "High frequency communication pattern between planning and agent services",
        "recommendations": [
          {
            "type": "optimization",
            "description": "Consider implementing request batching"
          }
        ]
      }
    ],
    "total_patterns": 1,
    "analysis_window": {
      "start_time": "2024-01-14T10:30:00Z",
      "end_time": "2024-01-15T10:30:00Z"
    }
  }
}
```

## ToolCapture Proxy API

**Base URL**: `http://tool-capture-proxy:8000`
**Purpose**: Intelligent proxy for service-to-service communication with adaptive caching

### Proxy Endpoints

#### Proxy Tool Execution
```http
POST /proxy/tool/{tool_name}
```

**Request Body**:
```json
{
  "target_service": "omniagent",
  "input_data": {
    "code": "def hello(): return 'world'",
    "language": "python",
    "analysis_type": "quality"
  },
  "execution_options": {
    "timeout_seconds": 30,
    "priority": "normal",
    "cache_policy": "adaptive"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_550e8400-e29b-41d4-a716-446655440000",
    "result": {
      "score": 85,
      "issues": [
        {
          "type": "naming",
          "severity": "low",
          "message": "Function name could be more descriptive"
        }
      ],
      "recommendations": [
        "Consider adding type hints",
        "Add docstring for better documentation"
      ]
    },
    "execution_metadata": {
      "target_instance": "omniagent-prod-002",
      "execution_time_ms": 1250,
      "cache_hit": false,
      "routing_strategy": "load_balanced"
    }
  }
}
```

#### Proxy Health Check
```http
GET /proxy/health/{service_name}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "service_name": "omniagent",
    "instances": [
      {
        "instance_id": "omniagent-prod-001",
        "status": "healthy",
        "load": 0.45,
        "response_time_avg_ms": 120
      },
      {
        "instance_id": "omniagent-prod-002",
        "status": "healthy",
        "load": 0.32,
        "response_time_avg_ms": 95
      }
    ],
    "overall_health": "healthy",
    "load_balancing_active": true
  }
}
```

### Cache Management Endpoints

#### Get Cache Statistics
```http
GET /cache/stats
```

**Response**:
```json
{
  "success": true,
  "data": {
    "cache_statistics": {
      "total_entries": 15420,
      "hit_rate": 0.78,
      "miss_rate": 0.22,
      "eviction_rate": 0.05,
      "memory_usage_mb": 245,
      "memory_limit_mb": 512
    },
    "adaptive_caching": {
      "learning_enabled": true,
      "confidence_score": 0.85,
      "ttl_adjustments_today": 127,
      "performance_improvement": "23% faster avg response"
    }
  }
}
```

#### Clear Cache
```http
DELETE /cache/clear?pattern={pattern}
```

**Query Parameters**:
- `pattern` (optional): Cache key pattern to clear (default: all)

**Response**:
```json
{
  "success": true,
  "data": {
    "entries_cleared": 1547,
    "memory_freed_mb": 89,
    "operation_time_ms": 150
  }
}
```

#### Force Cache Entry
```http
PUT /cache/entry
```

**Request Body**:
```json
{
  "cache_key": "custom_key_12345",
  "value": {"result": "cached_data"},
  "ttl_seconds": 600,
  "tags": ["manual", "high_priority"]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "cache_key": "custom_key_12345",
    "cached": true,
    "ttl_seconds": 600,
    "size_bytes": 156
  }
}
```

### Circuit Breaker Endpoints

#### Get Circuit Breaker Status
```http
GET /circuit-breaker/status
```

**Response**:
```json
{
  "success": true,
  "data": {
    "circuit_breakers": [
      {
        "service": "omniagent",
        "state": "closed",
        "failure_count": 2,
        "failure_threshold": 5,
        "success_count": 145,
        "last_failure_time": "2024-01-15T09:45:00Z"
      },
      {
        "service": "omnimemory",
        "state": "half_open",
        "failure_count": 5,
        "failure_threshold": 5,
        "test_requests_sent": 3,
        "circuit_opened_at": "2024-01-15T10:20:00Z"
      }
    ]
  }
}
```

#### Reset Circuit Breaker
```http
POST /circuit-breaker/reset/{service_name}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "service": "omniagent",
    "previous_state": "open",
    "new_state": "closed",
    "reset_time": "2024-01-15T10:35:00Z",
    "failure_count_reset": true
  }
}
```

## Service Registry API

**Base URL**: `http://service-registry:8000`
**Purpose**: Dynamic service discovery and health monitoring

### Service Registration Endpoints

#### Register Service
```http
POST /registry/services
```

**Request Body**:
```json
{
  "service_name": "omniagent",
  "version": "1.2.3",
  "instance_id": "omniagent-prod-003",
  "host": "10.0.1.18",
  "port": 8000,
  "health_check_url": "http://10.0.1.18:8000/health",
  "capabilities": ["code_generation", "documentation"],
  "metadata": {
    "environment": "production",
    "region": "us-west-2",
    "deployment_id": "deploy_20240115_103000"
  },
  "ttl_seconds": 60
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "registration_id": "reg_550e8400-e29b-41d4-a716-446655440000",
    "service_name": "omniagent",
    "instance_id": "omniagent-prod-003",
    "registered_at": "2024-01-15T10:30:00Z",
    "expires_at": "2024-01-15T10:31:00Z",
    "consul_registered": true
  }
}
```

#### Update Service Registration
```http
PUT /registry/services/{instance_id}
```

**Request Body**:
```json
{
  "health_status": "healthy",
  "load_factor": 0.45,
  "metadata": {
    "current_connections": 23,
    "avg_response_time_ms": 120
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "instance_id": "omniagent-prod-003",
    "updated_at": "2024-01-15T10:35:00Z",
    "health_status": "healthy",
    "consul_updated": true
  }
}
```

#### Deregister Service
```http
DELETE /registry/services/{instance_id}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "instance_id": "omniagent-prod-003",
    "deregistered_at": "2024-01-15T10:40:00Z",
    "consul_deregistered": true
  }
}
```

### Service Discovery Endpoints

#### Discover Services
```http
GET /registry/services?service_name={name}&healthy_only={bool}&tags={tags}
```

**Query Parameters**:
- `service_name` (optional): Filter by service name
- `healthy_only` (optional): Return only healthy instances (default: true)
- `tags` (optional): Comma-separated list of required tags

**Response**:
```json
{
  "success": true,
  "data": {
    "services": [
      {
        "service_name": "omniagent",
        "instances": [
          {
            "instance_id": "omniagent-prod-001",
            "host": "10.0.1.15",
            "port": 8000,
            "health_status": "healthy",
            "version": "1.2.3",
            "capabilities": ["code_generation", "documentation"],
            "load_factor": 0.45,
            "last_heartbeat": "2024-01-15T10:34:30Z",
            "metadata": {
              "environment": "production",
              "region": "us-west-2"
            }
          }
        ]
      }
    ],
    "total_services": 1,
    "total_instances": 1,
    "query_time_ms": 15
  }
}
```

#### Get Service Health
```http
GET /registry/services/{service_name}/health
```

**Response**:
```json
{
  "success": true,
  "data": {
    "service_name": "omniagent",
    "overall_health": "healthy",
    "instance_count": 3,
    "healthy_instances": 3,
    "unhealthy_instances": 0,
    "instances": [
      {
        "instance_id": "omniagent-prod-001",
        "health_status": "healthy",
        "last_check": "2024-01-15T10:35:00Z",
        "response_time_ms": 15,
        "health_details": {
          "database": "healthy",
          "memory_usage": "normal",
          "cpu_usage": "normal"
        }
      }
    ]
  }
}
```

### Tool Registry Endpoints

#### Register Tool
```http
POST /registry/tools
```

**Request Body**:
```json
{
  "tool_name": "code_analyzer",
  "version": "2.1.0",
  "service_source": "omniagent",
  "category": "code_analysis",
  "description": "Analyze code quality and patterns",
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {"type": "string"},
      "language": {"type": "string"}
    },
    "required": ["code", "language"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "score": {"type": "number"},
      "issues": {"type": "array"}
    }
  },
  "execution_requirements": {
    "max_execution_time_ms": 30000,
    "memory_limit_mb": 512
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "tool_id": "tool_550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "code_analyzer",
    "registered_at": "2024-01-15T10:30:00Z",
    "registry_entry_created": true
  }
}
```

#### Discover Tools
```http
GET /registry/tools?category={category}&service={service}&capability={capability}
```

**Query Parameters**:
- `category` (optional): Filter by tool category
- `service` (optional): Filter by source service
- `capability` (optional): Filter by capability

**Response**:
```json
{
  "success": true,
  "data": {
    "tools": [
      {
        "tool_id": "tool_550e8400-e29b-41d4-a716-446655440000",
        "tool_name": "code_analyzer",
        "version": "2.1.0",
        "service_source": "omniagent",
        "category": "code_analysis",
        "description": "Analyze code quality and patterns",
        "available_instances": [
          "omniagent-prod-001",
          "omniagent-prod-002"
        ],
        "avg_execution_time_ms": 1250,
        "success_rate": 0.98
      }
    ],
    "total_tools": 1
  }
}
```

## Common Administrative Endpoints

All services expose these standard administrative endpoints:

### Configuration Endpoints

#### Get Configuration
```http
GET /admin/config
```

**Response**:
```json
{
  "success": true,
  "data": {
    "configuration": {
      "service_name": "hook-receiver",
      "version": "1.0.0",
      "environment": "production",
      "database": {
        "host": "postgres",
        "database": "omninode_bridge",
        "max_connections": 20
      },
      "kafka": {
        "bootstrap_servers": "redpanda:9092",
        "consumer_group": "hook-receiver-group"
      }
    },
    "last_updated": "2024-01-15T10:00:00Z"
  }
}
```

#### Update Configuration
```http
PUT /admin/config
```

**Request Body**:
```json
{
  "database.max_connections": 30,
  "kafka.consumer_group": "hook-receiver-group-v2"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "updated_keys": [
      "database.max_connections",
      "kafka.consumer_group"
    ],
    "restart_required": true,
    "updated_at": "2024-01-15T10:35:00Z"
  }
}
```

### Debug Endpoints

#### Get Service Information
```http
GET /admin/info
```

**Response**:
```json
{
  "success": true,
  "data": {
    "service": {
      "name": "hook-receiver",
      "version": "1.0.0",
      "build_time": "2024-01-15T08:00:00Z",
      "git_commit": "a1b2c3d4",
      "environment": "production"
    },
    "runtime": {
      "uptime_seconds": 86400,
      "memory_usage_mb": 245,
      "cpu_usage_percent": 12.5,
      "active_connections": 15
    },
    "dependencies": {
      "database": "connected",
      "kafka": "connected",
      "consul": "connected",
      "redis": "connected"
    }
  }
}
```

#### Get Logs
```http
GET /admin/logs?level={level}&lines={count}&since={timestamp}
```

**Query Parameters**:
- `level` (optional): Filter by log level (DEBUG, INFO, WARN, ERROR)
- `lines` (optional): Number of recent log lines (default: 100)
- `since` (optional): ISO timestamp to get logs since

**Response**:
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "timestamp": "2024-01-15T10:35:00Z",
        "level": "INFO",
        "message": "Hook processed successfully",
        "module": "hook_processor",
        "extra": {
          "hook_id": "hook_12345",
          "processing_time_ms": 245
        }
      }
    ],
    "total_lines": 1,
    "query_time_ms": 50
  }
}
```

This comprehensive API documentation provides complete reference for all service endpoints in the OmniNode Bridge, enabling efficient integration and management of the bridge services.
