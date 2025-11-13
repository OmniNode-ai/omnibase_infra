# MetadataStampingService API Documentation

## Overview

High-performance metadata stamping service implementing **O.N.E. v0.1 protocol compliance** with sub-2ms BLAKE3 hash generation, namespace support, unified response format, and Kafka event publishing.

**Base URL**: `http://localhost:8053/api/v1/metadata-stamping`

## Authentication & Headers

### Standard Headers

```http
Content-Type: application/json
Accept: application/json
X-Request-ID: uuid4
X-Correlation-ID: uuid4
X-Namespace: omninode.services.metadata  # Optional namespace override
```

### O.N.E. v0.1 Compliance Headers

```http
X-Protocol-Version: 1.0
X-Metadata-Version: 0.1
X-Operation-ID: uuid4  # Auto-generated if not provided
```

## Unified Response Format

All endpoints return the standardized `UnifiedResponse` format:

```json
{
  "status": "success|error|partial",
  "data": { /* Response data */ },
  "error": [ /* Error details array */ ],
  "message": "Operation completed successfully",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "operation": "create_stamp",
    "protocol_version": "1.0",
    "metadata_version": "0.1",
    "execution_time_ms": 1.5,
    "op_id": "uuid4"
  }
}
```

### Error Response Format

```json
{
  "status": "error",
  "data": null,
  "error": [
    {
      "code": "VALIDATION_ERROR",
      "field": "content",
      "message": "Content cannot be empty"
    }
  ],
  "message": "Validation failed",
  "metadata": {
    "operation": "create_stamp",
    "status_code": 422
  }
}
```

## Core Endpoints

### Create Metadata Stamp

Generate cryptographic metadata stamps with O.N.E. v0.1 compliance.

```http
POST /stamp
```

**Request Body**:
```json
{
  "content": "Content to be stamped",
  "file_path": "/optional/file/path.txt",
  "options": {
    "stamp_type": "lightweight|rich",
    "include_metadata": true,
    "validate_integrity": true
  },
  "metadata": {
    "author": "user123",
    "purpose": "document_verification"
  },
  "protocol_version": "1.0",
  "namespace": "omninode.services.metadata",
  "intelligence_data": {
    "context": "automated_processing",
    "priority": "high"
  },
  "version": 1,
  "op_id": "custom-operation-id",
  "metadata_version": "0.1"
}
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "success": true,
    "stamp_id": "stamp_550e8400-e29b-41d4-a716-446655440000",
    "file_hash": "a1b2c3d4e5f6789...",
    "stamped_content": "Content to be stamped\n<!-- METADATA_STAMP:...",
    "stamp": "<!-- METADATA_STAMP:lightweight:blake3:a1b2c3d4... -->",
    "stamp_type": "lightweight",
    "performance_metrics": {
      "execution_time_ms": 1.2,
      "file_size_bytes": 256,
      "cpu_usage_percent": 5.2,
      "performance_grade": "A"
    },
    "created_at": "2025-09-28T10:30:00Z",
    "op_id": "custom-operation-id",
    "namespace": "omninode.services.metadata",
    "version": 1,
    "metadata_version": "0.1"
  },
  "message": "Metadata stamp created successfully",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "protocol_version": "1.0",
    "operation": "create_stamp",
    "execution_time_ms": 2.1
  }
}
```

### Validate Stamps

Validate existing metadata stamps in content with namespace filtering.

```http
POST /validate
```

**Request Body**:
```json
{
  "content": "Content with stamps to validate",
  "options": {
    "strict_mode": true,
    "expected_hash": "a1b2c3d4e5f6789...",
    "check_tampering": true
  },
  "namespace": "omninode.services.metadata"
}
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "success": true,
    "is_valid": true,
    "stamps_found": 3,
    "current_hash": "a1b2c3d4e5f6789...",
    "validation_details": [
      {
        "stamp_type": "lightweight",
        "stamp_hash": "a1b2c3d4e5f6789...",
        "is_valid": true,
        "current_hash": "a1b2c3d4e5f6789..."
      }
    ],
    "performance_metrics": {
      "execution_time_ms": 5.2,
      "file_size_bytes": 1024,
      "performance_grade": "A"
    }
  },
  "message": "Stamp validation completed successfully",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "validation_summary": {
      "valid": true,
      "stamps_found": 3
    },
    "operation": "validate_stamps"
  }
}
```

### Generate Hash

Generate BLAKE3 hash for files with performance optimization.

```http
POST /hash
```

**Request Body**: `multipart/form-data`
- `file`: File to hash
- `namespace` (optional): Namespace (default: "default")

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "file_hash": "a1b2c3d4e5f6789abc...",
    "execution_time_ms": 0.8,
    "file_size_bytes": 2048,
    "performance_grade": "A"
  },
  "message": "File hash generated successfully",
  "metadata": {
    "namespace": "default",
    "filename": "document.pdf",
    "operation": "generate_hash"
  }
}
```

### Retrieve Stamp

Retrieve existing metadata stamp by file hash with namespace filtering.

```http
GET /stamp/{file_hash}?namespace={namespace}
```

**Path Parameters**:
- `file_hash`: BLAKE3 hash (64-character hexadecimal)

**Query Parameters**:
- `namespace` (optional): Filter by namespace

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "id": "stamp_550e8400-e29b-41d4-a716-446655440000",
    "file_hash": "a1b2c3d4e5f6789...",
    "file_path": "/path/to/file.txt",
    "file_size": 1024,
    "content_type": "text/plain",
    "stamp_data": {
      "stamp_type": "lightweight",
      "stamp": "<!-- METADATA_STAMP:... -->",
      "metadata": {
        "author": "user123"
      },
      "namespace": "omninode.services.metadata"
    },
    "protocol_version": "1.0",
    "intelligence_data": {},
    "version": 1,
    "op_id": "uuid4",
    "namespace": "omninode.services.metadata",
    "metadata_version": "0.1",
    "created_at": "2025-09-28T10:30:00Z"
  },
  "message": "Metadata stamp retrieved successfully",
  "metadata": {
    "file_hash": "a1b2c3d4e5f6789...",
    "namespace": "omninode.services.metadata",
    "operation": "get_stamp"
  }
}
```

## Batch Operations

### Batch Stamp Operations

Process multiple stamping operations for high throughput.

```http
POST /batch
```

**Request Body**:
```json
{
  "items": [
    {
      "id": "item1",
      "content": "First content to stamp",
      "file_path": "/path/to/file1.txt",
      "namespace": "test",
      "metadata": {"type": "document"}
    },
    {
      "id": "item2",
      "content": "Second content to stamp",
      "namespace": "production"
    }
  ],
  "options": {
    "stamp_type": "lightweight",
    "include_metadata": true,
    "validate_integrity": true
  },
  "protocol_version": "1.0",
  "namespace": "omninode.services.metadata"
}
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "total_items": 2,
    "successful_items": 2,
    "failed_items": 0,
    "results": [
      {
        "id": "item1",
        "success": true,
        "stamp_id": "stamp_uuid1",
        "file_hash": "hash1...",
        "stamp": "<!-- METADATA_STAMP:... -->",
        "performance_metrics": {
          "execution_time_ms": 1.1,
          "file_size_bytes": 256,
          "performance_grade": "A"
        }
      },
      {
        "id": "item2",
        "success": true,
        "stamp_id": "stamp_uuid2",
        "file_hash": "hash2...",
        "stamp": "<!-- METADATA_STAMP:... -->",
        "performance_metrics": {
          "execution_time_ms": 0.9,
          "file_size_bytes": 512,
          "performance_grade": "A"
        }
      }
    ],
    "overall_performance": {
      "execution_time_ms": 45.2,
      "file_size_bytes": 768,
      "performance_grade": "A"
    }
  },
  "message": "Batch processing completed: 2/2 successful",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "protocol_version": "1.0",
    "operation": "batch_stamp"
  }
}
```

## Protocol Compliance

### Validate Protocol Compliance

Validate content against O.N.E. v0.1 protocol standards.

```http
POST /validate-protocol
```

**Request Body**:
```json
{
  "content": "Content to validate against protocol",
  "target_protocol": "O.N.E.v0.1",
  "validation_level": "strict|moderate|lenient",
  "namespace": "omninode.services.metadata"
}
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "validation_result": {
      "is_valid": true,
      "protocol_version": "1.0",
      "compliance_level": "full",
      "issues": [],
      "recommendations": [
        "Consider adding explicit namespace declaration"
      ]
    },
    "performance_metrics": {
      "execution_time_ms": 3.2,
      "file_size_bytes": 1024,
      "performance_grade": "A"
    }
  },
  "message": "Protocol validation completed: full compliance",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "target_protocol": "O.N.E.v0.1",
    "validation_level": "strict",
    "operation": "validate_protocol"
  }
}
```

## Namespace Management

### Query Namespace Stamps

Query stamps within a specific namespace with pagination.

```http
GET /namespace/{namespace}?limit={limit}&offset={offset}
```

**Path Parameters**:
- `namespace`: Namespace to query

**Query Parameters**:
- `limit` (optional): Maximum results (1-1000, default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "namespace": "omninode.services.metadata",
    "total_stamps": 150,
    "stamps": [
      {
        "stamp_id": "stamp_uuid1",
        "file_hash": "hash1...",
        "file_path": "/path/to/file1.txt",
        "stamp_type": "lightweight",
        "created_at": "2025-09-28T10:30:00Z",
        "metadata": {"author": "user123"},
        "op_id": "op_uuid1",
        "namespace": "omninode.services.metadata",
        "version": 1,
        "metadata_version": "0.1",
        "intelligence_data": {}
      }
    ],
    "pagination": {
      "limit": 50,
      "offset": 0,
      "total": 150,
      "has_more": true
    }
  },
  "message": "Namespace query completed: 1 stamps found",
  "metadata": {
    "namespace": "omninode.services.metadata",
    "pagination": {
      "limit": 50,
      "offset": 0,
      "total": 150
    },
    "operation": "query_namespace"
  }
}
```

## Health & Monitoring

### Health Check

Comprehensive service health status with component monitoring.

```http
GET /health
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "components": {
      "service": {
        "status": "healthy",
        "response_time_ms": 2.1,
        "details": {
          "version": "0.1.0",
          "uptime_seconds": 86400
        }
      },
      "database": {
        "status": "healthy",
        "response_time_ms": 3.5,
        "details": {
          "connection_pool": "20/50 active",
          "query_performance": "optimal"
        }
      },
      "hash_engine": {
        "status": "healthy",
        "response_time_ms": 0.8,
        "details": {
          "pool_utilization": "15/100",
          "avg_hash_time_ms": 0.9
        }
      }
    },
    "uptime_seconds": 86400,
    "version": "0.1.0"
  },
  "message": "Health check completed: healthy",
  "metadata": {
    "operation": "health_check",
    "components_checked": 3
  }
}
```

### Performance Metrics

Detailed performance metrics with namespace filtering.

```http
GET /metrics?namespace={namespace}
```

**Response** (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "hash_generation": {
      "total_operations": 15420,
      "avg_execution_time_ms": 0.95,
      "p99_execution_time_ms": 1.8,
      "performance_grade_distribution": {
        "A": 0.87,
        "B": 0.11,
        "C": 0.02
      },
      "throughput_ops_per_second": 1250
    },
    "database": {
      "total_queries": 8945,
      "avg_query_time_ms": 2.3,
      "connection_pool": {
        "active_connections": 15,
        "max_connections": 50,
        "utilization": 0.3
      }
    },
    "service_uptime": 86400
  },
  "message": "Performance metrics retrieved successfully",
  "metadata": {
    "namespace": null,
    "metrics_collected_at": 1727529600,
    "operation": "get_metrics"
  }
}
```

## Error Codes

### Standard Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Input validation failed |
| `INVALID_HASH_FORMAT` | 400 | Hash format is invalid |
| `STAMP_NOT_FOUND` | 404 | Stamp not found |
| `STAMP_NOT_FOUND_IN_NAMESPACE` | 404 | Stamp not found in specified namespace |
| `DATABASE_NOT_CONFIGURED` | 503 | Database not available |
| `DATABASE_CONNECTION_ERROR` | 503 | Database connection failed |
| `MEMORY_ERROR` | 507 | Insufficient memory |
| `RUNTIME_ERROR` | 500 | Service runtime error |
| `UNEXPECTED_ERROR` | 500 | Unexpected internal error |
| `HEALTH_CHECK_FAILED` | 500 | Health check failed |
| `EMPTY_BATCH_REQUEST` | 400 | Batch request contains no items |

### O.N.E. v0.1 Compliance Errors

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `PROTOCOL_COMPLIANCE_FAILED` | 422 | Content fails O.N.E. v0.1 compliance |
| `NAMESPACE_INVALID` | 400 | Invalid namespace format |
| `INTELLIGENCE_DATA_INVALID` | 422 | Intelligence data validation failed |
| `VERSION_MISMATCH` | 409 | Protocol version mismatch |
| `OPERATION_ID_CONFLICT` | 409 | Operation ID already exists |

## Rate Limits

- **Standard Endpoints**: 1000 requests/minute per IP
- **Batch Endpoints**: 100 requests/minute per IP
- **Hash Generation**: 2000 requests/minute per IP (file size limited)
- **Health/Metrics**: 100 requests/minute per IP

## Performance SLA

- **Hash Generation**: <2ms p99, <1ms average
- **API Response**: <10ms p95 under normal load
- **Throughput**: 1000+ concurrent requests supported
- **Availability**: 99.9% uptime guaranteed
- **Database Queries**: <5ms p95 response time

## Kafka Events

Events are automatically published to Kafka topics with OnexEnvelopeV1 format:

### Topics

- `metadata.stamp.created` - Stamp creation events
- `metadata.stamp.validated` - Validation events
- `metadata.batch.processed` - Batch operation events
- `metadata.protocol.validated` - Protocol compliance events

### Event Schema

```json
{
  "envelope": {
    "version": "1.0",
    "event_type": "MetadataStampCreatedEvent",
    "timestamp": "2025-09-28T10:30:00Z",
    "correlation_id": "uuid4",
    "source_service": "metadata-stamping-service"
  },
  "payload": {
    "stamp_id": "uuid4",
    "file_hash": "blake3_hash",
    "namespace": "omninode.services.metadata",
    "op_id": "uuid4",
    "created_at": "2025-09-28T10:30:00Z"
  }
}
```

This comprehensive API documentation provides complete reference for integrating with the MetadataStampingService, enabling efficient metadata stamping operations with O.N.E. v0.1 protocol compliance.
