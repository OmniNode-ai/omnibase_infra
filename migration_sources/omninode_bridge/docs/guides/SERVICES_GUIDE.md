# OmniNode Bridge Services - Complete Guide

## Overview

OmniNode Bridge provides two core containerized services:

1. **OnexTree Service** (Port 8058) - Fast project structure intelligence
2. **Metadata Stamping Service** (Port 8057) - Cryptographic file stamping

Both services are production-ready and accessible via REST APIs.

---

## OnexTree Service

### What It Does

OnexTree provides **sub-5ms file lookup** across your entire project by building an in-memory index of the project structure.

**Key Features:**
- Generate complete project tree in <100ms (10K+ files)
- Sub-5ms file lookups (typically <1ms)
- Query by extension, name, or exact path
- Memory efficient: <20MB for typical projects
- BLAKE3-based change detection

### How It Works

#### 1. Architecture

```
┌─────────────────────────────────────────────┐
│         OnexTree Service (Port 8058)        │
├─────────────────────────────────────────────┤
│  FastAPI REST API                           │
├─────────────────────────────────────────────┤
│  Query Engine (In-Memory Indexes)           │
│  ├─ Exact Path Index (O(1) lookup)         │
│  ├─ Extension Index (O(1) by type)         │
│  ├─ Directory Index (O(1) children)        │
│  └─ Name Index (O(1) by name)              │
├─────────────────────────────────────────────┤
│  Tree Generator                             │
│  ├─ Filesystem Walker                      │
│  ├─ BLAKE3 Hash Generator                  │
│  └─ Statistics Collector                   │
└─────────────────────────────────────────────┘
```

#### 2. Tree Generation Process

```python
# Step 1: Walk filesystem
for file in walk_directory(project_root):
    - Calculate BLAKE3 hash
    - Extract metadata (size, extension, modified time)
    - Build hierarchical tree structure

# Step 2: Build indexes
for node in tree:
    exact_path_index[node.path] = node
    extension_index[node.extension].append(node)
    directory_index[parent].append(node)
    name_index[node.name].append(node)

# Step 3: Calculate statistics
- Total files/directories
- File type distribution
- Total size
- Last updated timestamp
```

#### 3. Query Execution

**Path Lookup (O(1) - <1ms):**
```python
node = exact_path_index.get("src/main.py")
```

**Extension Query (O(1) - <1ms):**
```python
py_files = extension_index.get("py")
```

**Name Search (O(1) - <2ms):**
```python
main_files = name_index.get("main")
```

### API Usage

#### Generate Tree

```bash
curl -X POST http://localhost:8058/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project_root": "/app/src"
  }'
```

**Response:**
```json
{
  "success": true,
  "total_files": 671,
  "total_directories": 134,
  "total_size_mb": 38.53,
  "generation_time_ms": 95.2
}
```

#### Query Files

**By Extension:**
```bash
curl -X POST http://localhost:8058/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": ".py",
    "query_type": "extension",
    "limit": 10
  }'
```

**By Name:**
```bash
curl -X POST http://localhost:8058/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "main",
    "query_type": "name"
  }'
```

**By Exact Path:**
```bash
curl -X POST http://localhost:8058/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "src/metadata_stamping/main.py",
    "query_type": "path"
  }'
```

**Auto-detect (default):**
```bash
curl -X POST http://localhost:8058/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": ".py"
  }'
```

**Response:**
```json
{
  "success": true,
  "query": ".py",
  "query_type": "extension",
  "results": [
    {
      "path": "src/main.py",
      "name": "main.py",
      "type": "file",
      "size": 1234,
      "extension": "py"
    }
  ],
  "count": 1,
  "execution_time_ms": 0.48
}
```

#### Get Statistics

```bash
curl http://localhost:8058/stats
```

**Response:**
```json
{
  "tree_loaded": true,
  "statistics": {
    "total_files": 671,
    "total_directories": 134,
    "file_type_distribution": {
      "py": 245,
      "md": 45,
      "json": 12
    },
    "total_size_bytes": 40401920,
    "index_sizes": {
      "exact_path_entries": 805,
      "extension_types": 15,
      "directories_indexed": 134,
      "unique_names": 312
    }
  }
}
```

### Performance Characteristics

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| Tree Generation (10K files) | <100ms | 48-95ms | ✅ |
| Path Lookup | <5ms | <1ms | ✅ |
| Extension Query | <5ms | <1ms | ✅ |
| Name Query | <5ms | <0.5ms | ✅ |
| Memory Usage | <20MB | ~15MB | ✅ |

---

## Metadata Stamping Service

### What It Does

Generates **cryptographic stamps** for files and content using BLAKE3 hashing with **sub-2ms** performance.

**Key Features:**
- BLAKE3 hash generation in <2ms
- O.N.E. v0.1 protocol compliance
- Namespace support for multi-tenancy
- PostgreSQL persistence
- Kafka event publishing
- Redis caching (optional)
- Batch processing

### How It Works

#### 1. Architecture

```
┌─────────────────────────────────────────────┐
│    Metadata Stamping Service (Port 8057)    │
├─────────────────────────────────────────────┤
│  FastAPI REST API                           │
│  └─ Unified Response Format                │
├─────────────────────────────────────────────┤
│  Stamping Engine                            │
│  ├─ BLAKE3 Hash Generator (<2ms)           │
│  ├─ Protocol Handler (O.N.E. v0.1)         │
│  └─ Namespace Manager                      │
├─────────────────────────────────────────────┤
│  Storage Layer                              │
│  ├─ PostgreSQL (persistent storage)        │
│  ├─ Redis Cache (optional)                 │
│  └─ Kafka Events (optional)                │
├─────────────────────────────────────────────┤
│  Batch Processor                            │
│  └─ Priority Queue + Worker Pool           │
└─────────────────────────────────────────────┘
```

#### 2. Stamping Process

```python
# Step 1: Generate BLAKE3 hash
file_hash = blake3(file_data)  # <2ms

# Step 2: Create stamp metadata
stamp = {
    "uid": uuid4(),
    "hash": file_hash,
    "timestamp": utc_now(),
    "namespace": "omninode.services.metadata",
    "protocol_version": "1.0"
}

# Step 3: Store in database
db.insert(stamp)  # <5ms with connection pooling

# Step 4: Publish event (async)
kafka.publish("metadata.stamp.created", stamp)

# Step 5: Return stamped content
return original_content + stamp_comment
```

#### 3. Hash Generation Performance

**Performance Paths:**

```python
if file_size <= 1KB:
    # Direct hash (fastest)
    hash = blake3(data)  # <0.5ms

elif file_size <= 1MB:
    # Pooled hasher (medium)
    hash = hasher_pool.hash(data)  # <1ms

else:
    # Streaming hash (large files)
    hash = stream_hash(data)  # <2ms
```

**Performance Grades:**
- Grade A: <1ms execution time
- Grade B: 1-2ms execution time
- Grade C: >2ms execution time

### API Usage

#### Create Stamp

```bash
curl -X POST http://localhost:8057/api/v1/metadata-stamping/stamp \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your file content here",
    "file_path": "/path/to/file.txt",
    "options": {
      "stamp_type": "lightweight"
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "success": true,
    "stamp_id": "87c49e7a-b837-4416-bdad-94a565080b38",
    "file_hash": "c53eb0f2a711cd6a47089c8d28e87cca...",
    "stamped_content": "# ONEX:uid=3fd9bef3...\nYour file content here",
    "stamp": "# ONEX:uid=3fd9bef3,hash=c53eb0f2...",
    "stamp_type": "lightweight",
    "performance_metrics": {
      "execution_time_ms": 1.2,
      "file_size_bytes": 12,
      "performance_grade": "A"
    },
    "created_at": "2025-09-30T20:05:21Z",
    "namespace": "omninode.services.metadata"
  }
}
```

#### Generate Hash Only

```bash
curl -X POST http://localhost:8057/api/v1/metadata-stamping/hash \
  -F "file=@README.md"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "file_hash": "2c0dc7cf5cb3178969170fa43be72ba4...",
    "execution_time_ms": 0.8,
    "file_size_bytes": 8013,
    "performance_grade": "A"
  }
}
```

#### Validate Stamps

```bash
curl -X POST http://localhost:8057/api/v1/metadata-stamping/validate \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Content with embedded stamps to validate"
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "success": true,
    "is_valid": true,
    "stamps_found": 1,
    "current_hash": "c53eb0f2a711cd6a...",
    "validation_details": [
      {
        "stamp_type": "lightweight",
        "stamp_hash": "c53eb0f2a711cd6a...",
        "is_valid": true
      }
    ]
  }
}
```

#### Retrieve Stamp by Hash

```bash
curl http://localhost:8057/api/v1/metadata-stamping/stamp/c53eb0f2a711cd6a...
```

#### Batch Stamping

```bash
curl -X POST http://localhost:8057/api/v1/metadata-stamping/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "file1",
        "content": "First file content",
        "file_path": "file1.txt"
      },
      {
        "id": "file2",
        "content": "Second file content",
        "file_path": "file2.txt"
      }
    ]
  }'
```

### O.N.E. v0.1 Protocol Compliance

**Stamp Format:**
```
# ONEX:uid=<uuid>,hash=<blake3_hash>,ts=<iso_timestamp>
```

**Example:**
```
# ONEX:uid=3fd9bef3-2fa7-4d1a-985d-120af49acfe3,hash=c53eb0f2a711cd6a47089c8d28e87cca81d2b2b13c375c54d9ab1c5e48113f8b,ts=2025-09-30T20:05:21.746974+00:00
```

**Fields:**
- `uid`: Unique identifier (UUID v4)
- `hash`: BLAKE3 cryptographic hash (64 hex chars)
- `ts`: ISO 8601 timestamp with timezone
- `namespace`: Organization namespace (default: `omninode.services.metadata`)
- `version`: Protocol version (default: 1)
- `metadata_version`: Metadata schema version (default: 0.1)

### Performance Characteristics

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| BLAKE3 Hash | <2ms | 0.5-1.2ms | ✅ |
| API Response | <10ms | 5-8ms | ✅ |
| Database Insert | <5ms | 2-4ms | ✅ |
| Batch Processing | 1000+ req/s | 1200+ req/s | ✅ |

---

## Service Integration

### Container Status

```bash
# Check running services
docker ps | grep -E "onextree|metadata"

# Expected output:
# omninode-bridge-onextree (8058:8058)
# omninode-bridge-metadata-stamping (8057:8053)
```

### Health Checks

```bash
# OnexTree health
curl http://localhost:8058/health

# Metadata Stamping health
curl http://localhost:8057/api/v1/metadata-stamping/health
```

### View Logs

```bash
# OnexTree logs
docker logs omninode-bridge-onextree -f

# Metadata Stamping logs
docker logs omninode-bridge-metadata-stamping -f
```

### Restart Services

```bash
# Restart OnexTree
docker-compose restart onextree

# Restart Metadata Stamping
docker-compose restart metadata-stamping

# Restart both
docker-compose restart onextree metadata-stamping
```

---

## Use Cases

### Use Case 1: Index and Stamp Project

```bash
# 1. Generate OnexTree
curl -X POST http://localhost:8058/generate \
  -H "Content-Type: application/json" \
  -d '{"project_root": "/app/src"}' | jq .

# 2. Query Python files
curl -X POST http://localhost:8058/query \
  -H "Content-Type: application/json" \
  -d '{"query": ".py", "limit": 100}' | jq -r '.results[].path' > files.txt

# 3. Stamp each file (example with first file)
FILE_PATH=$(head -1 files.txt)
curl -X POST http://localhost:8057/api/v1/metadata-stamping/stamp \
  -H "Content-Type: application/json" \
  -d "{\"content\":\"$(cat $FILE_PATH)\",\"file_path\":\"$FILE_PATH\"}" | jq .
```

### Use Case 2: Monitor File Changes

```bash
# 1. Generate initial tree
curl -X POST http://localhost:8058/generate \
  -d '{"project_root": "/app/src"}' | jq .

# 2. Get initial stats
curl http://localhost:8058/stats | jq '.statistics' > stats_before.json

# ... make changes to files ...

# 3. Regenerate tree
curl -X POST http://localhost:8058/generate \
  -d '{"project_root": "/app/src"}' | jq .

# 4. Compare stats
curl http://localhost:8058/stats | jq '.statistics' > stats_after.json
diff stats_before.json stats_after.json
```

### Use Case 3: Validate File Integrity

```bash
# 1. Stamp file
STAMP_RESPONSE=$(curl -s -X POST http://localhost:8057/api/v1/metadata-stamping/stamp \
  -H "Content-Type: application/json" \
  -d '{"content":"Important data","file_path":"data.txt"}')

STAMPED_CONTENT=$(echo $STAMP_RESPONSE | jq -r '.data.stamped_content')

# 2. Save stamped content
echo "$STAMPED_CONTENT" > data.txt

# 3. Later: validate
curl -X POST http://localhost:8057/api/v1/metadata-stamping/validate \
  -H "Content-Type: application/json" \
  -d "{\"content\":\"$(cat data.txt)\"}" | jq '.data.is_valid'
```

---

## API Documentation

### OnexTree Service
- **Swagger UI**: http://localhost:8058/docs
- **OpenAPI JSON**: http://localhost:8058/openapi.json

### Metadata Stamping Service
- **Swagger UI**: http://localhost:8057/docs
- **OpenAPI JSON**: http://localhost:8057/openapi.json

---

## Troubleshooting

### OnexTree Service Issues

**Service not responding:**
```bash
# Check container status
docker ps -a | grep onextree

# Check logs
docker logs omninode-bridge-onextree --tail 50

# Restart
docker-compose restart onextree
```

**Tree generation fails:**
```bash
# Ensure project_root path is accessible in container
# Check volume mounts in docker-compose.yml

# Verify path inside container
docker exec omninode-bridge-onextree ls -la /app/src
```

### Metadata Stamping Service Issues

**Database connection errors:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test database connection
docker exec omninode-bridge-postgres psql -U postgres -c "SELECT 1"
```

**Kafka errors (optional features):**
```bash
# Check RedPanda is running
docker ps | grep redpanda

# Kafka features are optional - service works without them
```

---

## Performance Tuning

### OnexTree

**For large projects (10K+ files):**
- Increase memory limits in docker-compose.yml
- Consider periodic tree regeneration vs continuous watching

### Metadata Stamping

**For high throughput:**
- Enable Redis caching: `ENABLE_REDIS_CACHE=true`
- Increase database connection pool: `DATABASE_POOL_MAX_SIZE=100`
- Enable batch processing: `ENABLE_BATCH_PROCESSING=true`

---

## Summary

**OnexTree Service (8058):**
- ✅ Sub-5ms file lookups
- ✅ Fast tree generation (<100ms for 10K files)
- ✅ Multiple query types (extension, name, path)
- ✅ Minimal memory footprint

**Metadata Stamping Service (8057):**
- ✅ Sub-2ms BLAKE3 hashing
- ✅ O.N.E. v0.1 protocol compliance
- ✅ PostgreSQL persistence
- ✅ Batch processing support

Both services are production-ready and accessible via REST APIs with comprehensive Swagger documentation.
