# Getting Started with OmniNode Bridge

**Quick Start**: Get from zero to running system in 5 minutes

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **Docker** and Docker Compose
- **Git** for cloning the repository
- **Poetry** for dependency management (`pip install poetry`)

## 5-Minute Quick Start

### 1. Clone and Install (2 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd omninode_bridge

# Install dependencies
poetry install

# Expected output: ✅ Dependencies installed successfully
```

### 2. Configure Kafka/Redpanda Hostname (ONE-TIME, 1 minute)

**Important**: Kafka/Redpanda requires hostname resolution due to its two-step broker discovery protocol.

**Linux/macOS:**
```bash
# Add Docker container hostname to /etc/hosts (required once)
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# Verify it was added
grep omninode-bridge-redpanda /etc/hosts
# Expected output: 127.0.0.1 omninode-bridge-redpanda
```

**Windows (requires Administrator privileges):**
```powershell
# Open Command Prompt or PowerShell as Administrator, then run:
echo 127.0.0.1 omninode-bridge-redpanda >> C:\Windows\System32\drivers\etc\hosts

# Verify it was added
type C:\Windows\System32\drivers\etc\hosts | findstr omninode-bridge-redpanda
# Expected output: 127.0.0.1 omninode-bridge-redpanda
```

**Why is this needed?** Kafka uses a two-step discovery: (1) bootstrap connection, (2) broker address resolution. Other services (PostgreSQL, HTTP) don't need this.

### 3. Start Services (1 minute)

```bash
# Start PostgreSQL, Redpanda (Kafka), and supporting services
docker compose -f deployment/docker-compose.yml up -d

# Verify services are running
docker compose -f deployment/docker-compose.yml ps

# Expected output:
# ✅ omninode-bridge-postgres   (port 5432)
# ✅ omninode-bridge-redpanda   (ports 9092, 29092)
# ✅ omninode-bridge-consul     (port 8500)
```

### 4. Run Database Migrations (30 seconds)

```bash
# Apply all database migrations
poetry run alembic upgrade head

# Expected output:
# INFO  [alembic.runtime.migration] Running upgrade -> 001
# INFO  [alembic.runtime.migration] Running upgrade 001 -> 002
# ...
# INFO  [alembic.runtime.migration] Running upgrade 008 -> 009
```

### 5. Start the Service (30 seconds)

```bash
# Start the MetadataStampingService
poetry run uvicorn src.metadata_stamping.main:app --reload --port 8053

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8053
# INFO:     Application startup complete
```

**Access Points**:
- 🌐 **API**: http://localhost:8053
- 📚 **Interactive Docs**: http://localhost:8053/docs
- 📊 **Service Metrics**: http://localhost:8053/metrics
- 📈 **Prometheus UI**: http://localhost:9090 (if Prometheus is enabled in docker-compose)

### 6. Test Your Setup (30 seconds)

**Option A: Interactive API Docs**

1. Open http://localhost:8053/docs in your browser
2. Click on `POST /stamp` endpoint
3. Click "Try it out"
4. Paste this test payload:
```json
{
  "content": "Hello, OmniNode Bridge!",
  "namespace": "quickstart.test",
  "file_path": "/tmp/test.txt"
}
```
5. Click "Execute"
6. ✅ **Success**: You should see a 200 response with stamp metadata

**Option B: Command Line**

```bash
# Test stamp creation endpoint
curl -X POST "http://localhost:8053/stamp" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello, OmniNode Bridge!",
    "namespace": "quickstart.test",
    "file_path": "/tmp/test.txt"
  }'

# Expected response (truncated):
# {
#   "stamp_id": "stamp_abc123...",
#   "file_hash": "blake3_def456...",
#   "namespace": "quickstart.test",
#   "status": "success"
# }
```

**Option C: Health Check**

```bash
# Check service health
curl http://localhost:8053/health

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "database": "healthy",
#     "kafka": "healthy"
#   }
# }
```

---

## What You Just Did

Congratulations! You now have a running OmniNode Bridge system with:

1. ✅ **MetadataStampingService** - BLAKE3 hash generation and metadata stamping
2. ✅ **PostgreSQL Database** - Persistent storage with ONEX v2.0 schema
3. ✅ **Kafka (Redpanda)** - Event streaming infrastructure
4. ✅ **Bridge Nodes** - ONEX-compliant orchestrator and reducer nodes

---

## Next Steps

### Explore the Bridge Nodes

```bash
# Run bridge orchestrator node
python -m omninode_bridge.nodes.orchestrator.v1_0_0.node

# Run bridge reducer node (in separate terminal)
python -m omninode_bridge.nodes.reducer.v1_0_0.node

# Run bridge registry node (in separate terminal)
python -m omninode_bridge.nodes.registry.v1_0_0.node
```

### Run Tests

```bash
# Run all tests
poetry run pytest tests/

# Run specific test categories
poetry run pytest tests/unit/              # Unit tests
poetry run pytest tests/integration/       # Integration tests
poetry run pytest tests/performance/       # Performance tests

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Explore Event Infrastructure

```bash
# View Kafka topics (requires kcat or rpk)
docker exec omninode-bridge-redpanda rpk topic list

# Expected output: 13+ topics including:
# - omninode_codegen_request_analyze_v1
# - omninode_codegen_response_analyze_v1
# - omninode_codegen_status_session_v1
# - omninode_codegen_dlq_*_v1 (Dead Letter Queue topics)
```

### Monitor Performance

```bash
# Access service metrics (Prometheus format)
curl http://localhost:8053/metrics

# Key metrics to watch:
# - hash_generation_duration_seconds (target: <2ms)
# - api_request_duration_seconds (target: <10ms)
# - database_query_duration_seconds (target: <5ms)

# Optional: Access Prometheus UI (if enabled in docker-compose)
# http://localhost:9090
```

---

## Troubleshooting Quick Fixes

### Issue: Kafka Connection Failures

```bash
# Symptom: "Failed to resolve 'omninode-bridge-redpanda'"

# Solution 1: Verify hostname configuration
grep omninode-bridge-redpanda /etc/hosts
# If nothing shows, run step 2 again

# Solution 2: Restart Redpanda
docker compose -f deployment/docker-compose.yml restart redpanda

# Solution 3: Test Kafka connectivity
docker exec omninode-bridge-redpanda rpk cluster info
```

### Issue: PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is running
docker compose -f deployment/docker-compose.yml ps postgres

# If not running, start it
docker compose -f deployment/docker-compose.yml up -d postgres

# Check logs for errors
docker compose -f deployment/docker-compose.yml logs postgres
```

### Issue: Port 8053 Already in Use

```bash
# Find process using port 8053
lsof -i :8053

# Kill the process (replace <PID> with actual process ID)
kill -9 <PID>

# Or use a different port
poetry run uvicorn src.metadata_stamping.main:app --reload --port 8054
```

### Issue: Poetry Dependency Conflicts

```bash
# Clear Poetry cache
poetry cache clear --all pypi

# Remove lock file and reinstall
rm poetry.lock
poetry install
```

### Issue: Database Migration Failures

```bash
# Check current migration version
poetry run alembic current

# Downgrade to previous version
poetry run alembic downgrade -1

# Re-run migration
poetry run alembic upgrade head
```

---

## Common Development Tasks

### Format Code

```bash
# Format with Black
poetry run black src/ tests/

# Expected output: "All done! ✨ 🍰 ✨"
```

### Run Type Checking

```bash
# Run mypy
poetry run mypy src/

# Expected output: "Success: no issues found"
```

### Run Linting

```bash
# Run ruff
poetry run ruff check src/ tests/

# Auto-fix issues
poetry run ruff check --fix src/ tests/
```

### Generate Database Migration

```bash
# Auto-generate migration from schema changes
poetry run alembic revision --autogenerate -m "Add new table"

# Review generated migration in migrations/versions/
# Edit if needed, then apply
poetry run alembic upgrade head
```

### Stop All Services

```bash
# Stop Docker services
docker compose -f deployment/docker-compose.yml down

# Stop and remove volumes (⚠️ deletes all data)
docker compose -f deployment/docker-compose.yml down -v
```

---

## Project Structure Quick Reference

```
omninode_bridge/
├── src/                          # Source code
│   ├── metadata_stamping/        # MetadataStampingService
│   ├── omninode_bridge/          # Bridge nodes and infrastructure
│   │   ├── nodes/                # ONEX v2.0 bridge nodes
│   │   │   ├── orchestrator/     # Workflow coordination
│   │   │   ├── reducer/          # Aggregation and state
│   │   │   └── registry/         # Service discovery
│   │   ├── events/               # Kafka event infrastructure
│   │   ├── persistence/          # Database layer
│   │   └── models/               # Data models
│   └── onextree_service/         # OnexTree intelligence client
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── performance/              # Performance tests
│
├── migrations/                   # Database migrations
│   ├── 001_create_workflow_executions.sql
│   ├── 002_create_workflow_steps.sql
│   └── ...                       # 9 migrations total
│
├── docs/                         # Documentation
│   ├── guides/                   # Implementation guides
│   ├── api/                      # API references
│   ├── architecture/             # Architecture docs
│   └── events/                   # Event infrastructure
│
├── deployment/
│   ├── docker-compose.yml            # Main development services
│   ├── docker-compose.bridge.yml     # Bridge nodes services
│   ├── docker-compose.codegen.yml    # Code generation services
│   └── docker-compose.phase2.yml     # Phase 2 services
├── pyproject.toml                # Python dependencies
└── alembic.ini                   # Database migration config
```

---

## Key Concepts

### ONEX v2.0 Compliance

OmniNode Bridge implements the **ONEX v2.0** (One.Node.Enterprise) architecture:

- **Suffix-Based Naming**: `NodeBridgeOrchestrator`, `ModelBridgeState`
- **Contract-Driven**: Nodes configured via YAML contracts
- **Subcontract Composition**: Capabilities added via subcontracts
- **Dependency Injection**: ModelONEXContainer provides services

### Bridge Node Types

1. **NodeBridgeOrchestrator** (Orchestrator)
   - Coordinates stamping workflows
   - Routes to MetadataStamping and OnexTree services
   - Manages FSM state transitions
   - Publishes Kafka events

2. **NodeBridgeReducer** (Reducer)
   - Aggregates metadata across workflows
   - Groups by namespace, time, file type
   - Tracks FSM states
   - Persists to PostgreSQL

3. **NodeBridgeRegistry** (Registry)
   - Service discovery and registration
   - Health monitoring
   - Circuit breaker integration

### Event Infrastructure

- **13 Kafka Topics**: Request, response, status, and DLQ topics
- **OnexEnvelopeV1 Format**: Standardized event envelope
- **Correlation Tracking**: UUID-based request/response correlation
- **DLQ Monitoring**: Dead Letter Queue for failed events

### Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| BLAKE3 hash generation | <2ms (p99) | <1ms (avg) |
| API response time | <10ms (p95) | <5ms |
| Database queries | <5ms (p95) | <3ms |
| Event log queries | <100ms (p95) | <50ms |
| CRUD operations | <20ms (p95) | <10ms |

---

## Further Reading

Now that you have a running system, explore these guides:

- **[Setup Guide](./SETUP.md)** - Complete development environment setup
- **[Architecture Guide](./architecture/ARCHITECTURE.md)** - System architecture and design
- **[Bridge Nodes Guide](./guides/BRIDGE_NODES_GUIDE.md)** - Bridge node implementation
- **[API Reference](./api/API_REFERENCE.md)** - Complete API documentation
- **[Event System Guide](./events/EVENT_SYSTEM_GUIDE.md)** - Kafka event infrastructure
- **[Testing Guide](./testing/TESTING_GUIDE.md)** - Test organization and execution
- **[Database Guide](./database/DATABASE_GUIDE.md)** - Database schema and migrations
- **[Operations Guide](./operations/OPERATIONS_GUIDE.md)** - Deployment and monitoring

---

## Get Help

### Documentation

- **Project README**: `README.md`
- **Complete Setup**: [docs/SETUP.md](./SETUP.md)
- **All Documentation**: [docs/INDEX.md](./INDEX.md)

### Issues and Support

- Check [Troubleshooting](#troubleshooting-quick-fixes) section
- Review [Common Issues](./SETUP.md#troubleshooting) in Setup Guide
- File an issue on GitHub (if applicable)

### Community

- Contributing: [docs/CONTRIBUTING.md](./CONTRIBUTING.md)
- Code of Conduct: Follow best practices and be respectful

---

**Welcome to OmniNode Bridge!** 🚀

You're now ready to explore the MVP foundation for the omninode ecosystem. Happy coding!
