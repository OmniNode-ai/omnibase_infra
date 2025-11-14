# Database Migrations

PostgreSQL database schema and migrations for OmniBase Infrastructure.

## Structure

```
database/
├── migrations/
│   └── 001_init_infrastructure_schema.sql
└── README.md
```

## Migrations

### 001_init_infrastructure_schema.sql

**Purpose:** Initialize the infrastructure database schema

**Creates:**
- `infrastructure` schema
- `infrastructure.service_registry` table for service discovery

**Service Registry Schema:**
- `id` - Serial primary key
- `service_name` - Service identifier (e.g., 'postgres-adapter')
- `service_version` - Semantic version (e.g., 'v1.0.0')
- `endpoint` - Service HTTP endpoint
- `health_endpoint` - Health check endpoint
- `status` - Service status ('active', 'inactive', 'degraded')
- `created_at` - Registration timestamp
- `updated_at` - Last update timestamp

**Seed Data:**
- PostgreSQL Adapter (port 8080)
- Consul Adapter (port 8081)

## Running Migrations

### Docker Compose

Migrations are automatically applied when PostgreSQL container starts:

```bash
docker-compose -f docker-compose.infrastructure.yml up -d postgres
```

### Manual Execution

```bash
# Connect to PostgreSQL
docker exec -it omnibase-infra-postgres psql -U postgres -d omnibase_infrastructure

# Run migration
\i /path/to/001_init_infrastructure_schema.sql
```

### Via psql

```bash
psql -h localhost -p 5435 -U postgres -d omnibase_infrastructure -f database/migrations/001_init_infrastructure_schema.sql
```

## Future Migrations

Additional migrations should follow the naming convention:
- `002_add_infrastructure_state_tables.sql` - For reducer state storage
- `003_add_infrastructure_intents_tables.sql` - For orchestrator intents
- `004_add_audit_tables.sql` - For security audit logging

## Schema Design Principles

1. **Namespacing:** Use `infrastructure` schema for all infrastructure tables
2. **Timestamps:** Include `created_at` and `updated_at` on all tables
3. **Idempotency:** Use `IF NOT EXISTS` and `ON CONFLICT DO NOTHING`
4. **Versioning:** Include version in migration filename
5. **Documentation:** Comment each table and significant column

## Integration with Nodes

### PostgreSQL Adapter
- Uses connection pool to execute queries
- Stores infrastructure state in tables
- Manages transactions

### Reducer
- Stores state in `infrastructure_state` table (future migration)
- Stores intents in `infrastructure_intents` table (future migration)

### Orchestrator
- Reads intents from database
- Updates workflow state
- Coordinates multi-node operations

## Monitoring

Check database health:
```bash
# Via PostgreSQL adapter
curl http://localhost:8081/health

# Direct database connection
docker exec omnibase-infra-postgres pg_isready -U postgres -d omnibase_infrastructure
```

## Backup and Recovery

```bash
# Backup
docker exec omnibase-infra-postgres pg_dump -U postgres omnibase_infrastructure > backup.sql

# Restore
docker exec -i omnibase-infra-postgres psql -U postgres omnibase_infrastructure < backup.sql
```

## Production Considerations

1. **Use managed PostgreSQL** (AWS RDS, Google Cloud SQL)
2. **Enable automated backups**
3. **Configure read replicas** for high availability
4. **Set up monitoring** and alerting
5. **Use connection pooling** (PgBouncer) for large deployments
6. **Enable SSL/TLS** for encrypted connections
7. **Implement row-level security** for multi-tenant deployments
