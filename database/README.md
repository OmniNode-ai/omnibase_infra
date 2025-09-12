# Database Migrations

This directory contains SQL migration scripts for the omnibase_infrastructure database.

## Migration Files

- `001_init_infrastructure_schema.sql` - Initial infrastructure schema setup

## Usage

### Development Environment
```bash
# Apply migrations to local PostgreSQL instance
POSTGRES_HOST=localhost POSTGRES_PORT=5435 \
POSTGRES_PASSWORD="9#mK2$vP8@xL3&nQ7*wR5!zE6^uY4%tA1$bN3" \
psql -h localhost -p 5435 -U postgres -d omnibase_infrastructure -f database/migrations/001_init_infrastructure_schema.sql
```

### Docker Environment
```bash
# Apply migrations inside Docker container
docker exec -i omnibase_infra-postgres-1 psql -U postgres -d omnibase_infrastructure < database/migrations/001_init_infrastructure_schema.sql
```

## Migration Naming Convention

- `<number>_<description>.sql` (e.g., `001_init_infrastructure_schema.sql`)
- Numbers are zero-padded and sequential
- Descriptions use snake_case

## Schema Structure

### `infrastructure` Schema
- `service_registry` - Central registry for all infrastructure services
  - Tracks service endpoints, versions, and health status
  - Used by service discovery and health monitoring systems