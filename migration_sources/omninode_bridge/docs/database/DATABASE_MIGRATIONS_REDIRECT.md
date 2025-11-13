# Database Migrations

**For migration procedures, see:** [migrations/README.md](../../migrations/README.md)

## Quick Start

### Setup PostgreSQL Extensions

```bash
# Automated setup (one-time)
bash deployment/scripts/setup_postgres_extensions.sh
```

### Run Migrations

```bash
# Apply all migrations
for migration in migrations/00*.sql migrations/01*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        psql -h localhost -U postgres -d omninode_bridge -f "$migration"
    fi
done
```

### Verify

```bash
# Check tables created
psql -h localhost -U postgres -d omninode_bridge -c "\dt"
```

---

**Note**: The comprehensive deployment guide (1044 lines) has been archived to `docs/archive/DATABASE_MIGRATIONS_COMPREHENSIVE.md` for future use when there are external consumers of this repository. For MVP phase, the technical migration guide in `migrations/README.md` provides all necessary information.
