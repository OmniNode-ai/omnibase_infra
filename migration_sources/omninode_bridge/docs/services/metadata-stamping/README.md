# MetadataStamping Service Documentation

## Overview

High-performance microservice for cryptographic metadata stamps in the omninode ecosystem. Implements O.N.E. v0.1 protocol compliance with sub-2ms BLAKE3 hash generation, namespace support, unified response format, and Kafka event publishing.

**Status:** Phase 1+ Complete with Enhanced Schema Compliance and Event-Driven Architecture

## Documentation

- **[Deployment Guide](./deployment-guide.md)** - Complete deployment instructions and configuration
- **[API Reference](./api-reference.md)** - Detailed API specifications and examples
- **[CLAUDE.md](../../../CLAUDE.md)** - Primary implementation guide with architecture and development details

## Quick Links

### Service Endpoints
- Health: `GET /api/v1/metadata-stamping/health`
- Metrics: `GET /api/v1/metadata-stamping/metrics`
- API Docs: `http://localhost:8053/docs` (when running)

### Related Documentation
- [Architecture](../../architecture/service-architecture.md)
- [Deployment](../../deployment/)
- [Operations](../../operations/)
- [Protocol Compliance](../../protocol/)

## Performance Targets

- BLAKE3 hashing: < 2ms per operation
- API response: < 10ms for standard operations
- Throughput: 1000+ concurrent requests
- Memory: < 512MB under normal load

## Quick Start

```bash
# Start services
docker-compose up -d

# Initialize database
poetry run alembic upgrade head

# Start development server
poetry run uvicorn src.omninode_bridge.services.metadata_stamping.main:app --reload --port 8053
```

## Support

- Issues: GitHub Issues
- Performance: Sub-2ms BLAKE3 hashing guaranteed
- SLA: 99.9% uptime, <10ms API response times
