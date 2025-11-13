# OmniNode Bridge Services

## Overview

Service-specific documentation for all microservices in the omninode_bridge ecosystem.

## Services

### MetadataStamping Service
High-performance cryptographic metadata stamping with O.N.E. v0.1 protocol compliance.

- **Documentation:** [metadata-stamping/](./metadata-stamping/)
- **Status:** Phase 1+ Complete
- **Port:** 8053
- **Key Features:**
  - Sub-2ms BLAKE3 hash generation
  - Namespace support
  - Kafka event publishing
  - Unified API responses

### OnexTree Service
Standalone agent intelligence system for project structure awareness and duplicate detection.

- **Documentation:** [onextree/](./onextree/)
- **Status:** Phase 3 - Distributed Architecture
- **Key Features:**
  - Project structure caching
  - 80% reduction in duplicate file creation
  - <5ms lookup speed
  - Event-driven updates

## Service Architecture

All services follow consistent patterns:
- FastAPI async architecture
- PostgreSQL persistence
- Kafka event streaming
- Prometheus metrics
- Health check endpoints

## Related Documentation

- [Overall Architecture](../architecture/)
- [Deployment Guides](../deployment/)
- [Operations](../operations/)
- [API Documentation](../api/)
