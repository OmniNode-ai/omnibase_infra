# ONEX Infrastructure Project Overview

## Purpose
Infrastructure automation, deployment orchestration, and operational tooling for the ONEX framework ecosystem. Provides tooling for provisioning, configuration management, monitoring, and operations across cloud providers and on-premises environments.

## Tech Stack
- **Python 3.12** with Poetry for dependency management
- **ONEX Framework Dependencies**: omnibase_spi, omnibase_core
- **Database**: PostgreSQL with asyncpg/psycopg2-binary
- **Message Bus**: Kafka with aiokafka
- **Service Discovery**: Consul with python-consul
- **Secret Management**: Vault with hvac
- **Web Framework**: FastAPI with uvicorn
- **CLI**: Click with Rich for enhanced output

## Key Architecture Principles
- **Agent-Driven Development**: ALL coding tasks MUST use sub-agents (NO direct coding)
- **Contract-Driven**: All tools/services follow contract patterns
- **Strong Typing**: NEVER use `Any` types, all Pydantic models
- **Container Injection**: All dependencies via ONEXContainer
- **Protocol Resolution**: Duck typing through protocols, never isinstance
- **OnexError Only**: All exceptions converted to OnexError with chaining

## Repository Structure
```
src/omnibase_infra/
├── nodes/                    # Contract-driven node architecture
├── models/                   # Shared Pydantic models (DRY pattern)
├── infrastructure/           # Core infrastructure components
├── security/                 # Security tooling and policies
├── observability/           # Monitoring and metrics
├── automation/              # Automation scripts
└── cli/                     # Command-line interfaces
```

## Current State
- Feature branch: postgres-redpanda-event-bus-integration
- Production-ready PostgreSQL + RedPanda event bus integration
- 98/100 compliance score, targeting 100% with minor optimizations
