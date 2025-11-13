# OmniNode Bridge Project Overview

## Purpose
OmniNode Bridge is an intelligent service lifecycle management and adaptive event processing system for the omninode ecosystem. It provides:

- **HookReceiver Service**: Webhook processing and event lifecycle management
- **Model Metrics API**: AI Lab integration and performance tracking
- **Workflow Coordinator**: Multi-step task orchestration and workflow management
- **PostgreSQL Client**: Database connection pooling and management
- **Kafka Integration**: Event streaming via RedPanda (Kafka-compatible)

## Tech Stack
- **Python**: 3.12
- **Web Framework**: FastAPI with Uvicorn
- **Database**: PostgreSQL with asyncpg (async connections)
- **Streaming**: RedPanda (Kafka-compatible)
- **Dependency Management**: Poetry
- **Containerization**: Docker and Docker Compose
- **Service Discovery**: Consul
- **Monitoring**: Prometheus metrics, structured logging (structlog)
- **Security**: Circuit breakers, rate limiting (slowapi), SSL/TLS support

## Project Structure
```
src/omninode_bridge/
├── main.py                 # HookReceiver entry point
├── constants.py            # Application constants
├── models/                 # Pydantic models
│   ├── events.py          # Event models
│   ├── hooks.py           # Hook models
│   └── model_metrics.py   # Metrics models
├── services/              # Core services
│   ├── hook_receiver.py   # FastAPI webhook service
│   ├── postgres_client.py # PostgreSQL connection management
│   ├── kafka_client.py    # Kafka/RedPanda client
│   ├── workflow_coordinator.py # Task orchestration
│   └── model_metrics_api.py    # AI Lab integration
└── cli/                   # Command-line interface
    └── workflow_submit.py # Workflow submission CLI
```

## Key Features
- SSL/TLS PostgreSQL connections
- Connection pooling with resource management
- Event-driven architecture with Kafka topics
- Rate limiting and authentication
- Health checks and monitoring
- Docker containerized deployment
- ONEX compliance for microservices
