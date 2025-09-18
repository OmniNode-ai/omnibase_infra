# ONEX Infrastructure Code Standards

## Zero Tolerance Policies
- **`Any` types**: Absolutely forbidden under all circumstances
- **Direct coding**: Prohibited - must use agent delegation
- **Hand-written models**: Must be contract-generated
- **Hardcoded configs**: Must be contract-driven
- **isinstance usage**: Use protocol resolution instead

## Naming Conventions
- **Model classes**: CamelCase with `Model` prefix (e.g., `ModelUserData`)
- **File names**: snake_case (e.g., `model_user_data.py`)
- **One model per file**: Each file contains exactly one `Model*` class
- **Contract naming**: Consistent with node names

## Architecture Patterns
- **Contract-Driven**: All tools/services follow contract patterns
- **Container Injection**: `def __init__(self, container: ONEXContainer)`
- **Protocol Resolution**: Duck typing through protocols
- **Error Handling**: All exceptions → `OnexError` with chaining
- **Strong Typing**: Pydantic models for all data structures

## Code Quality Tools
- **Ruff**: Linting and formatting (replaces black + isort)
- **MyPy**: Strict type checking
- **Pytest**: Comprehensive testing with asyncio support
- **Pre-commit**: Automated quality checks

## Documentation Requirements
- Type hints mandatory for all functions/methods
- Docstrings for public APIs
- Contract documentation for all nodes
- README updates for architectural changes

## Infrastructure Specific Standards
- **Service Integration**: Use adapter pattern for external services
- **Connection Pooling**: Database connections via dedicated managers
- **Event-Driven**: Infrastructure events through Kafka adapters
- **Security-First**: All components must pass security audits
- **Observability**: All tools must include monitoring/metrics
