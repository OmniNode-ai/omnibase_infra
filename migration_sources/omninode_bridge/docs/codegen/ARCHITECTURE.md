# Code Generation System Architecture

**Version**: 1.0.0
**Last Updated**: 2025-11-01
**Status**: Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Relationships](#component-relationships)
4. [Strategy Pattern](#strategy-pattern)
5. [Quality Gates Pipeline](#quality-gates-pipeline)
6. [Extension Points](#extension-points)
7. [Design Decisions](#design-decisions)
8. [Performance Characteristics](#performance-characteristics)

---

## System Overview

The Code Generation System is a **unified facade** that provides a single entry point for generating ONEX v2.0 compliant node code. It uses the **Strategy Pattern** to support multiple generation approaches while maintaining a consistent API.

### Core Principles

1. **Unified API**: Single entry point (`CodeGenerationService`) for all code generation
2. **Strategy Pattern**: Pluggable generation strategies selected at runtime
3. **Automatic Classification**: Intelligent node type classification from requirements
4. **Comprehensive Validation**: Multi-stage quality gates with configurable strictness
5. **Performance Monitoring**: Built-in metrics and observability
6. **Intelligence Integration**: Optional RAG intelligence from Archon MCP

### System Goals

- ✅ **Simplicity**: Single service replaces multiple parallel systems
- ✅ **Flexibility**: Support multiple generation strategies
- ✅ **Quality**: Comprehensive validation and quality gates
- ✅ **Performance**: Sub-second generation for common use cases
- ✅ **Extensibility**: Easy to add new strategies and features
- ✅ **Observability**: Rich metrics and tracing

---

## Architecture Diagram

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Code                            │
│                   (User Applications)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ generate_node()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               CodeGenerationService                          │
│                    (Unified Facade)                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ - Requirement validation                              │  │
│  │ - Node type classification                            │  │
│  │ - Strategy selection                                  │  │
│  │ - Result aggregation                                  │  │
│  │ - Performance monitoring                              │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ select_strategy()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   StrategyRegistry                           │
│                (Strategy Management)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ - Strategy registration                               │  │
│  │ - Strategy discovery                                  │  │
│  │ - Runtime strategy selection                          │  │
│  │ - Capability matching                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Jinja2      │ │ Template     │ │   Hybrid     │
│  Strategy    │ │ Loading      │ │  Strategy    │
│              │ │ Strategy     │ │              │
│ (Template-   │ │ (LLM-        │ │ (Combined)   │
│  based)      │ │  powered)    │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Generated Artifacts                          │
│            (ModelGeneratedArtifacts)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ - node.py (Node implementation)                       │  │
│  │ - contract.yaml (Node contract)                       │  │
│  │ - models.py (Data models)                             │  │
│  │ - tests/*.py (Unit tests)                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Component Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  CodeGenerationService                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │  PRD Analyzer    │      │ Node Classifier  │               │
│  │                  │      │                  │               │
│  │ - Prompt parsing │      │ - Type detection │               │
│  │ - Requirement    │      │ - Template       │               │
│  │   extraction     │      │   selection      │               │
│  └────────┬─────────┘      └────────┬─────────┘               │
│           │                         │                          │
│           │                         │                          │
│           ▼                         ▼                          │
│  ┌──────────────────────────────────────────────┐             │
│  │         Strategy Registry                    │             │
│  │                                               │             │
│  │  ┌────────────────────────────────────────┐  │             │
│  │  │  Strategy Selection Logic               │  │             │
│  │  │                                         │  │             │
│  │  │  if prefer_strategy:                   │  │             │
│  │  │      use preferred                     │  │             │
│  │  │  elif enable_llm:                      │  │             │
│  │  │      use template_loading              │  │             │
│  │  │  elif fast_generation:                 │  │             │
│  │  │      use jinja2                        │  │             │
│  │  │  else:                                  │  │             │
│  │  │      use default                       │  │             │
│  │  └────────────────────────────────────────┘  │             │
│  └──────────────────┬───────────────────────────┘             │
│                     │                                          │
│                     │                                          │
└─────────────────────┼──────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   Jinja2     │ │  Template   │ │   Hybrid     │
│   Strategy   │ │  Loading    │ │   Strategy   │
│              │ │  Strategy   │ │              │
├──────────────┤ ├─────────────┤ ├──────────────┤
│              │ │             │ │              │
│ TemplateEngine│ │ TemplateEngine│ │ Both systems│
│     ↓        │ │   Loader    │ │     ↓        │
│ Jinja2       │ │     ↓       │ │ Jinja2 base  │
│ Rendering    │ │ Pre-written │ │ + LLM enhance│
│     ↓        │ │ Templates   │ │     ↓        │
│ Basic        │ │     ↓       │ │ Enhanced     │
│ Validation   │ │ BusinessLogic│ │ Validation   │
│              │ │  Generator  │ │              │
│              │ │     ↓       │ │              │
│              │ │ LLM-powered │ │              │
│              │ │  Methods    │ │              │
└──────────────┘ └─────────────┘ └──────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Quality Validator    │
         │                        │
         │ - ONEX compliance      │
         │ - Type safety          │
         │ - Code quality         │
         │ - Documentation        │
         │ - Test coverage        │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ ModelGenerationResult  │
         │                        │
         │ - artifacts            │
         │ - strategy_used        │
         │ - generation_time_ms   │
         │ - validation_passed    │
         │ - metadata             │
         └────────────────────────┘
```

---

## Component Relationships

### 1. CodeGenerationService (Facade)

**Responsibility**: Unified entry point for all code generation

**Key Methods**:
```python
class CodeGenerationService:
    def __init__(
        templates_directory: Optional[Path],
        archon_mcp_url: Optional[str],
        enable_intelligence: bool,
    )

    async def generate_node(
        requirements: ModelPRDRequirements,
        strategy: str,
        enable_llm: bool,
        validation_level: str,
        ...
    ) -> ModelGenerationResult

    def list_strategies() -> list[dict]
    def get_strategy_info(strategy_type: str) -> dict
```

**Dependencies**:
- `StrategyRegistry`: Strategy management
- `NodeClassifier`: Node type classification
- `PRDAnalyzer`: Requirement extraction (via direct usage)

**Design Pattern**: **Facade Pattern**
- Simplifies complex subsystem (multiple strategies)
- Provides unified interface
- Delegates to appropriate strategy

### 2. StrategyRegistry

**Responsibility**: Manage and select generation strategies

**Key Methods**:
```python
class StrategyRegistry:
    def register(
        strategy: BaseGenerationStrategy,
        is_default: bool,
    ) -> None

    def get_strategy(
        strategy_type: EnumStrategyType,
    ) -> Optional[BaseGenerationStrategy]

    def select_strategy(
        node_type: EnumNodeType,
        enable_llm: bool,
        prefer_strategy: Optional[EnumStrategyType],
    ) -> BaseGenerationStrategy
```

**Strategy Selection Algorithm**:
```
1. If prefer_strategy specified and supports node_type:
   → Use preferred strategy

2. If enable_llm and TemplateLoadStrategy available:
   → Use LLM-powered strategy

3. If Jinja2Strategy available:
   → Use template-based strategy

4. Fall back to default strategy

5. If no suitable strategy found:
   → Raise RuntimeError
```

**Design Pattern**: **Registry Pattern**
- Centralized strategy management
- Runtime strategy discovery
- Pluggable architecture

### 3. BaseGenerationStrategy (Abstract)

**Responsibility**: Define interface for all strategies

**Key Methods**:
```python
class BaseGenerationStrategy(ABC):
    @abstractmethod
    async def generate(
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult

    @abstractmethod
    def supports_node_type(
        node_type: EnumNodeType,
    ) -> bool

    @abstractmethod
    def get_strategy_info() -> dict

    def validate_requirements(
        requirements: ModelPRDRequirements,
        validation_level: EnumValidationLevel,
    ) -> tuple[bool, list[str]]
```

**Design Pattern**: **Strategy Pattern**
- Defines family of algorithms (generation strategies)
- Makes them interchangeable
- Encapsulates each algorithm

### 4. Jinja2Strategy (Concrete Strategy)

**Responsibility**: Template-based code generation

**Implementation**:
```python
class Jinja2Strategy(BaseGenerationStrategy):
    def __init__(
        templates_directory: Optional[Path],
        enable_inline_templates: bool,
        enable_validation: bool,
    )

    async def generate(
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        # 1. Load Jinja2 templates
        # 2. Render templates with requirements
        # 3. Generate artifacts
        # 4. Validate if enabled
        # 5. Return result
```

**Characteristics**:
- ⚡ **Fast**: ~200ms generation time
- 📝 **Template-based**: Uses Jinja2 templates
- 🎯 **Best for**: Simple CRUD, well-defined patterns
- 🔧 **No LLM**: Purely template-based

### 5. TemplateLoadStrategy (Concrete Strategy)

**Responsibility**: LLM-powered code generation

**Implementation**:
```python
class TemplateLoadStrategy(BaseGenerationStrategy):
    def __init__(
        template_dir: Optional[Path],
        enable_llm: bool,
        enable_validation: bool,
    )

    async def generate(
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        # 1. Load pre-written templates
        # 2. Identify stubs (TODO, NotImplementedError)
        # 3. Use LLM to generate implementations
        # 4. Inject generated code
        # 5. Validate and return
```

**Characteristics**:
- 🤖 **LLM-powered**: Uses AI for complex logic
- ⏱️ **Slower**: ~3000ms generation time
- 🎯 **Best for**: Complex logic, novel requirements
- 💡 **Intelligent**: Learns from RAG patterns

### 6. HybridStrategy (Concrete Strategy)

**Responsibility**: Combine template-based and LLM-powered

**Implementation**:
```python
class HybridStrategy(BaseGenerationStrategy):
    async def generate(
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        # 1. Generate base with Jinja2Strategy
        # 2. Identify areas for enhancement
        # 3. Use LLM to enhance critical sections
        # 4. Merge results
        # 5. Comprehensive validation
```

**Characteristics**:
- ⚖️ **Balanced**: Template base + LLM enhancement
- ⏱️ **Medium speed**: ~800ms generation time
- 🎯 **Best for**: Production-critical features
- 🏆 **Best quality**: Combines strengths of both

---

## Strategy Pattern

### Why Strategy Pattern?

The Strategy Pattern was chosen for several key reasons:

#### 1. **Pluggable Algorithms**
```python
# Easy to add new strategies
class CustomStrategy(BaseGenerationStrategy):
    async def generate(self, request):
        # Custom generation logic
        pass

# Register and use
service.strategy_registry.register(CustomStrategy())
result = await service.generate_node(strategy="custom")
```

#### 2. **Runtime Selection**
```python
# Select strategy at runtime based on requirements
strategy = registry.select_strategy(
    node_type=node_type,
    enable_llm=enable_llm,
    prefer_strategy=user_preference,
)
```

#### 3. **Encapsulation**
```python
# Each strategy encapsulates its algorithm
class Jinja2Strategy:
    # Template-based logic isolated here
    pass

class TemplateLoadStrategy:
    # LLM-powered logic isolated here
    pass
```

#### 4. **Open/Closed Principle**
```python
# Open for extension (add new strategies)
# Closed for modification (existing code unchanged)

# Add new strategy without changing CodeGenerationService
class MLStrategy(BaseGenerationStrategy):
    pass
```

### Strategy Selection Flow

```
┌─────────────────────────────────────────────────┐
│       Client calls generate_node()              │
│                                                 │
│  service.generate_node(                         │
│      requirements=req,                          │
│      strategy="auto",                           │
│      enable_llm=True,                           │
│  )                                              │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│    CodeGenerationService._select_strategy()     │
│                                                 │
│    prefer_strategy = (                          │
│        None if strategy == "auto"               │
│        else parse_strategy(strategy)            │
│    )                                            │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│    StrategyRegistry.select_strategy()           │
│                                                 │
│    Decision Tree:                               │
│                                                 │
│    1. prefer_strategy specified?                │
│       → Use preferred (if supports node_type)   │
│                                                 │
│    2. enable_llm=True?                          │
│       → Try TemplateLoadStrategy                │
│                                                 │
│    3. Jinja2Strategy available?                 │
│       → Use Jinja2Strategy                      │
│                                                 │
│    4. Default strategy set?                     │
│       → Use default                             │
│                                                 │
│    5. No suitable strategy?                     │
│       → Raise RuntimeError                      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│         Selected Strategy.generate()            │
└─────────────────────────────────────────────────┘
```

---

## Quality Gates Pipeline

### Overview

Quality gates are validation checkpoints that ensure generated code meets quality standards.

### Validation Levels

| Level | Checks | Use Case | Performance |
|-------|--------|----------|-------------|
| **none** | No validation | Prototyping | ⚡⚡⚡ Fastest |
| **basic** | Syntax only | Development | ⚡⚡ Fast |
| **standard** | Syntax + types + structure | General use | ⚡ Moderate |
| **strict** | All checks + quality metrics | Production | 🐌 Thorough |

### Validation Pipeline

```
┌────────────────────────────────────────────────┐
│         Generated Artifacts                    │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│   Stage 1: Syntax Validation                   │
│                                                │
│   ✓ Python syntax correct                     │
│   ✓ YAML syntax correct                       │
│   ✓ Imports valid                             │
└────────────────┬───────────────────────────────┘
                 │
                 ▼ (if basic+)
┌────────────────────────────────────────────────┐
│   Stage 2: Type Safety                         │
│                                                │
│   ✓ Type annotations present                  │
│   ✓ Pydantic models valid                     │
│   ✓ Method signatures correct                 │
└────────────────┬───────────────────────────────┘
                 │
                 ▼ (if standard+)
┌────────────────────────────────────────────────┐
│   Stage 3: ONEX Compliance                     │
│                                                │
│   ✓ Node naming conventions                   │
│   ✓ Contract structure                        │
│   ✓ Required methods present                  │
│   ✓ Model naming conventions                  │
└────────────────┬───────────────────────────────┘
                 │
                 ▼ (if strict)
┌────────────────────────────────────────────────┐
│   Stage 4: Quality Metrics                     │
│                                                │
│   ✓ Documentation coverage >70%               │
│   ✓ Test coverage >80%                        │
│   ✓ Code complexity <10                       │
│   ✓ No code smells                            │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│   Validation Result                            │
│                                                │
│   - passed: bool                               │
│   - errors: list[str]                          │
│   - warnings: list[str]                        │
│   - quality_score: float                       │
└────────────────────────────────────────────────┘
```

### Quality Scores

```python
class ModelValidationResult:
    # Overall scores
    quality_score: float  # 0.0-1.0
    passed: bool  # quality_score >= threshold

    # Component scores
    onex_compliance_score: float
    type_safety_score: float
    code_quality_score: float
    documentation_score: float
    test_coverage_score: float

    # Details
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]
```

---

## Extension Points

### Adding a New Strategy

```python
from omninode_bridge.codegen.strategies.base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    ModelGenerationRequest,
    ModelGenerationResult,
)

class CustomStrategy(BaseGenerationStrategy):
    """Custom code generation strategy."""

    def __init__(self):
        super().__init__(
            strategy_name="Custom Strategy",
            strategy_type=EnumStrategyType.CUSTOM,  # Add to enum
            enable_validation=True,
        )

    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """Generate code using custom logic."""

        # 1. Implement custom generation logic
        artifacts = await self._custom_generation(request)

        # 2. Validate if enabled
        validation = await self._validate(artifacts)

        # 3. Return result
        return ModelGenerationResult(
            artifacts=artifacts,
            strategy_used=self.strategy_type,
            generation_time_ms=generation_time,
            validation_passed=validation.passed,
            validation_errors=validation.errors,
            correlation_id=request.correlation_id,
        )

    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """Check if strategy supports node type."""
        return node_type in [EnumNodeType.EFFECT, EnumNodeType.COMPUTE]

    def get_strategy_info(self) -> dict:
        """Get strategy information."""
        return {
            "name": self.strategy_name,
            "type": self.strategy_type.value,
            "supported_node_types": ["effect", "compute"],
            "requires_llm": False,
            "performance_profile": "fast",
        }

# Register custom strategy
service = CodeGenerationService()
service.strategy_registry.register(CustomStrategy())

# Use custom strategy
result = await service.generate_node(
    requirements=requirements,
    strategy="custom",
)
```

### Adding Custom Validation Rules

```python
from omninode_bridge.codegen import QualityValidator

class CustomValidator(QualityValidator):
    """Custom quality validator with additional rules."""

    async def validate(self, artifacts):
        """Validate with custom rules."""

        # Run standard validation
        result = await super().validate(artifacts)

        # Add custom validation
        custom_errors = self._validate_custom_rules(artifacts)
        result.errors.extend(custom_errors)

        # Recalculate scores
        result.quality_score = self._calculate_quality_score(result)
        result.passed = result.quality_score >= self.min_quality_threshold

        return result

    def _validate_custom_rules(self, artifacts):
        """Implement custom validation rules."""
        errors = []

        # Example: Check for specific patterns
        if "TODO" in artifacts.node_file:
            errors.append("Node contains TODO comments")

        return errors
```

---

## Design Decisions

### 1. Facade Pattern for Service

**Decision**: Use Facade pattern for `CodeGenerationService`

**Rationale**:
- ✅ Simplifies complex subsystem (multiple strategies)
- ✅ Provides single entry point
- ✅ Hides internal complexity
- ✅ Easier to test and mock

**Alternative Considered**: Direct strategy usage
**Rejected Because**: Too complex for users, no unified API

### 2. Strategy Pattern for Generation

**Decision**: Use Strategy pattern for generation approaches

**Rationale**:
- ✅ Pluggable algorithms
- ✅ Runtime strategy selection
- ✅ Open/Closed principle
- ✅ Easy to extend

**Alternative Considered**: Factory pattern
**Rejected Because**: Less flexible, harder to add new strategies

### 3. Pydantic Models for Type Safety

**Decision**: Use Pydantic v2 for all data models

**Rationale**:
- ✅ Runtime type validation
- ✅ Automatic documentation
- ✅ JSON serialization
- ✅ IDE support

**Alternative Considered**: Plain dataclasses
**Rejected Because**: No runtime validation, less features

### 4. Async-First API

**Decision**: All generation methods are async

**Rationale**:
- ✅ Non-blocking I/O (LLM calls, file I/O)
- ✅ Better concurrency
- ✅ Modern Python best practices
- ✅ Future-proof

**Alternative Considered**: Sync API with threading
**Rejected Because**: More complex, less performant

### 5. Built-in Validation

**Decision**: Include validation in service (not optional separate step)

**Rationale**:
- ✅ Consistent quality
- ✅ Easier to use correctly
- ✅ Configurable strictness
- ✅ Better default behavior

**Alternative Considered**: Separate validation step
**Rejected Because**: Users might skip it, inconsistent quality

---

## Performance Characteristics

### Generation Time Benchmarks

| Strategy | Node Type | Time (avg) | Time (p95) | Time (p99) |
|----------|-----------|-----------|-----------|-----------|
| **jinja2** | Effect | 200ms | 350ms | 500ms |
| **jinja2** | Compute | 180ms | 320ms | 450ms |
| **jinja2** | Reducer | 220ms | 380ms | 550ms |
| **template_loading** | Effect | 2800ms | 4200ms | 5500ms |
| **template_loading** | Compute | 3100ms | 4500ms | 6000ms |
| **hybrid** | Effect | 800ms | 1200ms | 1500ms |

### Memory Usage

| Strategy | Memory (avg) | Memory (peak) |
|----------|-------------|--------------|
| **jinja2** | 30MB | 50MB |
| **template_loading** | 120MB | 180MB |
| **hybrid** | 80MB | 120MB |

### Validation Overhead

| Level | Time | Impact |
|-------|------|--------|
| **none** | 0ms | 0% |
| **basic** | 50ms | +25% |
| **standard** | 150ms | +75% |
| **strict** | 400ms | +200% |

### Scalability

- **Concurrent Requests**: Supports 100+ concurrent generations
- **Memory Scaling**: Linear with number of concurrent requests
- **CPU Scaling**: Parallelizable across multiple cores
- **I/O Scaling**: Async I/O prevents blocking

---

## Next Steps

1. ✅ **Read Usage Guide**: See [USAGE_GUIDE.md](./USAGE_GUIDE.md)
2. ✅ **Review Migration**: See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
3. ✅ **Try Examples**: See [examples/codegen/](../../examples/codegen/)
4. ✅ **Extend System**: Add custom strategies or validation rules

---

## Support

- **Documentation**: [docs/codegen/](.)
- **Examples**: [examples/codegen/](../../examples/codegen/)
- **Issues**: GitHub Issues
- **Questions**: Team Slack #code-generation

---

**Status**: ✅ Production ready, comprehensive architecture documented
