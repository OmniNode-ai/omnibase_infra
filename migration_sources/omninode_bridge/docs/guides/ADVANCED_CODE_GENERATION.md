# Advanced Code Generation Guide

**Status**: ✅ Complete
**Version**: 2.0.0
**Last Updated**: 2025-11-05

---

## Table of Contents

1. [Overview](#overview)
2. [Advanced ContractInferencer Usage](#advanced-contractinferencer-usage)
3. [Custom Mixin Configuration](#custom-mixin-configuration)
4. [LLM Prompt Engineering](#llm-prompt-engineering)
5. [Batch Processing Patterns](#batch-processing-patterns)
6. [CI/CD Integration](#cicd-integration)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Troubleshooting](#advanced-troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

This guide covers advanced topics for the OmniNode code generation system, building upon the foundation established in:

- **[Code Generation Guide](./CODE_GENERATION_GUIDE.md)** - Core concepts and 9-stage pipeline
- **[Mixin-Enhanced Generation Quickstart](./MIXIN_ENHANCED_GENERATION_QUICKSTART.md)** - Quick start guide
- **[Contract Inferencer Documentation](../codegen/CONTRACT_INFERENCER.md)** - Contract inference details

### What This Guide Covers

**For Experienced Users**:
- Deep dive into ContractInferencer internals
- Custom mixin catalog creation
- LLM prompt optimization techniques
- Large-scale batch processing
- Production CI/CD integration
- Performance tuning strategies
- Complex troubleshooting scenarios

**Prerequisites**:
- Familiarity with ONEX v2.0 architecture
- Understanding of basic code generation workflow
- Experience with Python async/await
- Knowledge of Jinja2 templating

---

## Advanced ContractInferencer Usage

The ContractInferencer is the intelligent bridge between existing code and v2.0 contracts. This section covers advanced usage patterns.

### Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                   ContractInferencer Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: node.py file                                         │
│     ↓                                                         │
│  ┌────────────────────────────────────────────────┐         │
│  │ 1. AST Parser                                   │         │
│  │    - Extract class structure                    │         │
│  │    - Detect inheritance chain                   │         │
│  │    - Parse method signatures                    │         │
│  │    - Extract docstrings and comments            │         │
│  └──────────────┬─────────────────────────────────┘         │
│                 ▼                                             │
│  ┌────────────────────────────────────────────────┐         │
│  │ 2. I/O Operation Detector                       │         │
│  │    - Scan imports for operation types            │         │
│  │    - Pattern match on method bodies              │         │
│  │    - Infer capabilities from operations          │         │
│  └──────────────┬─────────────────────────────────┘         │
│                 ▼                                             │
│  ┌────────────────────────────────────────────────┐         │
│  │ 3. Mixin Detector                               │         │
│  │    - Match against mixin catalog                 │         │
│  │    - Extract mixin-specific methods              │         │
│  │    - Determine mixin dependencies                │         │
│  └──────────────┬─────────────────────────────────┘         │
│                 ▼                                             │
│  ┌────────────────────────────────────────────────┐         │
│  │ 4. LLM Config Inferencer                        │         │
│  │    - Build context-rich prompts                  │         │
│  │    - Query LLM (Z.ai GLM-4.5)                   │         │
│  │    - Parse YAML configurations                   │         │
│  │    - Extract confidence scores                   │         │
│  └──────────────┬─────────────────────────────────┘         │
│                 ▼                                             │
│  ┌────────────────────────────────────────────────┐         │
│  │ 5. Contract Builder                             │         │
│  │    - Construct ModelEnhancedContract             │         │
│  │    - Configure advanced features                 │         │
│  │    - Generate subcontract references             │         │
│  └──────────────┬─────────────────────────────────┘         │
│                 ▼                                             │
│  Output: contract.yaml (ONEX v2.0 compliant)                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Advanced AST Parsing

The inferencer uses Python's `ast` module for comprehensive code analysis. Here's how to extend it:

#### Custom AST Visitors

```python
import ast
from typing import List, Dict, Any

class AdvancedNodeVisitor(ast.NodeVisitor):
    """
    Custom AST visitor for extracting advanced patterns.

    Extends the built-in inferencer with domain-specific logic.
    """

    def __init__(self):
        self.async_methods: List[str] = []
        self.decorator_usage: Dict[str, List[str]] = {}
        self.type_hints: Dict[str, str] = {}
        self.exception_handling: List[Dict[str, Any]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async method information."""
        self.async_methods.append(node.name)

        # Extract decorators
        decorators = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        self.decorator_usage[node.name] = decorators

        # Extract return type hint
        if node.returns:
            self.type_hints[node.name] = ast.unparse(node.returns)

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Analyze exception handling patterns."""
        exceptions_caught = []
        for handler in node.handlers:
            if handler.type:
                exc_type = ast.unparse(handler.type)
                exceptions_caught.append(exc_type)

        self.exception_handling.append({
            'lineno': node.lineno,
            'exceptions': exceptions_caught,
            'has_finally': len(node.finalbody) > 0,
        })

        self.generic_visit(node)


async def enhanced_inference_example():
    """
    Demonstrate enhanced contract inference with custom AST analysis.
    """
    from omninode_bridge.codegen.contract_inferencer import ContractInferencer
    from pathlib import Path

    # Initialize inferencer with custom options
    inferencer = ContractInferencer(
        enable_llm=True,
        llm_tier=EnumLLMTier.CLOUD_FAST,
    )

    # Path to existing node
    node_path = Path("src/omninode_bridge/nodes/database_effect/v1_0_0/node.py")

    # Step 1: Parse node with built-in parser
    analysis = inferencer._parse_node_file(node_path)

    # Step 2: Enhance with custom visitor
    tree = ast.parse(node_path.read_text())
    visitor = AdvancedNodeVisitor()
    visitor.visit(tree)

    # Step 3: Merge insights
    enhanced_analysis = {
        **analysis.__dict__,
        'async_methods': visitor.async_methods,
        'decorator_usage': visitor.decorator_usage,
        'exception_patterns': visitor.exception_handling,
    }

    # Step 4: Generate contract with enhanced context
    contract_yaml = await inferencer.infer_from_node(
        node_path=node_path,
        output_path=node_path.parent / "contract_enhanced.yaml"
    )

    print(f"✅ Enhanced contract generated")
    print(f"   Async methods detected: {len(visitor.async_methods)}")
    print(f"   Exception handlers: {len(visitor.exception_handling)}")

    await inferencer.cleanup()
```

### Multi-File Contract Generation

Generate contracts for entire node families:

```python
import asyncio
from pathlib import Path
from typing import List, Dict
from omninode_bridge.codegen.contract_inferencer import ContractInferencer

async def batch_infer_contracts(
    base_path: Path,
    enable_llm: bool = True,
    parallel: bool = True,
) -> Dict[str, str]:
    """
    Generate contracts for all nodes in a directory tree.

    Args:
        base_path: Root directory containing nodes
        enable_llm: Whether to use LLM for config inference
        parallel: Generate contracts in parallel

    Returns:
        Dict mapping node paths to generated contract YAML
    """
    inferencer = ContractInferencer(enable_llm=enable_llm)

    # Find all node.py files
    node_files = list(base_path.rglob("*/node.py"))
    print(f"📋 Found {len(node_files)} node files")

    results = {}

    if parallel:
        # Parallel processing (faster)
        tasks = []
        for node_file in node_files:
            contract_path = node_file.parent / "contract.yaml"
            task = inferencer.infer_from_node(
                node_path=node_file,
                output_path=contract_path
            )
            tasks.append((node_file, task))

        # Execute all tasks concurrently
        completed = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )

        # Collect results
        for (node_file, _), result in zip(tasks, completed):
            if isinstance(result, Exception):
                print(f"❌ Failed: {node_file} - {result}")
            else:
                results[str(node_file)] = result
                print(f"✅ Generated: {node_file.parent.name}")

    else:
        # Sequential processing (more memory-efficient)
        for node_file in node_files:
            try:
                contract_path = node_file.parent / "contract.yaml"
                contract_yaml = await inferencer.infer_from_node(
                    node_path=node_file,
                    output_path=contract_path
                )
                results[str(node_file)] = contract_yaml
                print(f"✅ Generated: {node_file.parent.name}")
            except Exception as e:
                print(f"❌ Failed: {node_file} - {e}")

    await inferencer.cleanup()

    return results


# Usage
async def main():
    results = await batch_infer_contracts(
        base_path=Path("src/omninode_bridge/nodes"),
        enable_llm=True,
        parallel=True,
    )

    print(f"\n📊 Summary:")
    print(f"   Total: {len(results)} contracts generated")
```

### Selective Re-inference

Update only specific contract sections:

```python
import yaml
from pathlib import Path
from omninode_bridge.codegen.contract_inferencer import ContractInferencer

async def update_mixin_configs_only(
    node_path: Path,
    contract_path: Path,
):
    """
    Update only mixin configurations in existing contract.

    Preserves all other contract fields (version, description, etc.)
    """
    inferencer = ContractInferencer(enable_llm=True)

    # Step 1: Generate new contract
    new_contract_yaml = await inferencer.infer_from_node(
        node_path=node_path,
        output_path=None,  # Don't write yet
    )
    new_contract = yaml.safe_load(new_contract_yaml)

    # Step 2: Load existing contract
    existing_contract = yaml.safe_load(contract_path.read_text())

    # Step 3: Merge - keep existing, update only mixins
    existing_contract['mixins'] = new_contract.get('mixins', [])

    # Step 4: Write merged contract
    contract_path.write_text(yaml.dump(existing_contract, sort_keys=False))

    print(f"✅ Updated mixin configs in {contract_path}")

    await inferencer.cleanup()
```

### Custom I/O Operation Detection

Add domain-specific operation detection:

```python
from omninode_bridge.codegen.contract_inferencer import ContractInferencer

class CustomContractInferencer(ContractInferencer):
    """Extended inferencer with custom I/O detection."""

    def _detect_io_operations(self, analysis) -> List[str]:
        """
        Enhanced I/O operation detection with custom patterns.
        """
        # Start with base detection
        operations = super()._detect_io_operations(analysis)

        # Add domain-specific patterns
        imports = analysis.imports

        # Machine Learning operations
        if any(lib in imports for lib in ['torch', 'tensorflow', 'sklearn']):
            operations.append('ml_inference')

        # Blockchain operations
        if any(lib in imports for lib in ['web3', 'eth_account']):
            operations.append('blockchain_transaction')

        # GraphQL operations
        if 'gql' in imports or 'graphql' in imports:
            operations.append('graphql_query')

        # WebSocket operations
        if 'websockets' in imports or 'socketio' in imports:
            operations.append('websocket_connection')

        # Message queue operations (additional providers)
        if 'pika' in imports:
            operations.append('rabbitmq_queue')
        if 'celery' in imports:
            operations.append('task_queue')

        return list(set(operations))  # Deduplicate


# Usage
async def custom_inference_example():
    inferencer = CustomContractInferencer(enable_llm=True)

    contract_yaml = await inferencer.infer_from_node(
        node_path="path/to/ml_node.py"
    )

    # Contract will include 'ml_inference' capability
    await inferencer.cleanup()
```

---

## Custom Mixin Configuration

Mixins are the foundation of ONEX v2.0's extensibility. This section covers advanced mixin configuration techniques.

### Understanding the Mixin Catalog

The mixin catalog defines available mixins and their metadata:

```python
# From src/omninode_bridge/codegen/mixin_injector.py
MIXIN_CATALOG = {
    "MixinHealthCheck": {
        "import_path": "omnibase_core.mixins.mixin_health_check",
        "dependencies": [],
        "required_methods": ["get_health_checks"],
        "description": "Health check implementation with async support",
    },
    "MixinMetrics": {
        "import_path": "omnibase_core.mixins.mixin_metrics",
        "dependencies": [],
        "required_methods": ["record_metric", "get_metrics"],
        "description": "Performance metrics collection and aggregation",
    },
    # ... more mixins
}
```

### Creating a Custom Mixin Catalog

```python
from typing import Dict, Any
from pathlib import Path
import yaml

def create_custom_mixin_catalog(
    base_catalog: Dict[str, Any],
    custom_mixins: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extend the base mixin catalog with organization-specific mixins.

    Args:
        base_catalog: Standard ONEX mixin catalog
        custom_mixins: Organization-specific mixins

    Returns:
        Combined mixin catalog
    """
    catalog = base_catalog.copy()

    # Add custom mixins
    for mixin_name, mixin_config in custom_mixins.items():
        # Validate required fields
        required_fields = ['import_path', 'description']
        for field in required_fields:
            if field not in mixin_config:
                raise ValueError(f"Mixin {mixin_name} missing required field: {field}")

        # Add with defaults
        catalog[mixin_name] = {
            'import_path': mixin_config['import_path'],
            'description': mixin_config['description'],
            'dependencies': mixin_config.get('dependencies', []),
            'required_methods': mixin_config.get('required_methods', []),
            'config_schema': mixin_config.get('config_schema', {}),
        }

    return catalog


# Example: Organization-specific mixins
CUSTOM_MIXINS = {
    "MixinComplianceAudit": {
        "import_path": "myorg.mixins.compliance_audit",
        "description": "Automatic compliance audit logging for regulated industries",
        "required_methods": ["log_audit_event", "get_audit_trail"],
        "config_schema": {
            "retention_days": {"type": "int", "default": 2555},
            "include_pii": {"type": "bool", "default": False},
            "encryption_enabled": {"type": "bool", "default": True},
        },
    },
    "MixinRateLimiting": {
        "import_path": "myorg.mixins.rate_limiting",
        "description": "Token bucket rate limiting per user/IP",
        "required_methods": ["check_rate_limit", "reset_limit"],
        "dependencies": ["MixinMetrics"],  # Depends on metrics
        "config_schema": {
            "max_requests_per_minute": {"type": "int", "default": 100},
            "burst_size": {"type": "int", "default": 20},
            "strategy": {"type": "enum", "values": ["token_bucket", "sliding_window"]},
        },
    },
}

# Combine catalogs
from omninode_bridge.codegen.mixin_injector import MIXIN_CATALOG
EXTENDED_CATALOG = create_custom_mixin_catalog(MIXIN_CATALOG, CUSTOM_MIXINS)
```

### Dynamic Mixin Configuration Based on Environment

```python
import os
from typing import Dict, Any

def get_environment_specific_mixin_config(
    mixin_name: str,
    environment: str,
) -> Dict[str, Any]:
    """
    Generate mixin configuration based on deployment environment.

    Args:
        mixin_name: Name of the mixin
        environment: Target environment (dev, staging, prod)

    Returns:
        Environment-specific mixin configuration
    """
    # Base configurations per environment
    ENV_CONFIGS = {
        'dev': {
            'MixinMetrics': {
                'collect_latency': True,
                'percentiles': [50, 95, 99],
                'sampling_rate': 1.0,  # 100% sampling in dev
            },
            'MixinHealthCheck': {
                'check_interval_ms': 30000,  # 30s in dev
                'timeout_seconds': 5.0,
            },
        },
        'staging': {
            'MixinMetrics': {
                'collect_latency': True,
                'percentiles': [50, 90, 95, 99],
                'sampling_rate': 0.5,  # 50% sampling in staging
            },
            'MixinHealthCheck': {
                'check_interval_ms': 60000,  # 60s in staging
                'timeout_seconds': 10.0,
            },
        },
        'prod': {
            'MixinMetrics': {
                'collect_latency': True,
                'percentiles': [50, 90, 95, 99, 99.9],
                'sampling_rate': 0.1,  # 10% sampling in prod
            },
            'MixinHealthCheck': {
                'check_interval_ms': 120000,  # 120s in prod
                'timeout_seconds': 15.0,
            },
        },
    }

    env_config = ENV_CONFIGS.get(environment, ENV_CONFIGS['dev'])
    return env_config.get(mixin_name, {})


# Usage in contract generation
def generate_environment_aware_contract():
    """Generate contract with environment-specific mixin configs."""
    environment = os.getenv('DEPLOYMENT_ENV', 'dev')

    mixins = []
    for mixin_name in ['MixinMetrics', 'MixinHealthCheck']:
        config = get_environment_specific_mixin_config(mixin_name, environment)
        mixins.append({
            'name': mixin_name,
            'enabled': True,
            'config': config,
        })

    return mixins
```

### Mixin Dependency Resolution

```python
from typing import List, Dict, Set

def resolve_mixin_dependencies(
    requested_mixins: List[str],
    mixin_catalog: Dict[str, Any],
) -> List[str]:
    """
    Resolve mixin dependencies and return ordered list.

    Performs topological sort to ensure dependencies are initialized first.

    Args:
        requested_mixins: List of mixin names to enable
        mixin_catalog: Mixin catalog with dependency information

    Returns:
        Ordered list of mixins (dependencies first)

    Raises:
        ValueError: If circular dependency detected
    """
    resolved: List[str] = []
    seen: Set[str] = set()
    visiting: Set[str] = set()

    def visit(mixin_name: str):
        if mixin_name in visiting:
            raise ValueError(f"Circular dependency detected: {mixin_name}")

        if mixin_name in seen:
            return

        visiting.add(mixin_name)

        # Get dependencies
        mixin_info = mixin_catalog.get(mixin_name)
        if not mixin_info:
            raise ValueError(f"Unknown mixin: {mixin_name}")

        dependencies = mixin_info.get('dependencies', [])

        # Visit dependencies first
        for dep in dependencies:
            visit(dep)

        visiting.remove(mixin_name)
        seen.add(mixin_name)
        resolved.append(mixin_name)

    # Visit all requested mixins
    for mixin in requested_mixins:
        visit(mixin)

    return resolved


# Example usage
requested = ['MixinRateLimiting', 'MixinHealthCheck']
ordered_mixins = resolve_mixin_dependencies(requested, EXTENDED_CATALOG)
print(f"Mixins in initialization order: {ordered_mixins}")
# Output: ['MixinMetrics', 'MixinHealthCheck', 'MixinRateLimiting']
```

---

## LLM Prompt Engineering

The ContractInferencer uses LLM to infer intelligent mixin configurations. This section covers prompt optimization.

### Anatomy of an Inference Prompt

```python
# From ContractInferencer._infer_mixin_configs()

PROMPT_TEMPLATE = """You are an ONEX v2.0 configuration expert. Analyze the node and suggest optimal {mixin_name} configuration.

**Node Information:**
- Name: {node_name}
- Type: {node_type}
- Purpose: {node_purpose}
- I/O Operations: {io_operations}
- Methods: {methods}

**Mixin: {mixin_name}**
{mixin_description}

**Example Configurations:**
{example_configs}

**Task:**
Provide an optimal configuration for {mixin_name} based on the node's purpose and operations.

**Output Format (YAML only):**
```yaml
config_key_1: value1
config_key_2: value2
```

**Confidence Score (0.0-1.0):**
[Your confidence in this configuration]

**Reasoning:**
[Brief explanation of configuration choices]
"""
```

### Optimizing Prompts for Better Results

#### 1. Provide Rich Context

```python
def build_enhanced_prompt_context(
    node_analysis,
    mixin_name: str,
    mixin_catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build comprehensive context for LLM prompts.

    More context = better configuration inference.
    """
    # Extract node characteristics
    context = {
        'node_name': node_analysis.node_name,
        'node_type': node_analysis.node_type,
        'node_purpose': node_analysis.docstring or "Unknown",
        'io_operations': ', '.join(node_analysis.io_operations),
        'methods': ', '.join(node_analysis.methods),
        'has_async': any(m.startswith('async ') for m in node_analysis.methods),
    }

    # Add mixin information
    mixin_info = mixin_catalog.get(mixin_name, {})
    context['mixin_description'] = mixin_info.get('description', '')

    # Add relevant examples from similar nodes
    context['example_configs'] = get_similar_node_configs(
        node_type=node_analysis.node_type,
        mixin_name=mixin_name,
        io_operations=node_analysis.io_operations,
    )

    # Add domain-specific context
    if 'database' in node_analysis.io_operations:
        context['domain_notes'] = "Database operations: Consider connection pooling, retry logic, timeouts"
    elif 'http_request' in node_analysis.io_operations:
        context['domain_notes'] = "HTTP operations: Consider rate limiting, circuit breaker, timeouts"

    return context


def get_similar_node_configs(
    node_type: str,
    mixin_name: str,
    io_operations: List[str],
) -> str:
    """
    Fetch example configurations from similar nodes.

    In production, this would query a database of known-good configs.
    """
    # Example configurations (could be loaded from database)
    EXAMPLE_CONFIGS = {
        ('effect', 'MixinMetrics', 'database_query'): """
# Example from postgres_crud_effect
collect_latency: true
percentiles: [50, 95, 99]
histogram_buckets: [0.01, 0.05, 0.1, 0.5, 1.0]
track_per_operation: true
        """,
        ('effect', 'MixinHealthCheck', 'http_request'): """
# Example from api_client_effect
check_interval_ms: 60000
timeout_seconds: 10.0
healthy_threshold: 3
unhealthy_threshold: 2
        """,
    }

    # Find matching examples
    key = (node_type, mixin_name, io_operations[0] if io_operations else 'generic')
    return EXAMPLE_CONFIGS.get(key, "# No similar examples found")
```

#### 2. Use Few-Shot Learning

```python
def create_few_shot_prompt(
    mixin_name: str,
    node_context: Dict[str, Any],
) -> str:
    """
    Create prompt with few-shot examples for better LLM performance.
    """
    # Few-shot examples (input-output pairs)
    EXAMPLES = """
**Example 1:**
Node: NodePostgresCrudEffect
Type: effect
Operations: database_query
Purpose: PostgreSQL CRUD operations with connection pooling

Configuration:
```yaml
MixinMetrics:
  collect_latency: true
  percentiles: [50, 95, 99]
  track_per_operation: true
  histogram_buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
```
Confidence: 0.95
Reasoning: Database operations benefit from detailed latency tracking. Percentiles and histograms help identify slow queries.

**Example 2:**
Node: NodeHttpClientEffect
Type: effect
Operations: http_request
Purpose: External API client with retry logic

Configuration:
```yaml
MixinHealthCheck:
  check_interval_ms: 60000
  timeout_seconds: 10.0
  healthy_threshold: 3
  unhealthy_threshold: 2
  include_dependencies: true
```
Confidence: 0.90
Reasoning: HTTP clients need regular health checks. Conservative thresholds prevent flapping during transient failures.
"""

    prompt = f"""You are an ONEX v2.0 configuration expert. Study these examples, then configure {mixin_name} for the new node.

{EXAMPLES}

**New Node:**
Name: {node_context['node_name']}
Type: {node_context['node_type']}
Operations: {node_context['io_operations']}
Purpose: {node_context['node_purpose']}

**Task:**
Configure {mixin_name} for this node following the pattern from examples.

**Output Format (YAML only):**
```yaml
config_key: value
```

**Confidence Score (0.0-1.0):**
[Your confidence]

**Reasoning:**
[Explain your choices]
"""

    return prompt
```

#### 3. Temperature and Parameter Tuning

```python
from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier
from omninode_bridge.nodes.llm_effect.v1_0_0.models.model_request import ModelLLMRequest

async def tune_llm_parameters_for_inference(
    inferencer: ContractInferencer,
    prompt: str,
    task_type: str = "config_inference",
) -> str:
    """
    Use optimal LLM parameters for different inference tasks.
    """
    # Task-specific parameters
    PARAMETER_PROFILES = {
        'config_inference': {
            'temperature': 0.3,  # Low temp = consistent configs
            'max_tokens': 1000,
            'top_p': 0.9,
        },
        'creative_generation': {
            'temperature': 0.7,  # Higher temp = more variety
            'max_tokens': 2000,
            'top_p': 0.95,
        },
        'documentation': {
            'temperature': 0.5,  # Medium temp = readable but accurate
            'max_tokens': 1500,
            'top_p': 0.9,
        },
    }

    params = PARAMETER_PROFILES.get(task_type, PARAMETER_PROFILES['config_inference'])

    # Build request
    request = ModelLLMRequest(
        messages=[{"role": "user", "content": prompt}],
        tier=EnumLLMTier.CLOUD_FAST,
        temperature=params['temperature'],
        max_tokens=params['max_tokens'],
        top_p=params['top_p'],
        timeout_seconds=30.0,
    )

    # Call LLM through inferencer's LLM node
    response = await inferencer.llm_node.execute_effect(request)

    return response.content
```

#### 4. Prompt Validation and Iteration

```python
import yaml
from typing import Optional, Tuple

def validate_llm_response(
    response: str,
    mixin_name: str,
) -> Tuple[bool, Optional[Dict], Optional[float], Optional[str]]:
    """
    Validate and parse LLM response for configuration inference.

    Returns:
        (is_valid, config_dict, confidence_score, reasoning)
    """
    try:
        # Extract YAML block
        yaml_start = response.find('```yaml')
        yaml_end = response.find('```', yaml_start + 7)

        if yaml_start == -1 or yaml_end == -1:
            return False, None, None, "No YAML block found"

        yaml_content = response[yaml_start + 7:yaml_end].strip()
        config = yaml.safe_load(yaml_content)

        # Extract confidence score
        confidence_marker = "**Confidence Score"
        confidence_start = response.find(confidence_marker)
        confidence = 0.0
        if confidence_start != -1:
            conf_line = response[confidence_start:].split('\n')[1]
            try:
                confidence = float(conf_line.strip().strip('[]'))
            except ValueError:
                confidence = 0.5  # Default if parsing fails

        # Extract reasoning
        reasoning_marker = "**Reasoning:"
        reasoning_start = response.find(reasoning_marker)
        reasoning = ""
        if reasoning_start != -1:
            reasoning = response[reasoning_start + len(reasoning_marker):].strip()

        return True, config, confidence, reasoning

    except Exception as e:
        return False, None, None, f"Parse error: {e}"


async def iterative_prompt_refinement(
    inferencer: ContractInferencer,
    initial_prompt: str,
    max_iterations: int = 3,
) -> Dict:
    """
    Iteratively refine prompt until valid response obtained.
    """
    prompt = initial_prompt

    for iteration in range(max_iterations):
        response = await inferencer.llm_node.execute_effect(
            ModelLLMRequest(messages=[{"role": "user", "content": prompt}])
        )

        is_valid, config, confidence, reasoning = validate_llm_response(
            response.content,
            mixin_name="MixinMetrics",
        )

        if is_valid and confidence > 0.7:
            return {
                'config': config,
                'confidence': confidence,
                'reasoning': reasoning,
                'iterations': iteration + 1,
            }

        # Refine prompt with feedback
        prompt = f"""{initial_prompt}

**Previous attempt was invalid or low confidence. Issues:**
{reasoning if not is_valid else f"Confidence too low: {confidence}"}

**Requirements:**
- Provide YAML in ```yaml code block
- Include confidence score (0.0-1.0)
- Explain reasoning clearly

**Try again with these fixes:**
"""

    # Failed after max iterations
    return {'config': {}, 'confidence': 0.0, 'reasoning': 'Max iterations reached'}
```

---

## Batch Processing Patterns

For large-scale code generation, batch processing is essential. This section covers efficient batch patterns.

### Parallel Batch Generation

```python
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements

async def parallel_batch_generate(
    requirements_list: List[ModelPRDRequirements],
    output_base_dir: Path,
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate multiple nodes in parallel with concurrency limit.

    Args:
        requirements_list: List of node requirements
        output_base_dir: Base output directory
        max_concurrent: Maximum concurrent generations

    Returns:
        List of generation results
    """
    service = CodeGenerationService()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_limit(req: ModelPRDRequirements, index: int):
        async with semaphore:
            output_dir = output_base_dir / f"{req.service_name}_{index}"

            try:
                result = await service.generate_node(
                    requirements=req,
                    output_directory=output_dir,
                    strategy="auto",
                    validation_level="standard",
                )

                return {
                    'status': 'success',
                    'service_name': req.service_name,
                    'node_name': result.artifacts.node_name,
                    'generation_time_ms': result.generation_time_ms,
                    'output_dir': output_dir,
                }

            except Exception as e:
                return {
                    'status': 'failed',
                    'service_name': req.service_name,
                    'error': str(e),
                }

    # Create tasks for all requirements
    tasks = [
        generate_with_limit(req, idx)
        for idx, req in enumerate(requirements_list)
    ]

    # Execute with progress tracking
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

        # Progress update
        completed = len(results)
        total = len(requirements_list)
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    return results


# Usage example
async def main():
    # Load requirements from file or database
    requirements = [
        ModelPRDRequirements(
            node_type="effect",
            service_name=f"service_{i}",
            domain="api",
            business_description=f"Service {i} API client",
            operations=["get", "post", "delete"],
        )
        for i in range(20)  # Generate 20 nodes
    ]

    results = await parallel_batch_generate(
        requirements_list=requirements,
        output_base_dir=Path("./generated/batch"),
        max_concurrent=5,  # 5 concurrent generations
    )

    # Report
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ Successful: {successful}/{len(results)}")
```

### Resumable Batch Processing

```python
import json
from pathlib import Path
from typing import Optional

class ResumableBatchGenerator:
    """
    Batch generator with checkpointing for resumability.

    Saves progress after each successful generation.
    Can resume from checkpoint if interrupted.
    """

    def __init__(
        self,
        checkpoint_file: Path,
        output_base_dir: Path,
    ):
        self.checkpoint_file = checkpoint_file
        self.output_base_dir = output_base_dir
        self.service = CodeGenerationService()

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            return json.loads(self.checkpoint_file.read_text())
        return {'completed': [], 'failed': [], 'pending': []}

    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save checkpoint to disk."""
        self.checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

    async def generate_batch(
        self,
        requirements_list: List[ModelPRDRequirements],
        resume: bool = True,
    ) -> Dict[str, List]:
        """
        Generate batch with automatic checkpointing.

        Args:
            requirements_list: List of requirements
            resume: Whether to resume from checkpoint

        Returns:
            Dict with completed, failed, and pending lists
        """
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {
            'completed': [],
            'failed': [],
            'pending': [r.service_name for r in requirements_list],
        }

        # Filter to pending only
        pending_names = set(checkpoint['pending'])
        pending_reqs = [
            r for r in requirements_list
            if r.service_name in pending_names
        ]

        print(f"📋 Total: {len(requirements_list)}")
        print(f"✅ Already completed: {len(checkpoint['completed'])}")
        print(f"❌ Previously failed: {len(checkpoint['failed'])}")
        print(f"⏳ Pending: {len(pending_reqs)}")

        # Process pending
        for req in pending_reqs:
            try:
                output_dir = self.output_base_dir / req.service_name

                print(f"\n🔄 Generating: {req.service_name}")

                result = await self.service.generate_node(
                    requirements=req,
                    output_directory=output_dir,
                    strategy="auto",
                    validation_level="standard",
                )

                # Success
                checkpoint['completed'].append({
                    'service_name': req.service_name,
                    'node_name': result.artifacts.node_name,
                    'output_dir': str(output_dir),
                })
                checkpoint['pending'].remove(req.service_name)

                print(f"✅ Success: {result.artifacts.node_name}")

            except Exception as e:
                # Failure
                checkpoint['failed'].append({
                    'service_name': req.service_name,
                    'error': str(e),
                })
                checkpoint['pending'].remove(req.service_name)

                print(f"❌ Failed: {e}")

            # Save checkpoint after each operation
            self.save_checkpoint(checkpoint)

        return checkpoint


# Usage
async def resumable_batch_example():
    generator = ResumableBatchGenerator(
        checkpoint_file=Path("./batch_checkpoint.json"),
        output_base_dir=Path("./generated/resumable"),
    )

    requirements = [...]  # Your requirements list

    # First run
    checkpoint = await generator.generate_batch(requirements, resume=False)

    # If interrupted, resume later
    # checkpoint = await generator.generate_batch(requirements, resume=True)
```

### Load-Based Batch Scheduling

```python
import psutil
import time
from typing import Callable, Awaitable

class AdaptiveBatchScheduler:
    """
    Batch scheduler that adapts concurrency based on system load.

    Monitors CPU, memory, and adjusts concurrent tasks accordingly.
    """

    def __init__(
        self,
        min_concurrent: int = 1,
        max_concurrent: int = 10,
        cpu_threshold: float = 80.0,  # %
        memory_threshold: float = 85.0,  # %
    ):
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.current_concurrent = min_concurrent

    def get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io_busy': psutil.disk_io_counters().busy_time if hasattr(psutil.disk_io_counters(), 'busy_time') else 0,
        }

    def adjust_concurrency(self):
        """Adjust concurrent task limit based on system load."""
        load = self.get_system_load()

        if load['cpu_percent'] > self.cpu_threshold or load['memory_percent'] > self.memory_threshold:
            # System under stress - reduce concurrency
            self.current_concurrent = max(
                self.min_concurrent,
                self.current_concurrent - 1,
            )
            print(f"⚠️  High load detected - reducing concurrency to {self.current_concurrent}")

        elif load['cpu_percent'] < 50 and load['memory_percent'] < 70:
            # System has capacity - increase concurrency
            self.current_concurrent = min(
                self.max_concurrent,
                self.current_concurrent + 1,
            )
            print(f"✅ System capacity available - increasing concurrency to {self.current_concurrent}")

    async def execute_batch_adaptive(
        self,
        tasks: List[Callable[[], Awaitable]],
        adjustment_interval: int = 10,  # Adjust every N tasks
    ):
        """Execute tasks with adaptive concurrency."""
        results = []

        for i, task in enumerate(tasks):
            # Adjust concurrency periodically
            if i % adjustment_interval == 0:
                self.adjust_concurrency()

            # Execute with current concurrency limit
            semaphore = asyncio.Semaphore(self.current_concurrent)

            async def execute_limited():
                async with semaphore:
                    return await task()

            result = await execute_limited()
            results.append(result)

        return results
```

---

## CI/CD Integration

Integrate code generation into CI/CD pipelines for automated node creation and validation.

### GitHub Actions Integration

```yaml
# .github/workflows/generate-nodes.yml
name: Generate ONEX Nodes

on:
  workflow_dispatch:
    inputs:
      requirements_file:
        description: 'Path to requirements JSON file'
        required: true
        default: 'requirements/new_nodes.json'
      enable_llm:
        description: 'Enable LLM for config inference'
        required: false
        default: 'true'

jobs:
  generate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Load requirements
        id: load_reqs
        run: |
          echo "requirements=$(cat ${{ github.event.inputs.requirements_file }})" >> $GITHUB_OUTPUT

      - name: Generate nodes
        env:
          ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
          ENABLE_LLM: ${{ github.event.inputs.enable_llm }}
        run: |
          poetry run python scripts/ci_generate_nodes.py \
            --requirements "${{ steps.load_reqs.outputs.requirements }}" \
            --output-dir ./generated \
            --enable-llm ${{ github.event.inputs.enable_llm }}

      - name: Validate generated code
        run: |
          poetry run python scripts/validate_generated_code.py \
            --generated-dir ./generated

      - name: Run tests
        run: |
          poetry run pytest ./generated/*/tests/

      - name: Create PR with generated nodes
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'feat: Add generated ONEX nodes'
          title: 'Generated ONEX Nodes from Requirements'
          body: |
            ## Generated Nodes

            This PR contains automatically generated ONEX v2.0 nodes.

            **Source:** ${{ github.event.inputs.requirements_file }}
            **LLM Enabled:** ${{ github.event.inputs.enable_llm }}

            ## Validation

            - [x] Code generation successful
            - [x] ONEX v2.0 compliance validated
            - [x] All tests passing

            ## Files Changed

            See commit for generated node files.
          branch: generated-nodes-${{ github.run_number }}
```

### CI Generation Script

```python
# scripts/ci_generate_nodes.py
"""
CI/CD script for batch node generation.

Usage:
    python scripts/ci_generate_nodes.py \
        --requirements requirements.json \
        --output-dir ./generated \
        --enable-llm true
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements


async def ci_generate_nodes(
    requirements_file: Path,
    output_dir: Path,
    enable_llm: bool,
    fail_on_error: bool = True,
) -> int:
    """
    Generate nodes for CI/CD pipeline.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print("🚀 CI/CD Node Generation")
    print("=" * 60)

    # Load requirements
    print(f"\n📋 Loading requirements from: {requirements_file}")
    requirements_data = json.loads(requirements_file.read_text())
    requirements_list = [
        ModelPRDRequirements(**req)
        for req in requirements_data
    ]
    print(f"   Found {len(requirements_list)} node definitions")

    # Initialize service
    service = CodeGenerationService()

    # Generate all nodes
    results = []
    for i, req in enumerate(requirements_list, 1):
        print(f"\n[{i}/{len(requirements_list)}] Generating: {req.service_name}")

        node_output_dir = output_dir / req.service_name

        try:
            result = await service.generate_node(
                requirements=req,
                output_directory=node_output_dir,
                strategy="auto",
                validation_level="strict",  # Strict for CI
            )

            # Write files
            for filename, content in result.artifacts.get_all_files().items():
                file_path = node_output_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            results.append({
                'status': 'success',
                'service_name': req.service_name,
                'node_name': result.artifacts.node_name,
            })

            print(f"   ✅ Success: {result.artifacts.node_name}")

        except Exception as e:
            results.append({
                'status': 'failed',
                'service_name': req.service_name,
                'error': str(e),
            })

            print(f"   ❌ Failed: {e}")

            if fail_on_error:
                print("\n❌ Generation failed - aborting")
                return 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 Generation Summary")
    print("=" * 60)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   - {r['node_name']}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['service_name']}: {r['error']}")

    # Write results to file for next steps
    results_file = output_dir / "generation_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\n📄 Results written to: {results_file}")

    return 0 if len(failed) == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="CI/CD node generation")
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-llm", type=bool, default=True)
    parser.add_argument("--fail-on-error", type=bool, default=True)

    args = parser.parse_args()

    exit_code = asyncio.run(ci_generate_nodes(
        requirements_file=args.requirements,
        output_dir=args.output_dir,
        enable_llm=args.enable_llm,
        fail_on_error=args.fail_on_error,
    ))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

### Validation Script

```python
# scripts/validate_generated_code.py
"""
Validate generated code before merging.

Checks:
- ONEX v2.0 compliance
- Python syntax
- Contract validity
- Test coverage
"""

import ast
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def validate_python_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate Python file syntax."""
    try:
        ast.parse(file_path.read_text())
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def validate_contract_schema(contract_path: Path) -> Tuple[bool, str]:
    """Validate contract YAML schema."""
    try:
        contract = yaml.safe_load(contract_path.read_text())

        # Required fields
        required = ['schema_version', 'name', 'version', 'node_type']
        for field in required:
            if field not in contract:
                return False, f"Missing required field: {field}"

        # Version format
        if not isinstance(contract['version'], dict):
            return False, "Version must be dict with major/minor/patch"

        return True, "Valid contract"

    except Exception as e:
        return False, f"Contract error: {e}"


def validate_node_directory(node_dir: Path) -> Dict[str, Any]:
    """Validate complete node directory."""
    results = {
        'node_dir': str(node_dir),
        'checks': {},
        'overall_valid': True,
    }

    # Check required files exist
    required_files = ['node.py', 'contract.yaml', '__init__.py']
    for filename in required_files:
        file_path = node_dir / filename
        exists = file_path.exists()
        results['checks'][f'has_{filename}'] = exists
        if not exists:
            results['overall_valid'] = False

    # Validate Python files
    for py_file in node_dir.rglob("*.py"):
        is_valid, message = validate_python_syntax(py_file)
        results['checks'][f'syntax_{py_file.name}'] = is_valid
        if not is_valid:
            results['overall_valid'] = False
            results['checks'][f'syntax_{py_file.name}_error'] = message

    # Validate contract
    contract_path = node_dir / "contract.yaml"
    if contract_path.exists():
        is_valid, message = validate_contract_schema(contract_path)
        results['checks']['contract_valid'] = is_valid
        if not is_valid:
            results['overall_valid'] = False
            results['checks']['contract_error'] = message

    # Check for tests
    test_dir = node_dir / "tests"
    has_tests = test_dir.exists() and any(test_dir.glob("test_*.py"))
    results['checks']['has_tests'] = has_tests

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate generated code")
    parser.add_argument("--generated-dir", type=Path, required=True)
    args = parser.parse_args()

    print("🔍 Validating Generated Code")
    print("=" * 60)

    # Find all node directories
    node_dirs = [d for d in args.generated_dir.iterdir() if d.is_dir()]

    all_results = []
    for node_dir in node_dirs:
        print(f"\nValidating: {node_dir.name}")
        results = validate_node_directory(node_dir)
        all_results.append(results)

        status = "✅" if results['overall_valid'] else "❌"
        print(f"   {status} {node_dir.name}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)

    valid_count = sum(1 for r in all_results if r['overall_valid'])
    print(f"\n✅ Valid: {valid_count}/{len(all_results)}")
    print(f"❌ Invalid: {len(all_results) - valid_count}/{len(all_results)}")

    # Exit with error if any invalid
    if valid_count < len(all_results):
        print("\n❌ Validation failed - see errors above")
        sys.exit(1)

    print("\n✅ All validations passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## Performance Optimization

Optimize code generation performance for large-scale operations.

### Caching Strategies

```python
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Optional

class GenerationCache:
    """
    Cache for code generation artifacts.

    Caches based on requirements hash to avoid regenerating identical nodes.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, requirements: ModelPRDRequirements) -> str:
        """Compute hash of requirements for cache key."""
        req_dict = requirements.model_dump()
        req_json = json.dumps(req_dict, sort_keys=True)
        return hashlib.sha256(req_json.encode()).hexdigest()[:16]

    def get(self, requirements: ModelPRDRequirements) -> Optional[Dict]:
        """Get cached generation result."""
        cache_key = self._compute_hash(requirements)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            return json.loads(cache_file.read_text())

        return None

    def set(self, requirements: ModelPRDRequirements, result: Dict):
        """Cache generation result."""
        cache_key = self._compute_hash(requirements)
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(result, indent=2))

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


# Usage
async def cached_generation():
    cache = GenerationCache(cache_dir=Path("./.cache/codegen"))
    service = CodeGenerationService()

    requirements = ModelPRDRequirements(...)

    # Check cache first
    cached_result = cache.get(requirements)
    if cached_result:
        print("✅ Using cached result")
        return cached_result

    # Generate if not cached
    result = await service.generate_node(requirements, ...)

    # Cache result
    cache.set(requirements, result.model_dump())

    return result
```

### Template Pre-compilation

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PrecompiledTemplateEngine:
    """
    Template engine with pre-compiled templates for faster rendering.
    """

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            auto_reload=False,  # Disable auto-reload for performance
            cache_size=100,  # Cache compiled templates
        )

        # Pre-compile all templates
        self._precompile_templates()

    def _precompile_templates(self):
        """Pre-compile all templates on initialization."""
        template_files = Path(self.env.loader.searchpath[0]).rglob("*.j2")

        for template_file in template_files:
            template_name = str(template_file.relative_to(self.env.loader.searchpath[0]))
            # Accessing template compiles and caches it
            _ = self.env.get_template(template_name)

        print(f"✅ Pre-compiled {len(list(template_files))} templates")

    def render(self, template_name: str, context: Dict) -> str:
        """Render template with context (uses cached compiled version)."""
        template = self.env.get_template(template_name)
        return template.render(**context)
```

### Parallel LLM Inference

```python
async def parallel_mixin_config_inference(
    inferencer: ContractInferencer,
    node_analysis,
    mixin_names: List[str],
) -> Dict[str, Dict]:
    """
    Infer mixin configurations in parallel instead of sequentially.

    Significant speedup for nodes with multiple mixins.
    """
    async def infer_single_mixin(mixin_name: str):
        config = await inferencer._infer_single_mixin_config(
            node_analysis=node_analysis,
            mixin_name=mixin_name,
        )
        return mixin_name, config

    # Infer all mixins in parallel
    tasks = [infer_single_mixin(name) for name in mixin_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    configs = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"⚠️  Mixin inference failed: {result}")
            continue

        mixin_name, config = result
        configs[mixin_name] = config

    return configs
```

---

## Advanced Troubleshooting

Common advanced issues and solutions.

### Issue: LLM Inference Timeout

**Symptom**: ContractInferencer hangs or times out during LLM calls.

**Diagnosis**:
```python
import asyncio
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('omninode_bridge.codegen').setLevel(logging.DEBUG)

# Monitor LLM calls
async def diagnose_llm_timeout():
    inferencer = ContractInferencer(enable_llm=True)

    try:
        # Set shorter timeout for debugging
        inferencer.llm_timeout = 10.0  # 10 seconds

        contract_yaml = await asyncio.wait_for(
            inferencer.infer_from_node("path/to/node.py"),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        print("❌ LLM inference timed out")
        print("Check:")
        print("  1. ZAI_API_KEY is valid")
        print("  2. Z.ai service is reachable")
        print("  3. Node analysis isn't too large")
    finally:
        await inferencer.cleanup()
```

**Solutions**:
1. Increase timeout: `inferencer.llm_timeout = 60.0`
2. Use faster tier: `llm_tier=EnumLLMTier.CLOUD_FAST`
3. Simplify prompt: Reduce context size
4. Fallback to no-LLM mode: `enable_llm=False`

### Issue: Memory Usage During Batch Generation

**Symptom**: Memory usage grows unbounded during large batch operations.

**Diagnosis**:
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory: {memory_mb:.1f} MB")
    return memory_mb


async def memory_efficient_batch():
    initial_memory = monitor_memory()

    for i in range(100):
        # Generate node
        result = await service.generate_node(...)

        # Force garbage collection
        gc.collect()

        current_memory = monitor_memory()
        delta = current_memory - initial_memory

        if delta > 500:  # More than 500MB increase
            print(f"⚠️  Memory leak detected at iteration {i}")
            break
```

**Solutions**:
1. Process in smaller batches with cleanup between
2. Use resumable batch processing with checkpoints
3. Explicitly cleanup: `await inferencer.cleanup(); gc.collect()`
4. Reduce concurrency: Lower `max_concurrent` parameter

### Issue: Contract Validation Failures

**Symptom**: Generated contracts fail ONEX v2.0 validation.

**Diagnosis**:
```python
from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser

def diagnose_contract_validation(contract_yaml: str):
    """Diagnose contract validation issues."""
    parser = YAMLContractParser()

    try:
        contract = parser.parse_yaml_contract(contract_yaml)
        print("✅ Contract is valid")
        return contract
    except Exception as e:
        print(f"❌ Contract validation failed: {e}")

        # Detailed diagnosis
        try:
            import yaml
            contract_dict = yaml.safe_load(contract_yaml)

            # Check required fields
            required = ['schema_version', 'name', 'version', 'node_type']
            missing = [f for f in required if f not in contract_dict]

            if missing:
                print(f"Missing required fields: {missing}")

            # Check version format
            if 'version' in contract_dict:
                version = contract_dict['version']
                if not isinstance(version, dict):
                    print(f"Invalid version format: {version}")
                elif not all(k in version for k in ['major', 'minor', 'patch']):
                    print(f"Version missing major/minor/patch: {version}")

        except yaml.YAMLError as yaml_err:
            print(f"YAML parsing error: {yaml_err}")
```

**Solutions**:
1. Validate schema version: Ensure `schema_version: v2.0.0`
2. Check mixin references: All mixins must exist in catalog
3. Validate subcontracts: Must match node type
4. Use strict validation during generation

---

## Best Practices

### 1. Always Version Your Requirements

```python
# Good - version tracked
requirements_v1 = {
    "version": "1.0.0",
    "nodes": [...]
}

# Bad - no version tracking
requirements = [...]
```

### 2. Use Validation Checkpoints

```python
# Validate at multiple stages
async def robust_generation():
    # 1. Validate requirements
    service.validate_requirements(requirements)

    # 2. Generate with strict validation
    result = await service.generate_node(
        requirements=requirements,
        validation_level="strict",
    )

    # 3. Post-generation validation
    validate_generated_code(result.artifacts)

    # 4. Integration test
    run_integration_tests(result.artifacts)
```

### 3. Implement Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def generate_with_retry(requirements):
    return await service.generate_node(requirements, ...)
```

### 4. Monitor and Log Everything

```python
import logging
from omninode_bridge.codegen import CodeGenerationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codegen.log'),
        logging.StreamHandler(),
    ],
)

# All code generation operations are automatically logged
```

### 5. Use Type Hints Everywhere

```python
from typing import List, Dict, Optional
from pathlib import Path

async def well_typed_function(
    requirements: ModelPRDRequirements,
    output_dir: Path,
    enable_cache: bool = True,
) -> ModelGenerationResult:
    """Type hints make code self-documenting and enable IDE support."""
    ...
```

---

## Conclusion

This advanced guide covers sophisticated patterns for ONEX code generation. For additional help:

- **[Code Generation Guide](./CODE_GENERATION_GUIDE.md)** - Core concepts
- **[Mixin Catalog Reference](../reference/OMNIBASE_CORE_MIXIN_CATALOG.md)** - Available mixins
- **[Contract Inferencer Docs](../codegen/CONTRACT_INFERENCER.md)** - Inference details
- **[Examples](../../examples/codegen/)** - Working code examples

**Next Steps**:
1. Experiment with ContractInferencer on your nodes
2. Create custom mixin catalogs for your organization
3. Implement CI/CD integration
4. Optimize for your specific workload

**Questions or Issues?**
- Open an issue on GitHub
- Consult the OmniNode team
- Check existing examples in `examples/codegen/`

---

**Document Version**: 2.0.0
**Last Updated**: 2025-11-05
**Maintainer**: OmniNode Team
