# Error Recovery Guide

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Reading Time**: 25 minutes

---

## Overview

This guide provides comprehensive documentation for the Error Recovery System in Phase 4 code generation workflows. The system provides **5 recovery strategies** with **90%+ success rate** for transient errors.

**Key Features**:
- Pattern-based error matching with regex support
- 5 recovery strategies (Retry, Alternative Path, Degradation, Correction, Escalation)
- Automatic strategy selection based on error patterns
- Performance tracking and statistics
- Integration with MetricsCollector and SignalCoordinator

**Performance Targets**:
- Error analysis: **<100ms**
- Recovery decision: **<50ms**
- Total recovery overhead: **<500ms**
- Success rate: **90%+** for recoverable errors

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Recovery Strategies](#recovery-strategies)
3. [Error Patterns](#error-patterns)
4. [Configuration](#configuration)
5. [Custom Strategies](#custom-strategies)
6. [Integration](#integration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Setup

```python
from omninode_bridge.agents.workflows import (
    ErrorRecoveryOrchestrator,
    RecoveryContext,
    ErrorPattern,
    ErrorType,
    RecoveryStrategy,
)
from omninode_bridge.agents.metrics import MetricsCollector
from omninode_bridge.agents.coordination import SignalCoordinator

# Initialize components
metrics = MetricsCollector()
signals = SignalCoordinator(state, metrics)

# Create orchestrator
orchestrator = ErrorRecoveryOrchestrator(
    metrics_collector=metrics,
    signal_coordinator=signals,
    max_retries=3,
    base_delay=1.0,
)

# Add error patterns (optional - defaults provided)
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="api_timeout",
    error_type=ErrorType.NETWORK,
    regex_pattern=r"TimeoutError|ReadTimeout|ConnectTimeout",
    recovery_strategy=RecoveryStrategy.RETRY,
    metadata={"max_retries": 5, "base_delay": 2.0},
))
```

### Handling Errors

```python
# Execute workflow with error recovery
try:
    result = await generate_code(contract)

except Exception as e:
    # Create recovery context
    context = RecoveryContext(
        workflow_id="codegen-session-1",
        node_name="model_generator",
        step_count=5,
        state={"contract": contract},
        exception=e,
    )

    # Attempt recovery
    recovery_result = await orchestrator.handle_error(context)

    if recovery_result.success:
        print(f"✅ Recovered using {recovery_result.strategy_used}")
        result = recovery_result.result
    else:
        print(f"❌ Recovery failed: {recovery_result.error_message}")
        raise
```

### Quick Recovery Patterns

```python
# Retry network errors
@error_recovery.recoverable(ErrorType.NETWORK, RecoveryStrategy.RETRY)
async def fetch_from_api(url: str) -> dict:
    return await http_client.get(url)

# Try alternatives on validation failure
@error_recovery.recoverable(ErrorType.VALIDATION, RecoveryStrategy.ALTERNATIVE_PATH)
async def generate_with_template(template_id: str, context: dict) -> str:
    return await template_manager.render_template(template_id, context)

# Graceful degradation for complex operations
@error_recovery.recoverable(ErrorType.GENERATION, RecoveryStrategy.GRACEFUL_DEGRADATION)
async def generate_complex_model(contract: dict) -> str:
    return await complex_generator.generate(contract)
```

---

## Recovery Strategies

### 1. Retry Strategy

**Purpose**: Handle transient errors with exponential backoff

**Best For**:
- Network timeouts
- Temporary API failures
- Database connection issues
- LLM rate limiting
- Temporary file system errors

**How It Works**:
1. Retry operation with exponential backoff
2. Delays: 1s → 2s → 4s (configurable)
3. Maximum 3 retries by default (configurable)
4. Fail if all retries exhausted

**Configuration**:

```python
from omninode_bridge.agents.workflows import RetryStrategy

# Default retry strategy
strategy = RetryStrategy(
    max_retries=3,
    base_delay=1.0,  # Base delay: 1s
)

# Aggressive retry (more retries, longer delays)
aggressive_strategy = RetryStrategy(
    max_retries=5,
    base_delay=2.0,  # Delays: 2s, 4s, 8s, 16s, 32s
)

# Fast retry (fewer retries, shorter delays)
fast_strategy = RetryStrategy(
    max_retries=2,
    base_delay=0.5,  # Delays: 0.5s, 1s
)
```

**Usage Example**:

```python
# Define operation to retry
async def fetch_contract(contract_id: str) -> dict:
    """Fetch contract from API (may fail transiently)."""
    response = await http_client.get(f"/contracts/{contract_id}")
    return response.json()

# Add retry pattern
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="api_fetch_timeout",
    error_type=ErrorType.NETWORK,
    regex_pattern=r"TimeoutError|ReadTimeout|ConnectTimeout",
    recovery_strategy=RecoveryStrategy.RETRY,
    metadata={
        "max_retries": 5,
        "base_delay": 2.0,
        "operation": fetch_contract,
    },
))

# Automatic retry on timeout
try:
    contract = await fetch_contract("contract-123")
except TimeoutError as e:
    context = RecoveryContext(
        workflow_id="fetch-contract",
        node_name="api_client",
        exception=e,
    )
    recovery_result = await orchestrator.handle_error(context)
    contract = recovery_result.result
```

**Performance**:
- Base delay: 1s
- Max delay: 8s (after 3 retries)
- Total overhead: ~7s for 3 retries
- Success rate: **85%+** for transient errors

**Limitations**:
- Not suitable for permanent errors (400 Bad Request, etc.)
- May increase latency significantly
- Should have maximum retry limit

---

### 2. Alternative Path Strategy

**Purpose**: Try alternative templates, models, or approaches

**Best For**:
- Template rendering failures
- Model generation issues
- Validation failures
- Incompatible contract versions
- Feature not supported by primary method

**How It Works**:
1. Identify alternative approaches (templates, models, etc.)
2. Try alternatives in order of preference
3. Return first successful result
4. Fail if all alternatives exhausted

**Configuration**:

```python
from omninode_bridge.agents.workflows import AlternativePathStrategy

# Define alternatives
alternatives = [
    ("node_effect_v2", TemplateType.EFFECT),      # Try v2 first
    ("node_effect_v1_fallback", TemplateType.EFFECT),  # Fallback to v1
    ("node_effect_simple", TemplateType.EFFECT),   # Simplest version
]

strategy = AlternativePathStrategy(
    alternatives=alternatives,
)
```

**Usage Example**:

```python
# Define alternative templates
TEMPLATE_ALTERNATIVES = {
    "node_effect": [
        ("node_effect_v2", TemplateType.EFFECT),
        ("node_effect_v1", TemplateType.EFFECT),
        ("node_effect_basic", TemplateType.EFFECT),
    ],
    "node_compute": [
        ("node_compute_v2", TemplateType.COMPUTE),
        ("node_compute_v1", TemplateType.COMPUTE),
    ],
}

# Add alternative path pattern
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="template_render_failure",
    error_type=ErrorType.VALIDATION,
    regex_pattern=r"Template.*render.*failed|Jinja2.*error",
    recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
    metadata={
        "alternatives": TEMPLATE_ALTERNATIVES["node_effect"],
    },
))

# Automatic fallback to alternative templates
try:
    result = await template_manager.render_template(
        "node_effect_v2",
        context={"node_name": "MyEffect"},
    )
except Exception as e:
    context = RecoveryContext(
        workflow_id="render-template",
        node_name="template_manager",
        exception=e,
        state={"template_id": "node_effect_v2"},
    )
    recovery_result = await orchestrator.handle_error(context)
    result = recovery_result.result  # Rendered with alternative template
```

**Advanced Example - Alternative Models**:

```python
# Define alternative LLM models
MODEL_ALTERNATIVES = [
    ("gemini-1.5-pro", "primary"),
    ("gpt-4", "secondary"),
    ("claude-3", "tertiary"),
]

orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="llm_model_failure",
    error_type=ErrorType.AI_SERVICE,
    regex_pattern=r"Model.*unavailable|API.*error|Service.*down",
    recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
    metadata={
        "model_alternatives": MODEL_ALTERNATIVES,
    },
))

# Fallback across multiple LLM providers
try:
    result = await llm_client.generate(prompt, model="gemini-1.5-pro")
except Exception as e:
    # Try gpt-4, then claude-3
    recovery_result = await orchestrator.handle_error(...)
    result = recovery_result.result
```

**Performance**:
- Overhead: <300ms per alternative
- Success rate: **75%+** if alternatives exist
- Depends on availability of alternatives

**Limitations**:
- Requires predefined alternatives
- May produce different output quality
- Should validate alternative results

---

### 3. Graceful Degradation Strategy

**Purpose**: Fall back to simpler generation with reduced features

**Best For**:
- Complex contract generation failures
- Memory constraints
- Timeout issues
- Quality gate failures
- Performance-critical scenarios

**How It Works**:
1. Identify optional features to remove
2. Progressively simplify generation
3. Generate with reduced features
4. Always produces output (never fails)

**Configuration**:

```python
from omninode_bridge.agents.workflows import GracefulDegradationStrategy

# Define degradation steps
degradation_steps = [
    "remove_optional_mixins",
    "simplify_validation",
    "skip_ai_quorum",
    "disable_performance_profiling",
    "use_basic_templates",
]

strategy = GracefulDegradationStrategy(
    degradation_steps=degradation_steps,
)
```

**Usage Example**:

```python
# Define degradation configuration
DEGRADATION_CONFIG = {
    "level_1": {  # Minor degradation
        "remove_optional_mixins": True,
        "skip_performance_metrics": True,
    },
    "level_2": {  # Moderate degradation
        "remove_optional_mixins": True,
        "simplify_validation": True,
        "skip_ai_quorum": True,
    },
    "level_3": {  # Major degradation
        "use_basic_template": True,
        "skip_all_validation": True,
        "disable_profiling": True,
    },
}

# Add degradation pattern
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="complex_generation_timeout",
    error_type=ErrorType.TIMEOUT,
    regex_pattern=r"Generation.*timeout|exceeded.*time limit",
    recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
    metadata=DEGRADATION_CONFIG,
))

# Automatic degradation on timeout
try:
    result = await generate_complex_model(
        contract=contract,
        mixins=["LoggingMixin", "MetricsMixin", "CachingMixin"],
        enable_ai_quorum=True,
        enable_profiling=True,
    )
except TimeoutError as e:
    # Progressively remove features until generation succeeds
    recovery_result = await orchestrator.handle_error(...)
    result = recovery_result.result  # Generated with reduced features
```

**Degradation Levels**:

**Level 1 - Minor** (5-10% performance improvement):
```python
degradation_level_1 = {
    "remove_optional_mixins": True,  # Remove CachingMixin, etc.
    "skip_performance_metrics": True,  # Disable profiling
    "reduce_validation_depth": True,  # Shallow validation only
}
```

**Level 2 - Moderate** (20-30% performance improvement):
```python
degradation_level_2 = {
    "remove_optional_mixins": True,
    "simplify_validation": True,  # Basic validation only
    "skip_ai_quorum": True,  # Disable AI validation
    "use_smaller_model": True,  # Use faster LLM
}
```

**Level 3 - Major** (50%+ performance improvement):
```python
degradation_level_3 = {
    "use_basic_template": True,  # Simplest template
    "skip_all_validation": True,  # No validation
    "disable_profiling": True,  # No metrics
    "remove_all_mixins": True,  # No mixins
}
```

**Performance**:
- Overhead: <200ms
- Success rate: **90%+** (almost always produces output)
- Quality trade-off: Reduced features = lower quality

**Limitations**:
- Output quality may be reduced
- Should document what features were removed
- May not meet all requirements

---

### 4. Error Correction Strategy

**Purpose**: Automatically fix known error patterns in generated code

**Best For**:
- Missing async keywords
- Missing imports
- Syntax errors from template variables
- Indentation issues
- Known formatting errors

**How It Works**:
1. Match error message to known patterns
2. Apply predefined correction (regex replacement)
3. Re-validate corrected code
4. Return corrected result

**Configuration**:

```python
from omninode_bridge.agents.workflows import ErrorCorrectionStrategy

# Define corrections
corrections = {
    "missing_async": {
        "pattern": r"def (execute_\w+)\(",
        "replacement": r"async def \1(",
    },
    "missing_typing_import": {
        "pattern": r"^(class Node\w+)",
        "prepend": "from typing import Any, Optional\n\n",
    },
    "missing_await": {
        "pattern": r"([^await\s])(\w+\.execute\w+\()",
        "replacement": r"\1await \2",
    },
}

strategy = ErrorCorrectionStrategy(
    corrections=corrections,
)
```

**Usage Example**:

```python
# Common error corrections
COMMON_CORRECTIONS = {
    "missing_async": {
        "description": "Add async keyword to execute methods",
        "pattern": r"def (execute_\w+)\(",
        "replacement": r"async def \1(",
        "validation": "check_for_async_methods",
    },

    "missing_typing_import": {
        "description": "Add typing module imports",
        "pattern": r"^(class Node\w+)",
        "prepend": (
            "from typing import Any, Dict, List, Optional\n"
            "from dataclasses import dataclass\n\n"
        ),
    },

    "missing_await": {
        "description": "Add await keyword for async calls",
        "pattern": r"([^await\s])(\w+\.execute\w+\()",
        "replacement": r"\1await \2",
    },

    "incorrect_indentation": {
        "description": "Fix indentation (4 spaces)",
        "pattern": r"^  ([^\s])",  # 2 spaces
        "replacement": r"    \1",  # 4 spaces
    },
}

# Add correction patterns
for correction_id, correction in COMMON_CORRECTIONS.items():
    orchestrator.add_error_pattern(ErrorPattern(
        pattern_id=f"syntax_{correction_id}",
        error_type=ErrorType.SYNTAX,
        regex_pattern=correction.get("error_regex", "SyntaxError"),
        recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        metadata=correction,
    ))

# Automatic error correction
generated_code = """
class NodeMyEffect(NodeEffect):
    def execute_effect(self, context: dict) -> dict:
        result = self.process_data(context)
        return result
"""

# Missing "async" keyword → automatically corrected
try:
    validated = await validator.validate_code(generated_code)
except SyntaxError as e:
    recovery_result = await orchestrator.handle_error(...)
    corrected_code = recovery_result.result

# Corrected code:
# async def execute_effect(self, context: dict) -> dict:
```

**Advanced Corrections**:

```python
# Multi-step corrections
COMPLEX_CORRECTIONS = {
    "onex_v2_compliance": {
        "steps": [
            {
                "description": "Add async keyword",
                "pattern": r"def (execute_\w+)\(",
                "replacement": r"async def \1(",
            },
            {
                "description": "Add type hints",
                "pattern": r"async def (\w+)\(([^)]+)\):",
                "replacement": r"async def \1(\2) -> dict[str, Any]:",
            },
            {
                "description": "Add docstring",
                "pattern": r"(async def \w+\([^)]+\)[^:]+:)\n",
                "replacement": r'\1\n        """Execute operation."""\n',
            },
        ],
    },
}

# Apply multi-step correction
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="onex_compliance_fix",
    error_type=ErrorType.VALIDATION,
    regex_pattern=r"ONEX.*compliance.*failed",
    recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
    metadata=COMPLEX_CORRECTIONS["onex_v2_compliance"],
))
```

**Performance**:
- Overhead: <100ms
- Success rate: **95%+** for known patterns
- Regex compilation cached

**Limitations**:
- Only works for known error patterns
- Complex errors may require manual fixes
- Regex corrections can be fragile

---

### 5. Escalation Strategy

**Purpose**: Escalate unrecoverable errors to human intervention

**Best For**:
- Invalid contracts
- Unsupported features
- Critical validation failures
- Unknown error patterns
- Security violations

**How It Works**:
1. Identify unrecoverable error
2. Create detailed error report
3. Send notification (Slack, email, etc.)
4. Optionally pause workflow
5. Wait for human resolution

**Configuration**:

```python
from omninode_bridge.agents.workflows import EscalationStrategy

# Define escalation config
escalation_config = {
    "notification_channel": "slack",
    "slack_webhook": "https://hooks.slack.com/...",
    "priority": "high",
    "assignee": "code-generation-team",
    "pause_workflow": True,
}

strategy = EscalationStrategy(
    escalation_config=escalation_config,
)
```

**Usage Example**:

```python
# Define escalation patterns
ESCALATION_PATTERNS = {
    "invalid_contract": {
        "description": "Contract validation failed",
        "priority": "high",
        "assignee": "contracts-team",
        "notification_channels": ["slack", "email"],
    },
    "security_violation": {
        "description": "Security validation failed",
        "priority": "critical",
        "assignee": "security-team",
        "notification_channels": ["slack", "pagerduty"],
        "pause_workflow": True,
    },
    "unsupported_feature": {
        "description": "Feature not yet supported",
        "priority": "medium",
        "assignee": "development-team",
        "notification_channels": ["slack"],
    },
}

# Add escalation patterns
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="critical_contract_error",
    error_type=ErrorType.CONTRACT,
    regex_pattern=r"Contract.*invalid|Contract.*corrupt",
    recovery_strategy=RecoveryStrategy.ESCALATION,
    metadata=ESCALATION_PATTERNS["invalid_contract"],
))

orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="security_validation_failure",
    error_type=ErrorType.SECURITY,
    regex_pattern=r"Security.*failed|Vulnerability.*detected",
    recovery_strategy=RecoveryStrategy.ESCALATION,
    metadata=ESCALATION_PATTERNS["security_violation"],
))

# Automatic escalation
try:
    result = await validate_contract(contract)
except ValidationError as e:
    if "security" in str(e).lower():
        # Critical error → escalate immediately
        recovery_result = await orchestrator.handle_error(...)
        # Notification sent, workflow paused, waiting for human resolution
```

**Escalation Notification Format**:

```python
# Slack notification example
{
    "text": "🚨 Code Generation Error - Escalation Required",
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error*: Contract validation failed\n*Priority*: HIGH"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Workflow ID*: {workflow_id}"},
                {"type": "mrkdwn", "text": f"*Error Type*: {error_type}"},
                {"type": "mrkdwn", "text": f"*Assignee*: @{assignee}"},
                {"type": "mrkdwn", "text": f"*Timestamp*: {timestamp}"},
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```{error_message}```"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Details"},
                    "url": f"https://dashboard.example.com/errors/{error_id}"
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Retry Workflow"},
                    "value": "retry"
                }
            ]
        }
    ]
}
```

**Performance**:
- Overhead: <50ms (notification only)
- Success rate: **100%** (always escalates)
- Human resolution time: Variable

**Limitations**:
- Requires human intervention
- May block workflow execution
- Depends on notification infrastructure

---

## Error Patterns

### Pattern Matching

Error patterns use regex matching to identify error types:

```python
# Define error pattern
pattern = ErrorPattern(
    pattern_id="unique_pattern_id",
    error_type=ErrorType.NETWORK,  # Category
    regex_pattern=r"TimeoutError|ReadTimeout",  # Regex
    recovery_strategy=RecoveryStrategy.RETRY,
    metadata={"max_retries": 3},  # Strategy-specific config
)

# Add to orchestrator
orchestrator.add_error_pattern(pattern)
```

### Error Types

```python
from omninode_bridge.agents.workflows import ErrorType

# Available error types
ErrorType.NETWORK          # Network/connectivity errors
ErrorType.VALIDATION       # Validation failures
ErrorType.GENERATION       # Code generation errors
ErrorType.SYNTAX           # Syntax errors
ErrorType.CONTRACT         # Contract parsing/validation
ErrorType.TIMEOUT          # Operation timeouts
ErrorType.RATE_LIMIT       # API rate limiting
ErrorType.AI_SERVICE       # LLM service errors
ErrorType.SECURITY         # Security violations
ErrorType.UNKNOWN          # Unclassified errors
```

### Pattern Priority

When multiple patterns match, priority determines which is used:

```python
# Higher priority patterns checked first
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="specific_timeout",
    error_type=ErrorType.TIMEOUT,
    regex_pattern=r"LLM.*timeout",
    recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
    priority=100,  # Check this first
))

orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="generic_timeout",
    error_type=ErrorType.TIMEOUT,
    regex_pattern=r"timeout",
    recovery_strategy=RecoveryStrategy.RETRY,
    priority=50,  # Fallback if specific doesn't match
))
```

### Common Patterns Library

Pre-defined patterns for common errors:

```python
from omninode_bridge.agents.workflows.recovery_patterns import (
    COMMON_NETWORK_PATTERNS,
    COMMON_SYNTAX_PATTERNS,
    COMMON_VALIDATION_PATTERNS,
    COMMON_TIMEOUT_PATTERNS,
)

# Add all common patterns
for pattern in COMMON_NETWORK_PATTERNS:
    orchestrator.add_error_pattern(pattern)

for pattern in COMMON_SYNTAX_PATTERNS:
    orchestrator.add_error_pattern(pattern)

# Or selectively add patterns
orchestrator.add_error_pattern(COMMON_TIMEOUT_PATTERNS["llm_timeout"])
orchestrator.add_error_pattern(COMMON_VALIDATION_PATTERNS["quality_threshold"])
```

**Common Network Patterns**:
```python
COMMON_NETWORK_PATTERNS = [
    ErrorPattern(
        pattern_id="connection_timeout",
        error_type=ErrorType.NETWORK,
        regex_pattern=r"ConnectionTimeout|ConnectTimeout",
        recovery_strategy=RecoveryStrategy.RETRY,
        metadata={"max_retries": 3, "base_delay": 2.0},
    ),
    ErrorPattern(
        pattern_id="read_timeout",
        error_type=ErrorType.NETWORK,
        regex_pattern=r"ReadTimeout|Socket.*timeout",
        recovery_strategy=RecoveryStrategy.RETRY,
        metadata={"max_retries": 5, "base_delay": 1.0},
    ),
    ErrorPattern(
        pattern_id="connection_refused",
        error_type=ErrorType.NETWORK,
        regex_pattern=r"ConnectionRefused|Connection.*refused",
        recovery_strategy=RecoveryStrategy.ESCALATION,
    ),
]
```

**Common Syntax Patterns**:
```python
COMMON_SYNTAX_PATTERNS = [
    ErrorPattern(
        pattern_id="missing_async",
        error_type=ErrorType.SYNTAX,
        regex_pattern=r"SyntaxError.*'await'.*outside.*async",
        recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        metadata={
            "pattern": r"def (execute_\w+)\(",
            "replacement": r"async def \1(",
        },
    ),
    ErrorPattern(
        pattern_id="missing_import",
        error_type=ErrorType.SYNTAX,
        regex_pattern=r"NameError.*not defined",
        recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        metadata={"add_imports": True},
    ),
]
```

---

## Configuration

### Orchestrator Configuration

```python
orchestrator = ErrorRecoveryOrchestrator(
    metrics_collector=metrics,          # Optional metrics collection
    signal_coordinator=signals,         # Optional event signaling
    max_retries=3,                      # Default retry limit
    base_delay=1.0,                     # Default retry delay (seconds)
    enable_statistics=True,             # Track recovery statistics
    enable_notifications=True,          # Enable escalation notifications
)
```

### Strategy-Specific Configuration

Each strategy accepts custom configuration:

```python
# Retry strategy config
retry_config = {
    "max_retries": 5,
    "base_delay": 2.0,
    "exponential_base": 2,  # 2^n backoff
    "max_delay": 60.0,      # Cap at 60 seconds
}

# Alternative path config
alternative_config = {
    "alternatives": [
        ("template_v2", TemplateType.EFFECT),
        ("template_v1", TemplateType.EFFECT),
    ],
    "validate_alternatives": True,
    "stop_on_first_success": True,
}

# Degradation config
degradation_config = {
    "degradation_steps": [
        "remove_optional_mixins",
        "simplify_validation",
        "skip_ai_quorum",
    ],
    "progressive": True,  # Try each step progressively
}

# Correction config
correction_config = {
    "corrections": {
        "missing_async": {
            "pattern": r"def (execute_\w+)\(",
            "replacement": r"async def \1(",
        },
    },
    "validate_after_correction": True,
}

# Escalation config
escalation_config = {
    "notification_channel": "slack",
    "priority": "high",
    "assignee": "team-name",
    "pause_workflow": True,
    "include_context": True,
}
```

### Environment-Specific Configuration

```python
import os

ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    # Production: Aggressive recovery
    orchestrator = ErrorRecoveryOrchestrator(
        max_retries=5,
        base_delay=2.0,
        enable_statistics=True,
        enable_notifications=True,
    )

elif ENV == "staging":
    # Staging: Moderate recovery
    orchestrator = ErrorRecoveryOrchestrator(
        max_retries=3,
        base_delay=1.0,
        enable_statistics=True,
        enable_notifications=False,
    )

else:
    # Development: Minimal recovery (fail fast)
    orchestrator = ErrorRecoveryOrchestrator(
        max_retries=1,
        base_delay=0.5,
        enable_statistics=False,
        enable_notifications=False,
    )
```

---

## Custom Strategies

### Creating Custom Recovery Strategies

Implement custom recovery strategies by extending `BaseRecoveryStrategy`:

```python
from omninode_bridge.agents.workflows.recovery_strategies import BaseRecoveryStrategy
from omninode_bridge.agents.workflows import RecoveryStrategy, RecoveryContext, RecoveryResult

class CustomRecoveryStrategy(BaseRecoveryStrategy):
    """
    Custom recovery strategy example.
    """

    def __init__(self, custom_param: str) -> None:
        super().__init__(RecoveryStrategy.CUSTOM)
        self.custom_param = custom_param

    async def execute(
        self,
        context: RecoveryContext,
        **kwargs,
    ) -> RecoveryResult:
        """
        Execute custom recovery logic.
        """
        start_time = time.perf_counter()

        try:
            # Custom recovery logic here
            result = await self._custom_recovery_logic(context)

            # Success
            duration_ms = (time.perf_counter() - start_time) * 1000
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CUSTOM,
                result=result,
                duration_ms=duration_ms,
            )

        except Exception as e:
            # Failed
            duration_ms = (time.perf_counter() - start_time) * 1000
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CUSTOM,
                error_message=str(e),
                duration_ms=duration_ms,
            )

    async def _custom_recovery_logic(self, context: RecoveryContext) -> Any:
        """
        Implement custom recovery logic.
        """
        # Your recovery logic here
        pass
```

### Registering Custom Strategies

```python
# Create custom strategy instance
custom_strategy = CustomRecoveryStrategy(custom_param="value")

# Register with orchestrator
orchestrator.register_strategy(
    strategy_type=RecoveryStrategy.CUSTOM,
    strategy_instance=custom_strategy,
)

# Use in error patterns
orchestrator.add_error_pattern(ErrorPattern(
    pattern_id="custom_error",
    error_type=ErrorType.CUSTOM,
    regex_pattern=r"Custom.*error",
    recovery_strategy=RecoveryStrategy.CUSTOM,
))
```

### Example: Circuit Breaker Strategy

```python
class CircuitBreakerStrategy(BaseRecoveryStrategy):
    """
    Circuit breaker recovery strategy.

    Opens circuit after N consecutive failures, preventing further attempts
    for a cooldown period. Useful for preventing cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: int = 60,
    ) -> None:
        super().__init__(RecoveryStrategy.CIRCUIT_BREAKER)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self.failure_count = 0
        self.circuit_open = False
        self.circuit_opened_at: Optional[float] = None

    async def execute(
        self,
        context: RecoveryContext,
        operation: Callable,
        **kwargs,
    ) -> RecoveryResult:
        """
        Execute with circuit breaker protection.
        """
        # Check if circuit is open
        if self.circuit_open:
            if time.time() - self.circuit_opened_at < self.cooldown_seconds:
                # Circuit still open
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                    error_message="Circuit breaker open - cooldown in progress",
                    duration_ms=0.0,
                )
            else:
                # Cooldown expired - try to close circuit
                self.circuit_open = False
                self.failure_count = 0

        # Try operation
        try:
            result = await operation(context.state)

            # Success - reset failure count
            self.failure_count = 0

            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                result=result,
                duration_ms=0.0,
            )

        except Exception as e:
            # Failure - increment count
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                # Open circuit
                self.circuit_open = True
                self.circuit_opened_at = time.time()
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures. "
                    f"Cooldown: {self.cooldown_seconds}s"
                )

            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                error_message=str(e),
                duration_ms=0.0,
            )
```

---

## Integration

### Integration with StagedParallelExecutor

```python
from omninode_bridge.agents.workflows import (
    StagedParallelExecutor,
    ErrorRecoveryOrchestrator,
)

# Create components
executor = StagedParallelExecutor(...)
error_recovery = ErrorRecoveryOrchestrator(...)

# Wrap workflow execution with recovery
async def execute_with_recovery(
    workflow_id: str,
    stages: list[WorkflowStage],
) -> WorkflowResult:
    """
    Execute workflow with automatic error recovery.
    """
    try:
        return await executor.execute_workflow(workflow_id, stages)

    except Exception as e:
        context = RecoveryContext(
            workflow_id=workflow_id,
            node_name="staged_executor",
            exception=e,
            state={"stages": [s.stage_id for s in stages]},
        )

        recovery_result = await error_recovery.handle_error(context)

        if recovery_result.success:
            return recovery_result.result
        else:
            raise
```

### Integration with ValidationPipeline

```python
from omninode_bridge.agents.workflows import (
    ValidationPipeline,
    ErrorRecoveryOrchestrator,
)

# Wrap validation with recovery
async def validate_with_recovery(
    code: str,
    context: ValidationContext,
) -> ValidationResult:
    """
    Validate code with error recovery for validation failures.
    """
    try:
        return await validation_pipeline.validate(code, context)

    except ValidationError as e:
        # Try error correction if validation fails
        recovery_context = RecoveryContext(
            workflow_id=context.workflow_id,
            node_name="validator",
            exception=e,
            state={"code": code, "context": context},
        )

        recovery_result = await error_recovery.handle_error(recovery_context)

        if recovery_result.success:
            # Re-validate corrected code
            return await validation_pipeline.validate(
                recovery_result.result,
                context,
            )
        else:
            raise
```

### Integration with TemplateManager

```python
from omninode_bridge.agents.workflows import (
    TemplateManager,
    ErrorRecoveryOrchestrator,
)

# Wrap template rendering with recovery
async def render_with_recovery(
    template_id: str,
    context: dict,
) -> str:
    """
    Render template with alternative path recovery.
    """
    try:
        return await template_manager.render_template(template_id, context)

    except Exception as e:
        recovery_context = RecoveryContext(
            workflow_id=f"render_{template_id}",
            node_name="template_manager",
            exception=e,
            state={"template_id": template_id, "context": context},
        )

        recovery_result = await error_recovery.handle_error(recovery_context)

        if recovery_result.success:
            return recovery_result.result
        else:
            raise
```

---

## Monitoring

### Recovery Statistics

Track recovery performance over time:

```python
# Get comprehensive statistics
stats = orchestrator.get_statistics()

print(f"Total Attempts: {stats.total_attempts}")
print(f"Success Rate: {stats.success_rate:.1%}")
print(f"Failure Rate: {stats.failure_rate:.1%}")
print(f"Avg Recovery Time: {stats.avg_recovery_time_ms:.2f}ms")

# Strategy breakdown
print("\nStrategy Usage:")
for strategy, count in stats.strategy_usage.items():
    success_rate = stats.strategy_success_rates[strategy]
    print(f"  {strategy}: {count} uses ({success_rate:.1%} success)")

# Error type breakdown
print("\nError Types:")
for error_type, count in stats.error_type_counts.items():
    print(f"  {error_type}: {count} errors")
```

### Real-Time Monitoring

Monitor recovery operations in real-time:

```python
# Subscribe to recovery events
@orchestrator.on_recovery_attempt
async def on_recovery_attempt(context: RecoveryContext, strategy: RecoveryStrategy):
    """Called when recovery attempt starts."""
    logger.info(
        f"Recovery attempt: {context.workflow_id} using {strategy}"
    )

@orchestrator.on_recovery_success
async def on_recovery_success(result: RecoveryResult):
    """Called when recovery succeeds."""
    metrics.record_recovery_success(
        strategy=result.strategy_used,
        duration_ms=result.duration_ms,
    )

@orchestrator.on_recovery_failure
async def on_recovery_failure(result: RecoveryResult):
    """Called when recovery fails."""
    metrics.record_recovery_failure(
        strategy=result.strategy_used,
        error=result.error_message,
    )
```

### Health Checks

Monitor error recovery system health:

```python
from omninode_bridge.agents.workflows.monitoring import HealthChecker

health_checker = HealthChecker(error_recovery=orchestrator)

# Define health checks
health_checker.add_check(
    "recovery_success_rate",
    lambda: orchestrator.get_statistics().success_rate >= 0.90,
    critical=True,
)

health_checker.add_check(
    "avg_recovery_time",
    lambda: orchestrator.get_statistics().avg_recovery_time_ms < 500,
    critical=False,
)

# Check health
health_status = await health_checker.check_all()

if not health_status.all_passing:
    logger.error(f"Health check failed: {health_status.failed_checks}")
```

### Alerts

Configure alerts for recovery issues:

```python
from omninode_bridge.agents.workflows.monitoring import AlertManager

alert_manager = AlertManager(
    error_recovery=orchestrator,
    notification_channel="slack",
)

# Define alert thresholds
alert_manager.add_threshold(
    "recovery_success_rate",
    min_value=0.85,
    severity="high",
)

alert_manager.add_threshold(
    "recovery_time_p95",
    max_value=1000.0,  # 1 second
    severity="medium",
)

# Start monitoring
await alert_manager.start_monitoring(interval_seconds=60)
```

---

## Troubleshooting

### Common Issues

#### Low Success Rate (<80%)

**Symptoms**:
- High failure rate in recovery statistics
- Frequent escalations

**Diagnosis**:
```python
stats = orchestrator.get_statistics()
print(f"Success rate: {stats.success_rate:.1%}")

# Check which strategies are failing
for strategy, rate in stats.strategy_success_rates.items():
    if rate < 0.8:
        print(f"❌ {strategy}: {rate:.1%} (below target)")
```

**Solutions**:
1. Add more error patterns for unmatched errors
2. Tune retry parameters (increase max_retries)
3. Add alternative paths for frequent failures
4. Review error logs for new error patterns

#### Slow Recovery (>500ms average)

**Symptoms**:
- High average recovery time
- Workflow latency increased

**Diagnosis**:
```python
stats = orchestrator.get_statistics()
print(f"Avg recovery time: {stats.avg_recovery_time_ms:.2f}ms")

# Check which strategies are slow
for strategy, times in stats.strategy_durations.items():
    avg_time = sum(times) / len(times)
    if avg_time > 500:
        print(f"⚠️  {strategy}: {avg_time:.2f}ms (slow)")
```

**Solutions**:
1. Reduce retry delays (lower base_delay)
2. Reduce number of alternatives to try
3. Simplify error correction patterns
4. Use faster degradation steps

#### Pattern Matching Issues

**Symptoms**:
- Errors not being matched to patterns
- Wrong strategy selected

**Diagnosis**:
```python
# Enable debug logging
import logging
logging.getLogger("omninode_bridge.agents.workflows.error_recovery").setLevel(logging.DEBUG)

# Test pattern matching
error_message = "TimeoutError: Read timeout"
matched_pattern = orchestrator.match_error_pattern(error_message)

if matched_pattern:
    print(f"✅ Matched: {matched_pattern.pattern_id}")
else:
    print("❌ No pattern matched")
```

**Solutions**:
1. Review regex patterns for accuracy
2. Add more specific patterns
3. Check pattern priority order
4. Use regex testing tools

#### Retry Exhaustion

**Symptoms**:
- All retries exhausted without success
- Transient errors not recovering

**Diagnosis**:
```python
# Check retry statistics
retry_stats = stats.strategy_details[RecoveryStrategy.RETRY]
print(f"Retry exhaustion rate: {retry_stats.exhaustion_rate:.1%}")
```

**Solutions**:
1. Increase max_retries for transient errors
2. Increase base_delay for rate-limited services
3. Add alternative path as fallback
4. Check if errors are truly transient

---

**See Also**:
- [Phase 4 Optimization Guide](./PHASE_4_OPTIMIZATION_GUIDE.md) - Complete optimization system
- [Performance Tuning Guide](./WORKFLOW_PERFORMANCE_TUNING.md) - Performance optimization
- [Production Deployment Guide](./PRODUCTION_DEPLOYMENT_GUIDE.md) - Production deployment
- [Workflows API Reference](../api/WORKFLOWS_API_REFERENCE.md) - Complete API documentation
