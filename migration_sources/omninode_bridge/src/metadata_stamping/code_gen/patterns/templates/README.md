# Pattern Code Templates

**Purpose**: Jinja2 templates for rendering pattern code

## Overview

This directory contains Jinja2 templates that are rendered when applying patterns to generated code.

## Template Format

Templates use Jinja2 syntax with common variables:

```jinja2
{# circuit_breaker.j2 #}
# Circuit breaker configuration
self._circuit_breakers: dict[str, ModelCircuitBreaker] = {}

{% for service in services %}
self._circuit_breakers["{{ service.name }}"] = ModelCircuitBreaker(
    failure_threshold={{ service.failure_threshold | default(5) }},
    recovery_timeout_seconds={{ service.recovery_timeout_ms | default(60000) // 1000 }},
)
{% endfor %}
```

## Common Variables

- `{{ node_name }}` - Node name
- `{{ node_type }}` - Node type (effect/compute/reducer/orchestrator)
- `{{ class_name }}` - Generated class name
- `{{ configuration }}` - Pattern configuration dict
- `{{ services }}` - List of external services
- `{{ features }}` - Enabled features

## Usage

Templates are loaded by PatternFormatter and rendered with context-specific variables during code generation.

## Next Steps

Templates will be created alongside patterns during Task F2.
