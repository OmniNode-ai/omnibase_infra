> **Navigation**: [Home](../index.md) > Plugins

# Plugin Documentation

Documentation for ONEX plugin system and plugin implementations.

Current plugin docs describe active requirements and examples. Point-in-time
optimization reports are not primary docs.

## Available Documents

| Document | Description |
|----------|-------------|
| [Compute Plugin Determinism](PLUGIN_DETERMINISM.md) | Required determinism guarantees, test patterns, and common violations for compute plugins |

## Example Plugins

| Example | Source | Tests |
|---------|--------|-------|
| JSON normalizer | `src/omnibase_infra/plugins/examples/plugin_json_normalizer.py` | `tests/unit/plugins/examples/test_plugin_json_normalizer.py`, `tests/unit/plugins/test_plugin_compute_determinism.py` |

## Related Documentation

- [Handler Plugin Loader](../patterns/handler_plugin_loader.md) - Plugin loading patterns
- [Protocol Patterns](../patterns/protocol_patterns.md) - Plugin interface patterns
