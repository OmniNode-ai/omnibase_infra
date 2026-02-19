# MCP Integration Guide

> **Status**: Current | **Last Updated**: 2026-02-19

This guide explains how to expose ONEX node operations as Model Context Protocol (MCP) tools so that AI agents (such as Claude) can discover and invoke them. It covers the full practical workflow: registering a tool, testing without a live server, and the current limitations to work around.

For architecture details — component inventory, request flow diagrams, and the full security model — see `docs/architecture/MCP_SERVICE_ARCHITECTURE.md`. This guide focuses on the *how-to* actions a developer takes.

---

## Table of Contents

1. [What MCP Integration Enables](#what-mcp-integration-enables)
2. [How It Fits Into ONEX](#how-it-fits-into-onex)
3. [Step 1 — Expose an Operation via HandlerMCP](#step-1-expose-an-operation-via-handlermcp)
4. [Step 2 — Register the Tool with ONEXToMCPAdapter](#step-2-register-the-tool-with-onextomcpadapter)
5. [Step 3 — Generate Parameters from a Pydantic Model](#step-3-generate-parameters-from-a-pydantic-model)
6. [Step 4 — Wire into ServiceMCPToolRegistry](#step-4-wire-into-servicemcptoolregistry)
7. [Step 5 — Use AdapterONEXToolExecution for Dispatch](#step-5-use-adapteronextoolexecution-for-dispatch)
8. [Step 6 — Use skip_server=True for Testing](#step-6-use-skip_servertrue-for-testing)
9. [Step 7 — Verify via the Envelope Interface](#step-7-verify-via-the-envelope-interface)
10. [Tool Discovery at Cold Start](#tool-discovery-at-cold-start)
11. [Hot Reload via Kafka](#hot-reload-via-kafka)
12. [Current Limitations](#current-limitations)
13. [Common Mistakes](#common-mistakes)

---

## What MCP Integration Enables

ONEX nodes are internal infrastructure primitives: they speak envelopes, emit events, and depend on contract YAML declarations. MCP integration bridges these nodes to the external world of AI agents.

When the MCP layer is running, an AI agent can:
- Query which ONEX operations are available (`mcp.list_tools`)
- Obtain JSON Schema parameter definitions for each operation
- Invoke a named operation by name with typed arguments (`mcp.call_tool`)
- Do all of this without knowing anything about envelopes, Kafka, or contract YAML

The MCP server runs on `0.0.0.0:8090` (default, configurable) using Streamable HTTP transport. It is backed by the official MCP Python SDK (`mcp.server.fastmcp`).

---

## How It Fits Into ONEX

```
AI Agent (MCP client)
        |
        | HTTP POST /mcp
        v
HandlerMCP  (handlers/handler_mcp.py)
        |
        | mcp.call_tool
        v
ServiceMCPToolRegistry  (services/mcp/service_mcp_tool_registry.py)
        |
        | get_tool(name) -> MCPToolDefinition
        v
AdapterONEXToolExecution  (adapters/adapter_onex_tool_execution.py)
        |
        | build envelope -> dispatch to ONEX node
        v
ONEX Orchestrator Node
        |
        | ModelHandlerOutput
        v
MCP JSON response -> AI Agent
```

The handler owns the uvicorn lifecycle. The registry is the in-memory tool cache. The adapter translates MCP calls to ONEX envelopes.

---

## Step 1 — Expose an Operation via HandlerMCP

`HandlerMCP` is already wired in the ONEX runtime. To use the MCP layer, you do not need to modify it — you add your tool to the registry and start the handler with the correct configuration.

The handler accepts configuration via its `initialize(config)` call:

```python
config = {
    "host": "0.0.0.0",
    "port": 8090,
    "path": "/mcp",
    "stateless": True,
    "json_response": True,
    "timeout_seconds": 30.0,
    "max_tools": 100,
    "consul_host": "consul",       # required (from .env / Infisical)
    "consul_port": 8500,           # required (int)
    "kafka_enabled": True,         # required
    "dev_mode": False,             # required
    "skip_server": False,          # False in production, True in tests
}
```

All required fields will raise `ProtocolConfigurationError` if absent. Do not add defaults for missing fields — the `.env` file is the single source of truth.

---

## Step 2 — Register the Tool with ONEXToMCPAdapter

`ONEXToMCPAdapter` converts ONEX node metadata into `MCPToolDefinition` structs. Registration is currently manual (automatic discovery from contract `mcp_enabled: true` is planned under OMN-1288).

```python
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import (
    MCPToolParameter,
    ONEXToMCPAdapter,
)

adapter = ONEXToMCPAdapter(
    node_executor=executor,    # AdapterONEXToolExecution instance
    container=container,       # ModelONEXContainer
)

# Register a node operation as an MCP tool
tool = await adapter.register_node_as_tool(
    node_name="node_compute_hash",
    description="Compute a SHA-256 hash of the provided input string. Returns the hex digest.",
    parameters=[
        MCPToolParameter(
            name="input",
            parameter_type="string",
            description="The string data to hash.",
            required=True,
        ),
        MCPToolParameter(
            name="algorithm",
            parameter_type="string",
            description="Hash algorithm: sha256 (default) or sha512.",
            required=False,
            default_value="sha256",
        ),
    ],
    version="1.0.0",
    tags=["compute", "crypto"],
    timeout_seconds=10,
)
```

The tool is stored in the adapter's `_tool_cache`. It will be available via `adapter.discover_tools()` and via tag-filtered queries.

Write AI-agent-friendly descriptions: they appear directly in the MCP tool listing that the AI agent uses to decide which tool to call. Be explicit about what arguments are required, what the operation does, and what the response contains.

---

## Step 3 — Generate Parameters from a Pydantic Model

If the ONEX node already has a Pydantic input model, you can derive the `MCPToolParameter` list automatically instead of writing it by hand.

```python
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import ONEXToMCPAdapter
from omnibase_infra.nodes.node_compute_hash.models.model_compute_hash_input import (
    ModelComputeHashInput,
)

# Step A: generate JSON Schema from the Pydantic model
schema = ONEXToMCPAdapter.pydantic_to_json_schema(ModelComputeHashInput)

# Step B: extract MCPToolParameter list from the schema
parameters = ONEXToMCPAdapter.extract_parameters_from_schema(schema)

# Step C: register with the derived parameters
tool = await adapter.register_node_as_tool(
    node_name="node_compute_hash",
    description="Compute a hash of the input data.",
    parameters=parameters,
)
```

`pydantic_to_json_schema` calls `model_json_schema()` on any `BaseModel` subclass. If the class is not a Pydantic model, it returns `{"type": "object"}` (soft failure). Pass `raise_on_error=True` to convert this to a hard `ProtocolConfigurationError`.

`extract_parameters_from_schema` honors the `required` list from the schema and handles union types (takes the first non-null type). For enum or format constraints, the full schema fragment is preserved in `MCPToolParameter.schema`.

---

## Step 4 — Wire into ServiceMCPToolRegistry

`ServiceMCPToolRegistry` is the in-memory cache that `HandlerMCP` queries at tool-call time. It is created inside `MCPServerLifecycle.start()` automatically when you run the full handler. If you need to access it directly (for example, in integration tests), retrieve it from the lifecycle object:

```python
from omnibase_infra.services.mcp.mcp_server_lifecycle import MCPServerLifecycle

lifecycle = MCPServerLifecycle(
    container=container,
    config=mcp_server_config,
    bus=None,
)
await lifecycle.start()

registry = lifecycle.registry  # ServiceMCPToolRegistry
executor = lifecycle.executor  # AdapterONEXToolExecution

# Manually upsert a tool definition (useful in integration test setup)
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition

tool_def = ModelMCPToolDefinition(name="node_compute_hash", description="...")
await registry.upsert_tool(tool_def, event_id="bootstrap-001")
```

The registry uses `asyncio.Lock` for coroutine-safe access within a single event loop. It is not safe to share across multiple threads.

---

## Step 5 — Use AdapterONEXToolExecution for Dispatch

`AdapterONEXToolExecution` implements `ProtocolToolExecutor`. It bridges MCP tool calls to the ONEX dispatcher by building an envelope and calling the appropriate ONEX orchestrator node.

In integrated mode (full runtime), `MCPServerLifecycle` creates and wires this automatically. You access it via `lifecycle.executor`.

In tests, you can construct it directly:

```python
from omnibase_infra.adapters.adapter_onex_tool_execution import AdapterONEXToolExecution

executor = AdapterONEXToolExecution(container=container)

result = await executor.execute(
    tool=tool_definition,
    arguments={"input": "hello", "algorithm": "sha256"},
    correlation_id=uuid4(),
)
# result is a dict returned to the MCP client
```

Note that `invoke_tool` on `ONEXToMCPAdapter` currently returns a placeholder (see [Current Limitations](#current-limitations)). Use `AdapterONEXToolExecution.execute()` directly for integrated dispatch.

---

## Step 6 — Use skip_server=True for Testing

Starting a full uvicorn server in unit tests is slow and creates port conflicts. The `skip_server=True` flag in `HandlerMCP` configuration skips uvicorn startup entirely while still initializing all other components (circuit breaker, registry, executor).

```python
import pytest
from unittest.mock import MagicMock
from omnibase_infra.handlers.handler_mcp import HandlerMCP


@pytest.mark.asyncio
async def test_mcp_handler_list_tools():
    container = MagicMock()
    handler = HandlerMCP(container)

    await handler.initialize({
        "host": "0.0.0.0",
        "port": 8090,
        "path": "/mcp",
        "stateless": True,
        "json_response": True,
        "timeout_seconds": 30.0,
        "max_tools": 100,
        "skip_server": True,   # <-- no uvicorn started
    })

    # Register a tool manually
    registered = await handler.register_tool(
        name="node_compute_hash",
        description="Compute a hash",
        parameters=[],
        version="1.0.0",
    )
    assert registered is True

    # Invoke via the envelope interface
    envelope = {
        "operation": "mcp.list_tools",
        "correlation_id": "test-cid-001",
        "payload": {},
    }
    output = await handler.execute(envelope)
    assert output.result is not None
    tools = output.result.get("tools", [])
    assert any(t["name"] == "node_compute_hash" for t in tools)

    await handler.shutdown()
```

With `skip_server=True`, `health_check()` returns `server_task_missing`. This is expected in test mode.

---

## Step 7 — Verify via the Envelope Interface

`HandlerMCP` can be invoked through the standard ONEX envelope system (not just via HTTP). Three operations are supported:

### mcp.list_tools

Returns all registered tools with their JSON Schema definitions.

```python
envelope = {
    "operation": "mcp.list_tools",
    "correlation_id": str(uuid4()),
    "payload": {},
}
output = await handler.execute(envelope)
tools = output.result["tools"]  # list of dicts with name, description, parameters
```

### mcp.call_tool

Invokes a named tool by name with typed arguments.

```python
envelope = {
    "operation": "mcp.call_tool",
    "correlation_id": str(uuid4()),
    "payload": {
        "tool_name": "node_compute_hash",
        "arguments": {
            "input": "hello world",
            "algorithm": "sha256",
        },
    },
}
output = await handler.execute(envelope)
result = output.result  # dict returned by AdapterONEXToolExecution
```

`mcp.call_tool` checks the circuit breaker before calling the executor. If the circuit is open, it raises `InfraUnavailableError`.

### mcp.describe

Returns handler metadata: type, category, tool count, configuration, and server state.

```python
envelope = {
    "operation": "mcp.describe",
    "correlation_id": str(uuid4()),
    "payload": {},
}
output = await handler.execute(envelope)
print(output.result["tool_count"])    # number of registered tools
print(output.result["server_state"])  # "running" or "not_started"
```

---

## Tool Discovery at Cold Start

When `MCPServerLifecycle.start()` runs in production, it performs a Consul scan to find MCP-enabled orchestrators before the HTTP server accepts requests. This is the "cold start" discovery phase.

`ServiceMCPToolDiscovery` queries Consul for services tagged with `mcp-enabled`. For each discovered service, it fetches the tool definitions and populates `ServiceMCPToolRegistry`.

In the current MVP, the Consul scan finds services but tool definition loading is not yet fully automatic — the `mcp_enabled: true` contract flag and automatic contract scanning is tracked under OMN-1288. Manual registration via `register_node_as_tool()` is the supported path until that ticket is complete.

---

## Hot Reload via Kafka

When `kafka_enabled=True`, `ServiceMCPToolSync` subscribes to Kafka events for real-time tool definition updates. This allows registering and deregistering tools without restarting the MCP server.

Tool sync events are published on the event bus when ORCHESTRATOR nodes register or deregister themselves. `ServiceMCPToolSync` calls `ServiceMCPToolRegistry.upsert_tool()` (for registration) or `ServiceMCPToolRegistry.remove_tool()` (for deregistration) with the event's `event_id` for idempotency.

The registry version-tracks each tool via `event_id`. Out-of-order Kafka messages (lower `event_id` than the current version) are silently ignored, preventing stale data from overwriting newer registrations.

To disable hot reload in local development (where Kafka may not be running), set `kafka_enabled=False` in the handler config.

---

## Current Limitations

| Limitation | Tracking | Workaround |
|------------|----------|------------|
| Tool registration is manual only | OMN-1288 | Call `register_node_as_tool()` explicitly during startup |
| `ONEXToMCPAdapter.invoke_tool()` returns a placeholder | OMN-1288 | Use `AdapterONEXToolExecution.execute()` directly |
| No authentication on the HTTP endpoint | OMN-1288 | Deploy behind an API gateway or restrict via firewall rules |
| `mcp.call_tool` in integrated mode requires registry + executor to be wired | OMN-1281 | Use `MCPServerLifecycle` to ensure both are initialized before calls |
| `max_tools` limit (default 100) silently returns `False` | — | Check the return value of `handler.register_tool()` |

---

## Common Mistakes

### Calling invoke_tool instead of AdapterONEXToolExecution.execute

`ONEXToMCPAdapter.invoke_tool()` returns a placeholder dict. It does not dispatch to ONEX nodes. Use `AdapterONEXToolExecution.execute()` for actual invocation.

### Using skip_server=False in unit tests

This starts a real uvicorn server and binds to a port. Use `skip_server=True` in all unit and integration tests that do not need the full HTTP stack.

### Registering the same tool name twice

`ServiceMCPToolRegistry.upsert_tool()` uses the tool name as the key. Registering with the same name and a stale `event_id` will be silently ignored. If you need to update a tool definition, use a newer `event_id`.

### Including argument values in error messages

`ProtocolToolExecutor` implementations must not log or include raw argument values in error messages. Arguments may contain secrets or user data. Only the tool name, operation name, correlation ID, and error code are safe to include.

### Not checking the return value of register_tool

`HandlerMCP.register_tool()` returns `False` when the tool limit (`max_tools`) is reached, without raising. The caller must check:

```python
ok = await handler.register_tool(name=..., description=..., parameters=..., version=...)
if not ok:
    logger.warning("Tool registry full — could not register %s", name)
```

---

## See Also

- `docs/architecture/MCP_SERVICE_ARCHITECTURE.md` — component inventory, flow diagrams, security model
- `src/omnibase_infra/handlers/handler_mcp.py` — top-level handler
- `src/omnibase_infra/handlers/mcp/adapter_onex_to_mcp.py` — contract-to-tool adapter
- `src/omnibase_infra/handlers/mcp/transport_streamable_http.py` — ASGI layer
- `src/omnibase_infra/services/mcp/` — registry, discovery, sync services
- `src/omnibase_infra/adapters/adapter_onex_tool_execution.py` — tool execution bridge
- `docs/patterns/circuit_breaker_implementation.md` — circuit breaker used by HandlerMCP
