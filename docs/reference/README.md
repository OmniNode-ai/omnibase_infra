> **Navigation**: [Home](../index.md) > Reference

# Reference Documentation

Comprehensive reference documentation for ONEX infrastructure.

## Core References

| Document | Description |
|----------|-------------|
| [Node Archetypes](node-archetypes.md) | EFFECT, COMPUTE, REDUCER, ORCHESTRATOR - complete documentation |
| [Contract.yaml Reference](contracts.md) | Complete contract format specification |

## API References

| Document | Description |
|----------|-------------|
| [Event Bus Integration](../architecture/EVENT_BUS_INTEGRATION_GUIDE.md) | Event bus adapters, protocols, and implementation details |

## onex CLI Reference

The `onex` CLI is the primary developer and operator interface. Entry point: `omnibase_infra.cli.commands:cli` (installed as `onex` via `pyproject.toml` entry-points).

### Core subcommands

| Subcommand | Purpose |
|------------|---------|
| `onex kafka produce <topic> --payload '<json>'` | Publish a ONEX command envelope to a Kafka topic from a dev machine without SSH. Supports `--dry-run` (print envelope, no publish) and `--envelope` (wrap payload in a standard correlation_id/timestamp envelope). PLAINTEXT auth only (LAN runtime host is unauthenticated from dev machines). (OMN-8435) |
| `onex skill <name> [args...]` | Dispatch an ONEX skill by name through the proven receipt-mode path. Skill-to-node mappings are declared in `src/omnibase_infra/cli/skill_mapping.yaml` — see below. |
| `onex node <name> [args...]` | Invoke a node directly by its entry-point name. |
| `onex delegate <args>` | Dispatch a delegation request through the bus. |
| `onex status` | Launch the status TUI (`onex-status` entry-point). |

### skill_mapping.yaml — declarative skill dispatch

`src/omnibase_infra/cli/skill_mapping.yaml` is a **YAML data file**, not generated code. It maps each `onex skill <name>` invocation to a backing node, typed result model, dispatch timeout, and CLI arg → payload field specs.

**Adding a skill does not require a code change** — add a YAML entry and a fixture. The `ModelSkillMappingRegistry` loads the file at runtime and builds the dispatch payload from the declared arg specs.

Key fields per entry:

| Field | Type | Description |
|-------|------|-------------|
| `skill_name` | string | Name as invoked: `onex skill <skill_name>` |
| `node_name` | string | Backing node (onex.nodes entry-point group) |
| `result_model` | string | FQN of the handler's typed result model (receipt schema identity) |
| `event_bus` | string | Backend override (default: `inmemory`) |
| `timeout` | int | Dispatch timeout in seconds (default: 300) |
| `args` | list | CLI arg → payload field specs (`arg_type`: string, integer, boolean, string_list) |
| `static_payload` | dict | Payload fields injected regardless of CLI args |
| `classifiers` | dict | Keyword classification for unset fields (used by delegate) |

### headless codex CLI for delegation (OMN-13158)

The delegation subsystem (`node_llm_inference_effect` via `HandlerLlmCliSubprocess`) uses the headless/non-interactive codex CLI (`codex --headless`) for delegation inference. The handler spawns the CLI as a subprocess in non-interactive mode, passes the prompt via stdin, and reads the response from stdout. This avoids the OAuth/browser flow required by the interactive CLI. The `codex-cli` entry in the handler's provider map resolves to the installed `codex` binary.

## Authoritative Source

For coding rules and standards, see [CLAUDE.md](../../CLAUDE.md) - the authoritative source for all development guidelines. Reference documentation here provides detailed explanations and examples.

## Related Documentation

- [Quick Start](../getting-started/quickstart.md) - Get running quickly
- [Architecture Overview](../architecture/overview.md) - System design
- [Patterns](../patterns/README.md) - Implementation patterns
- [ADRs](../decisions/README.md) - Why things work this way
- [CI Test Strategy](../testing/CI_TEST_STRATEGY.md) - CI gate inventory
