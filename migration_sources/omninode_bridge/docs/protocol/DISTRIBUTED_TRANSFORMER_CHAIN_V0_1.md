---
version: v0.1
status: Draft
last_updated: 2025-05-03
---

# ONEX Protocol Specification — OmniNode Execution Protocol (v0.1)

> **O.N.E. Framing:**
> ONEX defines the execution, introspection, and orchestration model for all schema-driven, agent-native processes within the OmniNode Environment (O.N.E.). It formalizes protocol, structure, and metadata for atomic, composable, and verifiable execution chains across distributed and federated systems.

---

## 🚀 Mission Statement

Establish a recursive, schema-governed, feedback-aware execution mesh where each task is decomposed into atomic transformers—functions, agents, or classes—bound by formal schemas. Nodes dynamically join, specialize, and execute transformers based on role, memory context, and schema compatibility.

---

## 🧱 Core Architectural Principles

- **Schema-first:** Typed, introspectable logic with strict I/O contracts
- **Composable:** DAG-modeled execution chains
- **Agent-native:** Agents mutate and reason over schemas and plans
- **Memory-backed:** All runs logged with input/output hashes and provenance
- **Trust-aware:** Explicit trust levels, attestation, and policy-bound execution
- **Simulation-capable:** All transformers support dry-run, budget estimation
- **Fallback-ready:** Supports retry, chaining, and failure routing
- **Green-aware:** Tracks and routes based on energy and resource footprint
- **Extensible:** Pluggable object types, execution logic, and registry models
- **Federated:** Supports multi-org registries and trust negotiation
- **Verifiable:** Declarative support for formal validation of critical chains
- **Chaos-tolerant:** Supports conformance chaos injection
- **Budget-constrained:** Controlled execution via programmable resource limits

---

## ✅ MVP Specification

### 1. Schema System

**Requirements:**

- MUST use Pydantic or JSON Schema
- MUST include:
  - Field validation
  - Versioning (`version`)
  - Examples
  - Migration hooks
- SHOULD support:
  - Multi-modal schemas (Avro, Protobuf)
  - Live documentation (`doc_url`)
  - Introspection metadata (`schema_url`)
  - `.pyi`, JSON Schema, and OpenAPI export

```python
class ParseCSVInput(BaseModel):
    csv_path: str
    delimiter: str = ","
    encoding: str = "utf-8"

    class Config:
        schema_extra = {
            "examples": [{"csv_path": "/data/input.csv"}],
            "version": "0.1.0",
            "migration": {
                "from": "0.0.9",
                "strategy": "add_default_field(delimiter, ',')"
            },
            "doc_url": "https://registry.omninode.org/schemas/ParseCSVInput"
        }
```

---

### 2. Transformer Definition

```python
@transformer(
    input_schema=ParseCSVInput,
    output_schema=ParseCSVOutput,
    namespace="data.parsing",
    version="0.1.0",
    role="parser",
    simulation_ready=True
)
def parse_csv(input: ParseCSVInput, context: ExecutionContext) -> ParseCSVOutput:
    if context.simulation:
        return simulate_output()
    try:
        return ParseCSVOutput(rows=parsed_rows)
    except Exception as e:
        raise CSVParseError(detail=str(e))
```

#### Additional Declarations

- `side_effect_type`: e.g. `"writes_disk"`, `"external_api_call"`
- `budget_aware`: whether transformer halts or yields on exhaustion
- `confidential_handling`: whether it requires enclave or zero-trust execution

---

### 3. Error Taxonomy

```python
class TransformerError(BaseModel):
    type: Literal["Transient", "Validation", "BudgetExceeded", "SecurityViolation"]
    detail: str
    retryable: bool
    fallback_hint: Optional[str]
```

---

### 4. Registry System

- Registry stores all entries as `RegistryEntry` objects
- Entries are signed, hash-addressed, and time-stamped
- Searchable by `object_type`, `namespace`, `tags`, `trust_level`
- Supports federation with manifest sync and trust anchor rotation
- Namespace format: `org.project.domain.transformer`

**Metadata Fields:**

| Field             | Required | Notes                                  |
|------------------|----------|----------------------------------------|
| object_type       | Yes      | `transformer`, `schema`, `plan`, etc.  |
| payload           | Yes      | Serialized version of object           |
| fingerprint       | Yes      | SHA-256 hash                           |
| trust_level       | Optional | UNVERIFIED / SIGNED / VERIFIED         |
| attestation       | Optional | Signed by org or federation root       |
| registered_at     | Yes      | UTC timestamp                          |
| doc_url           | Optional | Documentation link                     |
| schema_url        | Optional | Schema reference                       |

---

### 5. Execution Context and Budget

```python
class ExecutionContext(BaseModel):
    simulation: bool
    trace_id: str
    caller_id: Optional[str]
    execution_budget: Optional["ExecutionBudget"]
    policies: Optional[Dict[str, Any]]
```

```python
class ExecutionBudget(BaseModel):
    max_cost: Optional[float]
    max_steps: Optional[int]
    max_energy_kwh: Optional[float]
    time_limit_seconds: Optional[int]
```

**Simulation Example:**
```yaml
execution:
  mode: dry_run
  budget:
    max_cost: 0.01
    max_energy_kwh: 0.002
  trace_output: true
```

---

### 6. Orchestration and Plans

```python
class ExecutionPlan(BaseModel):
    plan_id: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    default_mode: Literal["local", "distributed", "hybrid"]
    description: Optional[str]
```

---

### 7. Agent Interface

```python
class AgentSpec(BaseModel):
    agent_id: str
    capabilities: Dict[str, Any]
    trust_level: Optional[str]
    region: Optional[str]
```

Supports:
- Dry-run routing
- Trust-level negotiation
- Policy enforcement introspection
- Federation-aware fallback

---

### 8. Feedback and Reward

```python
class ExecutionFeedback(BaseModel):
    transformer_id: str
    rating: float
    reward_signal: Optional[float]
    notes: Optional[str]
```

---

### 9. Federation Model

- **Trust Anchors:** Signed via shared org roots
- **Fallbacks:** Fallback to local transformer on:
  - Latency failure
  - Signature mismatch
  - Quorum disagreement
- **Conflict Resolution Policies:**
  `trust_policy: local_prefer | remote_prefer | quarantine`

---

### 10. Policy Model

```python
class ExecutionPolicy(BaseModel):
    type: Literal["governance", "security", "compliance", "privacy"]
    scope: Literal["transformer", "plan", "session", "org"]
    enforcement_level: Literal["soft", "strict"]
    payload: Dict[str, Any]
```

Future: runtime negotiation protocol for agents to adjust policies dynamically.

---

### 11. Observability and Telemetry

Emit logs for:
- `execution_start`, `execution_end`
- `retry`, `fallback`, `error`, `budget_exceeded`
- `deprecation_warning`, `policy_violation`
- `resource_usage` (CPU, RAM, carbon, time)

---

### 12. Chaos and Resilience

- Support `chaos_mode: true` in conformance runs
- Transformers SHOULD tag fault domain
- Simulations MUST validate fallback routing

---

### 13. Formal Verification (Optional)

- Transformers may include:
```json
{ "formally_verified": true, "proof_reference": "..." }
```

- DAGs can declare:
```yaml
verified_chain: true
```

---

### 14. Continuous Benchmarking

- Required metrics:
  - `latency_ms`
  - `memory_mb`
  - `energy_kwh`
  - `error_rate`
- Version promotion gated by:
  - Canary runs
  - Regression checks
  - Benchmark deltas

---

### 15. Monetization Hooks

```json
{
  "billing_model": "metered",
  "pricing": { "unit": "execution", "cost_usd": 0.0001 },
  "sla": { "uptime": "99.95%", "latency_ms": 100 }
}
```

---

### 16. Deprecation and Sunset Policy

```json
{
  "deprecated_since": "0.1.0",
  "sunset_date": "2026-01-01",
  "replacement": "org.foo.new_transformer"
}
```

Deprecated entries MUST emit warnings.

---

### 17. RegistryEntry Wrapper (Extensible)

```python
class RegistryEntry(BaseModel):
    object_type: Literal["transformer", "schema", "agent", "plan", "policy"]
    payload: Any
    fingerprint: str
    trust_level: Optional[str]
    registered_at: datetime
    doc_url: Optional[str]
```

---

## 🧪 Conformance Suite

- Schema validation
- Retry/fallback conformance
- Budget exhaustion tracing
- Policy injection enforcement
- Federation failover simulation
- Chaos test validation

---

## 🏗 Developer Tooling

- `.pyi`, JSON Schema, OpenAPI exports
- CLI: `onex scaffold`, `onex simulate`, `onex validate`
- IDE plugins (VS Code baseline)
- Plan visualizer (Graphviz, Mermaid)

---

## 📚 References

- [OmniNode Tool Metadata Standard](omninode_tool_metadata_standard_v0_1.md)
- [O.N.E. Protocol Layer Overview](o_n_e__protocol_spec_v0_1.md)
- [ONEX Registry Field Reference](onex_registry_fields.md)

---
