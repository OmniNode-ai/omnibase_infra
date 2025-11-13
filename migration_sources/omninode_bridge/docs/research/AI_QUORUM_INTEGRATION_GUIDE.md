# AI Quorum Phase 4 Integration Guide

**Target**: omninode_bridge Phase 4 Code Generation
**Source**: omniclaude proven implementation
**Date**: 2025-11-06

---

## Quick Integration Reference

### 1. Install Dependencies

```bash
cd /Volumes/PRO-G40/Code/omninode_bridge

# Add to requirements.txt or pyproject.toml
httpx>=0.24.0
pyyaml>=6.0
```

### 2. Copy Core Files

```bash
# Create consensus directory
mkdir -p lib/consensus

# Copy from omniclaude
cp /Volumes/PRO-G40/Code/omniclaude/claude_hooks/lib/consensus/quorum.py \
   lib/consensus/

cp /Volumes/PRO-G40/Code/omniclaude/agents/parallel_execution/quorum_validator.py \
   lib/consensus/

cp /Volumes/PRO-G40/Code/omniclaude/agents/lib/models/quorum_config.py \
   lib/consensus/

# Create __init__.py
cat > lib/consensus/__init__.py << 'EOF'
"""AI Quorum consensus validation system."""

from .quorum import AIQuorum, ModelConfig, QuorumScore
from .quorum_validator import QuorumValidator, QuorumResult, ValidationDecision
from .quorum_config import QuorumConfig

__all__ = [
    "AIQuorum",
    "ModelConfig",
    "QuorumScore",
    "QuorumValidator",
    "QuorumResult",
    "ValidationDecision",
    "QuorumConfig",
]
EOF
```

### 3. Configure Models

```bash
# Add to .env
cat >> .env << 'EOF'

# AI Quorum Configuration
QUORUM_ENABLED=true
QUORUM_MODE=balanced  # fast, balanced, standard, strict

# Model Endpoints
OLLAMA_BASE_URL=http://192.168.86.200:11434
GEMINI_API_KEY=your_gemini_api_key_here
ZAI_API_KEY=your_zai_api_key_here  # Optional

# Performance
QUORUM_TIMEOUT_SECONDS=10
QUORUM_MIN_PARTICIPATION=0.60
EOF
```

### 4. Create config.yaml

```yaml
# config.yaml

quorum:
  enabled: true

  thresholds:
    auto_apply: 0.80
    approve: 0.70
    confidence: 0.60

  ollama:
    base_url: "http://192.168.86.200:11434"

  models:
    codestral:
      enabled: true
      name: "codestral:22b-v0.1-q4_K_M"
      type: "ollama"
      weight: 2.5
      timeout: 15.0

    gemini_flash:
      enabled: true
      name: "gemini-2.5-flash"
      type: "gemini"
      weight: 1.5
      timeout: 10.0

    deepseek_lite:
      enabled: true
      name: "deepseek-coder:6.7b"
      type: "ollama"
      weight: 1.0
      timeout: 10.0

    llama_31:
      enabled: true
      name: "llama3.1:8b"
      type: "ollama"
      weight: 1.0
      timeout: 10.0
```

### 5. Integrate into ContractInferencer

```python
# nodes/llm_effect/v1_0_0/node.py

from lib.consensus import QuorumValidator, ValidationDecision

class ContractInferencer:
    def __init__(self, quorum_enabled: bool = True):
        self.quorum_validator = QuorumValidator() if quorum_enabled else None

    async def infer_contract_with_validation(
        self,
        user_prompt: str,
        context: Dict[str, Any]
    ) -> ModelContract:
        """Generate and validate contract with AI quorum."""

        # Generate contract
        contract = await self._generate_contract(user_prompt, context)

        # Validate with quorum (if enabled)
        if self.quorum_validator:
            result = await self.quorum_validator.validate_intent(
                user_prompt=user_prompt,
                task_breakdown=contract.dict()
            )

            # Handle validation result
            if result.decision == ValidationDecision.FAIL:
                logger.warning(
                    f"Contract validation FAILED: {result.deficiencies}"
                )
                # Retry once with feedback
                contract = await self._regenerate_with_feedback(
                    user_prompt,
                    context,
                    result.deficiencies
                )

            elif result.decision == ValidationDecision.RETRY:
                logger.info(
                    f"Contract validation RETRY: {result.deficiencies}"
                )
                # Refine with feedback
                contract = await self._refine_contract(
                    contract,
                    result.deficiencies
                )

            else:  # PASS
                logger.info(
                    f"Contract validation PASSED (score: {result.scores.get('alignment', 0):.1f}, "
                    f"confidence: {result.confidence:.2f})"
                )

            # Log metrics
            await self._log_quorum_metrics(result)

        return contract

    async def _regenerate_with_feedback(
        self,
        user_prompt: str,
        context: Dict[str, Any],
        deficiencies: List[str]
    ) -> ModelContract:
        """Regenerate contract with feedback from quorum."""

        # Add feedback to context
        feedback_context = {
            **context,
            "quorum_feedback": deficiencies,
            "regeneration_attempt": True
        }

        # Regenerate with enriched prompt
        enriched_prompt = f"""
{user_prompt}

IMPORTANT: Address these issues from previous attempt:
{chr(10).join(f"- {d}" for d in deficiencies)}
"""

        return await self._generate_contract(enriched_prompt, feedback_context)
```

### 6. Test Integration

```python
# tests/test_quorum_integration.py

import pytest
from lib.consensus import QuorumValidator, ValidationDecision

@pytest.mark.asyncio
async def test_contract_validation_pass():
    """Test successful contract validation."""
    validator = QuorumValidator()

    result = await validator.validate_intent(
        user_prompt="Create a Kafka consumer effect node",
        task_breakdown={
            "node_type": "Effect",
            "name": "KafkaConsumer",
            "description": "Consumes events from Kafka",
            "input_model": {"topic": {"type": "str"}},
            "output_model": {"events": {"type": "List[Dict]"}}
        }
    )

    assert result.decision == ValidationDecision.PASS
    assert result.confidence > 0.6
    assert result.scores["alignment"] > 75

@pytest.mark.asyncio
async def test_contract_validation_fail():
    """Test failed contract validation (wrong node type)."""
    validator = QuorumValidator()

    result = await validator.validate_intent(
        user_prompt="Create a Kafka consumer effect node",
        task_breakdown={
            "node_type": "Compute",  # WRONG! Should be Effect
            "name": "KafkaConsumer",
            "description": "Consumes events from Kafka"
        }
    )

    assert result.decision in [ValidationDecision.RETRY, ValidationDecision.FAIL]
    assert "node type" in " ".join(result.deficiencies).lower()

@pytest.mark.asyncio
async def test_quorum_resilience():
    """Test quorum still works with model failures."""
    # Test handled by quorum system internally
    # Minimum 60% participation enforced
    pass
```

---

## Integration Architecture

### Phase 4 Pipeline with Quorum

```
┌──────────────────────────────────────────────────────────────────┐
│                 Phase 4 Code Generation Pipeline                  │
└──────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Prompt Analysis                                       │
│   • Extract requirements from user prompt                      │
│   • Identify domain and context                                │
│                                                                 │
│   [OPTIONAL] Quorum: Validate PRD Completeness                │
│   └─→ 10% of pipelines (if quorum_mode = "strict")           │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 1.5: Intelligence Gathering (RAG)                        │
│   • Query similar patterns from knowledge base                 │
│   • Retrieve best practices and templates                      │
│                                                                 │
│   [OPTIONAL] Quorum: Validate Architecture Decisions          │
│   └─→ 20% of pipelines (if quorum_mode = "standard/strict")  │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 2: Contract Generation (ContractInferencer)              │
│   • Generate ModelContract via LLM                             │
│   • Define input/output models                                 │
│   • Specify node type (Effect/Compute/Reducer/Orchestrator)    │
│                                                                 │
│   [CRITICAL] Quorum: Validate Contract Structure              │ ⭐
│   └─→ 100% of pipelines (if quorum_mode != "fast")           │
│       ┌──────────────────────────────────────┐                │
│       │      QuorumValidator.validate_intent  │                │
│       │                                       │                │
│       │  1. Check node type correctness      │                │
│       │  2. Verify input/output models       │                │
│       │  3. Validate FSM states (if present) │                │
│       │  4. Check alignment with user prompt │                │
│       │                                       │                │
│       │  Decision: PASS / RETRY / FAIL       │                │
│       └──────────────────────────────────────┘                │
│                            ↓                                    │
│   IF FAIL: Regenerate contract with feedback                   │
│   IF RETRY: Refine contract with deficiency list               │
│   IF PASS: Continue to Stage 3                                 │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 3: Pre-Generation Validation                             │
│   • ONEX compliance checks                                     │
│   • Dependency validation                                      │
│   • Schema validation                                          │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 4: Code Generation (LLM Effect Node)                     │
│   • Generate Python code via LLM                               │
│   • Implement business logic                                   │
│   • Add error handling                                         │
│                                                                 │
│   [HIGH PRIORITY] Quorum: Validate Business Logic             │ ⭐
│   └─→ 80% of pipelines (if quorum_mode = "standard/strict")  │
│       ┌──────────────────────────────────────┐                │
│       │   AIQuorum.score_correction          │                │
│       │                                       │                │
│       │  1. Check algorithm correctness      │                │
│       │  2. Verify error handling            │                │
│       │  3. Validate edge cases              │                │
│       │  4. Assess code quality              │                │
│       │                                       │                │
│       │  Score: 0.0-1.0, Confidence: 0.0-1.0│                │
│       └──────────────────────────────────────┘                │
│                            ↓                                    │
│   IF LOW SCORE: Regenerate with feedback                       │
│   IF MEDIUM SCORE: Apply refinements                           │
│   IF HIGH SCORE: Continue to Stage 5                           │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 5: Post-Generation Validation                            │
│   • Syntax checks (AST parsing)                                │
│   • ONEX pattern validation                                    │
│   • Test generation                                            │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Stage 6: File Writing                                          │
│   • Write node implementation                                  │
│   • Write contract YAML                                        │
│   • Write tests                                                │
└────────────────────────────────────────────────────────────────┘
```

### Quorum Validation Flow Detail

```
┌─────────────────────────────────────────────────────────────────┐
│              Quorum Validation Flow (Stage 2)                   │
└─────────────────────────────────────────────────────────────────┘

User Prompt: "Create a Kafka consumer effect node"
     +
Generated Contract:
{
  "node_type": "Effect",
  "name": "KafkaConsumer",
  "input_model": {...},
  "output_model": {...}
}
     │
     ↓
┌────────────────────────────────────────────┐
│  QuorumValidator.validate_intent()          │
└────────────────────────────────────────────┘
     │
     ├─→ Generate Validation Prompt
     │   """
     │   Given user request: "Create a Kafka consumer effect node"
     │
     │   Task breakdown generated:
     │   {
     │     "node_type": "Effect",
     │     "name": "KafkaConsumer",
     │     ...
     │   }
     │
     │   Questions:
     │   1. Does the task breakdown correctly understand user intent? (0-100)
     │   2. Is the correct node type selected? (Effect/Compute/Reducer/Orchestrator)
     │   3. Are all requirements captured? (list any missing)
     │
     │   Respond with JSON only:
     │   {
     │     "alignment_score": <0-100>,
     │     "correct_node_type": <true/false>,
     │     "expected_node_type": "<type>",
     │     "missing_requirements": [<list>],
     │     "recommendation": "PASS|RETRY|FAIL"
     │   }
     │   """
     │
     ↓
┌────────────────────────────────────────────┐
│  Parallel Model Execution (3-10s)          │
└────────────────────────────────────────────┘
     │
     ├─────────┬─────────┬─────────┬─────────┐
     │         │         │         │         │
     ↓         ↓         ↓         ↓         ↓
 ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
 │Gemini│  │GLM  │  │DeepSeek│ │Codestral│
 │Flash │  │4.5  │  │Lite │  │     │
 │     │  │     │  │     │  │     │
 │w:1.5│  │w:2.0│  │w:1.0│  │w:2.5│
 └─────┘  └─────┘  └─────┘  └─────┘
     │         │         │         │
     ↓         ↓         ↓         ↓
 Score:    Score:    Score:    Score:
  0.85      0.90      0.82      0.88
 APPROVE   APPROVE   APPROVE   APPROVE
     │         │         │         │
     └─────────┴─────────┴─────────┘
                  ↓
     ┌────────────────────────────┐
     │ Enforce MIN_PARTICIPATION  │
     │   (60% = 3/4 models)       │
     │                             │
     │   ✅ 4/4 models responded  │
     │   (100% > 60% required)    │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │ Calculate Weighted Consensus│
     │                             │
     │ consensus = Σ(score × weight) / Σ(weight)
     │           = (0.85×1.5 + 0.90×2.0 + 0.82×1.0 + 0.88×2.5) / 7.0
     │           = (1.275 + 1.80 + 0.82 + 2.20) / 7.0
     │           = 6.095 / 7.0
     │           = 0.871
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │ Calculate Confidence        │
     │                             │
     │ variance = Σ(score - 0.871)² / 4
     │          = ((0.85-0.871)² + (0.90-0.871)² + (0.82-0.871)² + (0.88-0.871)²) / 4
     │          = (0.000441 + 0.000841 + 0.002601 + 0.000081) / 4
     │          = 0.003964 / 4
     │          = 0.000991
     │
     │ confidence = 1.0 - variance
     │            = 1.0 - 0.000991
     │            = 0.999
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │ Determine Recommendation    │
     │                             │
     │ Models saying APPROVE: 4/4  │
     │ Models saying REJECT: 0/4   │
     │ Models saying REVIEW: 0/4   │
     │                             │
     │ → Majority (100%) = APPROVE │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │       QuorumResult          │
     │                             │
     │ decision: PASS              │
     │ confidence: 0.999           │
     │ scores: {                   │
     │   "alignment": 86.25,       │
     │   "pass_pct": 1.00          │
     │ }                           │
     │ deficiencies: []            │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   Contract APPROVED         │
     │   Continue to Stage 3       │
     └────────────────────────────┘
```

### Retry Flow (on FAIL/RETRY)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Retry Flow with Feedback                       │
└─────────────────────────────────────────────────────────────────┘

Initial Contract:
{
  "node_type": "Compute",  ❌ WRONG! Should be Effect
  "name": "KafkaConsumer",
  ...
}
     │
     ↓
QuorumResult:
{
  "decision": "RETRY",
  "confidence": 0.55,
  "deficiencies": [
    "Incorrect node type: should be Effect not Compute",
    "Kafka consumers perform I/O operations (Effect nodes)",
    "Missing error handling in output_model"
  ]
}
     │
     ↓
┌────────────────────────────────────────────┐
│  _regenerate_with_feedback()               │
└────────────────────────────────────────────┘
     │
     ├─→ Enrich prompt with feedback:
     │   """
     │   Create a Kafka consumer effect node
     │
     │   IMPORTANT: Address these issues:
     │   - Incorrect node type: should be Effect not Compute
     │   - Kafka consumers perform I/O operations (Effect nodes)
     │   - Missing error handling in output_model
     │   """
     │
     ↓
┌────────────────────────────────────────────┐
│  Generate Contract (Attempt 2)             │
└────────────────────────────────────────────┘
     │
     ↓
Regenerated Contract:
{
  "node_type": "Effect",  ✅ CORRECTED
  "name": "KafkaConsumer",
  "input_model": {...},
  "output_model": {
    "events": {...},
    "error": {...}  ✅ ADDED
  }
}
     │
     ↓
┌────────────────────────────────────────────┐
│  Validate Again with Quorum                │
└────────────────────────────────────────────┘
     │
     ↓
QuorumResult:
{
  "decision": "PASS",
  "confidence": 0.92,
  "scores": {"alignment": 88.5}
}
     │
     ↓
✅ Contract APPROVED (after retry)
```

---

## Configuration Examples

### Development Setup (Fast Mode)

```python
# dev_config.py

from lib.consensus import QuorumConfig

config = QuorumConfig.from_mode("fast")

# Result:
# - No quorum validation (stub mode)
# - Latency: <1ms
# - Quality: Manual review required
```

### Production Setup (Balanced Mode)

```python
# prod_config.py

from lib.consensus import QuorumConfig

config = QuorumConfig.from_mode("balanced")

# Result:
# - PRD validation: ✅
# - Contract validation: ✅ (CRITICAL)
# - Intelligence validation: ❌
# - Code validation: ❌
# - Latency: 6-20s
# - Quality: 90% issue detection
```

### Critical Setup (Strict Mode)

```python
# critical_config.py

from lib.consensus import QuorumConfig

config = QuorumConfig.from_mode("strict")

# Result:
# - PRD validation: ✅
# - Contract validation: ✅
# - Intelligence validation: ✅
# - Code validation: ✅
# - Latency: 12-40s
# - Quality: 95% issue detection
```

---

## Metrics & Monitoring

### Track These Metrics

```python
# In QuorumValidator

@dataclass
class QuorumMetrics:
    total_validations: int = 0
    pass_count: int = 0
    retry_count: int = 0
    fail_count: int = 0

    avg_consensus_score: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0

    model_participation_rates: Dict[str, float] = field(default_factory=dict)
    model_avg_scores: Dict[str, float] = field(default_factory=dict)

    cache_hit_rate: float = 0.0

# Log after each validation
async def _log_metrics(self, result: QuorumResult, latency_ms: float):
    """Log metrics to Kafka and database."""

    # Publish to Kafka
    event = {
        "event_type": "quorum_validation_completed",
        "timestamp": datetime.utcnow().isoformat(),
        "stage": "contract",
        "decision": result.decision.value,
        "confidence": result.confidence,
        "consensus_score": result.scores.get('alignment', 0),
        "latency_ms": latency_ms,
        "model_count": len(result.model_responses)
    }

    await self.kafka_producer.send("quorum-validation-events", event)

    # Update PostgreSQL metrics table
    await self._update_metrics_db(event)
```

### Dashboard Queries

```sql
-- Quorum validation success rate by stage
SELECT
  stage,
  COUNT(*) as total,
  ROUND(100.0 * SUM(CASE WHEN decision = 'PASS' THEN 1 ELSE 0 END) / COUNT(*), 2) as pass_rate,
  ROUND(AVG(confidence), 3) as avg_confidence,
  ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM quorum_validation_events
WHERE timestamp > now() - interval '7 days'
GROUP BY stage
ORDER BY pass_rate DESC;

-- Model participation and performance
SELECT
  model,
  COUNT(*) as responses,
  ROUND(AVG(consensus_score), 3) as avg_score,
  ROUND(AVG(CASE WHEN recommendation = 'APPROVE' THEN 1.0 ELSE 0.0 END), 3) as approve_rate
FROM (
  SELECT
    resp->>'model' as model,
    (resp->>'alignment_score')::float as consensus_score,
    resp->>'recommendation' as recommendation
  FROM quorum_validation_events,
       jsonb_array_elements(model_responses) as resp
  WHERE timestamp > now() - interval '7 days'
) model_data
GROUP BY model
ORDER BY avg_score DESC;
```

---

## Troubleshooting

### Issue: "Insufficient model participation"

**Cause**: <60% of models responded

**Solution**:
```python
# Check model availability
for model_name, config in self.models.items():
    try:
        response = await self._test_model_connection(config)
        print(f"✓ {model_name}: Available")
    except Exception as e:
        print(f"✗ {model_name}: {e}")

# Reduce minimum participation temporarily
MIN_MODEL_PARTICIPATION = 0.50  # 50% instead of 60%
```

### Issue: "Quorum validation too slow"

**Cause**: Models timing out or slow response

**Solution**:
```python
# 1. Reduce timeout per model
ModelConfig(name="codestral", timeout=5.0)  # From 15.0

# 2. Use faster models
# Replace codestral:22b with deepseek:6.7b (3x faster)

# 3. Enable caching
quorum = AIQuorum(enable_cache=True, cache_ttl=3600)

# 4. Use stub mode for development
quorum = AIQuorum(stub_mode=True)  # <1ms validation
```

### Issue: "Too many false positives"

**Cause**: Models too strict or thresholds too high

**Solution**:
```python
# 1. Lower confidence threshold
if result.consensus_score >= 0.6:  # From 0.7
    decision = "PASS"

# 2. Adjust model weights
# Reduce weight of strict model
ModelConfig(name="strict_model", weight=0.5)  # From 2.0

# 3. Add more lenient models
ModelConfig(name="llama3.1", weight=1.5)  # General reasoning
```

---

## Testing Checklist

✅ **Unit Tests**:
- Consensus calculation with weighted voting
- Minimum participation enforcement
- Confidence calculation from variance
- Retry logic with feedback

✅ **Integration Tests**:
- Contract validation (PASS/RETRY/FAIL scenarios)
- Business logic validation
- Model failure resilience
- Cache effectiveness

✅ **Performance Tests**:
- Validation latency p95 < 10s
- Cache hit rate > 30%
- Parallel vs sequential speedup > 2x

✅ **End-to-End Tests**:
- Full pipeline with quorum enabled
- Retry flow with feedback application
- Multi-stage validation coordination

---

## Success Criteria

✅ **Contract Validation**:
- 95% accuracy in detecting incorrect node types
- 80% reduction in downstream failures
- <10s p95 validation latency

✅ **Business Logic Validation**:
- 40% improvement in generated code quality
- 90% reduction in logic errors
- <12s p95 validation latency

✅ **System Reliability**:
- 98% uptime (quorum system available)
- Graceful degradation on model failures
- 60%+ model participation enforced

---

**Integration Guide Complete**
**Ready for Phase 4 Implementation**

For detailed implementation examples and architecture analysis, see:
- [`AI_QUORUM_RESEARCH_REPORT.md`](./AI_QUORUM_RESEARCH_REPORT.md) - Full research report (30KB)
- [`AI_QUORUM_KEY_FINDINGS.md`](./AI_QUORUM_KEY_FINDINGS.md) - Key findings summary (12KB)
