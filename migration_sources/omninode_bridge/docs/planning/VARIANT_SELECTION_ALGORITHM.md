# Template Variant Selection Algorithm Design

**Task**: F4 - Foundation Task
**Version**: 1.0
**Created**: 2025-11-06
**Status**: ✅ Design Complete
**Performance Target**: <5ms selection time, >95% accuracy

---

## Executive Summary

The **Variant Selection Algorithm** intelligently chooses the optimal template variant (MINIMAL, STANDARD, PRODUCTION, CUSTOM) based on multi-dimensional analysis of node requirements. The algorithm uses a **weighted scoring system** with **confidence-based fallback** to ensure reliable template selection across diverse use cases.

### Key Features

- **Multi-Criteria Decision Making**: Analyzes complexity, features, environment, and patterns
- **Weighted Scoring**: Configurable weights for different selection factors
- **Confidence Scoring**: Provides confidence metric (0-1) for each selection
- **Graceful Degradation**: Falls back to STANDARD when confidence is low
- **Fast Selection**: <5ms target through efficient decision tree
- **Explainable AI**: Provides rationale for each selection decision

### Quick Reference

| Complexity | Environment | Features | Pattern Matches | → Variant |
|-----------|-------------|----------|-----------------|-----------|
| Simple | Development | 0-2 | 0-1 | **MINIMAL** |
| Simple | Production | 0-2 | 0-2 | **STANDARD** |
| Medium | Development | 3-5 | 2-4 | **STANDARD** |
| Medium | Production | 3-5 | 3-5 | **PRODUCTION** |
| Complex | Any | 6+ | 5+ | **PRODUCTION** |
| Any | Any | Custom patterns | 8+ | **CUSTOM** |

---

## 1. Algorithm Architecture

### 1.1 Selection Pipeline

```
Input: ModelPRDRequirements
    ↓
[1. Complexity Assessment] → ComplexityScore
    ↓
[2. Feature Extraction] → FeatureSet
    ↓
[3. Environment Detection] → EnvironmentType
    ↓
[4. Pattern Matching] → PatternMatchCount
    ↓
[5. Variant Scoring] → VariantScores (MINIMAL/STANDARD/PRODUCTION/CUSTOM)
    ↓
[6. Confidence Calculation] → ConfidenceScore (0-1)
    ↓
[7. Fallback Logic] → Final Variant Selection
    ↓
Output: ModelTemplateSelection (variant, confidence, rationale)
```

### 1.2 Core Components

```python
# Component hierarchy
VariantSelector
├── ComplexityAnalyzer
│   ├── assess_operation_count()
│   ├── assess_dependency_count()
│   └── assess_data_model_count()
├── FeatureExtractor
│   ├── extract_resilience_features()
│   ├── extract_observability_features()
│   └── extract_integration_features()
├── EnvironmentDetector
│   ├── detect_from_requirements()
│   └── detect_from_config()
├── PatternMatcher
│   ├── match_production_patterns()
│   └── count_pattern_matches()
├── VariantScorer
│   ├── score_minimal()
│   ├── score_standard()
│   ├── score_production()
│   └── score_custom()
└── ConfidenceCalculator
    ├── calculate_confidence()
    └── apply_fallback_logic()
```

---

## 2. Stage 1: Complexity Assessment

### 2.1 Complexity Metrics

**Definition**: Node complexity is determined by the **scale and interconnectedness** of operations.

**Scoring Dimensions**:

| Dimension | Weight | Simple (0-3) | Medium (4-7) | Complex (8-10) |
|-----------|--------|--------------|--------------|----------------|
| **Operation Count** | 35% | 1-2 ops | 3-5 ops | 6+ ops |
| **Dependency Count** | 30% | 0 deps | 1-2 deps | 3+ deps |
| **Data Model Count** | 20% | 0-1 models | 2-3 models | 4+ models |
| **Performance Requirements** | 15% | None | 1-2 metrics | 3+ metrics |

### 2.2 Complexity Scoring Formula

```python
def assess_complexity(requirements: ModelPRDRequirements) -> tuple[str, float]:
    """
    Assess node complexity and return classification + score.

    Returns:
        tuple[str, float]: ("simple"|"medium"|"complex", score 0-10)
    """
    # Operation count score (0-10, weight 0.35)
    operation_count = len(requirements.operations)
    if operation_count <= 2:
        op_score = 2.0
    elif operation_count <= 5:
        op_score = 5.0
    else:
        op_score = 9.0

    # Dependency count score (0-10, weight 0.30)
    dependency_count = len(requirements.dependencies)
    if dependency_count == 0:
        dep_score = 1.0
    elif dependency_count <= 2:
        dep_score = 5.0
    else:
        dep_score = 9.0

    # Data model count score (0-10, weight 0.20)
    model_count = len(requirements.data_models)
    if model_count <= 1:
        model_score = 2.0
    elif model_count <= 3:
        model_score = 5.0
    else:
        model_score = 9.0

    # Performance requirements score (0-10, weight 0.15)
    perf_count = len(requirements.performance_requirements)
    if perf_count == 0:
        perf_score = 1.0
    elif perf_count <= 2:
        perf_score = 5.0
    else:
        perf_score = 9.0

    # Weighted complexity score
    complexity_score = (
        op_score * 0.35 +
        dep_score * 0.30 +
        model_score * 0.20 +
        perf_score * 0.15
    )

    # Classify based on score
    if complexity_score <= 3.0:
        classification = "simple"
    elif complexity_score <= 7.0:
        classification = "medium"
    else:
        classification = "complex"

    return classification, complexity_score
```

### 2.3 Complexity Examples

**Simple Node** (score: 2.1):
```python
operations = ["fetch_user"]  # 1 operation
dependencies = {}  # 0 dependencies
data_models = ["User"]  # 1 model
performance_requirements = {}  # 0 metrics
# → score = 2.0*0.35 + 1.0*0.30 + 2.0*0.20 + 1.0*0.15 = 2.1
# → classification = "simple"
```

**Medium Node** (score: 5.3):
```python
operations = ["create_order", "update_inventory", "send_notification"]  # 3 ops
dependencies = {"database": "postgresql", "kafka": "redpanda"}  # 2 deps
data_models = ["Order", "LineItem", "Inventory"]  # 3 models
performance_requirements = {"latency_ms": 100}  # 1 metric
# → score = 5.0*0.35 + 5.0*0.30 + 5.0*0.20 + 1.0*0.15 = 4.4
# → classification = "medium"
```

**Complex Node** (score: 8.9):
```python
operations = ["orchestrate_workflow", "validate_state", "execute_rollback",
              "publish_events", "update_metrics", "trigger_alerts", "log_audit"]  # 7 ops
dependencies = {"db": "postgres", "kafka": "redpanda", "redis": "cache",
                "consul": "discovery"}  # 4 deps
data_models = ["Workflow", "Step", "State", "Event", "Metric"]  # 5 models
performance_requirements = {"latency_p99": 50, "throughput_rps": 1000,
                            "memory_mb": 512}  # 3 metrics
# → score = 9.0*0.35 + 9.0*0.30 + 9.0*0.20 + 9.0*0.15 = 9.0
# → classification = "complex"
```

---

## 3. Stage 2: Feature Extraction

### 3.1 Feature Categories

**Resilience Features**:
- `retry_policy`: Automatic retry on transient failures
- `circuit_breaker`: Prevent cascade failures
- `timeout_handling`: Request timeout management
- `fallback_strategy`: Graceful degradation

**Observability Features**:
- `health_checks`: Liveness/readiness probes
- `metrics_collection`: Prometheus/statsd metrics
- `distributed_tracing`: OpenTelemetry tracing
- `structured_logging`: JSON logging with correlation IDs

**Integration Features**:
- `event_publishing`: Kafka event publishing
- `event_consumption`: Kafka event consumption
- `api_integration`: HTTP API calls
- `database_integration`: Database operations
- `cache_integration`: Redis/memcached caching

**Security Features**:
- `input_validation`: Request validation
- `authentication`: Auth token validation
- `authorization`: RBAC/ABAC checks
- `secrets_management`: Vault integration
- `rate_limiting`: Request rate limiting

### 3.2 Feature Extraction Algorithm

```python
def extract_features(requirements: ModelPRDRequirements) -> set[str]:
    """
    Extract feature flags from requirements.

    Returns:
        set[str]: Set of feature identifiers
    """
    features = set()

    # Analyze features list
    for feature in requirements.features:
        feature_lower = feature.lower()

        # Resilience features
        if any(keyword in feature_lower for keyword in ["retry", "retries"]):
            features.add("retry_policy")
        if any(keyword in feature_lower for keyword in ["circuit", "breaker"]):
            features.add("circuit_breaker")
        if "timeout" in feature_lower:
            features.add("timeout_handling")
        if "fallback" in feature_lower:
            features.add("fallback_strategy")

        # Observability features
        if any(keyword in feature_lower for keyword in ["health", "healthcheck"]):
            features.add("health_checks")
        if any(keyword in feature_lower for keyword in ["metrics", "prometheus"]):
            features.add("metrics_collection")
        if any(keyword in feature_lower for keyword in ["tracing", "trace"]):
            features.add("distributed_tracing")
        if any(keyword in feature_lower for keyword in ["logging", "logs"]):
            features.add("structured_logging")

        # Integration features
        if any(keyword in feature_lower for keyword in ["event", "kafka", "publish"]):
            features.add("event_publishing")
        if any(keyword in feature_lower for keyword in ["consume", "consumer", "subscription"]):
            features.add("event_consumption")
        if any(keyword in feature_lower for keyword in ["api", "http", "rest"]):
            features.add("api_integration")
        if any(keyword in feature_lower for keyword in ["database", "db", "sql"]):
            features.add("database_integration")
        if any(keyword in feature_lower for keyword in ["cache", "redis", "memcached"]):
            features.add("cache_integration")

        # Security features
        if any(keyword in feature_lower for keyword in ["validation", "validate"]):
            features.add("input_validation")
        if any(keyword in feature_lower for keyword in ["auth", "authentication"]):
            features.add("authentication")
        if any(keyword in feature_lower for keyword in ["authorization", "rbac", "abac"]):
            features.add("authorization")
        if any(keyword in feature_lower for keyword in ["secrets", "vault"]):
            features.add("secrets_management")
        if any(keyword in feature_lower for keyword in ["rate", "throttle", "limit"]):
            features.add("rate_limiting")

    # Analyze dependencies
    for dep_name, dep_type in requirements.dependencies.items():
        dep_lower = f"{dep_name} {dep_type}".lower()

        if any(keyword in dep_lower for keyword in ["kafka", "redpanda", "event"]):
            features.add("event_publishing")
        if any(keyword in dep_lower for keyword in ["postgres", "mysql", "database"]):
            features.add("database_integration")
        if any(keyword in dep_lower for keyword in ["redis", "memcached"]):
            features.add("cache_integration")

    return features
```

### 3.3 Feature Scoring

**Feature Count → Variant Mapping**:

| Feature Count | Interpretation | Suggested Variant |
|--------------|----------------|------------------|
| 0-2 | Minimal requirements | MINIMAL or STANDARD |
| 3-5 | Standard production needs | STANDARD or PRODUCTION |
| 6-10 | Full production features | PRODUCTION |
| 11+ | Highly specialized | PRODUCTION or CUSTOM |

---

## 4. Stage 3: Environment Detection

### 4.1 Environment Types

| Environment | Characteristics | Variant Preference |
|------------|----------------|-------------------|
| **development** | Local dev, rapid iteration, debugging | MINIMAL → STANDARD |
| **testing** | CI/CD, automated testing, reproducibility | STANDARD |
| **staging** | Pre-production, performance testing | STANDARD → PRODUCTION |
| **production** | Live traffic, high reliability, monitoring | PRODUCTION |

### 4.2 Environment Detection Logic

```python
def detect_environment(requirements: ModelPRDRequirements) -> str:
    """
    Detect target environment from requirements.

    Returns:
        str: "development" | "testing" | "staging" | "production"
    """
    # Check performance requirements
    if requirements.performance_requirements:
        # If strict SLAs defined, likely production/staging
        if any(
            key in requirements.performance_requirements
            for key in ["latency_p99", "availability_sla", "throughput_rps"]
        ):
            return "production"

    # Check features for production indicators
    feature_keywords = " ".join(requirements.features).lower()

    if any(keyword in feature_keywords for keyword in
           ["high-availability", "disaster-recovery", "multi-region", "production"]):
        return "production"

    if any(keyword in feature_keywords for keyword in
           ["staging", "pre-production", "canary"]):
        return "staging"

    if any(keyword in feature_keywords for keyword in
           ["test", "ci", "cd", "integration-test"]):
        return "testing"

    # Default to development for simple cases
    return "development"
```

---

## 5. Stage 4: Pattern Matching

### 5.1 Pattern Matching Score

**Concept**: Count how many production patterns from the pattern library match the node requirements.

**Pattern Categories** (from PRODUCTION_NODE_PATTERNS.md):
1. Standard imports
2. Class declaration & docstring
3. Initialization pattern
4. Execute method pattern
5. Event publishing
6. Error handling
7. Lifecycle methods
8. Consul service discovery
9. Logging pattern
10. Metrics tracking
11. Main entry point
12. Type patterns

### 5.2 Pattern Matching Algorithm

```python
def count_pattern_matches(requirements: ModelPRDRequirements) -> int:
    """
    Count how many production patterns match the requirements.

    Returns:
        int: Pattern match count (0-12+)
    """
    match_count = 0

    # Pattern 1: Standard imports (always matches)
    match_count += 1

    # Pattern 2: Class declaration (always matches)
    match_count += 1

    # Pattern 3: Initialization (always matches)
    match_count += 1

    # Pattern 4: Execute method (always matches)
    match_count += 1

    # Pattern 5: Event publishing
    if any(
        keyword in requirements.features
        for keyword in ["event", "kafka", "publish"]
    ):
        match_count += 1

    # Pattern 6: Error handling (production requirement)
    if any(
        keyword in requirements.features
        for keyword in ["error", "retry", "circuit"]
    ):
        match_count += 1

    # Pattern 7: Lifecycle methods
    if any(
        keyword in requirements.features
        for keyword in ["lifecycle", "startup", "shutdown"]
    ):
        match_count += 1

    # Pattern 8: Consul service discovery
    if "consul" in str(requirements.dependencies).lower():
        match_count += 1

    # Pattern 9: Logging (production requirement)
    if any(
        keyword in requirements.features
        for keyword in ["logging", "observability", "tracing"]
    ):
        match_count += 1

    # Pattern 10: Metrics tracking
    if any(
        keyword in requirements.features
        for keyword in ["metrics", "monitoring", "prometheus"]
    ):
        match_count += 1

    # Pattern 11: Main entry point (always matches)
    match_count += 1

    # Pattern 12: Type patterns (always matches with Pydantic)
    match_count += 1

    return match_count
```

### 5.3 Pattern Match Interpretation

| Pattern Matches | Interpretation | Variant Suggestion |
|----------------|----------------|-------------------|
| 0-4 | Minimal patterns | MINIMAL |
| 5-7 | Standard patterns | STANDARD |
| 8-10 | Production patterns | PRODUCTION |
| 11-12 | Full production suite | PRODUCTION |
| 13+ | Custom patterns beyond standard | CUSTOM |

---

## 6. Stage 5: Variant Scoring

### 6.1 Scoring Matrix

Each variant receives a **score (0-100)** based on the input criteria.

**Scoring Formula**:
```
variant_score = (
    complexity_weight * complexity_match +
    feature_weight * feature_match +
    environment_weight * environment_match +
    pattern_weight * pattern_match
)
```

**Weight Configuration**:
```python
VARIANT_WEIGHTS = {
    "complexity": 0.30,      # 30% - Node complexity
    "features": 0.25,        # 25% - Feature requirements
    "environment": 0.25,     # 25% - Target environment
    "patterns": 0.20,        # 20% - Pattern matches
}
```

### 6.2 Variant Scoring Functions

```python
def score_minimal(
    complexity: str,
    feature_count: int,
    environment: str,
    pattern_count: int,
) -> float:
    """
    Score MINIMAL variant suitability.

    MINIMAL is best for:
    - Simple complexity
    - 0-2 features
    - Development environment
    - 0-4 pattern matches
    """
    score = 0.0

    # Complexity match (0-30 points)
    if complexity == "simple":
        score += 30.0
    elif complexity == "medium":
        score += 10.0
    else:  # complex
        score += 0.0

    # Feature match (0-25 points)
    if feature_count <= 2:
        score += 25.0
    elif feature_count <= 4:
        score += 15.0
    else:
        score += 0.0

    # Environment match (0-25 points)
    if environment == "development":
        score += 25.0
    elif environment == "testing":
        score += 15.0
    else:
        score += 0.0

    # Pattern match (0-20 points)
    if pattern_count <= 4:
        score += 20.0
    elif pattern_count <= 6:
        score += 10.0
    else:
        score += 0.0

    return score


def score_standard(
    complexity: str,
    feature_count: int,
    environment: str,
    pattern_count: int,
) -> float:
    """
    Score STANDARD variant suitability.

    STANDARD is best for:
    - Simple to medium complexity
    - 2-6 features
    - Development to staging environments
    - 4-8 pattern matches
    """
    score = 0.0

    # Complexity match (0-30 points)
    if complexity == "simple":
        score += 20.0
    elif complexity == "medium":
        score += 30.0
    else:  # complex
        score += 10.0

    # Feature match (0-25 points)
    if 2 <= feature_count <= 6:
        score += 25.0
    elif feature_count < 2:
        score += 15.0
    elif feature_count <= 8:
        score += 20.0
    else:
        score += 10.0

    # Environment match (0-25 points)
    if environment in ["development", "testing"]:
        score += 25.0
    elif environment == "staging":
        score += 20.0
    else:  # production
        score += 10.0

    # Pattern match (0-20 points)
    if 4 <= pattern_count <= 8:
        score += 20.0
    elif pattern_count < 4:
        score += 10.0
    else:
        score += 15.0

    return score


def score_production(
    complexity: str,
    feature_count: int,
    environment: str,
    pattern_count: int,
) -> float:
    """
    Score PRODUCTION variant suitability.

    PRODUCTION is best for:
    - Medium to complex complexity
    - 5+ features
    - Staging to production environments
    - 7+ pattern matches
    """
    score = 0.0

    # Complexity match (0-30 points)
    if complexity == "simple":
        score += 5.0
    elif complexity == "medium":
        score += 25.0
    else:  # complex
        score += 30.0

    # Feature match (0-25 points)
    if feature_count >= 7:
        score += 25.0
    elif feature_count >= 5:
        score += 20.0
    elif feature_count >= 3:
        score += 10.0
    else:
        score += 0.0

    # Environment match (0-25 points)
    if environment == "production":
        score += 25.0
    elif environment == "staging":
        score += 20.0
    elif environment == "testing":
        score += 10.0
    else:  # development
        score += 0.0

    # Pattern match (0-20 points)
    if pattern_count >= 9:
        score += 20.0
    elif pattern_count >= 7:
        score += 15.0
    elif pattern_count >= 5:
        score += 10.0
    else:
        score += 0.0

    return score


def score_custom(
    complexity: str,
    feature_count: int,
    environment: str,
    pattern_count: int,
) -> float:
    """
    Score CUSTOM variant suitability.

    CUSTOM is best for:
    - Complex nodes with specialized needs
    - 10+ features
    - Any environment
    - 11+ pattern matches (beyond standard patterns)
    """
    score = 0.0

    # Complexity match (0-30 points)
    if complexity == "complex":
        score += 30.0
    elif complexity == "medium":
        score += 15.0
    else:
        score += 0.0

    # Feature match (0-25 points)
    if feature_count >= 12:
        score += 25.0
    elif feature_count >= 10:
        score += 20.0
    elif feature_count >= 8:
        score += 10.0
    else:
        score += 0.0

    # Environment match (0-25 points)
    # Custom templates can work in any environment
    score += 20.0

    # Pattern match (0-20 points)
    if pattern_count >= 13:
        score += 20.0
    elif pattern_count >= 11:
        score += 15.0
    else:
        score += 0.0

    return score
```

### 6.3 Scoring Example

**Input**:
- Complexity: "medium" (score: 5.3)
- Feature count: 5
- Environment: "production"
- Pattern matches: 8

**Variant Scores**:
```python
minimal_score = score_minimal("medium", 5, "production", 8)
# = 10.0 (complexity) + 0.0 (features) + 0.0 (environment) + 0.0 (patterns)
# = 10.0 / 100

standard_score = score_standard("medium", 5, "production", 8)
# = 30.0 (complexity) + 20.0 (features) + 10.0 (environment) + 15.0 (patterns)
# = 75.0 / 100

production_score = score_production("medium", 5, "production", 8)
# = 25.0 (complexity) + 20.0 (features) + 25.0 (environment) + 15.0 (patterns)
# = 85.0 / 100  ← WINNER

custom_score = score_custom("medium", 5, "production", 8)
# = 15.0 (complexity) + 0.0 (features) + 20.0 (environment) + 0.0 (patterns)
# = 35.0 / 100
```

**Selected Variant**: **PRODUCTION** (score: 85.0)

---

## 7. Stage 6: Confidence Calculation

### 7.1 Confidence Formula

**Confidence Score** (0-1) indicates how certain the algorithm is about the selection.

```python
def calculate_confidence(
    selected_variant: str,
    variant_scores: dict[str, float],
) -> float:
    """
    Calculate confidence score for variant selection.

    Confidence is based on:
    1. Score margin (winner vs runner-up)
    2. Absolute score of winner
    3. Score distribution (entropy)

    Returns:
        float: Confidence score (0-1)
    """
    scores = sorted(variant_scores.values(), reverse=True)
    winner_score = scores[0]
    runner_up_score = scores[1] if len(scores) > 1 else 0.0

    # Factor 1: Score margin (0-0.5)
    # Larger margin = higher confidence
    margin = (winner_score - runner_up_score) / 100.0
    margin_confidence = min(margin * 2.0, 0.5)  # Cap at 0.5

    # Factor 2: Absolute score (0-0.3)
    # Higher absolute score = higher confidence
    absolute_confidence = (winner_score / 100.0) * 0.3

    # Factor 3: Score distribution entropy (0-0.2)
    # Lower entropy (concentrated scores) = higher confidence
    total_score = sum(variant_scores.values())
    if total_score > 0:
        probabilities = [s / total_score for s in variant_scores.values()]
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
        max_entropy = math.log2(len(variant_scores))
        entropy_confidence = (1.0 - entropy / max_entropy) * 0.2
    else:
        entropy_confidence = 0.0

    # Total confidence
    confidence = margin_confidence + absolute_confidence + entropy_confidence

    return min(confidence, 1.0)  # Cap at 1.0
```

### 7.2 Confidence Thresholds

| Confidence Range | Interpretation | Action |
|-----------------|----------------|--------|
| 0.85 - 1.00 | **Very High** | Use selected variant with confidence |
| 0.70 - 0.84 | **High** | Use selected variant, log rationale |
| 0.50 - 0.69 | **Medium** | Use selected variant, warn user |
| 0.30 - 0.49 | **Low** | Consider fallback to STANDARD |
| 0.00 - 0.29 | **Very Low** | Fallback to STANDARD, require review |

---

## 8. Stage 7: Fallback Logic

### 8.1 Fallback Decision Tree

```
IF confidence >= 0.70:
    → Use selected variant
ELIF confidence >= 0.50:
    → Use selected variant with warning
ELIF confidence >= 0.30:
    → IF selected_variant == PRODUCTION:
        → Use PRODUCTION
      ELIF selected_variant == CUSTOM:
        → Fallback to PRODUCTION
      ELSE:
        → Use STANDARD
ELSE:  # confidence < 0.30
    → Fallback to STANDARD (safe default)
```

### 8.2 Fallback Implementation

```python
def apply_fallback_logic(
    selected_variant: EnumTemplateVariant,
    confidence: float,
) -> tuple[EnumTemplateVariant, str]:
    """
    Apply fallback logic based on confidence.

    Returns:
        tuple[EnumTemplateVariant, str]: (final_variant, reason)
    """
    if confidence >= 0.70:
        # High confidence - use as-is
        return selected_variant, "High confidence selection"

    elif confidence >= 0.50:
        # Medium confidence - use with warning
        return (
            selected_variant,
            f"Medium confidence ({confidence:.2f}) - manual review recommended"
        )

    elif confidence >= 0.30:
        # Low confidence - selective fallback
        if selected_variant == EnumTemplateVariant.PRODUCTION:
            # Keep production if scores indicate it
            return (
                EnumTemplateVariant.PRODUCTION,
                f"Low confidence ({confidence:.2f}) but PRODUCTION scores highest"
            )
        elif selected_variant == EnumTemplateVariant.CUSTOM:
            # CUSTOM too risky at low confidence
            return (
                EnumTemplateVariant.PRODUCTION,
                f"Low confidence ({confidence:.2f}) - fallback from CUSTOM to PRODUCTION"
            )
        else:
            # Fall back to safe default
            return (
                EnumTemplateVariant.STANDARD,
                f"Low confidence ({confidence:.2f}) - fallback to STANDARD"
            )

    else:  # confidence < 0.30
        # Very low confidence - always use STANDARD
        return (
            EnumTemplateVariant.STANDARD,
            f"Very low confidence ({confidence:.2f}) - fallback to STANDARD safe default"
        )
```

---

## 9. Complete Algorithm Pseudocode

### 9.1 Main Selection Function

```python
def select_template_variant(
    requirements: ModelPRDRequirements,
    node_type: str,
    target_environment: Optional[str] = None,
) -> ModelTemplateSelection:
    """
    Select optimal template variant for requirements.

    Args:
        requirements: PRD requirements with feature flags
        node_type: Node type (effect/compute/reducer/orchestrator)
        target_environment: Optional target environment override

    Returns:
        ModelTemplateSelection with variant, confidence, and rationale
    """
    start_time = time.perf_counter()

    # Stage 1: Complexity Assessment
    complexity, complexity_score = assess_complexity(requirements)

    # Stage 2: Feature Extraction
    features = extract_features(requirements)
    feature_count = len(features)

    # Stage 3: Environment Detection
    environment = target_environment or detect_environment(requirements)

    # Stage 4: Pattern Matching
    pattern_count = count_pattern_matches(requirements)

    # Stage 5: Variant Scoring
    variant_scores = {
        "minimal": score_minimal(complexity, feature_count, environment, pattern_count),
        "standard": score_standard(complexity, feature_count, environment, pattern_count),
        "production": score_production(complexity, feature_count, environment, pattern_count),
        "custom": score_custom(complexity, feature_count, environment, pattern_count),
    }

    # Select variant with highest score
    selected_variant_str = max(variant_scores, key=variant_scores.get)
    selected_variant = EnumTemplateVariant(selected_variant_str)

    # Stage 6: Confidence Calculation
    confidence = calculate_confidence(selected_variant_str, variant_scores)

    # Stage 7: Fallback Logic
    final_variant, fallback_reason = apply_fallback_logic(selected_variant, confidence)

    # Build rationale
    rationale = _build_rationale(
        complexity=complexity,
        complexity_score=complexity_score,
        feature_count=feature_count,
        environment=environment,
        pattern_count=pattern_count,
        variant_scores=variant_scores,
        selected_variant=final_variant,
        confidence=confidence,
        fallback_reason=fallback_reason,
    )

    # Determine template path
    template_path = Path(f"templates/node_variants/{node_type}/{final_variant.value}.py.j2")

    # Determine base class
    base_class = _get_base_class(node_type)

    # Performance check
    selection_time_ms = (time.perf_counter() - start_time) * 1000
    if selection_time_ms > 5.0:
        logger.warning(
            f"Template selection took {selection_time_ms:.2f}ms (target: <5ms)"
        )

    return ModelTemplateSelection(
        variant=final_variant,
        template_path=template_path,
        base_class=base_class,
        mixins=list(features),  # Use extracted features as mixin hints
        patterns=[f"pattern_{i}" for i in range(pattern_count)],
        rationale=rationale,
        confidence_score=confidence,
    )
```

### 9.2 Rationale Builder

```python
def _build_rationale(
    complexity: str,
    complexity_score: float,
    feature_count: int,
    environment: str,
    pattern_count: int,
    variant_scores: dict[str, float],
    selected_variant: EnumTemplateVariant,
    confidence: float,
    fallback_reason: str,
) -> str:
    """Build human-readable rationale for selection."""

    rationale_parts = [
        f"Selected: {selected_variant.value.upper()}",
        f"Confidence: {confidence:.2f}",
        "",
        "Analysis:",
        f"  - Complexity: {complexity} (score: {complexity_score:.1f}/10)",
        f"  - Features: {feature_count} identified",
        f"  - Environment: {environment}",
        f"  - Pattern Matches: {pattern_count}/12",
        "",
        "Variant Scores:",
    ]

    for variant, score in sorted(variant_scores.items(), key=lambda x: x[1], reverse=True):
        marker = "✓" if variant == selected_variant.value else " "
        rationale_parts.append(f"  {marker} {variant.upper()}: {score:.1f}/100")

    if fallback_reason != "High confidence selection":
        rationale_parts.extend([
            "",
            f"Note: {fallback_reason}"
        ])

    return "\n".join(rationale_parts)
```

---

## 10. Performance Optimization

### 10.1 Optimization Strategies

**Target**: <5ms selection time

**Optimizations**:
1. **Lazy Evaluation**: Only compute scores for viable variants
2. **Early Exit**: If one variant scores >90, skip remaining
3. **Caching**: Cache pattern match results for similar nodes
4. **Precomputed Lookups**: Use dictionaries instead of lists
5. **Minimal String Operations**: Reduce string parsing overhead

### 10.2 Performance Profiling

```python
import cProfile
import pstats

def profile_selection():
    """Profile variant selection performance."""

    profiler = cProfile.Profile()
    profiler.enable()

    # Run selection 1000 times
    for _ in range(1000):
        select_template_variant(sample_requirements, "effect")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
```

**Expected Bottlenecks**:
- Feature extraction (string matching): ~30% of time
- Pattern matching: ~25% of time
- Variant scoring: ~20% of time
- Confidence calculation: ~15% of time
- Other: ~10%

---

## 11. Edge Cases & Special Handling

### 11.1 Edge Cases

**Case 1: Empty Requirements**
```python
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="unknown",
    domain="generic",
    business_description="No description provided",
)
# → Fallback to STANDARD (safe default)
```

**Case 2: Conflicting Signals**
```python
# Simple complexity but production environment
complexity = "simple"
environment = "production"
# → Score both MINIMAL and PRODUCTION
# → Select based on other factors (features, patterns)
# → If tie, prefer STANDARD as compromise
```

**Case 3: Custom Pattern Request**
```python
# User explicitly requests custom template
if "custom_template" in requirements.features:
    # Override algorithm and use CUSTOM
    return EnumTemplateVariant.CUSTOM
```

### 11.2 Special Node Types

**Orchestrator Nodes**:
- Default to PRODUCTION (coordination requires resilience)
- Minimum pattern count: 8

**Reducer Nodes**:
- Default to STANDARD or PRODUCTION
- Require event publishing patterns

**Effect Nodes**:
- Most flexible - can use any variant
- Selection based purely on requirements

**Compute Nodes**:
- Often MINIMAL or STANDARD (pure functions)
- Production only if high complexity

---

## 12. Testing Strategy

### 12.1 Test Coverage

**Unit Tests**:
- `test_assess_complexity()` - All complexity ranges
- `test_extract_features()` - All feature categories
- `test_detect_environment()` - All environment types
- `test_count_pattern_matches()` - Pattern matching accuracy
- `test_score_variants()` - Scoring functions
- `test_calculate_confidence()` - Confidence calculation
- `test_apply_fallback_logic()` - Fallback scenarios

**Integration Tests**:
- `test_select_template_variant_e2e()` - End-to-end selection
- `test_performance_target()` - <5ms selection time
- `test_accuracy_benchmark()` - >95% accuracy on known cases

### 12.2 Accuracy Validation

**Validation Dataset**: 50+ real node examples with known "correct" variants

**Validation Process**:
1. Run algorithm on all examples
2. Compare selected variant to expected variant
3. Calculate accuracy percentage
4. Analyze mismatches for algorithm improvement

**Target Accuracy**: >95% (47+ correct out of 50)

---

## 13. Monitoring & Observability

### 13.1 Metrics to Track

**Selection Metrics**:
- `variant_selection_time_ms`: P50, P95, P99 selection time
- `variant_selection_by_type`: Count of each variant selected
- `variant_confidence_distribution`: Histogram of confidence scores
- `fallback_triggered_count`: Number of fallback decisions

**Accuracy Metrics**:
- `manual_override_rate`: How often users override selection
- `variant_change_rate`: How often selected variant is changed
- `user_satisfaction_score`: Post-generation feedback

### 13.2 Logging

```python
logger.info(
    "Template variant selected",
    extra={
        "variant": final_variant.value,
        "confidence": confidence,
        "complexity": complexity,
        "feature_count": feature_count,
        "environment": environment,
        "pattern_count": pattern_count,
        "selection_time_ms": selection_time_ms,
        "fallback_applied": fallback_reason != "High confidence selection",
    }
)
```

---

## 14. Future Enhancements

### 14.1 Machine Learning Integration

**Phase 4+**: Train ML model on historical selection data

**Features for ML Model**:
- All current algorithm inputs
- User feedback (correct/incorrect)
- Post-generation metrics (test coverage, complexity)
- Historical pattern usage

**Expected Improvement**: Accuracy 95% → 98%+

### 14.2 User Feedback Loop

**Collect Feedback**:
- Did the selected variant work well? (Yes/No)
- What would you have preferred? (Variant selection)
- Why? (Free text)

**Use Feedback**:
- Adjust scoring weights
- Identify new patterns
- Refine fallback logic

### 14.3 Custom Variant Catalog

**Future Feature**: Allow users to create and register custom variants

**Custom Variant Metadata**:
```yaml
name: "microservice_api"
base_variant: "production"
specialized_for:
  - complexity: "medium"
  - domain: "api"
  - features: ["rest", "authentication", "rate_limiting"]
```

---

## 15. Summary & Next Steps

### 15.1 Algorithm Strengths

✅ **Multi-Dimensional Analysis**: Considers complexity, features, environment, patterns
✅ **Explainable**: Provides clear rationale for every selection
✅ **Fast**: <5ms target through efficient scoring
✅ **Accurate**: >95% target through weighted scoring
✅ **Robust**: Graceful fallback for edge cases
✅ **Extensible**: Easy to add new variants or criteria

### 15.2 Implementation Checklist

- [ ] Implement `ComplexityAnalyzer` class
- [ ] Implement `FeatureExtractor` class
- [ ] Implement `EnvironmentDetector` class
- [ ] Implement `PatternMatcher` class
- [ ] Implement `VariantScorer` class
- [ ] Implement `ConfidenceCalculator` class
- [ ] Implement `VariantSelector` main class
- [ ] Create unit tests (90%+ coverage)
- [ ] Create integration tests
- [ ] Performance benchmark (<5ms)
- [ ] Accuracy validation (>95%)
- [ ] Documentation update

### 15.3 Ready for C8 Implementation

**Next Task**: C8 - Add variant selection logic to template_engine.py

**Integration Points**:
1. Import `VariantSelector`
2. Call `select_template_variant()` in template engine
3. Use returned `ModelTemplateSelection` for template loading
4. Log selection metrics
5. Handle fallback scenarios

---

**Document Status**: ✅ Complete - Ready for Implementation
**Next Steps**: Proceed to C8 (Add variant selection logic to template_engine.py)
