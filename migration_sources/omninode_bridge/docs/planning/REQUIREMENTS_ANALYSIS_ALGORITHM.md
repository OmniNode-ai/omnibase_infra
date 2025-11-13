# Requirements Analysis Algorithm - Phase 3 Intelligent Mixin Selection

**Version**: 1.0
**Created**: 2025-11-06
**Status**: 🎯 Design Complete
**Task**: C12 - Design Requirements Analysis Algorithm
**Stream**: Core Stream 3 - Intelligent Mixin Selection

---

## Executive Summary

This document defines the **Requirements Analysis Algorithm** for Phase 3 intelligent mixin selection. The algorithm analyzes `ModelPRDRequirements` from PRD analysis to automatically extract mixin requirements, categorize them, and score recommendations with >90% relevance.

### Key Innovations

- **Multi-dimensional requirement extraction**: Keyword, dependency, operation, and pattern-based analysis
- **Weighted scoring system**: Configurable weights per mixin category (database, resilience, observability, security)
- **Confidence calculation**: Statistical confidence based on multiple signal strengths
- **Conflict detection**: Automatic detection of mutually exclusive mixins
- **Usage learning**: Adaptive scoring based on historical success rates

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Analysis Time** | <50ms | Requirements analysis for single PRD |
| **Scoring Time** | <100ms | Score all 21 mixins against requirements |
| **Recommendation Time** | <20ms | Generate top-K recommendations |
| **Accuracy** | >90% | Recommendation relevance (manual validation) |
| **Confidence Calibration** | ±5% | Confidence score matches actual success rate |

---

## 1. Algorithm Overview

### 1.1 High-Level Flow

```
ModelPRDRequirements Input
         ↓
┌─────────────────────────────────────┐
│  Step 1: Feature Extraction         │
│  - Extract keywords from text       │
│  - Parse dependency lists           │
│  - Identify operation patterns      │
│  - Extract performance requirements │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Step 2: Requirement Categorization │
│  - Database requirements            │
│  - API requirements                 │
│  - Kafka/messaging requirements     │
│  - Security requirements            │
│  - Observability requirements       │
│  - Performance requirements         │
│  - Caching requirements             │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Step 3: Mixin Scoring              │
│  - Match requirements to mixins     │
│  - Calculate base scores            │
│  - Apply category weights           │
│  - Normalize scores (0-1)           │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Step 4: Conflict Detection         │
│  - Identify mutually exclusive      │
│  - Check prerequisite chains        │
│  - Detect redundancies              │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Step 5: Recommendation Generation  │
│  - Rank by score                    │
│  - Filter conflicts                 │
│  - Generate explanations            │
│  - Apply usage statistics           │
└─────────────┬───────────────────────┘
              ↓
List[ModelMixinRecommendation] Output
```

### 1.2 Core Data Structures

**Input**: `ModelPRDRequirements` (from prd_analyzer.py)
```python
class ModelPRDRequirements(BaseModel):
    node_type: str                        # effect/compute/reducer/orchestrator
    service_name: str                     # snake_case service name
    domain: str                           # database/api/ml/messaging/...
    operations: list[str]                 # ['create_record', 'update_record', ...]
    features: list[str]                   # ['connection_pooling', 'retry', ...]
    dependencies: dict[str, str]          # {'asyncpg': '>=0.28.0', ...}
    performance_requirements: dict        # {'latency_ms': 100, ...}
    business_description: str             # Natural language description
    best_practices: list[str]             # From RAG intelligence
    similar_patterns: list[str]           # Pattern IDs from intelligence
```

**Intermediate**: `ModelRequirementAnalysis` (new)
```python
class ModelRequirementAnalysis(BaseModel):
    """Structured analysis of PRD requirements."""

    # Extracted features
    keywords: set[str]                    # Normalized keywords from text
    dependency_packages: set[str]         # Dependency package names
    operation_types: set[str]             # Operation categories

    # Categorized requirements (0-10 score per category)
    database_score: float = 0.0           # Database operations strength
    api_score: float = 0.0                # API client operations strength
    kafka_score: float = 0.0              # Kafka/messaging operations strength
    security_score: float = 0.0           # Security requirements strength
    observability_score: float = 0.0      # Observability requirements strength
    resilience_score: float = 0.0         # Resilience requirements strength
    caching_score: float = 0.0            # Caching requirements strength
    performance_score: float = 0.0        # Performance optimization strength

    # Metadata
    confidence: float                      # Overall extraction confidence (0-1)
    rationale: str                        # Why these scores were assigned
```

**Output**: `ModelMixinRecommendation` (new)
```python
class ModelMixinRecommendation(BaseModel):
    """Recommendation for a specific mixin."""

    mixin_name: str                       # e.g., "MixinConnectionPooling"
    score: float                          # Confidence score (0-1)
    category: str                         # database/resilience/observability/security
    explanation: str                      # Why this mixin is recommended
    matched_requirements: list[str]       # Which requirements matched
    prerequisites: list[str]              # Other mixins or dependencies needed
    conflicts_with: list[str]             # Conflicting mixins
```

---

## 2. Feature Extraction Algorithm

### 2.1 Keyword Extraction

**Goal**: Extract domain-relevant keywords from text fields

**Sources**:
- `business_description` (weighted 1.0)
- `operations` (weighted 1.5 - operation names are highly relevant)
- `features` (weighted 2.0 - explicit feature requests)
- `best_practices` (weighted 0.8 - contextual information)

**Algorithm**:
```python
def extract_keywords(requirements: ModelPRDRequirements) -> set[str]:
    """
    Extract weighted keywords from all text fields.

    Process:
    1. Tokenize all text fields
    2. Normalize to lowercase
    3. Remove stopwords (the, a, an, is, are, ...)
    4. Stem/lemmatize (e.g., "connecting" → "connect")
    5. Filter to relevant domain terms (database, api, kafka, ...)
    6. Apply source weights
    7. Return unique keywords above threshold
    """
    keywords = set()

    # Extract from business_description
    for word in tokenize(requirements.business_description):
        if is_domain_relevant(word) and not is_stopword(word):
            keywords.add(normalize(word))

    # Extract from operations (higher weight)
    for operation in requirements.operations:
        for word in tokenize(operation):
            if is_domain_relevant(word):
                keywords.add(normalize(word))

    # Extract from features (highest weight)
    for feature in requirements.features:
        for word in tokenize(feature):
            keywords.add(normalize(word))

    return keywords
```

**Domain Keyword Dictionary** (used by `is_domain_relevant`):
```yaml
database_keywords:
  - database, db, postgres, postgresql, sql, query, transaction
  - connection, pool, pooling, crud, create, read, update, delete
  - table, schema, migration, orm, sqlalchemy, asyncpg
  - commit, rollback, isolation, lock

api_keywords:
  - api, rest, http, https, client, request, response
  - get, post, put, patch, delete, endpoint
  - json, xml, graphql, rest, soap
  - httpx, aiohttp, requests, urllib
  - retry, timeout, backoff, circuit, breaker

kafka_keywords:
  - kafka, redpanda, event, message, publish, consume
  - topic, partition, offset, broker, producer, consumer
  - stream, streaming, event-driven, async
  - aiokafka, confluent, serialization, deserialization

security_keywords:
  - secure, security, auth, authentication, authorization
  - token, jwt, oauth, api-key, credential, secret
  - encrypt, decrypt, hash, sign, verify
  - validate, validation, sanitize, escape
  - sensitive, pii, redact, mask

observability_keywords:
  - metrics, logging, tracing, monitoring, observability
  - prometheus, grafana, datadog, opentelemetry
  - health, healthcheck, liveness, readiness
  - log, logger, structured-logging, json-log
  - trace, span, baggage, context

resilience_keywords:
  - retry, circuit-breaker, fallback, timeout
  - fault-tolerant, resilient, robust, reliable
  - backoff, exponential-backoff, jitter
  - degradation, graceful-degradation

caching_keywords:
  - cache, caching, memoize, ttl, expiration
  - redis, memcached, in-memory, lru
  - invalidate, evict, warm-up

performance_keywords:
  - performance, optimize, optimization, fast, speed
  - throughput, latency, qps, rps, concurrent
  - batch, bulk, parallel, async, asyncio
  - pool, pooling, connection-reuse
```

### 2.2 Dependency Analysis

**Goal**: Extract package names and infer capabilities from dependencies

**Algorithm**:
```python
def analyze_dependencies(dependencies: dict[str, str]) -> set[str]:
    """
    Extract package names and infer technical stack.

    Process:
    1. Extract package names (keys from dict)
    2. Normalize package names (lowercase, strip extras)
    3. Map packages to capabilities (asyncpg → database)
    4. Return capability set
    """
    capabilities = set()

    for package_name, version_spec in dependencies.items():
        normalized = normalize_package_name(package_name)

        # Map to capabilities
        if normalized in ['asyncpg', 'psycopg2', 'sqlalchemy']:
            capabilities.add('database')
            capabilities.add('postgres')
        elif normalized in ['httpx', 'aiohttp', 'requests']:
            capabilities.add('api')
            capabilities.add('http-client')
        elif normalized in ['aiokafka', 'confluent-kafka']:
            capabilities.add('kafka')
            capabilities.add('messaging')
        elif normalized in ['redis', 'memcached']:
            capabilities.add('caching')
        elif normalized in ['prometheus-client', 'opentelemetry']:
            capabilities.add('metrics')
            capabilities.add('observability')

    return capabilities
```

**Dependency → Capability Mapping**:
```yaml
database:
  - asyncpg, psycopg2, psycopg3, sqlalchemy
  - databases, aiosqlite, asyncmy, aiomysql

http_client:
  - httpx, aiohttp, requests, urllib3

kafka:
  - aiokafka, confluent-kafka, kafka-python

caching:
  - redis, aioredis, redis-py, memcached, aiomemcache

metrics:
  - prometheus-client, opentelemetry-api, opentelemetry-sdk
  - datadog, statsd

security:
  - cryptography, pyjwt, python-jose, bcrypt, passlib

resilience:
  - tenacity, backoff, circuit-breaker, pybreaker
```

### 2.3 Operation Pattern Recognition

**Goal**: Identify operation patterns from operation names

**Patterns**:
```yaml
crud_pattern:
  keywords: [create, read, update, delete, insert, select, upsert, get, put, post, patch]
  category: database
  score_weight: 3.0

streaming_pattern:
  keywords: [stream, consume, process, handle, batch, bulk]
  category: performance
  score_weight: 2.0

api_call_pattern:
  keywords: [fetch, call, request, invoke, query, get, post]
  category: api
  score_weight: 2.5

event_pattern:
  keywords: [publish, emit, send, notify, broadcast, produce]
  category: kafka
  score_weight: 2.5

validation_pattern:
  keywords: [validate, check, verify, sanitize, clean]
  category: security
  score_weight: 2.0

aggregation_pattern:
  keywords: [aggregate, sum, count, reduce, collect, accumulate]
  category: performance
  score_weight: 1.5
```

**Algorithm**:
```python
def identify_operation_patterns(operations: list[str]) -> dict[str, float]:
    """
    Identify patterns in operation names and score categories.

    Returns:
        dict mapping category → pattern strength (0-10)
    """
    category_scores = defaultdict(float)

    for operation in operations:
        operation_lower = operation.lower()

        # Check each pattern
        for pattern_name, pattern_config in PATTERNS.items():
            keywords = pattern_config['keywords']
            category = pattern_config['category']
            weight = pattern_config['score_weight']

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in operation_lower)

            if matches > 0:
                category_scores[category] += matches * weight

    # Normalize scores to 0-10 range
    max_score = max(category_scores.values()) if category_scores else 1.0
    for category in category_scores:
        category_scores[category] = min(10.0, (category_scores[category] / max_score) * 10)

    return category_scores
```

### 2.4 Performance Requirement Analysis

**Goal**: Extract performance requirements and infer optimization needs

**Algorithm**:
```python
def analyze_performance_requirements(perf_reqs: dict) -> dict[str, float]:
    """
    Analyze performance requirements to determine optimization needs.

    Returns:
        dict mapping optimization_type → priority (0-10)
    """
    optimizations = {
        'caching': 0.0,
        'connection_pooling': 0.0,
        'retry_mechanism': 0.0,
        'circuit_breaker': 0.0,
        'batch_processing': 0.0,
    }

    # Check latency requirements
    if 'latency_ms' in perf_reqs:
        latency = perf_reqs['latency_ms']
        if latency < 100:  # Very low latency
            optimizations['caching'] = 8.0
            optimizations['connection_pooling'] = 10.0
        elif latency < 500:  # Low latency
            optimizations['caching'] = 6.0
            optimizations['connection_pooling'] = 7.0

    # Check throughput requirements
    if 'throughput_rps' in perf_reqs:
        rps = perf_reqs['throughput_rps']
        if rps > 1000:  # High throughput
            optimizations['connection_pooling'] = 10.0
            optimizations['batch_processing'] = 8.0
        elif rps > 100:  # Medium throughput
            optimizations['connection_pooling'] = 7.0

    # Check reliability requirements
    if 'availability' in perf_reqs:
        availability = perf_reqs['availability']
        if availability > 0.999:  # 99.9%+ availability
            optimizations['retry_mechanism'] = 9.0
            optimizations['circuit_breaker'] = 10.0
        elif availability > 0.99:  # 99%+ availability
            optimizations['retry_mechanism'] = 7.0
            optimizations['circuit_breaker'] = 8.0

    return optimizations
```

---

## 3. Requirement Categorization Algorithm

### 3.1 Category Score Calculation

**Goal**: Assign scores (0-10) to each requirement category based on extracted features

**Algorithm**:
```python
def categorize_requirements(
    keywords: set[str],
    capabilities: set[str],
    operation_patterns: dict[str, float],
    performance_optimizations: dict[str, float],
) -> ModelRequirementAnalysis:
    """
    Categorize requirements and calculate category scores.

    Scoring formula for each category:
        category_score = (
            keyword_match_score * 2.0 +
            capability_match_score * 3.0 +
            operation_pattern_score * 2.0 +
            performance_optimization_score * 1.5
        ) / (2.0 + 3.0 + 2.0 + 1.5)  # Normalize to 0-10

    Returns:
        ModelRequirementAnalysis with category scores
    """

    scores = {}

    # Database category
    db_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['database'])
    db_capabilities = 3.0 if 'database' in capabilities else 0.0
    db_operations = operation_patterns.get('database', 0.0)
    db_performance = performance_optimizations.get('connection_pooling', 0.0)

    scores['database'] = (
        db_keywords * 2.0 +
        db_capabilities * 3.0 +
        db_operations * 2.0 +
        db_performance * 1.5
    ) / 8.5

    # API category
    api_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['api'])
    api_capabilities = 3.0 if 'api' in capabilities else 0.0
    api_operations = operation_patterns.get('api', 0.0)
    api_performance = (
        performance_optimizations.get('retry_mechanism', 0.0) +
        performance_optimizations.get('circuit_breaker', 0.0)
    ) / 2.0

    scores['api'] = (
        api_keywords * 2.0 +
        api_capabilities * 3.0 +
        api_operations * 2.0 +
        api_performance * 1.5
    ) / 8.5

    # Kafka category
    kafka_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['kafka'])
    kafka_capabilities = 3.0 if 'kafka' in capabilities else 0.0
    kafka_operations = operation_patterns.get('kafka', 0.0)

    scores['kafka'] = (
        kafka_keywords * 2.0 +
        kafka_capabilities * 3.0 +
        kafka_operations * 2.0
    ) / 7.0

    # Security category
    sec_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['security'])
    sec_operations = operation_patterns.get('security', 0.0)

    scores['security'] = (
        sec_keywords * 3.0 +  # Security keywords are very important
        sec_operations * 2.0
    ) / 5.0

    # Observability category
    obs_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['observability'])
    obs_capabilities = 2.0 if 'metrics' in capabilities else 0.0

    scores['observability'] = (
        obs_keywords * 2.0 +
        obs_capabilities * 2.0
    ) / 4.0

    # Resilience category
    res_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['resilience'])
    res_performance = (
        performance_optimizations.get('retry_mechanism', 0.0) +
        performance_optimizations.get('circuit_breaker', 0.0)
    ) / 2.0

    scores['resilience'] = (
        res_keywords * 2.0 +
        res_performance * 3.0
    ) / 5.0

    # Caching category
    cache_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['caching'])
    cache_capabilities = 3.0 if 'caching' in capabilities else 0.0
    cache_performance = performance_optimizations.get('caching', 0.0)

    scores['caching'] = (
        cache_keywords * 2.0 +
        cache_capabilities * 3.0 +
        cache_performance * 2.0
    ) / 7.0

    # Performance category
    perf_keywords = count_keyword_matches(keywords, DOMAIN_KEYWORDS['performance'])
    perf_operations = operation_patterns.get('performance', 0.0)

    scores['performance'] = (
        perf_keywords * 2.0 +
        perf_operations * 2.0
    ) / 4.0

    # Calculate overall confidence
    confidence = calculate_confidence(scores, keywords, capabilities)

    return ModelRequirementAnalysis(
        keywords=keywords,
        dependency_packages=capabilities,
        operation_types=set(operation_patterns.keys()),
        database_score=min(10.0, scores['database']),
        api_score=min(10.0, scores['api']),
        kafka_score=min(10.0, scores['kafka']),
        security_score=min(10.0, scores['security']),
        observability_score=min(10.0, scores['observability']),
        resilience_score=min(10.0, scores['resilience']),
        caching_score=min(10.0, scores['caching']),
        performance_score=min(10.0, scores['performance']),
        confidence=confidence,
        rationale=generate_rationale(scores, keywords, capabilities),
    )
```

### 3.2 Keyword Match Scoring

**Algorithm**:
```python
def count_keyword_matches(extracted_keywords: set[str], category_keywords: list[str]) -> float:
    """
    Count keyword matches and return score (0-10).

    Scoring:
    - Each match: +2 points
    - Max score: 10 points
    """
    matches = len(extracted_keywords.intersection(set(category_keywords)))
    return min(10.0, matches * 2.0)
```

### 3.3 Confidence Calculation

**Goal**: Calculate overall confidence in requirement extraction

**Algorithm**:
```python
def calculate_confidence(
    scores: dict[str, float],
    keywords: set[str],
    capabilities: set[str],
) -> float:
    """
    Calculate confidence score (0-1) based on signal strength.

    Factors:
    - Number of non-zero scores (more signals = higher confidence)
    - Keyword count (more keywords = higher confidence)
    - Capability count (explicit dependencies = higher confidence)
    - Score variance (clear winner = higher confidence)

    Returns:
        Confidence score (0-1)
    """
    # Factor 1: Non-zero scores (0-0.3)
    non_zero_scores = sum(1 for s in scores.values() if s > 0.5)
    score_factor = min(0.3, non_zero_scores * 0.1)

    # Factor 2: Keyword count (0-0.3)
    keyword_factor = min(0.3, len(keywords) * 0.03)

    # Factor 3: Capability count (0-0.2)
    capability_factor = min(0.2, len(capabilities) * 0.05)

    # Factor 4: Score clarity (0-0.2)
    # Higher variance = clearer signal
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        clarity = (top_score - second_score) / 10.0 if top_score > 0 else 0.0
        clarity_factor = min(0.2, clarity * 0.2)
    else:
        clarity_factor = 0.1

    return score_factor + keyword_factor + capability_factor + clarity_factor
```

---

## 4. Mixin Scoring Algorithm

### 4.1 Mixin → Requirement Mapping

**Goal**: Map each mixin to requirement categories it addresses

**Mixin Mapping Table**:
```yaml
# Database Mixins
MixinConnectionPooling:
  primary_category: database
  required_scores:
    database: 5.0  # Minimum database score to recommend
  boost_factors:
    performance: 0.2  # Boost if performance is also high

MixinTransactionManagement:
  primary_category: database
  required_scores:
    database: 6.0
  keywords_match: [transaction, commit, rollback, isolation]

MixinDatabaseAdapter:
  primary_category: database
  required_scores:
    database: 4.0
  dependencies_match: [asyncpg, psycopg2, sqlalchemy]

# API Mixins
MixinCircuitBreaker:
  primary_category: resilience
  required_scores:
    api: 4.0
    resilience: 5.0
  or_logic: true  # Recommend if EITHER api OR resilience is high

MixinRetry:
  primary_category: resilience
  required_scores:
    api: 3.0
    resilience: 4.0
  or_logic: true

MixinAPIClient:
  primary_category: api
  required_scores:
    api: 5.0
  dependencies_match: [httpx, aiohttp, requests]

# Kafka Mixins
MixinEventDrivenNode:
  primary_category: kafka
  required_scores:
    kafka: 5.0
  keywords_match: [event, message, consume, produce]

MixinEventPublisher:
  primary_category: kafka
  required_scores:
    kafka: 4.0
  operation_match: [publish, emit, send]

MixinEventConsumer:
  primary_category: kafka
  required_scores:
    kafka: 4.0
  operation_match: [consume, handle, process]

# Security Mixins
MixinSecurityValidation:
  primary_category: security
  required_scores:
    security: 5.0
  keywords_match: [auth, token, validate, secure]

MixinSensitiveFieldRedaction:
  primary_category: security
  required_scores:
    security: 4.0
  keywords_match: [sensitive, pii, redact, mask]

# Observability Mixins (Always recommended at low score)
MixinHealthCheck:
  primary_category: observability
  required_scores:
    observability: 0.0  # Always include for production
  default_score: 0.7  # Default recommendation score

MixinMetrics:
  primary_category: observability
  required_scores:
    observability: 0.0  # Always include for production
  default_score: 0.8  # Higher priority than health checks

MixinStructuredLogging:
  primary_category: observability
  required_scores:
    observability: 2.0
  keywords_match: [log, logging, structured]

# Caching Mixins
MixinCaching:
  primary_category: caching
  required_scores:
    caching: 4.0
    performance: 3.0
  and_logic: true  # Recommend if BOTH caching AND performance are high

MixinCacheInvalidation:
  primary_category: caching
  required_scores:
    caching: 5.0
  keywords_match: [cache, invalidate, evict, ttl]

# Performance Mixins
MixinBatchProcessing:
  primary_category: performance
  required_scores:
    performance: 5.0
  keywords_match: [batch, bulk, parallel]

# Node Type Wrappers (Special handling)
ModelServiceEffect:
  primary_category: service
  required_scores:
    observability: 0.0  # Always available as convenience wrapper
  is_wrapper: true
  includes_mixins: [MixinNodeService, MixinHealthCheck, MixinEventBus, MixinMetrics]
```

### 4.2 Score Calculation Algorithm

**Goal**: Calculate score (0-1) for each mixin based on requirement analysis

**Algorithm**:
```python
def calculate_mixin_score(
    mixin_config: dict,
    requirement_analysis: ModelRequirementAnalysis,
) -> float:
    """
    Calculate score for single mixin.

    Scoring steps:
    1. Check required_scores thresholds
    2. Apply keyword matching bonus
    3. Apply dependency matching bonus
    4. Apply operation matching bonus
    5. Apply boost factors
    6. Normalize to 0-1

    Returns:
        Score (0-1) where >0.5 means "recommended"
    """
    score = 0.0

    # Step 1: Check required_scores
    primary_category = mixin_config['primary_category']
    required_scores = mixin_config.get('required_scores', {})

    # Get category score from analysis
    category_scores = {
        'database': requirement_analysis.database_score,
        'api': requirement_analysis.api_score,
        'kafka': requirement_analysis.kafka_score,
        'security': requirement_analysis.security_score,
        'observability': requirement_analysis.observability_score,
        'resilience': requirement_analysis.resilience_score,
        'caching': requirement_analysis.caching_score,
        'performance': requirement_analysis.performance_score,
    }

    # Check logic (AND vs OR)
    if mixin_config.get('or_logic'):
        # Any requirement met = pass
        meets_requirements = any(
            category_scores[cat] >= threshold
            for cat, threshold in required_scores.items()
        )
    elif mixin_config.get('and_logic'):
        # All requirements met = pass
        meets_requirements = all(
            category_scores[cat] >= threshold
            for cat, threshold in required_scores.items()
        )
    else:
        # Default: primary category must meet threshold
        primary_threshold = required_scores.get(primary_category, 0.0)
        meets_requirements = category_scores[primary_category] >= primary_threshold

    if not meets_requirements:
        return mixin_config.get('default_score', 0.0)

    # Base score from primary category (0-0.5)
    score += (category_scores[primary_category] / 10.0) * 0.5

    # Step 2: Keyword matching bonus (0-0.15)
    if 'keywords_match' in mixin_config:
        keyword_matches = len(
            requirement_analysis.keywords.intersection(
                set(mixin_config['keywords_match'])
            )
        )
        score += min(0.15, keyword_matches * 0.05)

    # Step 3: Dependency matching bonus (0-0.15)
    if 'dependencies_match' in mixin_config:
        dep_matches = len(
            requirement_analysis.dependency_packages.intersection(
                set(mixin_config['dependencies_match'])
            )
        )
        score += min(0.15, dep_matches * 0.075)

    # Step 4: Operation matching bonus (0-0.1)
    if 'operation_match' in mixin_config:
        op_matches = sum(
            1 for op_word in mixin_config['operation_match']
            if op_word in requirement_analysis.keywords
        )
        score += min(0.1, op_matches * 0.05)

    # Step 5: Apply boost factors (0-0.1)
    if 'boost_factors' in mixin_config:
        for boost_category, boost_weight in mixin_config['boost_factors'].items():
            if category_scores[boost_category] > 5.0:
                score += boost_weight * (category_scores[boost_category] / 10.0)

    return min(1.0, score)
```

### 4.3 Batch Scoring

**Goal**: Score all mixins efficiently

**Algorithm**:
```python
def score_all_mixins(
    requirement_analysis: ModelRequirementAnalysis,
    mixin_configs: dict[str, dict],
) -> dict[str, float]:
    """
    Score all mixins in single pass.

    Returns:
        dict mapping mixin_name → score (0-1)
    """
    scores = {}

    for mixin_name, mixin_config in mixin_configs.items():
        scores[mixin_name] = calculate_mixin_score(mixin_config, requirement_analysis)

    return scores
```

---

## 5. Conflict Detection Algorithm

### 5.1 Conflict Types

**Mutual Exclusion**: Two mixins that provide the same capability
```yaml
conflicts:
  - type: mutual_exclusion
    mixin_a: MixinCustomMetrics
    mixin_b: MixinMetrics
    reason: "Both provide metrics collection"
    resolution: prefer_higher_score
```

**Prerequisite Chains**: Mixin B requires Mixin A
```yaml
prerequisites:
  - mixin: MixinTransactionManagement
    requires: [MixinConnectionPooling]
    reason: "Transaction management requires connection pooling"
```

**Redundancies**: Mixin A already includes Mixin B
```yaml
redundancies:
  - mixin: ModelServiceEffect
    includes: [MixinHealthCheck, MixinEventBus, MixinMetrics]
    reason: "Service wrapper already includes these mixins"
```

### 5.2 Detection Algorithm

```python
def detect_conflicts(
    recommended_mixins: list[str],
    conflict_rules: dict,
) -> list[ModelMixinConflict]:
    """
    Detect all conflicts in recommended mixin list.

    Returns:
        List of conflicts with resolution strategies
    """
    conflicts = []

    # Check mutual exclusions
    for rule in conflict_rules.get('conflicts', []):
        if rule['mixin_a'] in recommended_mixins and rule['mixin_b'] in recommended_mixins:
            conflicts.append(ModelMixinConflict(
                type='mutual_exclusion',
                mixin_a=rule['mixin_a'],
                mixin_b=rule['mixin_b'],
                reason=rule['reason'],
                resolution=rule['resolution'],
            ))

    # Check missing prerequisites
    for rule in conflict_rules.get('prerequisites', []):
        if rule['mixin'] in recommended_mixins:
            for required in rule['requires']:
                if required not in recommended_mixins:
                    conflicts.append(ModelMixinConflict(
                        type='missing_prerequisite',
                        mixin_a=rule['mixin'],
                        mixin_b=required,
                        reason=rule['reason'],
                        resolution='add_prerequisite',
                    ))

    # Check redundancies
    for rule in conflict_rules.get('redundancies', []):
        if rule['mixin'] in recommended_mixins:
            for included in rule['includes']:
                if included in recommended_mixins:
                    conflicts.append(ModelMixinConflict(
                        type='redundancy',
                        mixin_a=rule['mixin'],
                        mixin_b=included,
                        reason=rule['reason'],
                        resolution='remove_redundant',
                    ))

    return conflicts
```

---

## 6. Recommendation Generation Algorithm

### 6.1 Top-K Selection

**Goal**: Select top-K mixins with highest scores and no conflicts

**Algorithm**:
```python
def generate_recommendations(
    mixin_scores: dict[str, float],
    requirement_analysis: ModelRequirementAnalysis,
    conflict_rules: dict,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[ModelMixinRecommendation]:
    """
    Generate top-K mixin recommendations.

    Process:
    1. Filter mixins by min_score
    2. Sort by score (descending)
    3. Detect and resolve conflicts
    4. Generate explanations
    5. Return top-K
    """
    # Step 1: Filter by min_score
    candidates = [
        (name, score)
        for name, score in mixin_scores.items()
        if score >= min_score
    ]

    # Step 2: Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Resolve conflicts iteratively
    selected_mixins = []
    for mixin_name, score in candidates:
        # Check if adding this mixin creates conflicts
        conflicts = detect_conflicts(
            selected_mixins + [mixin_name],
            conflict_rules
        )

        # If no unresolvable conflicts, add it
        if not has_unresolvable_conflicts(conflicts):
            selected_mixins.append(mixin_name)

        if len(selected_mixins) >= top_k:
            break

    # Step 4: Generate explanations
    recommendations = []
    for mixin_name in selected_mixins:
        explanation = generate_explanation(
            mixin_name,
            mixin_scores[mixin_name],
            requirement_analysis,
        )

        recommendations.append(ModelMixinRecommendation(
            mixin_name=mixin_name,
            score=mixin_scores[mixin_name],
            category=get_mixin_category(mixin_name),
            explanation=explanation,
            matched_requirements=get_matched_requirements(mixin_name, requirement_analysis),
            prerequisites=get_prerequisites(mixin_name, conflict_rules),
            conflicts_with=get_conflicts(mixin_name, conflict_rules),
        ))

    return recommendations
```

### 6.2 Explanation Generation

**Goal**: Generate human-readable explanation for each recommendation

**Algorithm**:
```python
def generate_explanation(
    mixin_name: str,
    score: float,
    requirement_analysis: ModelRequirementAnalysis,
) -> str:
    """
    Generate explanation for why mixin is recommended.

    Template:
        "Recommended because: {reasons}. Confidence: {score}."
    """
    reasons = []

    # Check category scores
    category = get_mixin_category(mixin_name)
    category_score = getattr(requirement_analysis, f"{category}_score")

    if category_score > 7.0:
        reasons.append(f"high {category} requirements (score: {category_score:.1f}/10)")
    elif category_score > 4.0:
        reasons.append(f"moderate {category} requirements (score: {category_score:.1f}/10)")

    # Check keyword matches
    mixin_config = get_mixin_config(mixin_name)
    if 'keywords_match' in mixin_config:
        keyword_matches = requirement_analysis.keywords.intersection(
            set(mixin_config['keywords_match'])
        )
        if keyword_matches:
            reasons.append(f"keywords: {', '.join(keyword_matches)}")

    # Check dependency matches
    if 'dependencies_match' in mixin_config:
        dep_matches = requirement_analysis.dependency_packages.intersection(
            set(mixin_config['dependencies_match'])
        )
        if dep_matches:
            reasons.append(f"dependencies: {', '.join(dep_matches)}")

    # Combine reasons
    if reasons:
        reason_text = "; ".join(reasons)
    else:
        reason_text = "general best practice for production nodes"

    return f"Recommended because: {reason_text}. Confidence: {score:.2f}."
```

---

## 7. Usage Statistics & Learning

### 7.1 Usage Tracking

**Goal**: Track which mixin recommendations were accepted/used

**Schema**:
```python
class ModelMixinUsageStats(BaseModel):
    """Track mixin usage statistics."""

    mixin_name: str
    recommended_count: int = 0     # Times recommended
    accepted_count: int = 0         # Times accepted by user
    success_count: int = 0          # Times resulted in successful generation
    failure_count: int = 0          # Times resulted in failure

    # Co-occurrence tracking
    often_used_with: dict[str, int]  # Other mixins frequently used together

    # Performance tracking
    avg_generation_time_ms: float = 0.0
    avg_code_quality_score: float = 0.0

    # Metadata
    last_updated: datetime
```

### 7.2 Adaptive Scoring

**Goal**: Adjust mixin scores based on historical success rates

**Algorithm**:
```python
def adjust_score_with_usage_stats(
    base_score: float,
    mixin_name: str,
    usage_stats: ModelMixinUsageStats,
) -> float:
    """
    Adjust base score using historical usage statistics.

    Adjustment factors:
    - Success rate: +0.1 if >90%, -0.1 if <50%
    - Acceptance rate: +0.05 if >80%, -0.05 if <30%
    - Code quality: +0.05 if >4.0/5.0, -0.05 if <3.0/5.0
    """
    adjusted_score = base_score

    # Success rate adjustment
    if usage_stats.recommended_count > 10:  # Require minimum sample size
        success_rate = usage_stats.success_count / usage_stats.recommended_count
        if success_rate > 0.9:
            adjusted_score += 0.1
        elif success_rate < 0.5:
            adjusted_score -= 0.1

    # Acceptance rate adjustment
    if usage_stats.recommended_count > 10:
        acceptance_rate = usage_stats.accepted_count / usage_stats.recommended_count
        if acceptance_rate > 0.8:
            adjusted_score += 0.05
        elif acceptance_rate < 0.3:
            adjusted_score -= 0.05

    # Code quality adjustment
    if usage_stats.avg_code_quality_score > 0:
        if usage_stats.avg_code_quality_score > 4.0:
            adjusted_score += 0.05
        elif usage_stats.avg_code_quality_score < 3.0:
            adjusted_score -= 0.05

    return max(0.0, min(1.0, adjusted_score))
```

---

## 8. Implementation Examples

### 8.1 Example 1: Database CRUD Node

**Input Requirements**:
```python
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="postgres_crud_adapter",
    domain="database",
    operations=["create_record", "read_record", "update_record", "delete_record"],
    features=["connection_pooling", "transaction_management", "error_handling"],
    dependencies={"asyncpg": ">=0.28.0"},
    performance_requirements={"latency_ms": 50, "throughput_rps": 500},
    business_description="PostgreSQL CRUD adapter with connection pooling and transaction support",
)
```

**Step-by-Step Analysis**:

1. **Feature Extraction**:
   ```python
   keywords = {'database', 'postgres', 'postgresql', 'crud', 'create', 'read',
               'update', 'delete', 'connection', 'pool', 'pooling', 'transaction',
               'error', 'handling'}
   capabilities = {'database', 'postgres'}
   operation_patterns = {'database': 8.0, 'performance': 2.0}
   performance_optimizations = {'connection_pooling': 7.0, 'caching': 6.0}
   ```

2. **Requirement Categorization**:
   ```python
   category_scores = {
       'database': 9.2,      # Very high (keywords + deps + operations)
       'api': 0.5,           # Very low
       'kafka': 0.0,         # None
       'security': 1.0,      # Minimal (error handling)
       'observability': 2.0, # Low (no explicit requirements)
       'resilience': 1.5,    # Low (error handling)
       'caching': 3.0,       # Low (inferred from performance)
       'performance': 5.0,   # Medium (latency requirements)
   }
   confidence = 0.82  # High confidence
   ```

3. **Mixin Scoring**:
   ```python
   mixin_scores = {
       'MixinConnectionPooling': 0.92,    # VERY HIGH (direct match)
       'MixinTransactionManagement': 0.85, # HIGH (explicit feature)
       'MixinDatabaseAdapter': 0.78,      # HIGH (dependencies + keywords)
       'MixinHealthCheck': 0.70,          # MEDIUM (default for production)
       'MixinMetrics': 0.80,              # HIGH (default for production)
       'MixinRetry': 0.35,                # LOW (no resilience requirements)
       'MixinCircuitBreaker': 0.30,       # LOW (no resilience requirements)
       'MixinCaching': 0.45,              # LOW (below threshold)
       # ... other mixins with low scores
   }
   ```

4. **Conflict Detection**: None (no conflicting mixins)

5. **Top-5 Recommendations**:
   ```python
   recommendations = [
       ModelMixinRecommendation(
           mixin_name='MixinConnectionPooling',
           score=0.92,
           category='database',
           explanation='Recommended because: high database requirements (score: 9.2/10); keywords: connection, pool, pooling; dependencies: asyncpg; performance requirements (latency: 50ms). Confidence: 0.92.',
           matched_requirements=['connection_pooling', 'database_operations', 'performance'],
           prerequisites=[],
           conflicts_with=[],
       ),
       ModelMixinRecommendation(
           mixin_name='MixinTransactionManagement',
           score=0.85,
           category='database',
           explanation='Recommended because: high database requirements (score: 9.2/10); keywords: transaction; explicit feature request. Confidence: 0.85.',
           matched_requirements=['transaction_management', 'database_operations'],
           prerequisites=['MixinConnectionPooling'],
           conflicts_with=[],
       ),
       ModelMixinRecommendation(
           mixin_name='MixinMetrics',
           score=0.80,
           category='observability',
           explanation='Recommended because: general best practice for production nodes. Confidence: 0.80.',
           matched_requirements=['observability'],
           prerequisites=[],
           conflicts_with=[],
       ),
       ModelMixinRecommendation(
           mixin_name='MixinDatabaseAdapter',
           score=0.78,
           category='database',
           explanation='Recommended because: high database requirements (score: 9.2/10); dependencies: asyncpg; keywords: database, adapter. Confidence: 0.78.',
           matched_requirements=['database_operations'],
           prerequisites=[],
           conflicts_with=[],
       ),
       ModelMixinRecommendation(
           mixin_name='MixinHealthCheck',
           score=0.70,
           category='observability',
           explanation='Recommended because: general best practice for production nodes. Confidence: 0.70.',
           matched_requirements=['observability'],
           prerequisites=[],
           conflicts_with=[],
       ),
   ]
   ```

### 8.2 Example 2: API Client with Retry

**Input Requirements**:
```python
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="external_api_client",
    domain="api",
    operations=["fetch_data", "post_data", "handle_errors"],
    features=["retry_logic", "circuit_breaker", "timeout_handling"],
    dependencies={"httpx": ">=0.24.0"},
    performance_requirements={"availability": 0.999, "timeout_ms": 5000},
    business_description="HTTP client for external API with fault tolerance and retry logic",
)
```

**Analysis Result**:
```python
category_scores = {
    'database': 0.0,
    'api': 8.5,           # HIGH
    'kafka': 0.0,
    'security': 1.0,
    'observability': 2.0,
    'resilience': 9.0,    # VERY HIGH (retry, circuit breaker, timeout)
    'caching': 0.5,
    'performance': 3.0,
}

mixin_scores = {
    'MixinRetry': 0.88,           # VERY HIGH
    'MixinCircuitBreaker': 0.85,  # HIGH
    'MixinAPIClient': 0.82,       # HIGH
    'MixinMetrics': 0.80,         # HIGH
    'MixinHealthCheck': 0.70,     # MEDIUM
}

recommendations = [
    # Top 5 mixins with explanations...
]
```

---

## 9. Performance Benchmarks

### 9.1 Timing Targets

| Operation | Target | Measured | Status |
|-----------|--------|----------|--------|
| Feature extraction | <20ms | TBD | - |
| Requirement categorization | <30ms | TBD | - |
| Mixin scoring (21 mixins) | <100ms | TBD | - |
| Conflict detection | <20ms | TBD | - |
| Recommendation generation | <20ms | TBD | - |
| **Total pipeline** | **<200ms** | **TBD** | **-** |

### 9.2 Accuracy Targets

| Metric | Target | Validation Method |
|--------|--------|------------------|
| Recommendation relevance | >90% | Manual review of 100+ cases |
| Conflict detection rate | 100% | Automated tests with known conflicts |
| False positive rate | <5% | Manual review |
| False negative rate | <10% | Comparison with expert recommendations |
| Confidence calibration | ±5% | Score vs actual success rate |

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Test Coverage**: >90% for all components

```python
# Test feature extraction
def test_keyword_extraction():
    requirements = ModelPRDRequirements(...)
    keywords = extract_keywords(requirements)
    assert 'database' in keywords
    assert 'connection' in keywords

# Test categorization
def test_database_category_scoring():
    requirements = ModelPRDRequirements(
        domain='database',
        dependencies={'asyncpg': '>=0.28.0'},
        operations=['create_record', 'read_record'],
    )
    analysis = categorize_requirements(...)
    assert analysis.database_score > 7.0

# Test mixin scoring
def test_connection_pooling_scoring():
    analysis = ModelRequirementAnalysis(database_score=9.0)
    score = calculate_mixin_score('MixinConnectionPooling', analysis)
    assert score > 0.8

# Test conflict detection
def test_mutual_exclusion_detection():
    mixins = ['MixinMetrics', 'MixinCustomMetrics']
    conflicts = detect_conflicts(mixins, conflict_rules)
    assert len(conflicts) == 1
    assert conflicts[0].type == 'mutual_exclusion'
```

### 10.2 Integration Tests

```python
def test_end_to_end_database_node():
    """Test complete pipeline for database node."""
    requirements = create_database_requirements()

    # Run full pipeline
    analysis = analyze_requirements(requirements)
    mixin_scores = score_all_mixins(analysis)
    recommendations = generate_recommendations(mixin_scores)

    # Verify recommendations
    assert 'MixinConnectionPooling' in [r.mixin_name for r in recommendations]
    assert all(r.score > 0.5 for r in recommendations)
    assert len(recommendations) <= 5

def test_end_to_end_api_client():
    """Test complete pipeline for API client."""
    requirements = create_api_client_requirements()

    analysis = analyze_requirements(requirements)
    mixin_scores = score_all_mixins(analysis)
    recommendations = generate_recommendations(mixin_scores)

    assert 'MixinRetry' in [r.mixin_name for r in recommendations]
    assert 'MixinCircuitBreaker' in [r.mixin_name for r in recommendations]
```

### 10.3 Validation Tests

**Validation against 50+ real node requirements**:
```python
def test_validation_against_production_nodes():
    """Validate recommendations against 50+ production node requirements."""

    test_cases = load_production_node_requirements()
    results = []

    for node_name, requirements, expected_mixins in test_cases:
        recommendations = generate_recommendations_pipeline(requirements)
        recommended_mixins = [r.mixin_name for r in recommendations]

        # Calculate metrics
        true_positives = len(set(recommended_mixins).intersection(expected_mixins))
        false_positives = len(set(recommended_mixins) - expected_mixins)
        false_negatives = len(expected_mixins - set(recommended_mixins))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        results.append({
            'node': node_name,
            'precision': precision,
            'recall': recall,
        })

    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)

    assert avg_precision > 0.90, f"Precision too low: {avg_precision}"
    assert avg_recall > 0.85, f"Recall too low: {avg_recall}"
```

---

## 11. Configuration Files

### 11.1 Scoring Configuration (scoring_config.yaml)

```yaml
# Mixin definitions with requirement mappings
version: "1.0"
updated_at: "2025-11-06"

# Global settings
global:
  min_recommendation_score: 0.5
  default_top_k: 5
  confidence_threshold: 0.6

# Category weights (applied to base scores)
category_weights:
  database: 1.0
  api: 1.0
  kafka: 1.0
  security: 0.95
  observability: 1.0
  resilience: 0.9
  caching: 0.85
  performance: 0.9

# Mixin configurations
mixins:
  # Database mixins
  MixinConnectionPooling:
    primary_category: database
    required_scores:
      database: 5.0
    boost_factors:
      performance: 0.2
    dependencies_match: [asyncpg, psycopg2, sqlalchemy]
    weight: 1.0

  MixinTransactionManagement:
    primary_category: database
    required_scores:
      database: 6.0
    keywords_match: [transaction, commit, rollback, isolation]
    prerequisites: [MixinConnectionPooling]
    weight: 0.95

  # ... (all 21 mixins defined)

# Conflict rules
conflicts:
  - mixin_a: MixinMetrics
    mixin_b: MixinCustomMetrics
    type: mutual_exclusion
    reason: "Both provide metrics collection"
    resolution: prefer_higher_score

  - mixin_a: MixinEventDrivenNode
    mixin_b: MixinSimpleNode
    type: mutual_exclusion
    reason: "Different node paradigms"
    resolution: prefer_event_driven

# Prerequisites
prerequisites:
  - mixin: MixinTransactionManagement
    requires: [MixinConnectionPooling]
    reason: "Transaction management requires connection pooling"

  - mixin: MixinCacheInvalidation
    requires: [MixinCaching]
    reason: "Cache invalidation requires caching to be enabled"
```

---

## 12. Success Criteria

### 12.1 Functional Requirements

- [ ] Feature extraction completes in <20ms
- [ ] Requirement categorization accuracy >90%
- [ ] Mixin scoring covers all 21 mixins
- [ ] Conflict detection catches all known conflicts
- [ ] Recommendations include explanations
- [ ] Usage statistics tracking implemented

### 12.2 Performance Requirements

- [ ] Total pipeline <200ms
- [ ] Scoring time <100ms per analysis
- [ ] Memory usage <50MB
- [ ] Handles 100+ concurrent requests

### 12.3 Quality Requirements

- [ ] Unit test coverage >90%
- [ ] Integration tests passing
- [ ] Validation against 50+ production nodes >90% accuracy
- [ ] Confidence calibration ±5%
- [ ] Documentation complete

---

## 13. References

### 13.1 Related Documents

- **Phase 3 Architecture Design**: Overall Phase 3 architecture
- **Phase 3 Task Breakdown**: Task dependencies and sequencing
- **Mixin Selector Quick Reference**: Phase 1 mixin selection implementation
- **OMNIBASE_CORE_MIXIN_CATALOG.md**: Complete mixin catalog

### 13.2 Code References

- `src/omninode_bridge/codegen/prd_analyzer.py`: PRD analysis (provides ModelPRDRequirements)
- `src/omninode_bridge/codegen/mixin_selector.py`: Phase 1 mixin selector
- `src/omninode_bridge/codegen/models/`: Pydantic models

---

## 14. Appendix

### 14.1 Complete Keyword Dictionary

See Section 2.1 for domain keyword definitions.

### 14.2 Algorithm Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Feature extraction | O(n) | O(n) | n = total text length |
| Categorization | O(k) | O(1) | k = number of keywords |
| Mixin scoring | O(m) | O(m) | m = number of mixins (21) |
| Conflict detection | O(m²) | O(c) | c = number of conflicts |
| Recommendation gen | O(m log m) | O(m) | Sorting mixins by score |
| **Total pipeline** | **O(n + m log m)** | **O(n + m)** | **Linear in input size** |

For typical inputs (n≈1000 chars, m=21 mixins), total time is <200ms.

### 14.3 Future Enhancements

**Phase 4+ Improvements**:
1. **Machine learning**: Train model on historical recommendations
2. **Contextual embeddings**: Use vector similarity for keyword matching
3. **Collaborative filtering**: "Users who used X also used Y"
4. **A/B testing**: Experiment with different scoring weights
5. **Explainable AI**: More detailed explanations using LIME/SHAP

---

**Document Status**: ✅ Complete - Ready for Implementation (Tasks C13-C15)
**Next Steps**: Implement RequirementsAnalyzer, MixinScorer, MixinRecommender, ConflictResolver
**Estimated Implementation Time**: 9 days (C13: 3 days, C14: 2 days, C15: 2 days, Tests: 2 days)
