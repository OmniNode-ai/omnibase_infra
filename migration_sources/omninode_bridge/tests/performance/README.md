# Performance Benchmarking Suite

Comprehensive performance benchmarks for NodeBridgeOrchestrator and NodeBridgeReducer using pytest-benchmark.

## Quick Start

```bash
# Run all performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Run specific benchmark category
pytest tests/performance/test_bridge_performance.py::TestReducerPerformance -v

# Generate detailed reports
pytest tests/performance/ --benchmark-only --benchmark-json=test-artifacts/benchmark.json
```

## Benchmark Categories

### 1. Reducer Performance Benchmarks

- **test_batch_aggregation_100_items**: Aggregate 100 items (Target: <100ms)
- **test_batch_aggregation_1000_items**: Aggregate 1000 items (Target: <1000ms)
- **test_batch_aggregation_10000_items**: Large batch processing (Target: <10s)
- **test_streaming_window_processing**: Windowed streaming (1000 items in batches of 100)
- **test_namespace_grouping_performance**: Multi-namespace aggregation (10 namespaces × 100 items)

### 2. Orchestrator Performance Benchmarks

- **test_single_workflow_execution**: Single workflow end-to-end (Target: <300ms)
- **test_concurrent_workflow_throughput**: 100 concurrent workflows (Target: <500ms total)
- **test_fsm_state_transition**: FSM transition overhead (Target: <50ms)
- **test_service_routing_overhead**: Service selection performance (Target: <10ms)

### 3. End-to-End Benchmarks

- **test_complete_workflow_pipeline**: Complete flow from request to persistence (Target: <500ms)
- **test_multi_service_coordination**: Multi-service coordination overhead (Target: <200ms)

### 4. Memory Profiling

- **test_memory_leak_detection**: 100 iterations with memory growth tracking

### 5. Introspection Load Tests (NEW)

**Location**: `test_introspection_load.py`

Tests for IntrospectionMixin heartbeat system performance and non-blocking behavior.

#### Single Node Performance
- **test_heartbeat_overhead_under_10ms**: Heartbeat overhead validation (Target: <10ms avg)
- **test_resource_collection_non_blocking**: Non-blocking resource collection verification
- **test_heartbeat_with_psutil_overhead**: Real psutil performance validation

#### Concurrent Load Tests
- **test_1000_concurrent_heartbeats**: 1000 nodes simultaneous broadcast (Target: <5000ms total)
- **test_sustained_heartbeat_load**: Memory leak detection under sustained load
- **test_event_loop_responsiveness_under_load**: Event loop health monitoring

#### Resource Collection
- **test_resource_info_caching_effectiveness**: Cache performance validation
- **test_concurrent_resource_collection**: 100 nodes concurrent resource access

#### Blocking Detection
- **test_no_blocking_psutil_calls_in_heartbeat**: High-frequency monitor detection
- **test_parallel_execution_proves_non_blocking**: Parallel speedup verification

**Usage**:
```bash
# Run all introspection load tests
pytest tests/performance/test_introspection_load.py -v -m performance

# Run without slow tests (skip 1000 node tests)
pytest tests/performance/test_introspection_load.py -v -m "performance and not slow"

# Run blocking detection only
pytest tests/performance/test_introspection_load.py::TestBlockingDetection -v
```

## Performance Thresholds

```python
PERFORMANCE_THRESHOLDS = {
    'orchestrator_workflow': {'max_ms': 300, 'p95_ms': 250},
    'orchestrator_concurrent': {'min_throughput': 100, 'max_latency_ms': 500},
    'reducer_batch_1000': {'max_ms': 1000, 'p95_ms': 800},
    'fsm_transition': {'max_ms': 50, 'p95_ms': 30},
    'end_to_end': {'max_ms': 500, 'p95_ms': 400},
    # Introspection load test thresholds
    'heartbeat_single': {'avg_ms': 10, 'p95_ms': 15, 'p99_ms': 25},
    'heartbeat_1000_concurrent': {'total_ms': 5000, 'success_rate': 1.0},
    'event_loop_responsiveness': {'min_iterations': 500},
    'blocking_detection': {'parallel_speedup_min': 2.0}
}
```

## Usage Examples

### Basic Benchmarking

```bash
# Run all performance tests
pytest tests/performance/ -v --benchmark-only

# Run with detailed statistics
pytest tests/performance/ -v --benchmark-only --benchmark-verbose

# Run with memory profiling
pytest tests/performance/ -v --benchmark-only -k "memory"
```

### Baseline Management

```bash
# Save current performance as baseline
pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

# Compare and fail if performance degrades >10%
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

### Report Generation

```bash
# JSON report for analysis
pytest tests/performance/ --benchmark-only --benchmark-json=test-artifacts/benchmark.json

# CSV export
pytest tests/performance/ --benchmark-only --benchmark-csv=test-artifacts/benchmark.csv

# Histogram visualization
pytest tests/performance/ --benchmark-only --benchmark-histogram=test-artifacts/benchmark-histogram
```

### CI/CD Integration

```bash
# Run in CI with strict thresholds
pytest tests/performance/ \
    --benchmark-only \
    --benchmark-json=test-artifacts/benchmark.json \
    --benchmark-compare=baseline \
    --benchmark-compare-fail=mean:10%,min:10%,max:10% \
    --maxfail=1
```

## Interpreting Results

### Statistical Measures

- **min**: Fastest execution time (best case)
- **max**: Slowest execution time (worst case)
- **mean**: Average execution time across all runs
- **stddev**: Standard deviation (lower = more consistent)
- **median**: Middle value (50th percentile)
- **iqr**: Interquartile range (25th-75th percentile spread)
- **ops**: Operations per second (throughput)

### Performance Grades

- **A**: <1ms for small operations, <100ms for batch operations
- **B**: <2ms for small operations, <200ms for batch operations
- **C**: >2ms for small operations, >200ms for batch operations

### Memory Statistics

- **baseline_mb**: Memory usage before test
- **peak_mb**: Maximum memory usage during test
- **delta_mb**: Memory increase after test
- **peak_delta_mb**: Maximum memory increase
- **samples**: Number of memory samples taken

## Configuration

Benchmark configuration is in `pytest-benchmark.ini`:

```ini
[pytest-benchmark]
autosave = true
json = true
csv = test-artifacts/benchmark.csv
rounds = 5
iterations = 10
compare-fail = mean:10%
```

## Troubleshooting

### Benchmarks are slow

- Reduce rounds/iterations: `--benchmark-min-rounds=3`
- Run specific tests: `-k "test_name"`
- Skip warmup: `--benchmark-disable-gc`

### Results are inconsistent

- Increase rounds: `--benchmark-min-rounds=10`
- Run on dedicated machine without background processes
- Use `--benchmark-calibrate` for automatic tuning

### Memory profiling not working

- Install memory_profiler: `pip install memory_profiler`
- Enable in test: Use `memory_tracker` fixture

## Best Practices

1. **Baseline Management**: Save baselines after major optimizations
2. **CI Integration**: Run benchmarks in CI to catch regressions early
3. **Isolation**: Run benchmarks on dedicated hardware for consistency
4. **Monitoring**: Track performance trends over time
5. **Documentation**: Document performance requirements and thresholds

## Performance Optimization

If benchmarks fail thresholds:

1. **Profile**: Use `--benchmark-columns=all` for detailed stats
2. **Compare**: Run `--benchmark-compare` to identify regressions
3. **Isolate**: Run specific tests with `-k` to narrow down issues
4. **Optimize**: Focus on high-impact areas (p95, p99 times)
5. **Validate**: Re-run benchmarks to confirm improvements

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance testing best practices](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Statistical analysis of benchmarks](https://www.brendangregg.com/blog/2018-06-30/benchmarking-checklist.html)
