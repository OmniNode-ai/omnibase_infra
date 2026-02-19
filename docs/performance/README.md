> **Navigation**: [Home](../index.md) > Performance

# Performance Documentation

This directory contains performance baselines, SLO definitions, and calibration
methodology for ONEX infrastructure services.

---

## Documents

| Document | Description |
|----------|-------------|
| [LLM_ENDPOINT_SLO.md](LLM_ENDPOINT_SLO.md) | Service Level Objectives for distributed LLM endpoints (TTFT, throughput, availability) |

---

## Related Decisions

- [ADR-004: Performance Baseline Thresholds](../decisions/adr-004-performance-baseline-thresholds.md) -
  Calibration methodology and E2E registration flow thresholds

---

## Calibration Methodology (Summary)

All thresholds in this directory follow the formula documented in ADR-004:

```
threshold = (base_operation_time + network_overhead) * safety_margin
```

- **base_operation_time**: Measured P95/P99 for the operation under representative load
- **network_overhead**: RTT to the service host (10–25ms for local network)
- **safety_margin**: 2x multiplier for GC pauses, thermal throttling, and load spikes

Recalibration is required after infrastructure changes, hardware upgrades, or sustained
threshold violations (>5% of requests).

---

## Adding New SLO Documents

When adding a new service or endpoint:

1. Create `docs/performance/<SERVICE>_SLO.md`
2. Follow the structure in `LLM_ENDPOINT_SLO.md`:
   - Endpoint inventory table
   - SLO targets (latency, throughput, availability)
   - Health check commands
   - Calibration methodology and triggers
   - Environment adjustment guidelines
3. Reference the new document in this README
4. Cross-reference from the relevant ADR or architecture document
