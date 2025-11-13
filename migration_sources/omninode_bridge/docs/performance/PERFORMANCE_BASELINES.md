# Performance Baselines - OmniNode Bridge Codegen System

**Generated:** 2025-10-22T21:10:49.490048
**Environment:** development

## Overview

This document contains performance baselines for the contract-first codegen system.
Baselines were established by running comprehensive benchmarks against deployed infrastructure.

## Cache Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Write Latency (p95) | 0.06ms | <5ms | ✅ |
| Read Hit Latency (p95) | 0.08ms | <5ms | ✅ |
| Cache Hit Rate | 50.0% | >70% | ❌ |

## Rate Limiter Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Check Overhead (p95) | 0.00ms | <1ms | ✅ |
| Average Overhead | 0.00ms | <1ms | - |

## Orchestrator Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Single Workflow (p50) | 45.20ms | <50ms | ✅ |
| Single Workflow (p95) | 89.50ms | <150ms | ✅ |
| Throughput | 12.5 workflows/sec | >10/sec | ✅ |
| Memory Usage | 85.3MB | <512MB | ✅ |

## Reducer Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Aggregation Throughput | 1250.5 items/sec | >1000/sec | ✅ |
| Batch 1000 Latency | 85.30ms | <100ms | ✅ |
| Streaming Latency (p95) | 45.80ms | <100ms | ✅ |
| Memory Usage | 120.5MB | <512MB | ✅ |
