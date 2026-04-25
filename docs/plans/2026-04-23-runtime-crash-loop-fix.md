---
epic_id: OMN-9551
---

# Runtime Crash Loop Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan phase-by-phase.

**Goal:** Fix the runtime crash loop caused by ServiceRuntimeHealthMonitor emitting to an unprovisioned Kafka topic, which trips the circuit breaker and shuts down the service.

**Architecture:** Two-pronged fix: (A) boot grace window suppresses emits for 120s after construction; (B) UnknownTopicOrPartitionError excluded from circuit breaker failure count.

**Tech Stack:** Python 3.12, aiokafka, Pydantic, asyncio

---

## Task 1: Add boot grace window to ServiceRuntimeHealthMonitor [OMN-9552]

Add boot_grace_seconds param to __init__, initialize _started_at = time.monotonic(), gate _emit() during grace window.

## Task 2: Exclude UnknownTopicOrPartitionError from circuit breaker [OMN-9553]

Catch UnknownTopicOrPartitionError before KafkaError in _publish_with_retry(), skip CB failure, raise ProtocolConfigurationError post-loop.

## Task 3: Wire RUNTIME_HEALTH_BOOT_GRACE_SECONDS in service_kernel [OMN-9554]

Read env var and pass to ServiceRuntimeHealthMonitor constructor.

## Task 4: Hostile reviewer gate fixes [OMN-9551]

Apply hostile reviewer findings: correct error type, IS-A ordering comment, fixture annotation, constant naming.
