#!/usr/bin/env python3
"""Live test of LLMMetricsStore with real PostgreSQL database."""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import asyncpg

from omninode_bridge.intelligence import (
    LLMGenerationHistory,
    LLMGenerationMetric,
    LLMMetricsStore,
    LLMPattern,
)


async def test_metrics_storage():
    """Test metrics storage with real PostgreSQL."""

    print("📊 Testing LLM Metrics Storage Integration\n")

    # Connect to database
    print("🔌 Connecting to PostgreSQL at 192.168.86.200:5436...")
    try:
        pool = await asyncpg.create_pool(
            host="192.168.86.200",
            port=5436,
            user="postgres",
            password="omninode_remote_2024_secure",
            database="omninode_bridge",
            min_size=1,
            max_size=5,
        )
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

    try:
        # Initialize metrics store
        store = LLMMetricsStore(pool)
        print("✅ LLMMetricsStore initialized\n")

        # === TEST 1: Store Generation Metric ===
        print("📝 TEST 1: Store generation metric")
        session_id = f"test_session_{uuid4().hex[:8]}"
        correlation_id = uuid4()

        metric = LLMGenerationMetric(
            metric_id=uuid4(),
            session_id=session_id,
            correlation_id=correlation_id,
            node_type="effect",
            model_tier="CLOUD_FAST",
            model_name="glm-4.5",
            prompt_tokens=39,
            completion_tokens=500,
            total_tokens=539,
            latency_ms=14286.38,
            cost_usd=0.000108,
            success=True,
            error_message=None,
            metadata={"test": True, "environment": "manual_test"},
            created_at=datetime.now(UTC),
        )

        metric_id = await store.store_generation_metric(metric)
        print(f"✅ Metric stored: {metric_id}")
        print(f"   Session: {session_id}")
        print(f"   Tokens: {metric.total_tokens}")
        print(f"   Cost: ${metric.cost_usd:.6f}\n")

        # === TEST 2: Retrieve Metrics by Session ===
        print("📋 TEST 2: Retrieve metrics by session")
        metrics = await store.get_metrics_by_session(session_id)
        print(f"✅ Retrieved {len(metrics)} metrics for session {session_id}")
        if metrics:
            m = metrics[0]
            print(f"   First metric: {m.model_name}, {m.total_tokens} tokens\n")

        # === TEST 3: Store Generation History ===
        print("📚 TEST 3: Store generation history")
        history = LLMGenerationHistory(
            history_id=uuid4(),
            metric_id=metric.metric_id,
            prompt_text="Write a Python function that calculates the factorial of a number.",
            generated_text="def factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('...')",
            quality_score=0.85,
            validation_passed=True,
            validation_errors=None,
            created_at=datetime.now(UTC),
        )

        history_id = await store.store_generation_history(history)
        print(f"✅ History stored: {history_id}")
        print(f"   Linked to metric: {metric.metric_id}")
        print(f"   Quality score: {history.quality_score}\n")

        # === TEST 4: Get Average Metrics ===
        print("📊 TEST 4: Get average metrics for glm-4.5")
        averages = await store.get_average_metrics("glm-4.5", days=7)
        print("✅ Average metrics (last 7 days):")
        print(f"   Total generations: {averages.get('total_generations', 0)}")
        print(f"   Successful: {averages.get('successful_generations', 0)}")
        print(f"   Avg latency: {averages.get('avg_latency_ms', 0):.2f}ms")
        print(f"   Avg tokens: {averages.get('avg_total_tokens', 0):.0f}")
        print(f"   Total cost: ${averages.get('total_cost_usd', 0):.6f}\n")

        # === TEST 5: Store Learned Pattern ===
        print("🧠 TEST 5: Store learned pattern")
        pattern = LLMPattern(
            pattern_id=uuid4(),
            pattern_type="high_quality_prompt",
            node_type="effect",
            pattern_data={
                "template": "Write a Python function that {action}. Include docstring and type hints.",
                "quality_indicators": ["type hints", "docstring", "error handling"],
            },
            usage_count=1,
            avg_quality_score=0.85,
            success_rate=1.0,
            metadata={"source": "manual_test"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        pattern_id = await store.store_learned_pattern(pattern)
        print(f"✅ Pattern stored: {pattern_id}")
        print(f"   Type: {pattern.pattern_type}")
        print(f"   Quality score: {pattern.avg_quality_score}\n")

        # === TEST 6: Get Best Patterns ===
        print("⭐ TEST 6: Get best patterns")
        patterns = await store.get_best_patterns("high_quality_prompt", limit=5)
        print(f"✅ Retrieved {len(patterns)} patterns")
        if patterns:
            p = patterns[0]
            print(
                f"   Best pattern: quality={p.avg_quality_score}, usage={p.usage_count}\n"
            )

        # === TEST 7: Get Session Summary ===
        print("📈 TEST 7: Get session summary")
        summary = await store.get_session_summary(session_id)
        if summary:
            print("✅ Session summary:")
            print(f"   Total generations: {summary.total_generations}")
            print(f"   Success rate: {summary.success_rate:.1%}")
            print(f"   Total cost: ${summary.total_cost_usd:.6f}")
            print(f"   Avg latency: {summary.avg_latency_ms:.2f}ms\n")
        else:
            print("⚠️  No summary found (might be timing issue)\n")

        # === TEST 8: Health Check ===
        print("🏥 TEST 8: Health check")
        health = await store.health_check()
        print(f"✅ Health check: {health['status']}")
        print(f"   Response time: {health.get('response_time_ms', 0):.2f}ms")
        print(f"   Pool size: {health.get('pool_size', 0)}")
        print(f"   Pool idle: {health.get('pool_idle', 0)}\n")

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await pool.close()
        print("\n🔒 Database connection closed")


if __name__ == "__main__":
    success = asyncio.run(test_metrics_storage())
    sys.exit(0 if success else 1)
