"""Performance optimization utilities for OmniNode Bridge services."""

import logging
import os
import time
from typing import Any, NotRequired, Optional, TypedDict

import asyncpg

logger = logging.getLogger(__name__)


class PerformanceAnalysis(TypedDict):
    """Type definition for performance analysis results."""

    status: str
    timestamp: float
    bottlenecks: list[str]
    recommendations: list[str]
    query_analysis: dict[str, Any]
    connection_analysis: dict[str, Any]
    message: NotRequired[str]  # Optional error message
    error: NotRequired[str]  # Optional error details


class OptimizationResult(TypedDict):
    """Type definition for optimization results."""

    status: str
    timestamp: float
    recommendations: list[str]
    applied_optimizations: list[str]
    connection_pool_analysis: dict[str, Any]
    environment_optimizations: dict[str, Any]
    current_config: NotRequired[dict[str, Any]]  # Current configuration details
    error: NotRequired[str]  # Optional error details


class DatabasePerformanceOptimizer:
    """Database performance optimization and analysis utilities."""

    def __init__(self, postgres_client):
        """Initialize with a PostgreSQL client instance."""
        self.client = postgres_client

    async def analyze_performance_bottlenecks(self) -> PerformanceAnalysis:
        """Analyze database performance and identify potential bottlenecks.

        Returns:
            Dictionary with performance analysis and recommendations
        """
        if not self.client.pool:
            return {
                "status": "error",
                "timestamp": time.time(),
                "bottlenecks": [],
                "recommendations": [],
                "query_analysis": {},
                "connection_analysis": {},
                "message": "No database connection",
            }

        analysis: PerformanceAnalysis = {
            "status": "success",
            "timestamp": time.time(),
            "bottlenecks": [],
            "recommendations": [],
            "query_analysis": {},
            "connection_analysis": {},
        }

        try:
            async with self.client.pool.acquire() as conn:
                # Analyze connection usage
                conn_stats = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """
                )

                if conn_stats:
                    total_conn = conn_stats["total_connections"]
                    active_conn = conn_stats["active_connections"]
                    idle_conn = conn_stats["idle_connections"]

                    analysis["connection_analysis"] = {
                        "total_connections": total_conn,
                        "active_connections": active_conn,
                        "idle_connections": idle_conn,
                        "utilization_percent": (active_conn / max(total_conn, 1)) * 100,
                    }

                    # Connection pool recommendations
                    if idle_conn > active_conn * 2:
                        analysis["recommendations"].append(
                            f"Consider reducing connection pool size - {idle_conn} idle vs {active_conn} active"
                        )
                    elif active_conn > total_conn * 0.8:
                        analysis["recommendations"].append(
                            f"Consider increasing connection pool size - {active_conn} active out of {total_conn} total"
                        )

                # Check for long-running queries
                long_queries = await conn.fetch(
                    """
                    SELECT pid, query_start, state, query
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                    AND query_start < NOW() - INTERVAL '30 seconds'
                    AND datname = current_database()
                    LIMIT 5
                """
                )

                if long_queries:
                    analysis["bottlenecks"].append("Long-running queries detected")
                    analysis["query_analysis"]["long_running"] = [
                        {
                            "duration_seconds": (
                                time.time() - row["query_start"].timestamp()
                            ),
                            "state": row["state"],
                            "query_preview": (
                                row["query"][:100] + "..."
                                if len(row["query"]) > 100
                                else row["query"]
                            ),
                        }
                        for row in long_queries
                    ]

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            analysis["status"] = "error"
            analysis["error"] = str(e)

        return analysis

    async def optimize_connection_pool_settings(self) -> OptimizationResult:
        """Optimize connection pool configuration based on current workload patterns.

        Returns:
            Dictionary with optimization results and new recommendations
        """
        analysis = await self.analyze_performance_bottlenecks()

        optimization_result: OptimizationResult = {
            "status": "success",
            "timestamp": time.time(),
            "recommendations": [],
            "applied_optimizations": [],
            "connection_pool_analysis": {},
            "environment_optimizations": {},
            "current_config": {
                "min_size": self.client.min_size,
                "max_size": self.client.max_size,
                "max_queries_per_connection": getattr(
                    self.client, "max_queries_per_connection", "N/A"
                ),
                "connection_max_age": getattr(
                    self.client, "connection_max_age_seconds", "N/A"
                ),
                "query_timeout": getattr(self.client, "query_timeout_seconds", "N/A"),
            },
        }

        try:
            # Analyze connection usage patterns
            conn_analysis = analysis.get("connection_analysis", {})
            utilization = conn_analysis.get("utilization_percent", 0)

            # Environment-based optimization recommendations
            environment = os.getenv("ENVIRONMENT", "development").lower()

            if environment == "production":
                # Production optimizations
                if utilization < 30:
                    optimization_result["recommendations"].extend(
                        [
                            f"Production: Consider reducing max_size from {self.client.max_size} to {max(self.client.max_size // 2, 10)}",
                            f"Production: Consider reducing min_size from {self.client.min_size} to {max(self.client.min_size // 2, 5)}",
                        ]
                    )
                elif utilization > 80:
                    optimization_result["recommendations"].extend(
                        [
                            f"Production: Consider increasing max_size from {self.client.max_size} to {min(self.client.max_size * 2, 100)}",
                            f"Production: Consider increasing min_size from {self.client.min_size} to {min(self.client.min_size * 2, 20)}",
                        ]
                    )

                optimization_result["environment_optimizations"]["production"] = {
                    "recommended_min_size": max(min(self.client.min_size, 10), 5),
                    "recommended_max_size": max(min(self.client.max_size, 50), 20),
                    "recommended_connection_max_age": 3600,  # 1 hour
                    "recommended_query_timeout": 60,  # 1 minute
                }

            elif environment == "staging":
                # Staging optimizations
                optimization_result["environment_optimizations"]["staging"] = {
                    "recommended_min_size": max(self.client.min_size // 2, 2),
                    "recommended_max_size": max(self.client.max_size // 2, 10),
                    "recommended_connection_max_age": 1800,  # 30 minutes
                    "recommended_query_timeout": 45,
                }

            else:  # development
                # Development optimizations
                optimization_result["environment_optimizations"]["development"] = {
                    "recommended_min_size": 2,
                    "recommended_max_size": 10,
                    "recommended_connection_max_age": 600,  # 10 minutes
                    "recommended_query_timeout": 30,
                }

            # Query timeout optimization based on analysis
            if "long_running" in analysis.get("query_analysis", {}):
                current_timeout = getattr(self.client, "query_timeout_seconds", 30)
                optimization_result["recommendations"].append(
                    f"Consider increasing query_timeout from {current_timeout}s to "
                    f"{current_timeout + 30}s due to long-running queries"
                )

        except Exception as e:
            logger.error(f"Error optimizing connection pool: {e}")
            optimization_result["status"] = "error"
            optimization_result["error"] = str(e)

        return optimization_result

    async def get_indexing_recommendations(self) -> dict[str, Any]:
        """Generate comprehensive database indexing recommendations for improved query performance.

        Returns:
            Dictionary with detailed indexing analysis and actionable recommendations
        """
        if not self.client.pool:
            return {
                "status": "error",
                "timestamp": time.time(),
                "recommendations": [],
                "specific_indexes": [],
                "performance_impact": "unknown",
                "summary": "No database connection available",
                "analysis_details": {},
                "message": "No database connection",
            }

        recommendations: dict[str, Any] = {
            "status": "success",
            "timestamp": time.time(),
            "recommendations": [],
            "specific_indexes": [],
            "performance_impact": "medium",
            "summary": "",
            "analysis_details": {},
        }

        try:
            async with self.client.pool.acquire() as conn:
                # Comprehensive table analysis with more detailed metrics
                table_analysis = await conn.fetch(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        n_tup_ins + n_tup_upd + n_tup_del as modifications,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del,
                        n_live_tup,
                        n_dead_tup
                    FROM pg_stat_user_tables
                    WHERE seq_scan > 10
                    ORDER BY seq_scan DESC
                    LIMIT 20
                """
                )

                # Analyze slow queries to understand index requirements
                slow_queries = (
                    await conn.fetch(
                        """
                    SELECT
                        query,
                        calls,
                        total_time,
                        mean_time,
                        rows
                    FROM pg_stat_statements
                    WHERE mean_time > 100
                    AND calls > 5
                    ORDER BY mean_time DESC
                    LIMIT 10
                """
                    )
                    if await self._check_pg_stat_statements(conn)
                    else []
                )

                # Check existing indexes for redundancy
                existing_indexes = await conn.fetch(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY tablename, indexname
                """
                )

                recommendations["analysis_details"] = {
                    "tables_analyzed": len(table_analysis),
                    "slow_queries_found": len(slow_queries),
                    "existing_indexes": len(existing_indexes),
                }

                # Analyze each table for indexing opportunities
                for table in table_analysis:
                    seq_scans = table["seq_scan"] or 0
                    idx_scans = table["idx_scan"] or 0
                    live_tuples = table["n_live_tup"] or 0
                    table_name = f"{table['schemaname']}.{table['tablename']}"

                    # High sequential scan analysis
                    if seq_scans > 0 and (idx_scans == 0 or seq_scans > idx_scans * 2):
                        priority = (
                            "critical"
                            if seq_scans > 5000 and live_tuples > 10000
                            else "high" if seq_scans > 1000 else "medium"
                        )

                        recommendations["recommendations"].append(
                            {
                                "type": "missing_index_opportunity",
                                "priority": priority,
                                "table": table_name,
                                "issue": f"High sequential scan ratio: {seq_scans} sequential vs {idx_scans} index scans",
                                "suggestion": f"Add indexes on frequently filtered columns for {table['tablename']}",
                                "impact": f"Could improve performance for {table['seq_tup_read']} tuples currently scanned",
                                "table_size": live_tuples,
                            }
                        )

                    # High modification rate analysis
                    modifications = table["modifications"] or 0
                    if (
                        modifications > live_tuples * 0.1 and live_tuples > 1000
                    ):  # >10% modification rate
                        recommendations["recommendations"].append(
                            {
                                "type": "high_modification_table",
                                "priority": "medium",
                                "table": table_name,
                                "issue": f"High modification rate: {modifications} changes vs {live_tuples} live tuples",
                                "suggestion": "Consider HOT updates optimization and selective indexing",
                                "impact": "Reduce index maintenance overhead while maintaining query performance",
                                "implementation": f"-- Review index usage on {table['tablename']}\n-- Consider partial indexes for frequently queried subsets",
                            }
                        )

                # OmniNode Bridge specific indexing patterns
                microservice_patterns = [
                    {
                        "type": "microservice_pattern",
                        "priority": "high",
                        "pattern": "Event Processing Tables",
                        "suggestion": "Create indexes for event processing workflows",
                        "impact": "Improves event query and correlation performance",
                        "specific_indexes": [
                            "CREATE INDEX CONCURRENTLY idx_events_timestamp_status ON events(timestamp, status) WHERE status IN ('pending', 'processing');",
                            "CREATE INDEX CONCURRENTLY idx_events_correlation_id ON events(correlation_id) WHERE correlation_id IS NOT NULL;",
                            "CREATE INDEX CONCURRENTLY idx_events_source_type ON events(source, event_type);",
                        ],
                    },
                    {
                        "type": "microservice_pattern",
                        "priority": "high",
                        "pattern": "Workflow State Management",
                        "suggestion": "Optimize workflow state tracking indexes",
                        "impact": "Faster workflow state transitions and queries",
                        "specific_indexes": [
                            "CREATE INDEX CONCURRENTLY idx_workflows_state_updated ON workflows(state, updated_at) WHERE state != 'completed';",
                            "CREATE INDEX CONCURRENTLY idx_workflow_executions_status ON workflow_executions(status, started_at) WHERE status IN ('running', 'pending');",
                            "CREATE INDEX CONCURRENTLY idx_workflow_tasks_parent ON workflow_tasks(parent_id) WHERE parent_id IS NOT NULL;",
                        ],
                    },
                    {
                        "type": "microservice_pattern",
                        "priority": "medium",
                        "pattern": "Performance Monitoring",
                        "suggestion": "Create time-series optimized indexes for metrics",
                        "impact": "Improves monitoring query performance and data retention",
                        "specific_indexes": [
                            "CREATE INDEX CONCURRENTLY idx_metrics_time_component ON performance_metrics(timestamp DESC, component);",
                            "CREATE INDEX CONCURRENTLY idx_metrics_recent ON performance_metrics(timestamp) WHERE timestamp > NOW() - INTERVAL '7 days';",
                            "CREATE INDEX CONCURRENTLY idx_audit_logs_time_level ON audit_logs(timestamp DESC, log_level) WHERE log_level IN ('ERROR', 'WARN');",
                        ],
                    },
                    {
                        "type": "microservice_pattern",
                        "priority": "medium",
                        "pattern": "User Session Management",
                        "suggestion": "Optimize user authentication and session indexes",
                        "impact": "Faster authentication checks and session management",
                        "specific_indexes": [
                            "CREATE INDEX CONCURRENTLY idx_sessions_active ON user_sessions(user_id, expires_at) WHERE expires_at > NOW();",
                            "CREATE INDEX CONCURRENTLY idx_auth_tokens_lookup ON auth_tokens(token_hash) WHERE revoked_at IS NULL;",
                            "CREATE INDEX CONCURRENTLY idx_users_login ON users(email) WHERE active = true;",
                        ],
                    },
                ]

                recommendations["recommendations"].extend(microservice_patterns)
                recommendations["specific_indexes"].extend(
                    [
                        index
                        for pattern in microservice_patterns
                        for index in pattern.get("specific_indexes", [])
                    ]
                )

                # Environment-specific optimizations
                environment = os.getenv("ENVIRONMENT", "development").lower()

                if environment == "production":
                    recommendations["recommendations"].append(
                        {
                            "type": "production_optimization",
                            "priority": "high",
                            "suggestion": "Use CONCURRENTLY for all index creation to avoid table locks",
                            "impact": "Prevents downtime during index creation",
                            "implementation": "Always use CREATE INDEX CONCURRENTLY in production",
                            "note": "Monitor pg_stat_progress_create_index for progress tracking",
                        }
                    )

                    recommendations["recommendations"].append(
                        {
                            "type": "production_maintenance",
                            "priority": "medium",
                            "suggestion": "Implement automated index maintenance scheduling",
                            "impact": "Maintains optimal index performance over time",
                            "implementation": "Schedule REINDEX CONCURRENTLY during low-traffic periods",
                        }
                    )

                elif environment == "development":
                    recommendations["recommendations"].append(
                        {
                            "type": "development_optimization",
                            "priority": "low",
                            "suggestion": "Development: Reduce index count for faster migrations",
                            "impact": "Speeds up development workflow and test execution",
                            "implementation": "Consider using partial indexes or fewer composite indexes in dev",
                        }
                    )

                # Advanced indexing strategies
                advanced_strategies = [
                    {
                        "type": "advanced_strategy",
                        "priority": "medium",
                        "strategy": "Covering Indexes for Read-Heavy Tables",
                        "suggestion": "Use covering indexes for frequently accessed columns",
                        "impact": "Eliminates table lookups for covered queries",
                        "example": "CREATE INDEX idx_table_covering ON table(id) INCLUDE (name, status, updated_at);",
                    },
                    {
                        "type": "advanced_strategy",
                        "priority": "medium",
                        "strategy": "Expression Indexes for Computed Queries",
                        "suggestion": "Create indexes on computed expressions used in WHERE clauses",
                        "impact": "Speeds up queries with computed conditions",
                        "example": "CREATE INDEX idx_table_computed ON table(LOWER(email)) WHERE active = true;",
                    },
                    {
                        "type": "advanced_strategy",
                        "priority": "low",
                        "strategy": "Bloom Indexes for Multi-Column Equality",
                        "suggestion": "Use Bloom indexes for tables with many optional filter columns",
                        "impact": "Reduces index storage while maintaining query performance",
                        "example": "CREATE INDEX idx_table_bloom ON table USING bloom(col1, col2, col3, col4);",
                        "note": "Requires bloom extension: CREATE EXTENSION IF NOT EXISTS bloom;",
                    },
                ]

                recommendations["recommendations"].extend(advanced_strategies)

                # Calculate performance impact
                critical_count = len(
                    [
                        r
                        for r in recommendations["recommendations"]
                        if r.get("priority") == "critical"
                    ]
                )
                high_priority_count = len(
                    [
                        r
                        for r in recommendations["recommendations"]
                        if r.get("priority") == "high"
                    ]
                )

                if critical_count > 0:
                    recommendations["performance_impact"] = "critical"
                    recommendations["summary"] = (
                        f"Found {critical_count} critical and {high_priority_count} high-priority indexing opportunities"
                    )
                elif high_priority_count > 0:
                    recommendations["performance_impact"] = "high"
                    recommendations["summary"] = (
                        f"Found {high_priority_count} high-priority indexing opportunities"
                    )
                else:
                    recommendations["summary"] = (
                        f"Found {len(table_analysis)} tables analyzed with general optimization opportunities"
                    )

        except Exception as e:
            logger.error(f"Error generating indexing recommendations: {e}")
            recommendations["status"] = "error"
            recommendations["error"] = str(e)

        return recommendations

    async def _check_pg_stat_statements(self, conn) -> bool:
        """Check if pg_stat_statements extension is available."""
        try:
            result = await conn.fetchrow(
                """
                SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
            """
            )
            return result is not None
        except (asyncpg.PostgresError, AttributeError, ConnectionError):
            # Database query failed due to postgres errors, connection issues, or missing connection method
            return False


class KafkaPerformanceOptimizer:
    """Kafka performance optimization and monitoring utilities."""

    def __init__(self, kafka_client):
        """Initialize with a Kafka client instance."""
        self.client = kafka_client

    async def get_partition_load_metrics(self) -> dict[str, Any]:
        """Get partition load balancing metrics.

        Returns:
            Dictionary with partition load analysis
        """
        metrics: dict[str, Any] = {
            "status": "success",
            "timestamp": time.time(),
            "topic_metrics": {},
            "overall_balance": {},
            "recommendations": [],
        }

        try:
            for topic, load_tracker in self.client.partition_load_tracker.items():
                if not load_tracker:
                    continue

                total_messages = sum(load_tracker)
                partition_count = len(load_tracker)
                avg_load = (
                    total_messages / partition_count if partition_count > 0 else 0
                )

                # Calculate load distribution
                max_load = max(load_tracker)
                min_load = min(load_tracker)
                load_skew = (max_load - min_load) / max_load if max_load > 0 else 0

                topic_metrics = {
                    "partition_count": partition_count,
                    "total_messages": total_messages,
                    "avg_messages_per_partition": avg_load,
                    "max_partition_load": max_load,
                    "min_partition_load": min_load,
                    "load_skew_percent": load_skew * 100,
                    "load_distribution": load_tracker,
                }

                metrics["topic_metrics"][topic] = topic_metrics

                # Generate recommendations
                if load_skew > self.client.max_partition_skew:
                    metrics["recommendations"].append(
                        {
                            "topic": topic,
                            "issue": f"High partition skew: {load_skew:.1%}",
                            "suggestion": "Consider using balanced partitioning strategy or redistricting data",
                            "priority": "high" if load_skew > 0.5 else "medium",
                        }
                    )

            # Overall balance assessment
            if metrics["topic_metrics"]:
                total_topics = len(metrics["topic_metrics"])
                skewed_topics = len(
                    [
                        t
                        for t in metrics["topic_metrics"].values()
                        if t["load_skew_percent"] > self.client.max_partition_skew * 100
                    ]
                )

                metrics["overall_balance"] = {
                    "total_topics_tracked": total_topics,
                    "skewed_topics": skewed_topics,
                    "balance_health": (
                        "good" if skewed_topics == 0 else "needs_attention"
                    ),
                    "partitioning_strategy": self.client.partitioning_strategy,
                }

        except Exception as e:
            logger.error(f"Error getting partition load metrics: {e}")
            metrics["status"] = "error"
            metrics["error"] = str(e)

        return metrics

    async def optimize_partitioning_strategy(
        self, topic: Optional[str] = None
    ) -> dict[str, Any]:
        """Optimize partitioning strategy for better load distribution.

        Args:
            topic: Specific topic to optimize, or None for all topics

        Returns:
            Dictionary with optimization results
        """
        optimization = {
            "status": "success",
            "timestamp": time.time(),
            "current_strategy": self.client.partitioning_strategy,
            "optimizations": [],
            "recommendations": [],
        }

        try:
            load_metrics = await self.get_partition_load_metrics()

            if topic:
                topics_to_analyze = (
                    [topic] if topic in load_metrics["topic_metrics"] else []
                )
            else:
                topics_to_analyze = list(load_metrics["topic_metrics"].keys())

            for topic_name in topics_to_analyze:
                topic_metrics = load_metrics["topic_metrics"][topic_name]
                load_skew = topic_metrics["load_skew_percent"]

                if load_skew > self.client.max_partition_skew * 100:
                    # High skew - recommend different strategy
                    current_strategy = self.client.partitioning_strategy

                    if current_strategy == "hash":
                        recommended_strategy = "balanced"
                        reason = "Hash partitioning showing high skew - balanced strategy may distribute load better"
                    elif current_strategy == "round_robin":
                        recommended_strategy = "balanced"
                        reason = "Round-robin showing skew - balanced strategy considers load history"
                    else:  # balanced
                        recommended_strategy = "round_robin"
                        reason = "Balanced strategy still showing skew - round-robin may help reset distribution"

                    optimization["optimizations"].append(
                        {
                            "topic": topic_name,
                            "current_skew": f"{load_skew:.1f}%",
                            "recommended_strategy": recommended_strategy,
                            "reason": reason,
                            "estimated_improvement": f"Reduce skew to <{self.client.max_partition_skew * 100:.1f}%",
                        }
                    )

                    optimization["recommendations"].append(
                        f"Set KAFKA_PARTITIONING_STRATEGY={recommended_strategy} for better load distribution on {topic_name}"
                    )

            # Environment-specific recommendations
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                optimization["recommendations"].append(
                    "Production: Consider using 'balanced' strategy with KAFKA_MAX_PARTITION_SKEW=0.15 for optimal performance"
                )
            elif environment == "development":
                optimization["recommendations"].append(
                    "Development: 'round_robin' strategy may be sufficient for testing with KAFKA_MAX_PARTITION_SKEW=0.3"
                )

        except Exception as e:
            logger.error(f"Error optimizing partitioning strategy: {e}")
            optimization["status"] = "error"
            optimization["error"] = str(e)

        return optimization


class WorkflowPerformanceOptimizer:
    """Workflow cache and memory performance optimization utilities."""

    def __init__(self, workflow_cache):
        """Initialize with a workflow cache instance."""
        self.cache = workflow_cache

    async def analyze_workflow_memory_usage(self) -> dict[str, Any]:
        """Analyze workflow cache memory usage and efficiency.

        Returns:
            Dictionary with memory analysis and optimization recommendations
        """
        stats = await self.cache.get_cache_stats()

        analysis = {
            "status": "success",
            "timestamp": time.time(),
            "current_usage": stats,
            "efficiency_metrics": {},
            "recommendations": [],
        }

        try:
            storage_metrics = stats["storage_metrics"]
            performance_metrics = stats["performance_metrics"]

            # Calculate efficiency metrics
            hit_rate = performance_metrics["hit_rate_percent"]
            memory_usage_percent = storage_metrics["memory_usage_percent"]
            disk_usage_percent = storage_metrics["disk_usage_percent"]

            analysis["efficiency_metrics"] = {
                "cache_efficiency": (
                    "excellent"
                    if hit_rate > 80
                    else "good" if hit_rate > 60 else "needs_improvement"
                ),
                "memory_efficiency": (
                    "optimal"
                    if memory_usage_percent < 80
                    else "high" if memory_usage_percent < 95 else "critical"
                ),
                "storage_efficiency": (
                    "optimal"
                    if disk_usage_percent < 70
                    else "high" if disk_usage_percent < 90 else "critical"
                ),
                "compression_ratio": performance_metrics["compressions"]
                / max(performance_metrics["decompressions"], 1),
            }

            # Generate recommendations
            if hit_rate < 60:
                analysis["recommendations"].append(
                    {
                        "priority": "high",
                        "issue": f"Low cache hit rate: {hit_rate:.1f}%",
                        "suggestion": "Consider increasing cache size or reviewing cache eviction policies",
                    }
                )

            if memory_usage_percent > 90:
                analysis["recommendations"].append(
                    {
                        "priority": "high",
                        "issue": f"High memory usage: {memory_usage_percent:.1f}%",
                        "suggestion": "Consider reducing WORKFLOW_CACHE_MEMORY_MB or enabling more aggressive compression",
                    }
                )

            if disk_usage_percent > 85:
                analysis["recommendations"].append(
                    {
                        "priority": "medium",
                        "issue": f"High disk usage: {disk_usage_percent:.1f}%",
                        "suggestion": "Consider increasing WORKFLOW_CACHE_DISK_MB or reducing cache retention time",
                    }
                )

            # Environment-specific optimization recommendations
            environment = os.getenv("ENVIRONMENT", "development").lower()
            config = stats["configuration"]

            if environment == "production":
                analysis["recommendations"].append(
                    {
                        "priority": "low",
                        "suggestion": f"Production optimization: Current memory limit {config['max_memory_size_mb']}MB is appropriate for production",
                    }
                )
            elif environment == "development":
                if config["max_memory_size_mb"] > 64:
                    analysis["recommendations"].append(
                        {
                            "priority": "low",
                            "suggestion": f"Development: Consider reducing memory limit from {config['max_memory_size_mb']}MB to 32-64MB",
                        }
                    )

        except Exception as e:
            logger.error(f"Error analyzing workflow memory usage: {e}")
            analysis["status"] = "error"
            analysis["error"] = str(e)

        return analysis
