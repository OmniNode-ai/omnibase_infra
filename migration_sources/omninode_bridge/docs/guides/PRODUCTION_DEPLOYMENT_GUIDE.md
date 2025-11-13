# Production Deployment Guide - Phase 4 Optimization

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Reading Time**: 30 minutes

---

## Overview

This guide provides step-by-step procedures for deploying Phase 4 Weeks 7-8 optimization components to production with monitoring, alerting, and SLA tracking.

**Components**:
- Error Recovery Orchestration
- Performance Optimizer
- Performance Profiler
- Health Monitoring
- Alert System
- SLA Tracking

**Target Metrics**:
- Error recovery success rate: **90%+**
- Overall workflow speedup: **2-3x**
- System availability: **99.9%+**
- p95 latency: **<5s**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Checklist](#deployment-checklist)
3. [Configuration](#configuration)
4. [Monitoring Setup](#monitoring-setup)
5. [Alert Configuration](#alert-configuration)
6. [SLA Configuration](#sla-configuration)
7. [Health Checks](#health-checks)
8. [Deployment Procedures](#deployment-procedures)
9. [Validation](#validation)
10. [Rollback Procedures](#rollback-procedures)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB SSD
- Python: 3.11+
- PostgreSQL: 15+
- Kafka/Redpanda: Latest stable

**Recommended**:
- CPU: 8-12 cores (for optimal parallelism)
- RAM: 16GB
- Disk: 100GB NVMe SSD
- Load balancer for horizontal scaling

### Software Dependencies

```bash
# Core dependencies
pip install omninode-bridge>=2.0.0

# Monitoring dependencies
pip install prometheus-client>=0.18.0
pip install grafana-client>=3.5.0

# Alert dependencies
pip install slack-sdk>=3.23.0

# Production dependencies
pip install gunicorn>=21.2.0
pip install uvicorn[standard]>=0.24.0
```

### Environment Variables

Create `.env.production`:

```bash
# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# PostgreSQL (Production)
POSTGRES_HOST=192.168.86.200
POSTGRES_PORT=5436
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=omninode_app
POSTGRES_PASSWORD=<secure-password>
POSTGRES_POOL_MIN=20
POSTGRES_POOL_MAX=50

# Kafka/Redpanda (Production)
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092
KAFKA_ENABLE_INTELLIGENCE=true
KAFKA_REQUEST_TIMEOUT_MS=10000

# Error Recovery
ERROR_RECOVERY_ENABLED=true
ERROR_RECOVERY_MAX_RETRIES=5
ERROR_RECOVERY_BASE_DELAY=2.0

# Performance Optimization
PERFORMANCE_OPTIMIZER_ENABLED=true
PERFORMANCE_AUTO_OPTIMIZE=true
PERFORMANCE_TARGET_CACHE_HIT_RATE=0.95
PERFORMANCE_TARGET_SPEEDUP=3.0

# Profiling
PROFILING_ENABLED=true
PROFILING_MEMORY_ENABLED=false  # Disable in production (5% overhead)
PROFILING_IO_ENABLED=true

# Monitoring
PROMETHEUS_PORT=9090
PROMETHEUS_PUSH_GATEWAY=http://localhost:9091
METRICS_COLLECTION_INTERVAL=60

# Alerting
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/...
ALERT_EMAIL_SMTP_HOST=smtp.example.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_FROM=alerts@example.com
ALERT_EMAIL_TO=team@example.com

# SLA
SLA_ENABLED=true
SLA_P95_LATENCY_MS=5000
SLA_SUCCESS_RATE=0.999
SLA_ERROR_RECOVERY_RATE=0.90
```

### Database Setup

```sql
-- Create production database user
CREATE USER omninode_app WITH PASSWORD '<secure-password>';

-- Grant permissions
GRANT CONNECT ON DATABASE omninode_bridge TO omninode_app;
GRANT USAGE ON SCHEMA public TO omninode_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO omninode_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO omninode_app;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_recovery_workflow_id
ON error_recovery_attempts(workflow_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_recovery_created_at
ON error_recovery_attempts(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_profiles_workflow_id
ON performance_profiles(workflow_id);
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] All prerequisites met (system requirements, dependencies)
- [ ] Production environment variables configured
- [ ] Database user and permissions created
- [ ] Database indexes created
- [ ] Kafka topics created and tested
- [ ] Error patterns configured for production errors
- [ ] Performance baseline established
- [ ] Monitoring infrastructure ready (Prometheus, Grafana)
- [ ] Alert channels configured (Slack, email)
- [ ] SLA thresholds defined
- [ ] Health check endpoints tested
- [ ] Rollback plan documented

### Deployment

- [ ] Application deployed to production servers
- [ ] Environment variables loaded
- [ ] Database migrations applied
- [ ] Error recovery orchestrator initialized
- [ ] Performance optimizer initialized
- [ ] Profiler initialized (if enabled)
- [ ] Health checks passing
- [ ] Metrics collection started
- [ ] Alerts configured and tested
- [ ] SLA tracking enabled

### Post-Deployment

- [ ] All health checks passing
- [ ] Metrics visible in Prometheus/Grafana
- [ ] Alerts tested (send test notifications)
- [ ] Error recovery tested with sample errors
- [ ] Performance optimization validated (2-3x speedup)
- [ ] Production traffic validated
- [ ] Monitoring dashboards reviewed
- [ ] On-call team notified
- [ ] Documentation updated

---

## Configuration

### Error Recovery Configuration

Create `config/error_recovery_production.py`:

```python
from omninode_bridge.agents.workflows import (
    ErrorPattern,
    ErrorType,
    RecoveryStrategy,
)

# Production error patterns
PRODUCTION_ERROR_PATTERNS = [
    # Network errors - aggressive retry
    ErrorPattern(
        pattern_id="production_api_timeout",
        error_type=ErrorType.NETWORK,
        regex_pattern=r"TimeoutError|ReadTimeout|ConnectTimeout",
        recovery_strategy=RecoveryStrategy.RETRY,
        metadata={"max_retries": 5, "base_delay": 2.0},
        priority=100,
    ),

    # Rate limiting - backoff retry
    ErrorPattern(
        pattern_id="production_rate_limit",
        error_type=ErrorType.RATE_LIMIT,
        regex_pattern=r"RateLimitError|429|quota.*exceeded",
        recovery_strategy=RecoveryStrategy.RETRY,
        metadata={"max_retries": 3, "base_delay": 5.0},
        priority=100,
    ),

    # LLM service failures - alternative models
    ErrorPattern(
        pattern_id="production_llm_failure",
        error_type=ErrorType.AI_SERVICE,
        regex_pattern=r"Model.*unavailable|API.*error",
        recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
        metadata={
            "model_alternatives": [
                ("gemini-1.5-pro", "primary"),
                ("gpt-4", "secondary"),
                ("claude-3", "tertiary"),
            ],
        },
        priority=90,
    ),

    # Template failures - fallback templates
    ErrorPattern(
        pattern_id="production_template_failure",
        error_type=ErrorType.VALIDATION,
        regex_pattern=r"Template.*render.*failed",
        recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
        metadata={
            "template_alternatives": [
                ("node_effect_v2", "primary"),
                ("node_effect_v1", "fallback"),
            ],
        },
        priority=80,
    ),

    # Complex generation - graceful degradation
    ErrorPattern(
        pattern_id="production_generation_timeout",
        error_type=ErrorType.TIMEOUT,
        regex_pattern=r"Generation.*timeout",
        recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
        metadata={
            "degradation_steps": [
                "remove_optional_mixins",
                "skip_ai_quorum",
                "simplify_validation",
            ],
        },
        priority=70,
    ),

    # Critical errors - escalate immediately
    ErrorPattern(
        pattern_id="production_critical_error",
        error_type=ErrorType.SECURITY,
        regex_pattern=r"Security.*failed|Vulnerability",
        recovery_strategy=RecoveryStrategy.ESCALATION,
        metadata={
            "notification_channel": "pagerduty",
            "priority": "critical",
            "assignee": "security-team",
        },
        priority=200,  # Highest priority
    ),
]
```

### Performance Optimization Configuration

Create `config/performance_production.py`:

```python
# Production optimization targets
PRODUCTION_OPTIMIZATION_CONFIG = {
    # Template cache
    "template_cache": {
        "target_hit_rate": 0.95,
        "cache_size": 200,  # Larger for production
        "preload_top_n": 50,
    },

    # Parallel execution
    "parallel_execution": {
        "target_concurrency": 12,  # Based on production CPU cores
        "target_speedup": 3.5,
        "max_parallel_tasks": 15,
    },

    # Memory
    "memory": {
        "target_overhead_mb": 50.0,
        "enable_gc_tuning": True,
        "gc_threshold": (700, 10, 10),
    },

    # I/O
    "io": {
        "target_async_ratio": 0.85,  # 85% for production
        "batch_size": 20,
    },

    # Profiling (disabled for production)
    "profiling": {
        "enabled": False,  # Only enable for debugging
        "memory_profiling": False,
        "io_profiling": True,
    },
}
```

---

## Monitoring Setup

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # OmniNode Bridge metrics
  - job_name: 'omninode_bridge'
    static_configs:
      - targets: ['localhost:8053']
    metrics_path: '/metrics'

  # Error recovery metrics
  - job_name: 'error_recovery'
    static_configs:
      - targets: ['localhost:9091']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'error_recovery_.*'
        action: keep

  # Performance metrics
  - job_name: 'performance'
    static_configs:
      - targets: ['localhost:9091']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'workflow_.*|template_.*|validation_.*'
        action: keep
```

### Grafana Dashboards

Create Grafana dashboard `dashboards/phase4_optimization.json`:

**Key Panels**:

1. **Error Recovery Success Rate** (Gauge)
   - Query: `rate(error_recovery_success_total[5m]) / rate(error_recovery_attempts_total[5m])`
   - Thresholds: Red <0.80, Yellow 0.80-0.90, Green >0.90

2. **Workflow Performance** (Graph)
   - Query: `histogram_quantile(0.95, rate(workflow_duration_ms_bucket[5m]))`
   - Target: <5000ms (p95)

3. **Cache Hit Rate** (Gauge)
   - Query: `rate(template_cache_hits_total[5m]) / rate(template_cache_requests_total[5m])`
   - Thresholds: Red <0.85, Yellow 0.85-0.95, Green >0.95

4. **Recovery Strategy Usage** (Pie Chart)
   - Query: `sum by (strategy) (error_recovery_attempts_total)`

5. **Memory Usage** (Graph)
   - Query: `process_resident_memory_bytes / 1024 / 1024`
   - Target: <512MB

### Metrics Export

Add to `src/omninode_bridge/agents/workflows/__init__.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Error Recovery Metrics
error_recovery_attempts = Counter(
    'error_recovery_attempts_total',
    'Total error recovery attempts',
    ['strategy', 'error_type']
)

error_recovery_success = Counter(
    'error_recovery_success_total',
    'Successful error recoveries',
    ['strategy']
)

error_recovery_duration = Histogram(
    'error_recovery_duration_ms',
    'Error recovery duration in milliseconds',
    ['strategy']
)

# Performance Metrics
workflow_duration = Histogram(
    'workflow_duration_ms',
    'Workflow execution duration',
    buckets=[100, 500, 1000, 2000, 5000, 10000, 30000]
)

workflow_speedup = Gauge(
    'workflow_speedup_factor',
    'Workflow parallelism speedup factor'
)

template_cache_hits = Counter(
    'template_cache_hits_total',
    'Template cache hits'
)

template_cache_misses = Counter(
    'template_cache_misses_total',
    'Template cache misses'
)

# Start Prometheus metrics server
def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
```

---

## Alert Configuration

### Slack Alerts

Create `config/alerts_slack.py`:

```python
from omninode_bridge.agents.workflows.monitoring import AlertManager

def configure_slack_alerts(webhook_url: str) -> AlertManager:
    """
    Configure Slack alerts for production.
    """
    alert_manager = AlertManager(
        notification_channel="slack",
        webhook_url=webhook_url,
    )

    # Critical: Error recovery success rate
    alert_manager.add_threshold(
        name="error_recovery_success_rate",
        metric="error_recovery_success_rate",
        min_value=0.85,
        severity="critical",
        evaluation_window_minutes=10,
        description="Error recovery success rate dropped below 85%",
    )

    # Critical: Workflow p95 latency
    alert_manager.add_threshold(
        name="workflow_p95_latency",
        metric="workflow_duration_p95_ms",
        max_value=5000.0,
        severity="critical",
        evaluation_window_minutes=5,
        description="Workflow p95 latency exceeded 5 seconds",
    )

    # High: Cache hit rate
    alert_manager.add_threshold(
        name="cache_hit_rate",
        metric="template_cache_hit_rate",
        min_value=0.90,
        severity="high",
        evaluation_window_minutes=15,
        description="Template cache hit rate dropped below 90%",
    )

    # Medium: Memory usage
    alert_manager.add_threshold(
        name="memory_usage",
        metric="process_memory_mb",
        max_value=512.0,
        severity="medium",
        evaluation_window_minutes=30,
        description="Memory usage exceeded 512MB",
    )

    return alert_manager
```

### Email Alerts

Create `config/alerts_email.py`:

```python
import smtplib
from email.message import EmailMessage

def send_email_alert(
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    from_addr: str,
    to_addr: str,
    smtp_user: str = None,
    smtp_password: str = None,
):
    """
    Send email alert via SMTP.
    """
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr

    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        if smtp_user and smtp_password:
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)
```

### PagerDuty Integration (Critical Alerts)

```python
from pypd import EventV2

def send_pagerduty_alert(
    routing_key: str,
    summary: str,
    severity: str,
    details: dict,
):
    """
    Send PagerDuty alert for critical incidents.
    """
    EventV2.create(
        routing_key=routing_key,
        event_action='trigger',
        payload={
            'summary': summary,
            'severity': severity,
            'source': 'omninode-bridge',
            'custom_details': details,
        }
    )
```

---

## SLA Configuration

### Define SLAs

Create `config/sla_production.py`:

```python
from omninode_bridge.agents.workflows.monitoring import SLATracker

def configure_production_slas(metrics_collector) -> SLATracker:
    """
    Configure production SLAs.
    """
    sla_tracker = SLATracker(metrics_collector=metrics_collector)

    # SLA 1: Workflow p95 latency <5s
    sla_tracker.add_sla(
        name="workflow_p95_latency",
        metric="workflow_duration_ms",
        percentile=95,
        threshold=5000.0,  # 5 seconds
        evaluation_window_hours=1,
        target_compliance=0.999,  # 99.9%
    )

    # SLA 2: System availability >99.9%
    sla_tracker.add_sla(
        name="system_availability",
        metric="health_check_success_rate",
        threshold=0.999,  # 99.9%
        evaluation_window_hours=24,
        target_compliance=1.0,
    )

    # SLA 3: Error recovery success rate >90%
    sla_tracker.add_sla(
        name="error_recovery_success_rate",
        metric="recovery_success_rate",
        threshold=0.90,  # 90%
        evaluation_window_hours=6,
        target_compliance=0.95,  # 95% compliance
    )

    # SLA 4: Cache hit rate >95%
    sla_tracker.add_sla(
        name="cache_hit_rate",
        metric="template_cache_hit_rate",
        threshold=0.95,  # 95%
        evaluation_window_hours=12,
        target_compliance=0.99,
    )

    return sla_tracker
```

### SLA Reporting

```python
async def generate_sla_report(sla_tracker: SLATracker) -> dict:
    """
    Generate SLA compliance report.
    """
    compliance = await sla_tracker.check_compliance()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "slas": [],
        "overall_compliance": 0.0,
    }

    passing_count = 0
    for sla in compliance:
        sla_status = {
            "name": sla.name,
            "passing": sla.passing,
            "current_value": sla.current_value,
            "threshold": sla.threshold,
            "compliance_rate": sla.compliance_rate,
        }
        report["slas"].append(sla_status)

        if sla.passing:
            passing_count += 1

    report["overall_compliance"] = passing_count / len(compliance)

    return report
```

---

## Health Checks

### Application Health Check

Create `/health` endpoint:

```python
from fastapi import FastAPI
from omninode_bridge.agents.workflows.monitoring import HealthChecker

app = FastAPI()

health_checker = HealthChecker(
    error_recovery=orchestrator,
    optimizer=optimizer,
    profiler=profiler,
)

# Add health checks
health_checker.add_check(
    "database_connection",
    lambda: check_database_connection(),
    critical=True,
)

health_checker.add_check(
    "kafka_connection",
    lambda: check_kafka_connection(),
    critical=True,
)

health_checker.add_check(
    "error_recovery_success_rate",
    lambda: orchestrator.get_statistics().success_rate >= 0.80,
    critical=False,
)

health_checker.add_check(
    "cache_hit_rate",
    lambda: template_manager.get_cache_stats().hit_rate >= 0.90,
    critical=False,
)

health_checker.add_check(
    "memory_usage",
    lambda: check_memory_usage() < 512.0,  # MB
    critical=False,
)

@app.get("/health")
async def health():
    """Health check endpoint."""
    status = await health_checker.check_all()

    return {
        "status": "healthy" if status.all_passing else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            check.name: {
                "passing": check.passing,
                "critical": check.critical,
                "message": check.message,
            }
            for check in status.checks
        },
    }

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    # Check critical dependencies
    db_ok = await check_database_connection()
    kafka_ok = await check_kafka_connection()

    if db_ok and kafka_ok:
        return {"status": "ready"}
    else:
        return {"status": "not_ready"}, 503
```

---

## Deployment Procedures

### Step-by-Step Deployment

**1. Prepare Production Environment**:

```bash
# SSH to production server
ssh production-server

# Create deployment directory
mkdir -p /opt/omninode_bridge/phase4_optimization
cd /opt/omninode_bridge/phase4_optimization

# Clone repository (or copy deployment package)
git clone https://github.com/OmniNode-ai/omninode_bridge.git .
git checkout phase-4-optimization

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-production.txt
```

**2. Configure Environment**:

```bash
# Copy production environment variables
cp .env.production.example .env

# Edit with production values
vim .env

# Verify configuration
python scripts/verify_production_config.py
```

**3. Run Database Migrations**:

```bash
# Backup database first
pg_dump -h 192.168.86.200 -p 5436 -U postgres omninode_bridge > backup_$(date +%Y%m%d).sql

# Run migrations
alembic upgrade head

# Verify migrations
python scripts/verify_database_schema.py
```

**4. Start Application**:

```bash
# Start with gunicorn (production WSGI server)
gunicorn src.omninode_bridge.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8053 \
  --timeout 300 \
  --access-logfile /var/log/omninode_bridge/access.log \
  --error-logfile /var/log/omninode_bridge/error.log \
  --log-level info \
  --daemon

# Or use systemd service
sudo systemctl start omninode-bridge
sudo systemctl enable omninode-bridge
```

**5. Verify Deployment**:

```bash
# Check health endpoint
curl http://localhost:8053/health

# Check metrics endpoint
curl http://localhost:8053/metrics

# Check application logs
tail -f /var/log/omninode_bridge/error.log
```

**6. Enable Monitoring**:

```bash
# Start Prometheus
sudo systemctl start prometheus
sudo systemctl enable prometheus

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Verify metrics collection
curl http://localhost:9090/api/v1/query?query=up
```

**7. Configure Alerts**:

```bash
# Test Slack webhook
python scripts/test_slack_webhook.py

# Start alert manager
python scripts/start_alert_manager.py --config config/alerts_slack.py
```

---

## Validation

### Post-Deployment Validation Checklist

```bash
# 1. Health checks
curl http://localhost:8053/health | jq '.status'
# Expected: "healthy"

# 2. Metrics collection
curl http://localhost:8053/metrics | grep error_recovery
# Expected: Metrics visible

# 3. Error recovery test
python scripts/test_error_recovery.py
# Expected: 90%+ success rate

# 4. Performance test
python scripts/test_workflow_performance.py
# Expected: 2-3x speedup

# 5. Cache hit rate
curl http://localhost:8053/metrics | grep template_cache
# Expected: >95% hit rate

# 6. Memory usage
curl http://localhost:8053/metrics | grep process_resident_memory
# Expected: <512MB

# 7. Alert test
python scripts/send_test_alert.py
# Expected: Alert received in Slack

# 8. SLA compliance
python scripts/check_sla_compliance.py
# Expected: All SLAs passing
```

---

## Rollback Procedures

### Quick Rollback

If deployment fails, follow these steps:

**1. Stop Application**:

```bash
sudo systemctl stop omninode-bridge
```

**2. Restore Previous Version**:

```bash
# Restore code
cd /opt/omninode_bridge
mv phase4_optimization phase4_optimization.failed
mv phase4_optimization.backup phase4_optimization

# Restore database (if needed)
psql -h 192.168.86.200 -p 5436 -U postgres omninode_bridge < backup_$(date +%Y%m%d).sql

# Or use alembic downgrade
cd phase4_optimization
alembic downgrade -1  # Downgrade one revision
```

**3. Restart Application**:

```bash
sudo systemctl start omninode-bridge
```

**4. Verify Rollback**:

```bash
curl http://localhost:8053/health
# Check logs
tail -f /var/log/omninode_bridge/error.log
```

---

## Troubleshooting

### Common Issues

#### Health Check Failing

**Symptoms**: `/health` endpoint returns "degraded" status

**Diagnosis**:
```bash
curl http://localhost:8053/health | jq '.checks'
```

**Solutions**:
1. Check database connection
2. Check Kafka connection
3. Review error logs
4. Restart application

#### Low Recovery Success Rate

**Symptoms**: Error recovery success rate <80%

**Diagnosis**:
```python
stats = orchestrator.get_statistics()
print(f"Success rate: {stats.success_rate:.1%}")
```

**Solutions**:
1. Review error patterns
2. Increase retry limits
3. Add more alternative paths
4. Check error logs for new patterns

#### High Memory Usage

**Symptoms**: Memory usage >512MB

**Diagnosis**:
```bash
curl http://localhost:8053/metrics | grep process_resident_memory
```

**Solutions**:
1. Reduce template cache size
2. Enable garbage collection tuning
3. Check for memory leaks
4. Restart application

---

**See Also**:
- [Phase 4 Optimization Guide](./PHASE_4_OPTIMIZATION_GUIDE.md) - Complete optimization system
- [Error Recovery Guide](./ERROR_RECOVERY_GUIDE.md) - Error recovery strategies
- [Performance Tuning Guide](./WORKFLOW_PERFORMANCE_TUNING.md) - Performance optimization
- [Workflows API Reference](../api/WORKFLOWS_API_REFERENCE.md) - Complete API documentation
