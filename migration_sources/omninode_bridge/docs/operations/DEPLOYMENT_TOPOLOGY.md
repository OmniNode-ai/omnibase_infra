# Deployment Topology and Operational Procedures

This document provides comprehensive deployment topology documentation and operational procedures for the OmniNode Bridge multi-service architecture.

## Production Deployment Topology

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Internet / External Clients                 │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                     Load Balancer / API Gateway                      │
│                    (NGINX Ingress / ALB)                           │
│                 SSL Termination & Rate Limiting                      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                    Kubernetes Cluster                               │
│  ┌─────────────────┬────────────────┬─────────────────────────────┐ │
│  │   Hook Receiver │  Model Metrics │  Workflow Coordinator      │ │
│  │   (3 replicas)  │   (2 replicas) │     (3 replicas)          │ │
│  │   Port: 8001    │   Port: 8002   │     Port: 8003            │ │
│  └─────────────────┼────────────────┼─────────────────────────────┘ │
│                    │                │                               │
│  ┌─────────────────▼────────────────▼─────────────────────────────┐ │
│  │                 Infrastructure Services                        │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │ │
│  │  │  PostgreSQL  │ │    Kafka     │ │       Redis          │   │ │
│  │  │  (Primary +  │ │   Cluster    │ │     (Cluster)        │   │ │
│  │  │  Read Replica)│ │ (3 brokers) │ │    (3 nodes)         │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  Monitoring & Observability                    │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │ │
│  │  │ Prometheus   │ │   Grafana    │ │    ELK Stack         │   │ │
│  │  │   Metrics    │ │  Dashboards  │ │  (Logs & Search)     │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Network Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Public Subnet                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Load Balancer                               │ │
│  │              (Public IP: 203.0.113.10)                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                       Private Subnet 1                              │
│                    (Application Tier)                               │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │  Hook Receiver  │ │  Model Metrics  │ │ Workflow Coordinator    │ │
│  │  10.0.1.10-12   │ │  10.0.1.20-21   │ │    10.0.1.30-32        │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                       Private Subnet 2                              │
│                     (Data Tier)                                     │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │   PostgreSQL    │ │     Kafka       │ │        Redis            │ │
│  │ 10.0.2.10 (PRI) │ │  10.0.2.20-22   │ │     10.0.2.30-32       │ │
│  │ 10.0.2.11 (REP) │ │   (3 brokers)   │ │     (3 nodes)          │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                       Private Subnet 3                              │
│                   (Monitoring Tier)                                 │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │   Prometheus    │ │    Grafana      │ │      ELK Stack          │ │
│  │   10.0.3.10     │ │   10.0.3.20     │ │    10.0.3.30-32        │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Environment Configurations

### Development Environment

```yaml
# environments/development/values.yaml
environment:
  name: development
  domain: dev.omninode-bridge.local

replicas:
  hookReceiver: 1
  modelMetrics: 1
  workflowCoordinator: 1

resources:
  hookReceiver:
    requests: { cpu: "100m", memory: "128Mi" }
    limits: { cpu: "500m", memory: "256Mi" }
  modelMetrics:
    requests: { cpu: "100m", memory: "128Mi" }
    limits: { cpu: "500m", memory: "256Mi" }
  workflowCoordinator:
    requests: { cpu: "100m", memory: "128Mi" }
    limits: { cpu: "500m", memory: "256Mi" }

database:
  host: postgres-dev
  name: omninode_bridge_dev
  poolSize: 5
  ssl: false

kafka:
  brokers: 1
  replicationFactor: 1
  autoCreateTopics: true

redis:
  nodes: 1
  persistence: false

monitoring:
  enabled: true
  retention: "7d"

security:
  rateLimiting:
    enabled: false
  authentication:
    keyRotationHours: 720  # 30 days
    disableAuth: true
  ssl:
    enabled: false

logging:
  level: DEBUG
  structuredLogging: true
```

### Staging Environment

```yaml
# environments/staging/values.yaml
environment:
  name: staging
  domain: staging.omninode-bridge.com

replicas:
  hookReceiver: 2
  modelMetrics: 2
  workflowCoordinator: 2

resources:
  hookReceiver:
    requests: { cpu: "250m", memory: "256Mi" }
    limits: { cpu: "1", memory: "512Mi" }
  modelMetrics:
    requests: { cpu: "250m", memory: "256Mi" }
    limits: { cpu: "1", memory: "512Mi" }
  workflowCoordinator:
    requests: { cpu: "250m", memory: "256Mi" }
    limits: { cpu: "1", memory: "512Mi" }

database:
  host: postgres-staging-cluster
  name: omninode_bridge_staging
  poolSize: 15
  ssl: true
  readReplica: postgres-staging-replica

kafka:
  brokers: 2
  replicationFactor: 2
  autoCreateTopics: false

redis:
  nodes: 2
  persistence: true
  backup: true

monitoring:
  enabled: true
  retention: "30d"
  alerting: true

security:
  rateLimiting:
    enabled: true
    multiplier: 2.0
  authentication:
    keyRotationHours: 336  # 14 days
    requireHttps: true
  ssl:
    enabled: true
    certificateIssuer: letsencrypt-staging

logging:
  level: INFO
  structuredLogging: true

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 6
  targetCPU: 70
```

### Production Environment

```yaml
# environments/production/values.yaml
environment:
  name: production
  domain: api.omninode-bridge.com

replicas:
  hookReceiver: 3
  modelMetrics: 2
  workflowCoordinator: 3

resources:
  hookReceiver:
    requests: { cpu: "500m", memory: "512Mi" }
    limits: { cpu: "2", memory: "1Gi" }
  modelMetrics:
    requests: { cpu: "500m", memory: "512Mi" }
    limits: { cpu: "2", memory: "1Gi" }
  workflowCoordinator:
    requests: { cpu: "500m", memory: "512Mi" }
    limits: { cpu: "2", memory: "1Gi" }

database:
  host: postgres-prod-cluster
  name: omninode_bridge
  poolSize: 25
  ssl: true
  sslMode: require
  readReplica: postgres-prod-replica
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: "30d"

kafka:
  brokers: 3
  replicationFactor: 3
  autoCreateTopics: false
  security:
    enabled: true
    protocol: SASL_SSL

redis:
  nodes: 3
  persistence: true
  backup:
    enabled: true
    schedule: "0 3 * * *"  # Daily at 3 AM
    retention: "14d"

monitoring:
  enabled: true
  retention: "90d"
  alerting: true
  oncall: true

security:
  rateLimiting:
    enabled: true
    multiplier: 1.0
    adaptive: true
  authentication:
    keyRotationHours: 168  # 7 days
    requireHttps: true
    jwtAlgorithm: RS256
  ssl:
    enabled: true
    certificateIssuer: letsencrypt-prod
    hstsEnabled: true
  networkPolicies:
    enabled: true
    defaultDeny: true

logging:
  level: WARNING
  structuredLogging: true
  retention: "90d"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 15
  targetCPU: 70
  targetMemory: 80
  scaleUpStabilization: 60s
  scaleDownStabilization: 300s

backup:
  enabled: true
  crossRegion: true
  encryption: true
```

## Infrastructure Requirements

### Minimum System Requirements

#### Development Environment
```yaml
Cluster:
  Nodes: 1
  CPU: 4 cores
  Memory: 8 GB
  Storage: 50 GB SSD

Services:
  Application Pods: 3 (1 per service)
  Infrastructure: PostgreSQL, Kafka, Redis (single instances)
  Monitoring: Basic Prometheus + Grafana

Estimated Monthly Cost: $50-100 (cloud)
```

#### Production Environment
```yaml
Cluster:
  Nodes: 6-9 (across 3 AZs)
  CPU: 32-48 cores total
  Memory: 64-96 GB total
  Storage: 500 GB SSD (application) + 2 TB (data)

Services:
  Application Pods: 8 (with autoscaling to 24)
  Infrastructure: HA PostgreSQL, Kafka cluster, Redis cluster
  Monitoring: Full observability stack

Estimated Monthly Cost: $2,000-5,000 (cloud)
```

### Storage Requirements

```yaml
PostgreSQL:
  Primary: 100 GB (production), 20 GB (staging), 5 GB (dev)
  Read Replica: Same as primary
  Backup: 3x primary size for retention
  IOPS: 3000 (production), 1000 (staging), 300 (dev)

Kafka:
  Log Storage: 200 GB (production), 50 GB (staging), 10 GB (dev)
  Retention: 7 days (production), 3 days (staging), 1 day (dev)
  Replication: 3x (production), 2x (staging), 1x (dev)

Redis:
  Memory: 8 GB (production), 4 GB (staging), 1 GB (dev)
  Persistence: RDB + AOF (production/staging), None (dev)
  Backup: Daily snapshots

Monitoring:
  Metrics: 50 GB (90d retention)
  Logs: 100 GB (90d retention)
  Traces: 25 GB (30d retention)
```

## Deployment Procedures

### 1. Pre-Deployment Checklist

```bash
#!/bin/bash
# scripts/pre-deployment-checklist.sh

echo "🔍 Pre-Deployment Checklist"
echo "============================"

# Check cluster health
echo "1. Checking cluster health..."
kubectl cluster-info
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running | grep -v Completed

# Check resource availability
echo "2. Checking resource availability..."
kubectl top nodes
kubectl describe nodes | grep -A 3 "Allocated resources"

# Verify secrets and configs
echo "3. Verifying secrets and configurations..."
kubectl get secrets -n omninode-bridge
kubectl get configmaps -n omninode-bridge

# Check infrastructure services
echo "4. Checking infrastructure services..."
kubectl get statefulsets -n omninode-bridge
kubectl get persistentvolumes

# Verify network policies
echo "5. Verifying network policies..."
kubectl get networkpolicies -n omninode-bridge

# Check monitoring stack
echo "6. Checking monitoring stack..."
kubectl get pods -n monitoring
curl -s http://prometheus.monitoring.svc.cluster.local:9090/-/healthy
curl -s http://grafana.monitoring.svc.cluster.local:3000/api/health

echo "✅ Pre-deployment checks complete"
```

### 2. Blue-Green Deployment Procedure

```bash
#!/bin/bash
# scripts/blue-green-deployment.sh

set -e

NAMESPACE="omninode-bridge"
VERSION=$1
ENVIRONMENT=$2

if [ -z "$VERSION" ] || [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <version> <environment>"
    exit 1
fi

echo "🚀 Starting Blue-Green Deployment"
echo "================================="
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"

# Determine current and new colors
CURRENT_COLOR=$(kubectl get service main-service -n $NAMESPACE \
    -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "Current: $CURRENT_COLOR → New: $NEW_COLOR"

# Step 1: Deploy to new color
echo "📦 Deploying to $NEW_COLOR environment..."
helm upgrade --install omninode-bridge-$NEW_COLOR ./helm/omninode-bridge \
    --namespace $NAMESPACE \
    --values environments/$ENVIRONMENT/values.yaml \
    --set image.tag=$VERSION \
    --set deployment.color=$NEW_COLOR \
    --wait --timeout=600s

# Step 2: Wait for deployment readiness
echo "⏳ Waiting for $NEW_COLOR deployment to be ready..."
kubectl rollout status deployment/hook-receiver-$NEW_COLOR -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/model-metrics-$NEW_COLOR -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/workflow-coordinator-$NEW_COLOR -n $NAMESPACE --timeout=300s

# Step 3: Run health checks
echo "🏥 Running health checks on $NEW_COLOR environment..."
if python scripts/deployment-health-check.py \
    --environment=$ENVIRONMENT \
    --color=$NEW_COLOR \
    --timeout=300; then
    echo "✅ Health checks passed"
else
    echo "❌ Health checks failed - aborting deployment"
    exit 1
fi

# Step 4: Run smoke tests
echo "🧪 Running smoke tests..."
if python scripts/smoke-tests.py \
    --environment=$ENVIRONMENT \
    --color=$NEW_COLOR; then
    echo "✅ Smoke tests passed"
else
    echo "❌ Smoke tests failed - aborting deployment"
    exit 1
fi

# Step 5: Gradual traffic switch (canary)
echo "🔄 Starting gradual traffic switch..."

# 10% traffic to new color
kubectl patch service main-service -n $NAMESPACE \
    --type='json' \
    -p='[{"op": "replace", "path": "/spec/selector", "value": {"app": "omninode-bridge", "canary": "10-90"}}]'

echo "📊 Directing 10% traffic to $NEW_COLOR..."
sleep 60

# Monitor metrics for 5 minutes
echo "📈 Monitoring metrics..."
python scripts/monitor-canary.py --duration=300 --threshold=0.95

# 50% traffic
kubectl patch service main-service -n $NAMESPACE \
    --type='json' \
    -p='[{"op": "replace", "path": "/spec/selector", "value": {"app": "omninode-bridge", "canary": "50-50"}}]'

echo "📊 Directing 50% traffic to $NEW_COLOR..."
sleep 120

python scripts/monitor-canary.py --duration=180 --threshold=0.95

# Full traffic switch
kubectl patch service main-service -n $NAMESPACE \
    --type='json' \
    -p='[{"op": "replace", "path": "/spec/selector", "value": {"color": "'$NEW_COLOR'"}}]'

echo "📊 Directing 100% traffic to $NEW_COLOR..."
sleep 60

# Step 6: Final validation
echo "🏥 Running final validation..."
if python scripts/deployment-health-check.py \
    --environment=$ENVIRONMENT \
    --color=$NEW_COLOR \
    --comprehensive; then
    echo "✅ Final validation passed"
else
    echo "❌ Final validation failed - initiating rollback"
    kubectl patch service main-service -n $NAMESPACE \
        --type='json' \
        -p='[{"op": "replace", "path": "/spec/selector", "value": {"color": "'$CURRENT_COLOR'"}}]'
    exit 1
fi

# Step 7: Clean up old deployment
echo "🧹 Cleaning up old $CURRENT_COLOR deployment..."
helm uninstall omninode-bridge-$CURRENT_COLOR -n $NAMESPACE || true

# Step 8: Update deployment records
echo "📝 Updating deployment records..."
kubectl annotate deployment hook-receiver-$NEW_COLOR -n $NAMESPACE \
    deployment.kubernetes.io/revision-history="$(date): Deployed version $VERSION"

echo "🎉 Blue-Green deployment completed successfully!"
echo "Active Color: $NEW_COLOR"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"
```

### 3. Rollback Procedure

```bash
#!/bin/bash
# scripts/rollback-deployment.sh

set -e

NAMESPACE="omninode-bridge"
TARGET_REVISION=${1:-"previous"}

echo "🔄 Starting Rollback Procedure"
echo "==============================="
echo "Target: $TARGET_REVISION"

# Step 1: Identify target revision
if [ "$TARGET_REVISION" = "previous" ]; then
    HOOK_REVISION=$(kubectl rollout history deployment/hook-receiver -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')
    METRICS_REVISION=$(kubectl rollout history deployment/model-metrics -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')
    WORKFLOW_REVISION=$(kubectl rollout history deployment/workflow-coordinator -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')
else
    HOOK_REVISION=$TARGET_REVISION
    METRICS_REVISION=$TARGET_REVISION
    WORKFLOW_REVISION=$TARGET_REVISION
fi

echo "Rollback targets:"
echo "  Hook Receiver: revision $HOOK_REVISION"
echo "  Model Metrics: revision $METRICS_REVISION"
echo "  Workflow Coordinator: revision $WORKFLOW_REVISION"

# Step 2: Perform rollback
echo "⏪ Rolling back deployments..."

kubectl rollout undo deployment/hook-receiver -n $NAMESPACE --to-revision=$HOOK_REVISION
kubectl rollout undo deployment/model-metrics -n $NAMESPACE --to-revision=$METRICS_REVISION
kubectl rollout undo deployment/workflow-coordinator -n $NAMESPACE --to-revision=$WORKFLOW_REVISION

# Step 3: Wait for rollback completion
echo "⏳ Waiting for rollback to complete..."

kubectl rollout status deployment/hook-receiver -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/model-metrics -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/workflow-coordinator -n $NAMESPACE --timeout=300s

# Step 4: Health check after rollback
echo "🏥 Running post-rollback health checks..."

if python scripts/deployment-health-check.py --timeout=180; then
    echo "✅ Rollback successful - system healthy"
else
    echo "❌ Rollback completed but health checks failed"
    echo "🚨 Immediate manual intervention required"
    exit 1
fi

echo "🎉 Rollback completed successfully!"
```

## Operational Procedures

### 1. Daily Operations Checklist

```bash
#!/bin/bash
# scripts/daily-operations-check.sh

echo "📋 Daily Operations Checklist - $(date)"
echo "========================================"

# Check cluster health
echo "1. 🏥 Cluster Health"
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running | grep -v Completed
echo ""

# Check application health
echo "2. 🔧 Application Health"
curl -sf http://hook-receiver.omninode-bridge.svc:8001/health
curl -sf http://model-metrics.omninode-bridge.svc:8002/health
curl -sf http://workflow-coordinator.omninode-bridge.svc:8003/health
echo "✅ All services healthy"
echo ""

# Check resource utilization
echo "3. 📊 Resource Utilization"
kubectl top nodes
kubectl top pods -n omninode-bridge
echo ""

# Check storage
echo "4. 💾 Storage Status"
kubectl get pv,pvc -n omninode-bridge
echo ""

# Check recent alerts
echo "5. 🚨 Recent Alerts (last 24h)"
python scripts/check-alerts.py --hours=24
echo ""

# Check backup status
echo "6. 💾 Backup Status"
python scripts/check-backups.py --last=1
echo ""

# Check certificate expiration
echo "7. 🔒 Certificate Status"
python scripts/check-certificates.py --days-warning=30
echo ""

echo "✅ Daily operations check complete"
```

### 2. Troubleshooting Runbooks

#### High Error Rate Runbook

```markdown
## High Error Rate Incident Response

### Alert: Error rate > 5% for 5 minutes

#### Immediate Actions (0-5 minutes)
1. **Acknowledge Alert**: Confirm receipt and assign primary responder
2. **Check Service Status**: Verify which services are affected
3. **Check Recent Deployments**: Identify any recent changes
4. **Enable Circuit Breakers**: If not already enabled

#### Investigation Steps (5-15 minutes)
1. **Check Logs**: Query centralized logs for error patterns
2. **Check Metrics**: Review Grafana dashboards for anomalies
3. **Check Infrastructure**: Verify database, Kafka, Redis health
4. **Check External Dependencies**: Verify third-party service status

#### Escalation Criteria
- Error rate > 10%
- Multiple services affected
- Database connectivity issues
- No clear root cause within 15 minutes

#### Resolution Actions
1. **Rollback**: If caused by recent deployment
2. **Scale Up**: If caused by traffic surge
3. **Circuit Breaker**: If external dependency issue
4. **Database**: Scale read replicas if database overload
```

#### Database Connection Exhaustion Runbook

```markdown
## Database Connection Exhaustion Response

### Alert: DB connection pool utilization > 90%

#### Immediate Actions (0-2 minutes)
1. **Check Pool Status**: Verify current pool utilization
2. **Check for Connection Leaks**: Review connection age metrics
3. **Identify High-Usage Services**: Check per-service connection usage

#### Resolution Steps
1. **Scale Pool Size**: Temporarily increase pool size
2. **Restart Problematic Services**: If connection leaks detected
3. **Enable Read Replicas**: Route read queries to replicas
4. **Optimize Queries**: Identify and optimize slow queries

#### Prevention
1. **Connection Monitoring**: Enhanced pool monitoring
2. **Query Optimization**: Regular query performance review
3. **Connection Auditing**: Automated leak detection
```

### 3. Monitoring and Alerting Configuration

#### Alert Manager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@omninode-bridge.com'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: 'critical'
      receiver: 'critical-alerts'
    - match:
        severity: 'warning'
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@company.com'
        subject: 'OmniNode Bridge Alert: {{ .GroupLabels.alertname }}'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@company.com'
        subject: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
    slack_configs:
      - channel: '#critical-alerts'
        api_url: 'https://hooks.slack.com/services/...'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'  # pragma: allowlist secret

  - name: 'warning-alerts'
    slack_configs:
      - channel: '#alerts'
        api_url: 'https://hooks.slack.com/services/...'
        title: 'Warning: {{ .GroupLabels.alertname }}'
```

#### Prometheus Alert Rules

```yaml
# monitoring/alert-rules.yml
groups:
  - name: omninode-bridge-alerts
    rules:
      # Service availability
      - alert: ServiceDown
        expr: up{job=~"hook-receiver|model-metrics|workflow-coordinator"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"

      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"

      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time on {{ $labels.service }}"

      # Database connection pool
      - alert: DatabasePoolExhaustion
        expr: db_connection_pool_utilization > 0.9
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Database connection pool near exhaustion"

      # Kafka consumer lag
      - alert: KafkaConsumerLag
        expr: kafka_consumer_lag_messages > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High Kafka consumer lag"
```

### 4. Backup and Disaster Recovery

#### Automated Backup Scripts

```bash
#!/bin/bash
# scripts/backup-databases.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgresql/$TIMESTAMP"
S3_BUCKET="omninode-bridge-backups"
RETENTION_DAYS=30

echo "🗄️ Starting database backup - $TIMESTAMP"

# Create backup directory
mkdir -p $BACKUP_DIR

# PostgreSQL backup
echo "📁 Backing up PostgreSQL..."
pg_dump -h postgres-service -U postgres -d omninode_bridge \
    --verbose --format=custom \
    --file=$BACKUP_DIR/omninode_bridge.dump

# Compress backup
echo "🗜️ Compressing backup..."
tar -czf $BACKUP_DIR.tar.gz -C /backups/postgresql $TIMESTAMP

# Upload to S3
echo "☁️ Uploading to S3..."
aws s3 cp $BACKUP_DIR.tar.gz s3://$S3_BUCKET/postgresql/

# Cleanup local files
rm -rf $BACKUP_DIR $BACKUP_DIR.tar.gz

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
aws s3 ls s3://$S3_BUCKET/postgresql/ | \
    awk '{print $4}' | \
    sort -r | \
    tail -n +$((RETENTION_DAYS + 1)) | \
    xargs -I {} aws s3 rm s3://$S3_BUCKET/postgresql/{}

echo "✅ Database backup completed: $TIMESTAMP"
```

#### Disaster Recovery Plan

```markdown
# Disaster Recovery Plan

## RTO/RPO Objectives
- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)

## Recovery Scenarios

### Scenario 1: Single Service Failure
**Detection**: Health checks fail, alerts triggered
**Response Time**: < 5 minutes
**Actions**:
1. Restart failed pods
2. Check resource constraints
3. Rollback if recent deployment
4. Scale up if needed

### Scenario 2: Database Failure
**Detection**: Database connectivity lost
**Response Time**: < 30 minutes
**Actions**:
1. Failover to read replica
2. Promote replica to primary
3. Update service configurations
4. Restore from backup if corruption

### Scenario 3: Complete Cluster Failure
**Detection**: All services unreachable
**Response Time**: < 4 hours
**Actions**:
1. Provision new cluster
2. Restore from backups
3. Deploy services
4. Validate functionality
5. Update DNS records
```

This comprehensive deployment topology and operational procedures documentation provides everything needed for production deployment readiness, including detailed environment configurations, deployment procedures, operational runbooks, and disaster recovery plans.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Update OpenAPI specification to include missing WorkflowCoordinator endpoints", "status": "completed", "activeForm": "Updated OpenAPI specification with WorkflowCoordinator endpoints"}, {"content": "Create Architecture Decision Records (ADRs) for key design decisions", "status": "completed", "activeForm": "Created comprehensive ADRs for architectural decisions"}, {"content": "Add sequence diagrams for key workflow flows", "status": "completed", "activeForm": "Created sequence diagrams for workflow flows"}, {"content": "Document deployment topology and operational procedures", "status": "completed", "activeForm": "Documented deployment topology and operations"}, {"content": "Create performance benchmarks and SLA definitions", "status": "in_progress", "activeForm": "Creating performance benchmarks and SLA definitions"}]
