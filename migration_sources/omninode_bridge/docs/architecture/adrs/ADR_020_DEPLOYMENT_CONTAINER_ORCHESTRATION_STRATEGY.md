# ADR-020: Deployment and Container Orchestration Strategy

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of production-ready deployment and container orchestration strategy for multi-service architecture

## Context

The multi-service architecture (HookReceiver, ModelMetrics API, WorkflowCoordinator) requires a robust deployment and container orchestration strategy that can handle:

- **Multi-Service Coordination**: Deploy and manage three interconnected services with proper dependency ordering
- **Infrastructure Dependencies**: Manage PostgreSQL, Kafka, Redis, and other external dependencies
- **Environment Promotion**: Support development, staging, and production environments with proper configuration management
- **Scaling and Load Management**: Handle variable load patterns with automatic and manual scaling capabilities
- **Zero-Downtime Deployments**: Deploy updates without service interruption
- **Health Monitoring**: Ensure services are healthy before routing traffic to new deployments
- **Rollback Capabilities**: Quick rollback mechanisms for failed deployments
- **Secret Management**: Secure handling of API keys, database credentials, and other sensitive data
- **Networking and Service Discovery**: Proper inter-service communication and load balancing
- **Resource Management**: Efficient resource allocation and utilization across services

Traditional deployment approaches often lack the sophistication needed for microservices, leading to deployment complexity, coordination challenges, and operational overhead.

## Decision

We adopt a **Container-First Orchestration Strategy** using Docker containers with Kubernetes orchestration and comprehensive CI/CD automation:

### Container Architecture

#### 1. Multi-Stage Docker Build Strategy
```dockerfile
# Hook Receiver Service Dockerfile
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Development stage
FROM base as development
RUN poetry install --with dev
COPY . .
CMD ["python", "-m", "omninode_bridge.hook_receiver.main"]

# Production stage
FROM base as production
RUN poetry install --only main
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Create non-root user
RUN useradd --create-home --shell /bin/bash omninode
RUN chown -R omninode:omninode /app
USER omninode

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "omninode_bridge.hook_receiver.main"]
```

#### 2. Service-Specific Container Configuration
```yaml
# docker-compose.yml for local development
version: '3.8'

services:
  # Infrastructure services
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: omninode_bridge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 10s
      timeout: 5s
      retries: 5

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Application services
  hook-receiver:
    build:
      context: .
      target: development
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/omninode_bridge  # pragma: allowlist secret
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - REDIS_URL=redis://redis:6379
      - API_KEY=dev-api-key-12345
      - ENVIRONMENT=development
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  model-metrics:
    build:
      context: .
      target: development
    ports:
      - "8002:8002"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/omninode_bridge  # pragma: allowlist secret
      - API_KEY=dev-api-key-12345
      - ENVIRONMENT=development
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  workflow-coordinator:
    build:
      context: .
      target: development
    ports:
      - "8003:8003"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/omninode_bridge  # pragma: allowlist secret
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - API_KEY=dev-api-key-12345
      - ENVIRONMENT=development
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_healthy
      hook-receiver:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

### Kubernetes Orchestration

#### 1. Production Deployment Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: omninode-bridge
  labels:
    environment: production
    app: omninode-bridge

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: omninode-config
  namespace: omninode-bridge
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  KAFKA_BOOTSTRAP_SERVERS: "kafka-cluster:9092"
  REDIS_URL: "redis://redis-service:6379"
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "omninode_bridge"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: omninode-secrets
  namespace: omninode-bridge
type: Opaque
data:
  API_KEY: <base64-encoded-api-key>
  DATABASE_PASSWORD: <base64-encoded-db-password>
  KAFKA_SASL_PASSWORD: <base64-encoded-kafka-password>

---
# k8s/hook-receiver-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hook-receiver
  namespace: omninode-bridge
  labels:
    app: hook-receiver
    service: hook-receiver
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: hook-receiver
  template:
    metadata:
      labels:
        app: hook-receiver
        service: hook-receiver
    spec:
      containers:
      - name: hook-receiver
        image: omninode-bridge/hook-receiver:latest
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: omninode-config
        - secretRef:
            name: omninode-secrets
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:$(DATABASE_PASSWORD)@$(DATABASE_HOST):$(DATABASE_PORT)/$(DATABASE_NAME)"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 10

---
# k8s/hook-receiver-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: hook-receiver-service
  namespace: omninode-bridge
  labels:
    app: hook-receiver
spec:
  selector:
    app: hook-receiver
  ports:
  - port: 8001
    targetPort: 8001
    protocol: TCP
  type: ClusterIP

---
# k8s/hook-receiver-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hook-receiver-hpa
  namespace: omninode-bridge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hook-receiver
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

#### 2. Infrastructure Services Deployment
```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: omninode-bridge
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "omninode_bridge"
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: omninode-secrets
              key: DATABASE_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: omninode-ingress
  namespace: omninode-bridge
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.omninode-bridge.com
    secretName: omninode-tls
  rules:
  - host: api.omninode-bridge.com
    http:
      paths:
      - path: /api/v1/hooks
        pathType: Prefix
        backend:
          service:
            name: hook-receiver-service
            port:
              number: 8001
      - path: /api/v1/model-metrics
        pathType: Prefix
        backend:
          service:
            name: model-metrics-service
            port:
              number: 8002
      - path: /api/v1/workflows
        pathType: Prefix
        backend:
          service:
            name: workflow-coordinator-service
            port:
              number: 8003
```

### CI/CD Pipeline

#### 1. GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy OmniNode Bridge

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_omninode_bridge
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true

    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}

    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --no-interaction --no-root

    - name: Install project
      run: poetry install --no-interaction

    - name: Run linting
      run: |
        poetry run black --check .
        poetry run isort --check-only .
        poetry run flake8 .
        poetry run mypy .

    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_omninode_bridge  # pragma: allowlist secret
        REDIS_URL: redis://localhost:6379
        API_KEY: test-api-key
        ENVIRONMENT: testing
      run: poetry run pytest --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Deploy to Kubernetes
      run: |
        # Update image tags in deployment
        sed -i "s|omninode-bridge/hook-receiver:latest|${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}|" k8s/hook-receiver-deployment.yaml
        sed -i "s|omninode-bridge/model-metrics:latest|${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}|" k8s/model-metrics-deployment.yaml
        sed -i "s|omninode-bridge/workflow-coordinator:latest|${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}|" k8s/workflow-coordinator-deployment.yaml

        # Apply Kubernetes manifests
        kubectl apply -f k8s/namespace.yaml
        kubectl apply -f k8s/configmap.yaml
        kubectl apply -f k8s/secrets.yaml
        kubectl apply -f k8s/postgres-statefulset.yaml
        kubectl apply -f k8s/kafka-deployment.yaml
        kubectl apply -f k8s/redis-deployment.yaml
        kubectl apply -f k8s/hook-receiver-deployment.yaml
        kubectl apply -f k8s/model-metrics-deployment.yaml
        kubectl apply -f k8s/workflow-coordinator-deployment.yaml
        kubectl apply -f k8s/services.yaml
        kubectl apply -f k8s/ingress.yaml
        kubectl apply -f k8s/hpa.yaml

        # Wait for deployments to be ready
        kubectl rollout status deployment/hook-receiver -n omninode-bridge --timeout=300s
        kubectl rollout status deployment/model-metrics -n omninode-bridge --timeout=300s
        kubectl rollout status deployment/workflow-coordinator -n omninode-bridge --timeout=300s
```

#### 2. Deployment Health Checks
```python
# scripts/deployment-health-check.py
"""Comprehensive health check script for deployment validation."""

import asyncio
import aiohttp
import sys
from typing import List, Dict

class DeploymentHealthChecker:
    """Validate deployment health across all services."""

    def __init__(self, base_urls: Dict[str, str]):
        self.base_urls = base_urls
        self.health_endpoints = {
            service: f"{url}/health"
            for service, url in base_urls.items()
        }

    async def check_service_health(self, service: str, url: str) -> Dict:
        """Check health of a specific service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            "service": service,
                            "status": "healthy",
                            "response_time": response.headers.get('X-Response-Time', 'unknown'),
                            "details": health_data
                        }
                    else:
                        return {
                            "service": service,
                            "status": "unhealthy",
                            "http_status": response.status,
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "service": service,
                "status": "unreachable",
                "error": str(e)
            }

    async def validate_deployment(self) -> bool:
        """Validate entire deployment health."""
        print("🔍 Validating deployment health...")

        # Check all services
        health_checks = []
        for service, url in self.health_endpoints.items():
            health_checks.append(self.check_service_health(service, url))

        results = await asyncio.gather(*health_checks)

        # Analyze results
        healthy_services = 0
        total_services = len(results)

        for result in results:
            service = result["service"]
            status = result["status"]

            if status == "healthy":
                print(f"✅ {service}: Healthy")
                healthy_services += 1
            elif status == "unhealthy":
                print(f"⚠️  {service}: Unhealthy (HTTP {result.get('http_status')})")
            else:
                print(f"❌ {service}: Unreachable ({result.get('error')})")

        # Summary
        health_percentage = (healthy_services / total_services) * 100
        print(f"\n📊 Deployment Health: {healthy_services}/{total_services} services healthy ({health_percentage:.1f}%)")

        if healthy_services == total_services:
            print("🎉 Deployment is fully healthy!")
            return True
        elif health_percentage >= 75:
            print("⚠️  Deployment is partially healthy but may have issues")
            return False
        else:
            print("❌ Deployment has critical health issues")
            return False

# Usage in CI/CD pipeline
if __name__ == "__main__":
    services = {
        "hook-receiver": "https://api.omninode-bridge.com/api/v1/hooks",
        "model-metrics": "https://api.omninode-bridge.com/api/v1/model-metrics",
        "workflow-coordinator": "https://api.omninode-bridge.com/api/v1/workflows"
    }

    checker = DeploymentHealthChecker(services)
    is_healthy = asyncio.run(checker.validate_deployment())

    if not is_healthy:
        sys.exit(1)  # Fail CI/CD pipeline if deployment is unhealthy
```

### Environment Management

#### 1. Environment-Specific Configuration
```yaml
# environments/development.yaml
environment: development
debug: true
log_level: DEBUG

database:
  host: localhost
  port: 5432
  name: omninode_bridge_dev
  pool_size: 5

kafka:
  bootstrap_servers: localhost:9092
  auto_create_topics: true

rate_limiting:
  enabled: false

authentication:
  api_key_rotation_hours: 720  # 30 days
  disable_auth: true

---
# environments/staging.yaml
environment: staging
debug: false
log_level: INFO

database:
  host: staging-postgres
  port: 5432
  name: omninode_bridge_staging
  pool_size: 10

kafka:
  bootstrap_servers: staging-kafka:9092
  auto_create_topics: false

rate_limiting:
  enabled: true
  multiplier: 2.0

authentication:
  api_key_rotation_hours: 336  # 14 days

monitoring:
  prometheus_enabled: true
  jaeger_enabled: true

---
# environments/production.yaml
environment: production
debug: false
log_level: WARNING

database:
  host: prod-postgres-cluster
  port: 5432
  name: omninode_bridge
  pool_size: 25
  ssl_mode: require

kafka:
  bootstrap_servers: prod-kafka-cluster:9092
  security_protocol: SASL_SSL
  auto_create_topics: false

rate_limiting:
  enabled: true
  multiplier: 1.0
  adaptive_enabled: true

authentication:
  api_key_rotation_hours: 168  # 7 days
  jwt_algorithm: RS256
  require_https: true

monitoring:
  prometheus_enabled: true
  jaeger_enabled: true
  alert_manager_enabled: true

security:
  security_headers_enabled: true
  suspicious_activity_detection: true
```

### Blue-Green Deployment Strategy

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

NAMESPACE="omninode-bridge"
NEW_VERSION=$1
CURRENT_COLOR=$(kubectl get service main-service -n $NAMESPACE -o jsonpath='{.spec.selector.color}')

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "🚀 Starting blue-green deployment..."
echo "📊 Current color: $CURRENT_COLOR"
echo "🔄 Deploying to: $NEW_COLOR"
echo "📦 Version: $NEW_VERSION"

# Deploy to new color
echo "🔨 Deploying $NEW_COLOR environment..."
sed "s/{{COLOR}}/$NEW_COLOR/g; s/{{VERSION}}/$NEW_VERSION/g" k8s/deployment-template.yaml | kubectl apply -f -

# Wait for new deployment to be ready
echo "⏳ Waiting for $NEW_COLOR deployment to be ready..."
kubectl rollout status deployment/hook-receiver-$NEW_COLOR -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/model-metrics-$NEW_COLOR -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/workflow-coordinator-$NEW_COLOR -n $NAMESPACE --timeout=300s

# Health check new deployment
echo "🏥 Running health checks on $NEW_COLOR environment..."
if python scripts/deployment-health-check.py --color=$NEW_COLOR; then
    echo "✅ Health checks passed!"
else
    echo "❌ Health checks failed! Aborting deployment."
    exit 1
fi

# Switch traffic to new deployment
echo "🔄 Switching traffic to $NEW_COLOR..."
kubectl patch service main-service -n $NAMESPACE -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'

# Wait and verify traffic switch
sleep 30
echo "🏥 Verifying traffic switch..."
if python scripts/deployment-health-check.py --color=$NEW_COLOR; then
    echo "✅ Traffic switch successful!"

    # Cleanup old deployment
    echo "🧹 Cleaning up old $CURRENT_COLOR deployment..."
    kubectl delete deployment hook-receiver-$CURRENT_COLOR -n $NAMESPACE --ignore-not-found=true
    kubectl delete deployment model-metrics-$CURRENT_COLOR -n $NAMESPACE --ignore-not-found=true
    kubectl delete deployment workflow-coordinator-$CURRENT_COLOR -n $NAMESPACE --ignore-not-found=true

    echo "🎉 Blue-green deployment completed successfully!"
else
    echo "❌ Traffic switch failed! Rolling back..."
    kubectl patch service main-service -n $NAMESPACE -p '{"spec":{"selector":{"color":"'$CURRENT_COLOR'"}}}'
    exit 1
fi
```

## Consequences

### Positive Consequences

- **Production Readiness**: Comprehensive deployment strategy supports enterprise production requirements
- **Scalability**: Kubernetes orchestration provides automatic scaling and resource management
- **Zero-Downtime Deployments**: Blue-green strategy eliminates deployment-related downtime
- **Environment Consistency**: Containerization ensures consistent environments across development, staging, and production
- **Automated Operations**: CI/CD pipeline automates testing, building, and deployment processes
- **Health Monitoring**: Comprehensive health checks ensure deployment quality and early issue detection
- **Rollback Capabilities**: Quick rollback mechanisms minimize impact of failed deployments
- **Infrastructure as Code**: All deployment configuration is version-controlled and repeatable
- **Security**: Secure secret management and network policies protect sensitive data

### Negative Consequences

- **Operational Complexity**: Kubernetes orchestration adds significant operational complexity
- **Resource Overhead**: Running multiple environments and blue-green deployments requires additional resources
- **Learning Curve**: Team must develop Kubernetes and container orchestration expertise
- **Debugging Challenges**: Distributed container environments can be challenging to debug
- **Cost Implications**: Infrastructure for multiple environments and scaling can be expensive
- **Network Complexity**: Service mesh and networking configuration adds complexity
- **Storage Management**: Persistent storage and data migration strategies require careful planning

### Mitigation Strategies

- **Training and Documentation**: Comprehensive training programs for operational teams
- **Monitoring and Observability**: Extensive monitoring reduces debugging complexity
- **Automated Testing**: Comprehensive test suites catch issues before production
- **Gradual Rollout**: Phased deployment of Kubernetes features with careful monitoring
- **Cost Optimization**: Regular review and optimization of resource utilization
- **Disaster Recovery**: Comprehensive backup and recovery procedures
- **Security Scanning**: Automated security scanning of containers and configurations

## Implementation Details

### Resource Requirements
```yaml
# Resource allocation guidelines
development:
  services:
    hook-receiver: { cpu: "100m", memory: "128Mi" }
    model-metrics: { cpu: "100m", memory: "128Mi" }
    workflow-coordinator: { cpu: "100m", memory: "128Mi" }
  infrastructure:
    postgres: { cpu: "200m", memory: "256Mi" }
    kafka: { cpu: "200m", memory: "512Mi" }
    redis: { cpu: "50m", memory: "64Mi" }

staging:
  services:
    hook-receiver: { cpu: "250m", memory: "256Mi" }
    model-metrics: { cpu: "250m", memory: "256Mi" }
    workflow-coordinator: { cpu: "250m", memory: "256Mi" }
  infrastructure:
    postgres: { cpu: "500m", memory: "1Gi" }
    kafka: { cpu: "500m", memory: "1Gi" }
    redis: { cpu: "100m", memory: "128Mi" }

production:
  services:
    hook-receiver: { cpu: "500m", memory: "512Mi" }
    model-metrics: { cpu: "500m", memory: "512Mi" }
    workflow-coordinator: { cpu: "500m", memory: "512Mi" }
  infrastructure:
    postgres: { cpu: "2", memory: "4Gi" }
    kafka: { cpu: "1", memory: "2Gi" }
    redis: { cpu: "200m", memory: "256Mi" }
```

### Monitoring Integration
```yaml
# Prometheus ServiceMonitor for each service
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: omninode-bridge-monitor
  namespace: omninode-bridge
spec:
  selector:
    matchLabels:
      app: omninode-bridge
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

## Compliance

This deployment strategy aligns with ONEX standards by:

- **Operational Excellence**: Automated deployment pipelines and comprehensive monitoring
- **Security**: Secure container images, secret management, and network policies
- **Reliability**: High availability, zero-downtime deployments, and automatic failover
- **Performance**: Resource optimization and automatic scaling capabilities
- **Cost Optimization**: Efficient resource utilization and environment management
- **Infrastructure as Code**: All deployment configuration version-controlled and reproducible

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-017: Authentication and Authorization Strategy
- ADR-018: Rate Limiting and API Security Strategy
- ADR-019: Monitoring and Observability Strategy

## References

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Multi-Stage Builds](https://docs.docker.com/develop/dev-best-practices/)
- [Blue-Green Deployment Pattern](https://martinfowler.com/bliki/BlueGreenDeployment.html)
- [GitOps Principles](https://www.gitops.tech/)
- [Container Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Kubernetes Networking](https://kubernetes.io/docs/concepts/cluster-administration/networking/)
