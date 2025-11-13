# Security Implementation Guide

## Overview

This document provides comprehensive security implementation guidelines for the OmniNode Bridge system, covering authentication, authorization, rate limiting, audit logging, and infrastructure security.

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Rate Limiting](#rate-limiting)
3. [Audit Logging](#audit-logging)
4. [Infrastructure Security](#infrastructure-security)
5. [SSL/TLS Configuration](#ssltls-configuration)
6. [Security Monitoring](#security-monitoring)
7. [Incident Response](#incident-response)

## Authentication & Authorization

### API Key Authentication

All OmniNode Bridge services use API key authentication with dual header support:

```yaml
Security Methods:
  - Authorization: Bearer <api-key>
  - X-API-Key: <api-key>
```

#### Implementation Details

**Environment Configuration:**
```bash
# Primary API key (required)
API_KEY=omninode-bridge-api-key-2024

# Optional: Multiple API keys for different clients
API_KEYS_JSON='["key1", "key2", "key3"]'

# Key rotation schedule (recommended: 90 days)
API_KEY_ROTATION_DAYS=90
```

**Code Implementation:**
```python
# From hook_receiver.py and model_metrics_api.py
async def verify_api_key(
    authorization: HTTPAuthorizationCredentials | None = Depends(security),
    x_api_key: str | None = Header(None),
) -> bool:
    """Verify API key from Authorization header or X-API-Key header."""
    provided_key = None

    if authorization and authorization.scheme.lower() == "bearer":
        provided_key = authorization.credentials
    elif x_api_key:
        provided_key = x_api_key

    if not provided_key or provided_key != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
        )
    return True
```

#### Security Best Practices

1. **Key Management:**
   - Store API keys in environment variables, never in code
   - Use different keys for different environments (dev, staging, prod)
   - Implement key rotation every 90 days
   - Monitor key usage for anomalies

2. **Key Format:**
   - Minimum 32 characters
   - Include random alphanumeric characters
   - Consider adding environment prefix: `prod-omninode-bridge-xyz123`

3. **Access Control:**
   - Implement role-based access for different key types
   - Consider separate keys for read-only vs read-write operations
   - Log all authentication attempts for security monitoring

### Multi-Factor Authentication (Recommended Enhancement)

For production environments, consider implementing MFA for administrative access:

```python
# Future enhancement: JWT with MFA
from jose import JWTError, jwt
from passlib.context import CryptContext

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

async def verify_jwt_token(token: str) -> dict:
    """Verify JWT token with MFA claims."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("mfa_verified") is not True:
            raise HTTPException(status_code=401, detail="MFA required")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

## Rate Limiting

### Current Implementation

OmniNode Bridge implements sophisticated rate limiting using `slowapi` library:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
```

### Service-Specific Rate Limits

#### HookReceiver Service
```python
@app.post("/hooks")
@limiter.limit("100/minute")  # 100 hook requests per minute per IP
async def receive_hook(request: Request, _: bool = Depends(verify_api_key)):
    # Hook processing logic

@app.get("/sessions")
@limiter.limit("30/minute")  # 30 session queries per minute per IP
async def get_active_sessions(_: bool = Depends(verify_api_key)):
    # Session retrieval logic

@app.post("/sessions/{session_id}/end")
@limiter.limit("10/minute")  # 10 session end requests per minute per IP
async def end_session(session_id: UUID, _: bool = Depends(verify_api_key)):
    # Session termination logic
```

#### ModelMetrics API
```python
@app.post("/execute")
@limiter.limit("20/minute")  # 20 task executions per minute per IP
async def execute_task(request: TaskRequest, req: Request):
    # Task execution logic

@app.post("/compare")
@limiter.limit("5/minute")  # 5 model comparisons per minute per IP (resource intensive)
async def compare_models(request: ModelComparisonRequest, req: Request):
    # Model comparison logic
```

### Rate Limiting Strategy

#### Tier-Based Limits
```yaml
Rate Limit Tiers:
  Free Tier:
    - Hooks: 50/minute
    - Executions: 10/minute
    - Comparisons: 2/minute

  Standard Tier:
    - Hooks: 100/minute
    - Executions: 20/minute
    - Comparisons: 5/minute

  Premium Tier:
    - Hooks: 500/minute
    - Executions: 100/minute
    - Comparisons: 25/minute
```

#### Advanced Rate Limiting Features

1. **Sliding Window Implementation:**
```python
# Enhanced rate limiter with sliding window
from redis import Redis
import time

class SlidingWindowRateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if request is allowed under sliding window rate limit."""
        now = time.time()
        pipeline = self.redis.pipeline()

        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window_seconds)

        # Count current requests
        pipeline.zcard(key)

        # Add current request
        pipeline.zadd(key, {str(now): now})

        # Set expiration
        pipeline.expire(key, window_seconds)

        results = pipeline.execute()
        current_count = results[1]

        return current_count < limit
```

2. **Dynamic Rate Limiting:**
```python
# Adjust limits based on system load
async def get_dynamic_rate_limit(base_limit: int, system_load: float) -> int:
    """Adjust rate limits based on system performance."""
    if system_load > 0.8:
        return int(base_limit * 0.5)  # Reduce by 50% under high load
    elif system_load > 0.6:
        return int(base_limit * 0.7)  # Reduce by 30% under medium load
    return base_limit
```

### Rate Limit Monitoring

```python
# Prometheus metrics for rate limiting
RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Rate limit exceeded count",
    ["endpoint", "client_ip"]
)

# Rate limit monitoring
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Monitor and log rate limiting events."""
    try:
        response = await call_next(request)
        return response
    except RateLimitExceeded as e:
        client_ip = get_remote_address(request)
        endpoint = request.url.path

        RATE_LIMIT_EXCEEDED.labels(
            endpoint=endpoint,
            client_ip=client_ip
        ).inc()

        logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            endpoint=endpoint,
            user_agent=request.headers.get("User-Agent"),
            rate_limit=str(e)
        )

        raise
```

## Audit Logging

### Structured Logging Implementation

OmniNode Bridge uses `structlog` for comprehensive audit logging:

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
```

### Audit Event Categories

#### 1. Authentication Events
```python
# Successful authentication
logger.info(
    "Authentication successful",
    event_type="auth_success",
    client_ip=request.client.host,
    user_agent=request.headers.get("User-Agent"),
    endpoint=request.url.path,
    method=request.method,
    api_key_hash=hashlib.sha256(api_key.encode()).hexdigest()[:8]
)

# Failed authentication
logger.warning(
    "Authentication failed",
    event_type="auth_failure",
    client_ip=request.client.host,
    user_agent=request.headers.get("User-Agent"),
    endpoint=request.url.path,
    method=request.method,
    reason="invalid_api_key",
    provided_key_hash=hashlib.sha256(provided_key.encode()).hexdigest()[:8] if provided_key else None
)
```

#### 2. Hook Processing Events
```python
# Hook received and processed
logger.info(
    "Hook processed successfully",
    event_type="hook_processed",
    event_id=str(hook_event.id),
    source=hook_event.metadata.source,
    action=hook_event.payload.action,
    resource=hook_event.payload.resource,
    resource_id=hook_event.payload.resource_id,
    processing_time_ms=processing_time,
    correlation_id=hook_event.metadata.correlation_id,
    client_ip=request.client.host,
    success=all_published
)

# Hook processing failure
logger.error(
    "Hook processing failed",
    event_type="hook_error",
    event_id=str(hook_event.id),
    source=hook_event.metadata.source,
    action=hook_event.payload.action,
    error=str(e),
    client_ip=request.client.host
)
```

#### 3. Model Execution Events
```python
# Task execution
logger.info(
    "Task executed",
    event_type="task_execution",
    execution_id=str(execution_metrics.execution_id),
    task_type=request.task_type.value,
    model_used=response_data.get("model"),
    latency_ms=execution_metrics.latency_ms,
    success=execution_metrics.success,
    quality_score=execution_metrics.quality_score,
    client_ip=req.client.host,
    complexity=request.complexity
)
```

#### 4. Security Events
```python
# Rate limit exceeded
logger.warning(
    "Rate limit exceeded",
    event_type="rate_limit_exceeded",
    client_ip=client_ip,
    endpoint=endpoint,
    user_agent=request.headers.get("User-Agent"),
    rate_limit=str(e),
    timestamp=datetime.utcnow().isoformat()
)

# Suspicious activity detection
logger.error(
    "Suspicious activity detected",
    event_type="security_alert",
    client_ip=client_ip,
    pattern="multiple_failed_auth",
    count=failed_attempts,
    time_window_minutes=time_window,
    user_agent=user_agent
)
```

### Log Storage and Retention

#### Centralized Logging
```yaml
# docker-compose.yml logging configuration
services:
  hook-receiver:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=hook-receiver,environment=production"

  model-metrics:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=model-metrics,environment=production"
```

#### Log Forwarding (Recommended)
```yaml
# Example: ELK Stack integration
version: '3.8'
services:
  filebeat:
    image: docker.elastic.co/beats/filebeat:7.15.0
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
    environment:
      - ELASTICSEARCH_HOST=elasticsearch:9200
```

### Audit Log Analysis

#### Key Metrics to Monitor
```python
# Security metrics for monitoring
FAILED_AUTH_ATTEMPTS = Counter(
    "failed_auth_attempts_total",
    "Failed authentication attempts",
    ["client_ip", "endpoint"]
)

HOOK_PROCESSING_ERRORS = Counter(
    "hook_processing_errors_total",
    "Hook processing errors",
    ["source", "error_type"]
)

SUSPICIOUS_ACTIVITY = Counter(
    "suspicious_activity_total",
    "Suspicious activity alerts",
    ["activity_type", "client_ip"]
)
```

#### Automated Alert Rules
```yaml
# Example alerting rules (Prometheus/Grafana)
groups:
  - name: security_alerts
    rules:
      - alert: HighFailedAuthRate
        expr: rate(failed_auth_attempts_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High rate of failed authentication attempts"
          description: "{{ $labels.client_ip }} has {{ $value }} failed auth attempts per second"

      - alert: RateLimitExceeded
        expr: rate(rate_limit_exceeded_total[5m]) > 0.05
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Rate limit frequently exceeded"
          description: "Rate limit exceeded {{ $value }} times per second"
```

## Infrastructure Security

### Network Security

#### Firewall Configuration
```bash
# UFW firewall rules for production
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port from default 22)
sudo ufw allow 2222/tcp

# Allow specific service ports
sudo ufw allow 8001/tcp  # HookReceiver
sudo ufw allow 8002/tcp  # ModelMetrics
sudo ufw allow 8003/tcp  # WorkflowCoordinator

# Allow internal communication
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 9092  # Kafka
sudo ufw allow from 10.0.0.0/8 to any port 8500  # Consul

sudo ufw enable
```

#### VPC Configuration (AWS Example)
```yaml
# terraform/vpc.tf
resource "aws_vpc" "omninode_bridge" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "omninode-bridge-vpc"
    Environment = "production"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.omninode_bridge.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "omninode-bridge-private-${count.index + 1}"
    Type = "Private"
  }
}

resource "aws_security_group" "api_services" {
  name_prefix = "omninode-bridge-api-"
  vpc_id      = aws_vpc.omninode_bridge.id

  ingress {
    from_port   = 8001
    to_port     = 8003
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### Container Security

#### Dockerfile Security Best Practices
```dockerfile
# Use specific versions, not latest
FROM python:3.12.1-slim

# Create non-root user
RUN groupadd -r omninode && useradd --no-log-init -r -g omninode omninode

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=omninode:omninode . .

# Switch to non-root user
USER omninode

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["uvicorn", "omninode_bridge.services.hook_receiver:create_app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Container Runtime Security
```yaml
# docker-compose.yml security configurations
version: '3.8'
services:
  hook-receiver:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    user: "1001:1001"
```

### Database Security

#### PostgreSQL Security Configuration
```sql
-- Create dedicated database user
CREATE USER omninode_bridge WITH ENCRYPTED PASSWORD 'strong_password_here';  -- pragma: allowlist secret

-- Create database
CREATE DATABASE omninode_bridge OWNER omninode_bridge;

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE omninode_bridge TO omninode_bridge;
GRANT USAGE ON SCHEMA public TO omninode_bridge;
GRANT CREATE ON SCHEMA public TO omninode_bridge;

-- Enable row-level security
ALTER TABLE hook_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE service_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY hook_events_policy ON hook_events
    FOR ALL TO omninode_bridge
    USING (source = current_user OR source IS NULL);
```

#### Database Connection Security
```python
# PostgreSQL connection with SSL
import ssl
import asyncpg

async def create_secure_connection():
    """Create secure PostgreSQL connection."""
    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ssl_context.check_hostname = False  # Only for self-signed certificates
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    connection = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        ssl=ssl_context,
        command_timeout=30,
        server_settings={
            'application_name': 'omninode_bridge',
            'jit': 'off'  # Disable JIT for security
        }
    )
    return connection
```

## SSL/TLS Configuration

### Certificate Management

#### Let's Encrypt Integration
```bash
#!/bin/bash
# scripts/setup-ssl.sh

# Install certbot
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx

# Generate certificates
sudo certbot certonly --standalone \
    --email admin@omninode.ai \
    --agree-tos \
    --no-eff-email \
    -d api.omninode.ai \
    -d hooks.omninode.ai \
    -d metrics.omninode.ai

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Nginx SSL Configuration
```nginx
# /etc/nginx/sites-available/omninode-bridge
server {
    listen 80;
    server_name api.omninode.ai hooks.omninode.ai metrics.omninode.ai;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name hooks.omninode.ai;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/hooks.omninode.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hooks.omninode.ai/privkey.pem;

    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Proxy to HookReceiver service
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Security
        proxy_hide_header X-Powered-By;
        proxy_set_header X-Forwarded-Host $host;
    }
}
```

#### Application-Level TLS
```python
# TLS configuration for FastAPI
import ssl

def create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for HTTPS."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile="/etc/ssl/certs/omninode-bridge.crt",
        keyfile="/etc/ssl/private/omninode-bridge.key"
    )

    # Security configurations
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
    context.set_alpn_protocols(["h2", "http/1.1"])

    return context

# Run with SSL
if __name__ == "__main__":
    ssl_context = create_ssl_context()
    uvicorn.run(
        "omninode_bridge.services.hook_receiver:create_app",
        host="0.0.0.0",
        port=8001,
        ssl_context=ssl_context,
        access_log=True
    )
```

## Security Monitoring

### Real-Time Monitoring

#### Security Dashboard Metrics
```python
# Security monitoring metrics
SECURITY_EVENTS = Counter(
    "security_events_total",
    "Security events detected",
    ["event_type", "severity", "source"]
)

FAILED_LOGINS = Counter(
    "failed_logins_total",
    "Failed login attempts",
    ["client_ip", "user_agent_hash"]
)

RATE_LIMIT_VIOLATIONS = Histogram(
    "rate_limit_violations",
    "Rate limit violations",
    ["endpoint", "violation_severity"]
)

# Intrusion detection
async def detect_suspicious_activity(request: Request):
    """Detect and log suspicious activity patterns."""
    client_ip = get_remote_address(request)
    user_agent = request.headers.get("User-Agent", "")

    # Check for common attack patterns
    suspicious_patterns = [
        "sqlmap", "nikto", "dirb", "gobuster",
        "../", "script>", "union select"
    ]

    for pattern in suspicious_patterns:
        if pattern.lower() in str(request.url).lower() or pattern.lower() in user_agent.lower():
            SECURITY_EVENTS.labels(
                event_type="suspicious_pattern",
                severity="medium",
                source=client_ip
            ).inc()

            logger.warning(
                "Suspicious activity detected",
                client_ip=client_ip,
                pattern=pattern,
                url=str(request.url),
                user_agent=user_agent
            )
            break
```

#### Automated Threat Response
```python
# Automated blocking for repeat offenders
from typing import Dict, Set
import time

class ThreatResponseSystem:
    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.failed_attempts: Dict[str, list] = {}
        self.block_duration = 3600  # 1 hour

    async def record_failed_attempt(self, ip: str):
        """Record failed authentication attempt."""
        now = time.time()

        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []

        # Clean old attempts (older than 1 hour)
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip]
            if now - attempt < 3600
        ]

        self.failed_attempts[ip].append(now)

        # Block if more than 10 failed attempts in 1 hour
        if len(self.failed_attempts[ip]) > 10:
            self.blocked_ips.add(ip)

            logger.error(
                "IP blocked due to excessive failed attempts",
                ip=ip,
                attempts=len(self.failed_attempts[ip]),
                blocked_until=now + self.block_duration
            )

    def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        return ip in self.blocked_ips

# Middleware for threat response
@app.middleware("http")
async def threat_response_middleware(request: Request, call_next):
    """Implement automated threat response."""
    client_ip = get_remote_address(request)

    if threat_system.is_blocked(client_ip):
        raise HTTPException(
            status_code=403,
            detail="IP blocked due to suspicious activity"
        )

    return await call_next(request)
```

## Incident Response

### Security Incident Response Plan

#### 1. Detection and Analysis
```python
# Incident detection triggers
class SecurityIncident:
    def __init__(self, incident_type: str, severity: str, details: dict):
        self.incident_type = incident_type
        self.severity = severity
        self.details = details
        self.timestamp = datetime.utcnow()
        self.incident_id = str(uuid.uuid4())

    async def trigger_response(self):
        """Trigger appropriate incident response."""
        if self.severity in ["high", "critical"]:
            await self.notify_security_team()
            await self.implement_containment()

        await self.log_incident()

    async def notify_security_team(self):
        """Send immediate notification to security team."""
        notification = {
            "incident_id": self.incident_id,
            "type": self.incident_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }

        # Send to monitoring system (PagerDuty, Slack, etc.)
        await send_security_alert(notification)
```

#### 2. Containment Procedures
```bash
#!/bin/bash
# scripts/incident-containment.sh

# Emergency containment procedures

# 1. Block malicious IP immediately
block_ip() {
    local ip=$1
    sudo ufw deny from $ip
    echo "Blocked IP: $ip" | logger -t security-incident
}

# 2. Disable compromised API key
disable_api_key() {
    local key_hash=$1
    # Add to disabled keys list
    echo $key_hash >> /etc/omninode/disabled_keys.txt
    # Restart services to pick up changes
    docker-compose restart
}

# 3. Isolate affected services
isolate_service() {
    local service=$1
    docker-compose stop $service
    echo "Isolated service: $service" | logger -t security-incident
}

# 4. Create forensic snapshot
create_forensic_snapshot() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    docker exec postgres pg_dump omninode_bridge > /var/backups/forensic_${timestamp}.sql
    docker logs hook-receiver > /var/logs/forensic_hook_receiver_${timestamp}.log
    docker logs model-metrics > /var/logs/forensic_model_metrics_${timestamp}.log
}
```

#### 3. Recovery Procedures
```yaml
# Recovery checklist
Recovery Steps:
  1. Assess Damage:
     - Review audit logs for impact scope
     - Check data integrity
     - Verify system functionality

  2. System Restoration:
     - Restore from clean backups if needed
     - Update security configurations
     - Patch vulnerabilities

  3. Security Hardening:
     - Rotate all API keys
     - Update firewall rules
     - Enhance monitoring

  4. Validation:
     - Perform security scan
     - Test all functionality
     - Verify logs are clean

  5. Documentation:
     - Document incident timeline
     - Update security procedures
     - Conduct post-incident review
```

### Continuous Security Improvement

#### Security Scanning Integration
```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Run Bandit security linter
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-results.json

      - name: Upload results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

#### Security Metrics and KPIs
```python
# Security KPI tracking
SECURITY_METRICS = {
    "mean_time_to_detection": Histogram("security_detection_time_seconds"),
    "mean_time_to_response": Histogram("security_response_time_seconds"),
    "false_positive_rate": Gauge("security_false_positive_ratio"),
    "incidents_per_month": Counter("security_incidents_monthly"),
    "vulnerability_scan_score": Gauge("vulnerability_scan_score")
}

async def calculate_security_score() -> float:
    """Calculate overall security posture score."""
    factors = {
        "ssl_grade": 0.2,      # SSL Labs grade
        "vulnerability_score": 0.3,  # Vulnerability scan results
        "incident_frequency": 0.2,   # Frequency of security incidents
        "compliance_score": 0.3      # Compliance audit results
    }

    # Implementation would fetch actual scores
    total_score = sum(factors.values())
    return min(100.0, total_score * 100)
```

This comprehensive security implementation guide provides a robust foundation for securing the OmniNode Bridge system across all layers of the infrastructure stack.
