# Secrets Management - Best Practices

**Version:** 1.0
**Last Updated:** October 2025
**Status:** Production Ready

## Overview

This document outlines best practices for managing secrets, API keys, credentials, and sensitive configuration data in the omninode_bridge code generation system.

**Security Principle:** Never hardcode secrets in source code. Always use environment variables or dedicated secret management systems.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Secret Management Systems](#secret-management-systems)
3. [Development vs Production](#development-vs-production)
4. [API Key Configuration](#api-key-configuration)
5. [Key Rotation](#key-rotation)
6. [Auditing and Monitoring](#auditing-and-monitoring)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## Environment Variables

### Basic Setup

**DO NOT** hardcode secrets in your code:

```python
# ❌ BAD - Never do this
api_key = "sk-proj-abc123def456..."  # pragma: allowlist secret
password = "SuperSecret123!"  # pragma: allowlist secret
```

**DO** use environment variables:

```python
# ✅ GOOD - Load from environment
import os

api_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("DATABASE_PASSWORD")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
```

### Example .env File

Create a `.env` file for local development (add to `.gitignore`):

```bash
# .env (NEVER commit this file)

# OpenAI API
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_ORG_ID=org-your-org-id

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname  # pragma: allowlist secret
DATABASE_PASSWORD=your-db-password  # pragma: allowlist secret

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SASL_USERNAME=your-username
KAFKA_SASL_PASSWORD=your-password

# Service Authentication
JWT_SECRET_KEY=your-jwt-secret-key
API_KEY_SALT=your-api-key-salt

# External Services
GITHUB_TOKEN=ghp_your-github-token
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Feature Flags
ENABLE_AI_QUORUM=true
ENABLE_INTELLIGENCE_GATHERING=true
```

### Example .env.example File

Create a `.env.example` file as a template (safe to commit):

```bash
# .env.example - Template for environment variables

# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database  # pragma: allowlist secret
DATABASE_PASSWORD=your-secure-password  # pragma: allowlist secret
DATABASE_POOL_SIZE=20

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_SASL_USERNAME=your-username
KAFKA_SASL_PASSWORD=your-password

# Authentication
JWT_SECRET_KEY=generate-with-openssl-rand-base64-32
API_KEY_SALT=generate-with-openssl-rand-base64-16

# External Services
GITHUB_TOKEN=ghp_your-token-here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# AWS Configuration
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE  # pragma: allowlist secret
AWS_SECRET_ACCESS_KEY=your-secret-key  # pragma: allowlist secret
AWS_DEFAULT_REGION=us-east-1

# Feature Flags
ENABLE_AI_QUORUM=false
ENABLE_INTELLIGENCE_GATHERING=true
DEBUG_MODE=false
```

### Loading Environment Variables

Use `python-dotenv` for local development:

```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api_key = os.getenv("OPENAI_API_KEY")
```

In production, set environment variables directly in your deployment platform.

---

## Secret Management Systems

For production deployments, use dedicated secret management systems instead of plain environment variables.

### AWS Secrets Manager

**Setup:**

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str, region_name: str = "us-east-1") -> dict:
    """
    Retrieve secret from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret
        region_name: AWS region

    Returns:
        Secret value as dictionary
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        raise Exception(f"Failed to retrieve secret: {e}")

# Usage
secrets = get_secret("omninode/production/api-keys")
openai_key = secrets['openai_api_key']
db_password = secrets['database_password']
```

**Store secrets in AWS:**

```bash
# Create secret
aws secretsmanager create-secret \
    --name omninode/production/api-keys \
    --secret-string '{"openai_api_key":"sk-proj-...","database_password":"..."}'  # pragma: allowlist secret

# Update secret
aws secretsmanager update-secret \
    --secret-id omninode/production/api-keys \
    --secret-string '{"openai_api_key":"new-key"}'  # pragma: allowlist secret
```

### HashiCorp Vault

**Setup:**

```python
import hvac

def get_vault_secret(secret_path: str) -> dict:
    """
    Retrieve secret from HashiCorp Vault.

    Args:
        secret_path: Path to secret in Vault

    Returns:
        Secret data as dictionary
    """
    client = hvac.Client(
        url=os.getenv("VAULT_ADDR"),
        token=os.getenv("VAULT_TOKEN")
    )

    secret = client.secrets.kv.v2.read_secret_version(
        path=secret_path,
        mount_point='secret'
    )

    return secret['data']['data']

# Usage
secrets = get_vault_secret("omninode/production")
openai_key = secrets['openai_api_key']
```

**Store secrets in Vault:**

```bash
# Write secret
vault kv put secret/omninode/production \
    openai_api_key="sk-proj-..." \  # pragma: allowlist secret
    database_password="..."  # pragma: allowlist secret

# Read secret
vault kv get secret/omninode/production
```

### Google Cloud Secret Manager

**Setup:**

```python
from google.cloud import secretmanager

def get_gcp_secret(project_id: str, secret_id: str) -> str:
    """
    Retrieve secret from Google Cloud Secret Manager.

    Args:
        project_id: GCP project ID
        secret_id: Secret identifier

    Returns:
        Secret value as string
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
openai_key = get_gcp_secret("my-project", "openai-api-key")
```

---

## Development vs Production

### Development Environment

For local development, use `.env` files:

```bash
# .env.development
ENVIRONMENT=development
DEBUG_MODE=true
OPENAI_API_KEY=sk-test-...
DATABASE_URL=postgresql://localhost:5432/dev_db
```

**Load with python-dotenv:**

```python
from dotenv import load_dotenv

# Load development environment
load_dotenv('.env.development')
```

### Production Environment

For production, use secret management systems:

**Option 1: Kubernetes Secrets**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: omninode-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: sk-proj-your-key-here
  DATABASE_PASSWORD: your-password
```

**Option 2: AWS Parameter Store**

```bash
# Store parameter
aws ssm put-parameter \
    --name /omninode/production/openai-key \
    --value "sk-proj-..." \
    --type SecureString

# Retrieve in application
import boto3

ssm = boto3.client('ssm')
response = ssm.get_parameter(
    Name='/omninode/production/openai-key',
    WithDecryption=True
)
api_key = response['Parameter']['Value']
```

---

## API Key Configuration

### OpenAI API Keys

```python
from openai import OpenAI

# Load from environment
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

# Validate key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set")
```

### Database Credentials

```python
from sqlalchemy import create_engine

# Load from environment
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL must be set")

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=20
)
```

### Kafka Authentication

```python
from aiokafka import AIOKafkaProducer

producer = AIOKafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    security_protocol="SASL_SSL",
    sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
    sasl_plain_username=os.getenv("KAFKA_SASL_USERNAME"),
    sasl_plain_password=os.getenv("KAFKA_SASL_PASSWORD"),
)
```

---

## Key Rotation

### Automated Rotation Strategy

**Recommended Rotation Schedule:**
- API Keys: Every 90 days
- Database Passwords: Every 180 days
- JWT Secrets: Every 365 days
- Service Tokens: Every 30 days

### Rotation Procedure

1. **Generate new secret**
2. **Store new secret in secret manager**
3. **Update application to use new secret**
4. **Verify application works with new secret**
5. **Revoke old secret after grace period**

**Example Rotation Script:**

```bash
#!/bin/bash
# rotate-openai-key.sh

# Generate new key (manual step - get from OpenAI dashboard)
NEW_KEY="sk-proj-new-key-here"

# Update AWS Secrets Manager
aws secretsmanager update-secret \
    --secret-id omninode/production/openai-key \
    --secret-string "$NEW_KEY"

# Restart application to pick up new key
kubectl rollout restart deployment/omninode-bridge

# Wait for rollout
kubectl rollout status deployment/omninode-bridge

# Verify health
curl -f https://api.omninode.io/health || exit 1

echo "✅ Key rotation complete"
```

### Key Rotation with AWS Secrets Manager

Enable automatic rotation:

```python
import boto3

client = boto3.client('secretsmanager')

# Enable automatic rotation (requires Lambda function)
client.rotate_secret(
    SecretId='omninode/production/database',  # pragma: allowlist secret
    RotationLambdaARN='arn:aws:lambda:region:account:function:rotation-function',
    RotationRules={
        'AutomaticallyAfterDays': 90
    }
)
```

---

## Auditing and Monitoring

### Audit Logging

Log all secret access (but never log the secret values):

```python
import logging

logger = logging.getLogger(__name__)

def get_api_key(key_name: str) -> str:
    """Get API key with audit logging."""
    logger.info(
        "API key accessed",
        extra={
            "key_name": key_name,
            "user": get_current_user(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    return os.getenv(key_name)
```

### Monitoring Secret Usage

**CloudWatch Alarms (AWS):**

```bash
# Alert on secret retrieval failures
aws cloudwatch put-metric-alarm \
    --alarm-name omninode-secret-access-failures \
    --metric-name SecretRetrievalFailure \
    --namespace OmniNode \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold
```

### Security Scanning

**Pre-commit Hook to Detect Secrets:**

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: detect-private-key
      - id: detect-aws-credentials

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

**Run security scan:**

```bash
# Scan for hardcoded secrets
pip install detect-secrets
detect-secrets scan > .secrets.baseline

# Audit findings
detect-secrets audit .secrets.baseline
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Hardcoded Secrets

```python
# BAD
api_key = "sk-proj-abc123..."  # pragma: allowlist secret
password = "SuperSecret123!"  # pragma: allowlist secret
```

**Fix:** Use environment variables or secret managers.

### ❌ Mistake 2: Secrets in Git

```bash
# BAD - committing .env file
git add .env
git commit -m "Add environment config"
```

**Fix:** Add `.env` to `.gitignore`.

```bash
# .gitignore
.env
.env.*
!.env.example
```

### ❌ Mistake 3: Logging Secrets

```python
# BAD
logger.info(f"Using API key: {api_key}")
logger.debug(f"Database password: {password}")
```

**Fix:** Never log secret values.

```python
# GOOD
logger.info("API key loaded successfully")
logger.info(f"Using API key ending in: ...{api_key[-4:]}")
```

### ❌ Mistake 4: Secrets in Error Messages

```python
# BAD
raise Exception(f"Failed to connect with password: {password}")
```

**Fix:** Sanitize error messages.

```python
# GOOD
raise Exception("Failed to connect to database (check credentials)")
```

### ❌ Mistake 5: Default/Example Secrets

```python
# BAD
api_key = os.getenv("API_KEY", "default-key-12345")
```

**Fix:** Require secrets to be set, fail fast.

```python
# GOOD
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable must be set")
```

---

## Security Checklist

Use this checklist for code reviews:

- [ ] No hardcoded secrets in source code
- [ ] All secrets loaded from environment variables or secret managers
- [ ] `.env` files added to `.gitignore`
- [ ] `.env.example` template provided
- [ ] Secrets validated at startup (fail fast if missing)
- [ ] No secrets in log messages
- [ ] No secrets in error messages
- [ ] Secret access is audited
- [ ] Pre-commit hooks configured to detect secrets
- [ ] Key rotation procedure documented
- [ ] Production uses dedicated secret management system
- [ ] Secrets are encrypted at rest
- [ ] Secrets are encrypted in transit

---

## Additional Resources

- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [Google Cloud Secret Manager](https://cloud.google.com/secret-manager/docs)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12-Factor App: Config](https://12factor.net/config)

---

## Support

For questions about secrets management:
- Security Team: security@omninode.io
- Documentation: https://docs.omninode.io/security
- Slack: #security-questions
