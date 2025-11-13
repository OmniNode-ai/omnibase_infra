# Vault Infrastructure - OmniNode Bridge

## Overview

This directory contains Vault initialization scripts, policies, and secrets management tools for the OmniNode Bridge MVP foundation.

**Purpose**: Centralized secrets management for bridge nodes, services, and infrastructure components.

**Security Model**:
- Development: Read/write access for rapid iteration
- Staging: Read/write access for testing
- Production: Read-only access for bridge nodes (write via CI/CD only)

## Directory Structure

```
deployment/vault/
├── README.md                          # This file
├── seed_secrets.sh                    # Development secrets seeding script
├── policies/                          # Vault policies for access control
│   ├── bridge-nodes-read.hcl         # Read-only policy
│   └── bridge-nodes-write.hcl        # Write policy (dev/staging only)
└── keys/                              # Generated keys (gitignored)
    ├── init-keys.json                # Vault unseal keys (production)
    └── bridge-nodes-token.txt        # Bridge nodes access token
```

## Quick Start

### 1. Start Vault Container

```bash
# From project root
cd /Volumes/PRO-G40/Code/omninode_bridge
docker compose up -d vault

# Verify Vault is running
curl http://localhost:8200/v1/sys/health
```

### 2. Initialize Vault

```bash
# Set Vault environment variables
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=your-vault-token  # From deployment/.env

# Run initialization script
./deployment/scripts/init_vault.sh

# Script will:
# - Enable KV v2 secrets engine at omninode/
# - Create bridge-nodes-read and bridge-nodes-write policies
# - Generate access token for bridge nodes
# - Save token to deployment/vault/keys/bridge-nodes-token.txt
```

### 3. Seed Development Secrets

```bash
# Ensure .env file is configured
cat deployment/.env

# Run seeding script
./deployment/vault/seed_secrets.sh

# Script will populate:
# - omninode/development/* (PostgreSQL, Kafka, Consul, etc.)
# - omninode/staging/* (same structure for staging)
```

### 4. Verify Secrets

```bash
# List all secret paths
vault kv list omninode/development

# View specific secrets (masked passwords)
vault kv get omninode/development/postgres
vault kv get omninode/development/kafka
vault kv get omninode/development/consul
```

## Secrets Structure

### PostgreSQL (`omninode/{env}/postgres`)

```json
{
  "host": "192.168.86.200",
  "port": "5436",
  "database": "omninode_bridge",
  "username": "postgres",
  "password": "********",
  "max_connections": "50",
  "min_connections": "10"
}
```

### Kafka (`omninode/{env}/kafka`)

```json
{
  "bootstrap_servers": "192.168.86.200:9092",
  "enable_idempotence": "true",
  "acks": "all",
  "retries": "3",
  "compression_type": "snappy"
}
```

### Consul (`omninode/{env}/consul`)

```json
{
  "host": "192.168.86.200",
  "port": "28500",
  "datacenter": "omninode-bridge",
  "token": ""
}
```

### Service Config (`omninode/{env}/service_config`)

```json
{
  "log_level": "info",
  "environment": "development",
  "service_version": "1.0.0",
  "enable_metrics": "true",
  "enable_tracing": "true"
}
```

### OnexTree Intelligence (`omninode/{env}/onextree`)

```json
{
  "host": "192.168.86.200",
  "port": "8058",
  "api_url": "http://192.168.86.200:8058",
  "timeout_seconds": "30",
  "max_retries": "3"
}
```

### Authentication (`omninode/{env}/auth`)

```json
{
  "secret_key": "********",
  "algorithm": "HS256",
  "access_token_expire_minutes": "30",
  "refresh_token_expire_days": "7"
}
```

### Deployment (`omninode/{env}/deployment`)

```json
{
  "receiver_port": "8001",
  "allowed_ip_ranges": "192.168.86.0/24,10.0.0.0/8",
  "auth_secret_key": "********",
  "docker_host": "unix:///var/run/docker.sock"
}
```

## Policies

### bridge-nodes-read (Read-Only Access)

**Purpose**: Grant bridge nodes read access to all environment secrets.

**Capabilities**:
- ✅ Read secrets from development, staging, production
- ✅ List available secret paths
- ✅ Renew own token
- ❌ Create, update, or delete secrets
- ❌ Access system paths or auth configuration

**Usage**:
```bash
# Create token with read-only access
vault token create -policy=bridge-nodes-read -period=768h
```

### bridge-nodes-write (Write Access)

**Purpose**: Grant write access for development and staging environments only.

**Capabilities**:
- ✅ Full CRUD on development secrets
- ✅ Full CRUD on staging secrets
- ✅ Read-only access to production secrets
- ✅ Create child tokens with same policies
- ❌ Write to production environment (explicit deny)
- ❌ Modify auth or system configuration

**Security Note**: Production writes are explicitly denied in policy. Use CI/CD or manual operations for production secrets.

**Usage**:
```bash
# Create token with write access
vault token create -policy=bridge-nodes-write -period=768h
```

## Common Operations

### Update a Secret

```bash
# Update PostgreSQL password
vault kv put omninode/development/postgres password=new_password

# Update Kafka bootstrap servers
vault kv put omninode/development/kafka bootstrap_servers=new-server:9092
```

### Add a New Secret Path

```bash
# Add Redis secrets
vault kv put omninode/development/redis \
  host=192.168.86.200 \
  port=6379 \
  password=redis_password
```

### Rotate Tokens

```bash
# Generate new bridge nodes token
vault token create \
  -policy=bridge-nodes-read \
  -policy=bridge-nodes-write \
  -period=768h \
  -display-name="bridge-nodes" \
  -format=json | jq -r '.auth.client_token'

# Update .env file with new token
# Update bridge nodes configuration
docker compose restart orchestrator reducer registry
```

### Backup Secrets

```bash
# Export all development secrets
vault kv get -format=json omninode/development/postgres > backup-postgres.json
vault kv get -format=json omninode/development/kafka > backup-kafka.json

# Or use Vault snapshots (production)
vault operator raft snapshot save backup.snap
```

## Integration with Bridge Nodes

Bridge nodes retrieve secrets from Vault on startup using the generated token.

**Environment Variables**:
```bash
VAULT_ADDR=http://192.168.86.200:8200
VAULT_TOKEN=<token-from-keys-directory>
VAULT_ENABLED=true
VAULT_MOUNT_POINT=omninode
```

**Python Example**:
```python
import hvac

client = hvac.Client(
    url=os.getenv("VAULT_ADDR"),
    token=os.getenv("VAULT_TOKEN")
)

# Read PostgreSQL secrets
postgres_secrets = client.secrets.kv.v2.read_secret_version(
    path="development/postgres",
    mount_point="omninode"
)

host = postgres_secrets["data"]["data"]["host"]
password = postgres_secrets["data"]["data"]["password"]
```

## Production Deployment

### Pre-Production Checklist

- [ ] Run Vault in production mode (not dev mode)
- [ ] Configure TLS certificates for HTTPS
- [ ] Initialize Vault with `init_vault.sh` (sets DEV_MODE=false)
- [ ] Securely backup unseal keys to encrypted storage
- [ ] Rotate default tokens and passwords
- [ ] Configure auto-unseal (AWS KMS, Azure Key Vault, etc.)
- [ ] Set up Vault snapshots for disaster recovery
- [ ] Configure audit logging
- [ ] Restrict network access to Vault port (8200)

### Production Mode Differences

**Dev Mode** (current):
- Automatically initialized and unsealed
- In-memory storage (data lost on restart)
- Single root token
- HTTP only (no TLS)

**Production Mode**:
- Manual initialization required
- Persistent storage (consul, raft, etc.)
- Multiple unseal keys with threshold
- TLS/HTTPS required
- Audit logging enabled

### Migrating to Production

1. Update docker-compose.yml to use production Vault configuration
2. Configure persistent storage backend (Consul or Raft)
3. Run `init_vault.sh` with `DEV_MODE=false`
4. Securely store unseal keys
5. Configure TLS certificates
6. Enable audit logging
7. Test failover and recovery procedures

## Troubleshooting

### Vault Not Responding

```bash
# Check container status
docker ps | grep vault

# Check logs
docker logs omninode-bridge-vault

# Restart container
docker compose restart vault
```

### Invalid Token Error

```bash
# Check token validity
vault token lookup

# Renew token (if renewable)
vault token renew

# Generate new token
vault token create -policy=bridge-nodes-read -policy=bridge-nodes-write
```

### Secrets Not Found

```bash
# List available paths
vault kv list omninode/development

# Check if KV engine is enabled
vault secrets list

# Re-run initialization if needed
./deployment/scripts/init_vault.sh
```

### Permission Denied

```bash
# Check current token capabilities
vault token capabilities omninode/development/postgres

# Review policy
vault policy read bridge-nodes-read

# Ensure token has correct policies
vault token lookup
```

## Security Best Practices

1. **Never commit secrets**: Always use Vault or environment variables
2. **Rotate tokens regularly**: Use short-lived tokens with auto-renewal
3. **Principle of least privilege**: Use read-only tokens where possible
4. **Audit logging**: Enable and monitor Vault audit logs
5. **Secure unseal keys**: Store production unseal keys in separate secure locations
6. **TLS everywhere**: Always use HTTPS in production
7. **Network segmentation**: Restrict Vault access to trusted networks
8. **Regular backups**: Snapshot Vault data regularly

## References

- **Vault Documentation**: https://developer.hashicorp.com/vault/docs
- **KV Secrets Engine**: https://developer.hashicorp.com/vault/docs/secrets/kv/kv-v2
- **Policies**: https://developer.hashicorp.com/vault/docs/concepts/policies
- **Production Hardening**: https://developer.hashicorp.com/vault/tutorials/operations/production-hardening

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Vault logs: `docker logs omninode-bridge-vault`
3. Consult OmniNode Bridge documentation: `/docs/`
4. Open GitHub issue with relevant logs and environment details
