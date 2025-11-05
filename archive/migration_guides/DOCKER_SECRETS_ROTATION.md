# Docker Secrets Rotation Strategy

## Overview

This document outlines the automated secret rotation strategy for ONEX infrastructure services using Docker Swarm secrets and external secret management systems.

## Security Architecture

### Current Implementation

The infrastructure uses Docker Swarm secrets for secure credential management:

- **PostgreSQL credentials**: Stored as Docker secrets, read from `/run/secrets/postgres_password`
- **GitHub tokens**: Managed through Docker secrets for build processes
- **Future services**: Consul, Kafka, and Vault credentials will use the same pattern

### Secret Rotation Requirements

1. **Zero-downtime rotation**: Services must continue operating during credential updates
2. **Automated versioning**: Old credentials remain valid during transition periods
3. **Audit logging**: All rotation events must be logged for compliance
4. **Rollback capability**: Ability to revert to previous credentials if issues occur

## Implementation Strategy

### 1. Docker Secrets Versioning Pattern

```yaml
secrets:
  postgres_password_v1:
    environment: "POSTGRES_PASSWORD"
  postgres_password_v2:
    environment: "POSTGRES_PASSWORD_NEW"
    
  # GitHub token rotation
  github_token_v1:
    environment: "GITHUB_TOKEN"
  github_token_v2:
    environment: "GITHUB_TOKEN_NEW"
```

### 2. Service Configuration Updates

Services are configured to read from multiple secret versions during rotation:

```yaml
postgres-adapter:
  secrets:
    - postgres_password_v1
    - postgres_password_v2
  environment:
    # Primary credential (current)
    POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password_v1
    # Secondary credential (for rotation)
    POSTGRES_PASSWORD_FILE_NEW: /run/secrets/postgres_password_v2
```

### 3. Application-Level Secret Management

The PostgreSQL connection manager implements graceful credential rotation:

```python
def _read_rotated_password(self) -> str:
    """Read password with rotation support."""
    # Try primary password file
    primary_file = os.getenv("POSTGRES_PASSWORD_FILE")
    if primary_file and os.path.exists(primary_file):
        try:
            with open(primary_file, 'r') as f:
                return f.read().strip()
        except Exception:
            pass
    
    # Fall back to secondary password file during rotation
    secondary_file = os.getenv("POSTGRES_PASSWORD_FILE_NEW")
    if secondary_file and os.path.exists(secondary_file):
        try:
            with open(secondary_file, 'r') as f:
                return f.read().strip()
        except Exception:
            pass
    
    # Final fallback to environment variable (less secure)
    return os.getenv("POSTGRES_PASSWORD", "")
```

## Rotation Workflow

### Phase 1: Preparation

1. **Generate new credentials** in external secret management system
2. **Update Docker secrets** with versioned naming convention
3. **Deploy updated service configurations** with dual secret mounts

### Phase 2: Validation

1. **Health checks** verify new credentials work correctly
2. **Connection testing** ensures database accepts new passwords
3. **Rollback preparation** keeps old credentials available

### Phase 3: Cutover

1. **Update environment variables** to point to new secret files
2. **Restart services** to pick up new credential configuration
3. **Monitor service health** during transition

### Phase 4: Cleanup

1. **Remove old credentials** after successful rotation validation
2. **Update secret versions** for next rotation cycle
3. **Audit log completion** with rotation success confirmation

## Automated Rotation Schedule

### PostgreSQL Credentials
- **Rotation frequency**: Every 90 days
- **Automated trigger**: Kubernetes CronJob or GitHub Actions workflow
- **Validation window**: 24 hours overlap period

### GitHub Tokens
- **Rotation frequency**: Every 30 days
- **Automated trigger**: GitHub Apps token refresh
- **Validation window**: 2 hours overlap period

### Service Certificates
- **Rotation frequency**: Every 365 days
- **Automated trigger**: Cert-manager or external CA
- **Validation window**: 48 hours overlap period

## Monitoring and Alerting

### Success Metrics
- Rotation completion time < 5 minutes
- Zero service downtime during rotation
- All health checks pass within 30 seconds

### Failure Scenarios
- **Credential validation failure**: Automatic rollback to previous version
- **Service health degradation**: Alert operations team immediately
- **Audit log gaps**: Compliance violation reporting

### Observability Integration

```python
# Example rotation monitoring
from omnibase_infra.infrastructure.infrastructure_observability import (
    InfrastructureObservability,
    MetricType
)

async def log_rotation_event(service: str, status: str, duration_ms: float):
    """Log credential rotation events for monitoring."""
    observability = InfrastructureObservability()
    
    await observability.record_metric(
        metric_name=f"credential_rotation_{service}_{status}",
        value=duration_ms,
        metric_type=MetricType.HISTOGRAM,
        labels={
            "service": service,
            "rotation_status": status,
            "environment": os.getenv("NODE_ENV", "production")
        }
    )
```

## Security Best Practices

### 1. Credential Generation
- **High entropy**: Minimum 256-bit random passwords
- **No reuse**: Each rotation generates completely new credentials
- **Secure storage**: All credentials encrypted at rest

### 2. Access Control
- **Principle of least privilege**: Services access only required secrets
- **Role-based permissions**: Different rotation rights for different services
- **Audit trails**: All secret access logged and monitored

### 3. Network Security
- **TLS encryption**: All credential transmission uses TLS 1.3+
- **Certificate pinning**: Prevent man-in-the-middle attacks
- **Network isolation**: Secrets only accessible within service mesh

## Compliance Requirements

### SOC 2 Type II
- Quarterly rotation audit reports
- Continuous monitoring documentation
- Incident response procedures

### ISO 27001
- Risk assessment updates for rotation procedures
- Security controls validation
- Business continuity testing

## Emergency Procedures

### Credential Compromise Response
1. **Immediate revocation** of compromised credentials
2. **Force rotation** across all affected services
3. **Security incident logging** and stakeholder notification

### Rotation Failure Recovery
1. **Automatic rollback** to previous working credentials
2. **Service health validation** after rollback
3. **Root cause analysis** and procedure updates

## Implementation Checklist

- [x] PostgreSQL Docker secrets implementation
- [x] Connection manager rotation support
- [ ] GitHub token rotation automation
- [ ] Consul credential rotation
- [ ] Kafka authentication rotation
- [ ] Vault seal key rotation
- [ ] Certificate rotation automation
- [ ] Monitoring dashboard creation
- [ ] Compliance audit preparation

## Future Enhancements

### 1. External Secret Management Integration
- **HashiCorp Vault**: Full secret lifecycle management
- **AWS Secrets Manager**: Cloud-native rotation
- **Azure Key Vault**: Enterprise integration

### 2. Advanced Rotation Strategies
- **Blue/green credential deployment**: Zero-downtime guarantees
- **Canary rotation**: Gradual rollout validation
- **Multi-region coordination**: Global secret synchronization

### 3. Security Improvements
- **Hardware security modules**: Credential generation and storage
- **Zero-trust architecture**: Continuous credential verification
- **Quantum-safe cryptography**: Future-proof security

## References

- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [NIST SP 800-57 Key Management](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)