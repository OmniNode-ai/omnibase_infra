# ONEX Infrastructure (omnibase-infra)

Infrastructure automation, deployment orchestration, and operational tooling for the ONEX framework ecosystem.

## Overview

This repository contains the infrastructure automation and deployment tools for ONEX framework services. It provides tooling for provisioning, configuration management, monitoring, and operations across cloud providers and on-premises environments.

## Architecture Principles

- **Infrastructure as Code**: Version-controlled infrastructure definitions
- **ONEX Protocol Integration**: Follows the same protocol-driven patterns as omnibase-core
- **Container-First**: Kubernetes-native deployments with Docker containers
- **Security-First**: Automated security scanning and compliance checks
- **GitOps Workflow**: Infrastructure changes through Git with validation

## Repository Structure

```
src/omnibase_infra/
├── core/                              # Core infrastructure components
├── cli/                               # Command-line interfaces
├── automation/                        # Automation scripts and workflows
├── monitoring/                        # Monitoring and alerting configurations
└── security/                          # Security tooling and policies

terraform/                             # Terraform configurations
ansible/                              # Ansible playbooks
kubernetes/                           # Kubernetes manifests
docker/                              # Docker configurations
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/OmniNode-ai/omnibase-infra.git
cd omnibase-infra

# Install with Poetry
poetry install
```

### 2. Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install
```

## Integration with ONEX Ecosystem

This infrastructure repository integrates with:

- **omnibase-spi**: Protocol definitions for infrastructure services
- **omnibase-core**: Core framework services requiring infrastructure
- **ONEX Tools**: All ONEX tools requiring deployment and operational support

The exact integration patterns will follow the same protocol-driven architecture as established in the core omnibase repositories.

## Contributing

### Development Guidelines

1. **Infrastructure as Code**: All changes through version-controlled configurations
2. **Security First**: Security scanning and compliance checks for all changes
3. **Testing Required**: Infrastructure validation and testing before deployment
4. **Documentation**: Update documentation for all infrastructure changes
5. **Protocol-Driven**: Follow ONEX protocol patterns from omnibase-spi

### Code Standards

- **Python**: Black formatting, type hints, comprehensive testing
- **Terraform**: HCL formatting with `terraform fmt`
- **Ansible**: YAML formatting with proper variable naming
- **Security**: No secrets in code, all credentials through secure storage

## Security Documentation

- [Docker Secrets Rotation Strategy](docs/DOCKER_SECRETS_ROTATION.md): Comprehensive guide for automated credential rotation, security best practices, and compliance procedures.

## License

MIT License - see [LICENSE](LICENSE) file for details.
