# Bridge Nodes Write Policy
# Purpose: Grant write access to bridge nodes for managing development secrets
# Author: OmniNode Bridge Team
# Version: 1.0.0
# Security: Write access restricted to development environment only

# Development environment secrets (full CRUD)
path "omninode/data/development/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Development metadata (full CRUD)
path "omninode/metadata/development/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Production environment secrets (read-only for write policy)
# Write operations in production should be done manually or via CI/CD
path "omninode/data/production/*" {
  capabilities = ["read", "list"]
}

# Staging environment secrets (full CRUD for testing)
path "omninode/data/staging/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Staging metadata (full CRUD)
path "omninode/metadata/staging/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Allow listing available secret paths
path "omninode/metadata" {
  capabilities = ["list"]
}

# Allow nodes to renew their own tokens
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Allow nodes to lookup their own token
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

# Allow nodes to create child tokens with same or less privileges
path "auth/token/create" {
  capabilities = ["update"]
  allowed_policies = ["bridge-nodes-read", "bridge-nodes-write"]
}

# Deny access to sensitive system paths
path "sys/auth/*" {
  capabilities = ["deny"]
}

path "sys/policy/*" {
  capabilities = ["deny"]
}

# Deny production write operations (explicit safety check)
path "omninode/data/production/*" {
  capabilities = ["deny"]
  denied_parameters = {
    "create" = []
    "update" = []
    "delete" = []
  }
  # Only allow read and list
  allowed_parameters = {
    "read" = []
    "list" = []
  }
}
