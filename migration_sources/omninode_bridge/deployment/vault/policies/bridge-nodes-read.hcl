# Bridge Nodes Read Policy
# Purpose: Grant read-only access to bridge nodes for retrieving secrets
# Author: OmniNode Bridge Team
# Version: 1.0.0

# Development environment secrets (read + list)
path "omninode/data/development/*" {
  capabilities = ["read", "list"]
}

# Production environment secrets (read + list)
path "omninode/data/production/*" {
  capabilities = ["read", "list"]
}

# Staging environment secrets (read + list)
path "omninode/data/staging/*" {
  capabilities = ["read", "list"]
}

# Metadata access for discovery
path "omninode/metadata/development/*" {
  capabilities = ["read", "list"]
}

path "omninode/metadata/production/*" {
  capabilities = ["read", "list"]
}

path "omninode/metadata/staging/*" {
  capabilities = ["read", "list"]
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

# Deny access to sensitive system paths
path "sys/*" {
  capabilities = ["deny"]
}

path "auth/*" {
  capabilities = ["deny"]
}
