# Environment topology instances

These files are the authoritative, secret-free deployment topology for database
semantics. They are parsed as
`omnibase_core.models.core.ModelDeploymentTopology`; unknown fields and unresolved
schema, owner, principal, binding, or ledger references fail closed.

`omnibase_infra` owns these instances because it already owns the typed Docker
service catalog and ships the topology consumer library. Docker projections under
`docker/catalog/database-topology/` and Kubernetes projections in
`omninode_infra/k8s/database-topology/` are generated or parity-validated consumers.
The host-local `~/.omnibase/topology.yaml` file is never read here and is only a
projection for local setup.

The files contain environment-variable names and Kubernetes/Docker service names,
never passwords, tokens, or DSN values. The `omninode_runtime` service and the
`omninode_runtime` PostgreSQL principal are separate namespaces: the service consumes
the `omninode_runtime_service` binding.
