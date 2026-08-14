# Environment topology instances

These files are the authoritative, secret-free deployment topology for database
semantics. They are parsed as
`omnibase_core.models.core.ModelDeploymentTopology`; unknown fields and unresolved
schema, owner, principal, binding, or ledger references fail closed.

`omnibase_infra` owns these instances because it already owns the typed Docker
service catalog and ships the topology consumer library. Docker projections under
`docker/catalog/database-topology/` and Kubernetes projections in
`omninode_infra/topology/kubernetes/` are generated or parity-validated consumers.
The host-local `~/.omnibase/topology.yaml` file is never read here and is only a
projection for local setup.

`../application_database_profiles.yaml` is the typed, fail-closed mapping from
deployment profiles to these instances. The exact supported database profiles are
`local`, `test`, `stability-test`, `judge`, `prod`, `onex-dev`, and `onex-prod`.
The five Docker profiles intentionally share the `local` database instance because
their checked-in Compose surfaces use the same internal database name and DSN
environment contract; the mapping and each deployment injection are validated.
The two Kubernetes profiles use their matching cloud instances.

`ONEX_DATABASE_TOPOLOGY_PROFILE` is independent of `ONEX_ENVIRONMENT` and
`KAFKA_ENVIRONMENT`. The latter two are event namespaces and may carry values such
as `test-env`; they must never select or silently fall back to a database topology.

The files contain environment-variable names and Kubernetes/Docker service names,
never passwords, tokens, or DSN values. The `omninode_runtime` service and the
`omninode_runtime` PostgreSQL principal are separate namespaces: the service consumes
the `omninode_runtime_service` binding.
