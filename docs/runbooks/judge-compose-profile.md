# Judge Compose Profile

`docker/docker-compose.judge.yml` is the OMN-12924 Tier 2 judge reproduction
lane. It is a minimal local stack for renderable BYO-model-key proof:

- Postgres
- Redpanda
- Valkey
- forward and intelligence migrations
- main runtime
- effects runtime
- projection API

It intentionally does not include Keycloak, Infisical, deploy-agent, autoheal,
or dashboard services.

## Status

Tier 2 is best-effort unless this lane passes a cold-clone/cold-compose proof.
The current proof boundary is non-mutating compose render validation. Running
containers, changing `.201`, or performing a remote deploy requires explicit
operator approval.

## Inputs

Runtime policy comes from `docker/runtime-policy.env`, which is generated from
`contracts/services/runtime_policy.contract.yaml`.

Judge-provided secrets and local credentials come from
`docker/judge.env.example`. Copy it to a local ignored env file and replace:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY` only if the caller requires a separate alias
- `OPEN_ROUTER_API_KEY` only for OpenRouter fallback testing
- `LLM_GLM_API_KEY` only for GLM fallback testing
- `GITHUB_TOKEN` and `LINEAR_API_KEY` only for effects that create PRs or tickets

No source-level provider, model, endpoint, or secret literal should be required
for the judge reproduction path. Provider keys are resolved through
`ONEX_SECRET_RESOLVER_CONFIG_JSON` generated from the runtime policy contract,
with `enable_convention_fallback=false`.

## Non-Mutating Validation

```bash
docker compose \
  --env-file docker/runtime-policy.env \
  --env-file docker/judge.env.example \
  -f docker/docker-compose.judge.yml \
  --profile judge \
  config --quiet
```

```bash
docker compose \
  --env-file docker/runtime-policy.env \
  --env-file docker/judge.env.example \
  -f docker/docker-compose.judge.yml \
  --profile judge \
  config --services
```

Expected services:

```text
postgres
forward-migration
migration-gate
redpanda
projection-api
redpanda-partition-cap
valkey
intelligence-migration
omninode-runtime
runtime-effects
```

## Troubleshooting

Missing key:
Set `GEMINI_API_KEY` in the judge env file. The placeholder in
`docker/judge.env.example` is render-only.

Quota exceeded:
Use the recorded Gemini evidence path for the demo story, or provide an
alternate provider key through the same secret-ref contract path.

Local endpoint unavailable:
The judge lane defaults to cloud-compatible endpoint URLs in the compose
contract overlay. Do not edit source to route around local endpoint failures.

OpenRouter key name mismatch:
The logical ref is `llm.openrouter.api_key`; the env source path is
`OPEN_ROUTER_API_KEY`.

Stale overlay:
Regenerate and verify runtime policy before rendering:

```bash
uv run python scripts/render_runtime_policy_env.py --check-env-file docker/runtime-policy.env
```
