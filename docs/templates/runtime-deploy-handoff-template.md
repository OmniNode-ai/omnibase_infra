# Runtime Deploy Handoff — <ticket> — <YYYY-MM-DD>

> Template for any handoff/OCC doc that claims a runtime was deployed/redeployed.
> Every "deployed SHA" claim must be proven against the image's baked
> `/app/build-provenance.json`, never copied from a build log or memory.
> The `check-evidence-provenance-probe` pre-commit + CI gate rejects a
> deployed-SHA claim without an adjacent `docker exec ... cat
> /app/build-provenance.json` probe.

## Lane

- Lane: `<dev | stability-test | prod>`
- Compose project: `<omnibase-infra | omnibase-infra-stability-test | ...>`
- verified: `<YYYY-MM-DDTHH:MMZ>` via `<probe command>`

## Per-container deployed SHA table

Fill one row per running runtime container. The `infra_vcs_ref` and per-sibling
`vcs_ref` columns are read DIRECTLY from each container's baked provenance
manifest (see probe block below) — do not hand-copy from a build log.

| Container | `infra_vcs_ref` | `omnibase_core` vcs_ref | `omnimarket` vcs_ref | `onex_change_control` vcs_ref | dirty? |
|-----------|-----------------|-------------------------|----------------------|-------------------------------|--------|
| `<container-1>` | `<sha>` | `<sha>` | `<sha>` | `<sha>` | `<true/false>` |
| `<container-2>` | `<sha>` | `<sha>` | `<sha>` | `<sha>` | `<true/false>` |

## Provenance probe (EFFECT — reads the image's own manifest)

Paste the verbatim output of probing `/app/build-provenance.json` from each
running container. This is the citation that substantiates the SHA table above.

```bash
docker exec <runtime-container> cat /app/build-provenance.json
```

```json
{
  "build_source": "workspace",
  "build_time": "<...>",
  "infra_vcs_ref": "<...>",
  "per_repo_vcs_provenance": {
    "siblings": {
      "omnibase_core": {"vcs_ref": "<...>", "vcs_dirty": false, "vcs_branch": "dev"},
      "omnimarket": {"vcs_ref": "<...>", "vcs_dirty": false, "vcs_branch": "dev"}
    }
  }
}
```

## Health proof

- Probe: `<command>`
- Result: `<HEALTHY | ...>`
- verified: `<YYYY-MM-DDTHH:MMZ>` via `<command>`
