# OMN-20769: Redpanda v25.1.12 to v25.2.14 hop evidence

## Scope
Configuration advances from v25.1.12 to v25.2.14 as a necessary intermediate feature-release hop toward v25.3.18. This does not claim crash mitigation or resolution.

## Digest Derivation
Reproduced digests using the real CatalogResolver and generate_compose, SHA256(json.dumps(render, sort_keys=True, default=str).encode()). For all 17 pinned non-local bundles, old values reproduced by copying docker/catalog to a temporary directory and replacing the three catalog image pins with v25.1.12, comparing against the baseline at merge base 4090b069f179a06a5b1357f967b8e3d5730002e2; new values reproduced at dispatch head 18100e5f6b42ac9677e0237d11e054bacbe21d99.

12 bundle renders differed only at /services/redpanda/image: canary, core, fault-injection, omnidash, omnimarket-projections, runtime, runtime-core, runtime-infrastructure, runtime-integrations, runtime-observability-projections, secrets, tenant-path.
5 unchanged: auth, memgraph, observability, omnimemory, tracing.
Core and secrets render identical dictionaries; observability and tracing render identical dictionaries, so these pairs have equal hashes.
No digest baseline was edited by this landing fix.

Regeneration command:

```shell
env PYTHONDONTWRITEBYTECODE=1 uv run --frozen python - <<'PY'
import hashlib
import json
from pathlib import Path
import yaml
from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
catalog = Path('docker/catalog')
bundles = yaml.safe_load((catalog / 'bundles.yaml').read_text())
for name in sorted(set(bundles) - {'local', 'local-market', 'local-lane-mirror'}):
    rendered = generate_compose(CatalogResolver(catalog_dir=str(catalog)).resolve([name]))
    digest = hashlib.sha256(json.dumps(rendered, sort_keys=True, default=str).encode()).hexdigest()
    print(name, digest)
PY
```

## Lab Readback
On h101 focused test selection returned 34 passed, 90 deselected. Contract parser payload validation accepted 2 current payloads and refused 7 negative controls (old image with target comment/unrelated key, empty service, old deployed release with target comment, retained old hop, wrong first hop, empty declaration).

h201 live dev readback at 2026-10-28T21:46:24Z: image redpandadata/redpanda:v25.1.12, healthy, cluster_version=node_latest_version=15. Read-only; no lane container changed.

Isolated runtime probe completed 2026-10-28T21:48:42Z on h201, project landing-l18175-lab, container landing-l18175-lab-redpanda and volume landing-l18175-lab-data, network none, no published ports. Fresh volume started at v25.1.12 with cluster_version 15; created topic landing-l18175-proof and produced landing-l18175-hop-record at offset 0; stopped only the scratch container, then started v25.2.14 on the same volume, cluster_version=node_latest_version=16 and node_earliest_version=15, ready, restart count 0. Consumed retained record; rpk cluster health JSON is_healthy=true, no leaderless/under-replicated partitions; cleanup removed only scratch container and volume. Image digest sha256:b53a3a981be7536c2b5240efa13d7284e760ed01911627f3533e628117e427e2.

## Limits
This new probe uses a fresh scratch volume, not a current live-data copy or client load or long-term crash proof. Existing docs/incidents/OMN-21279/dev-broker-crash-cause.json fix_path.hop_proof.data_copy_run separately records the actual 49.2G dev-volume copy from 2026-10-26 and its v25.2.14 healthy result at logical version 16, 4635 topics, 14623 groups; scratch_run is a separate synthetic proof. Deployment must read live image and /v1/features again before upgrading; the hosted contract probe checks published repository configuration only and needs no lab access.
