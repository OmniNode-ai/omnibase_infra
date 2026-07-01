# Runtime ECR Build Failure — Root Cause + Fix (OMN-13654)

## Summary

`build-and-push-runtime.yml` (the workflow that builds the prod-bound
`omninode-runtime` image and pushes it to ECR) has been **red since at least
2026-06-07**. The Docker build itself succeeds; the failure is the **Trivy
vulnerability scanner gate** (`CRITICAL,HIGH --ignore-unfixed --exit-code 1`),
which flags 5 fixable HIGH CVEs in the resolved Python dependency tree. Because
Trivy fails, the ECR push and digest-resolution steps are skipped — so no fresh
runtime image has been published.

## Item 0 Classification (reproduced run 28275139143)

- Run: `28275139143`, workflow `build-and-push-runtime.yml`, `event=workflow_dispatch`, `headBranch=main`, `conclusion=failure`, created `2026-06-27T01:56:59Z`.
- Clean through step 14 "Build runtime image" (succeeds even with `--no-cache`).
- FAILS at step 15 "Run Trivy vulnerability scanner" (`exit-code 1`), 5 HIGH / 0 CRITICAL.
- Push + digest steps SKIPPED (gated behind Trivy).

### The 5 HIGH CVEs (all fixable)

| Package | Vulnerable (main) | Fixed-by | CVE |
|---------|-------------------|----------|-----|
| pyjwt | 2.12.1 | >=2.13.0 | CVE-2026-48526 (auth bypass via forged tokens) |
| python-multipart | 0.0.27 | >=0.0.30 | CVE-2026-53539 (streaming parser) |
| starlette | 1.0.1 | >=1.1.0 | CVE-2026-48818 (SSRF / UNC credential theft) |
| starlette | 1.0.1 | >=1.3.1 | CVE-2026-54283 (ASGI flaw) |

## Why main is vulnerable but dev was not

The build runs against **main** (push trigger is `branches: [main]`; only main
pushes to ECR). main's `uv.lock` still resolves pyjwt 2.12.1 /
python-multipart 0.0.27 / starlette 1.0.1. dev was already relocked on
2026-06-23 (PR #2081) to the fixed versions, but that fix never reached main —
the runtime build has had no green main relock since the CVEs were published.

## Fix

Added explicit security-floor constraints to `pyproject.toml` (matching the
existing pyjwt CVE-2026-32597 pattern) so the lock can never silently drift back
below the CVE-fixed versions:

```toml
"pyjwt>=2.13.0",
"python-multipart>=0.0.30",
"starlette>=1.3.1",
```

Re-locked with `uv lock`. Resolved versions in the new `uv.lock`:

- pyjwt **2.13.0**
- python-multipart **0.0.31**
- starlette **1.3.1**

All three are at or above the CVE-fixed minimums.

## Scan proof (Trivy)

Trivy is the same scanner the CI step uses, with the same vuln DB, run against
the resolved lockfile with the identical severity gate
(`--severity CRITICAL,HIGH --ignore-unfixed --exit-code 1`). Docker is not
available on the dev workstation, so the image-layer build could not be run
locally; the lockfile scan covers exactly the Python-dependency CVEs that the
failing CI build flagged (the only findings in run 28275139143).

| Lockfile | Trivy exit | HIGH/CRITICAL |
|----------|------------|---------------|
| main (stale) | 1 | 5 HIGH (the exact 5 CVEs above) — confirms detection |
| this branch (fixed) | 0 | 0 |

Command:
```bash
trivy fs --scanners vuln --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 uv.lock
```

## Remaining work for full DoD (green main build + ECR push)

The dep fix lands on **dev** via this PR. The failing build is on **main**.
Restoring a green `build-and-push-runtime.yml` + a fresh ECR push requires the
fix to reach main via a dev->main promotion. As of this PR, dev is ~160 commits
ahead of main, so a promotion would sweep substantial unrelated in-flight dev
work — that promotion is an operator decision and is intentionally NOT forced
here. The build-and-push-runtime.yml only triggers an ECR push from main, so the
green-main + fresh-digest DoD completes when the promotion lands.
