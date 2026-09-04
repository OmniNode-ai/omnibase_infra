# Model-review runner capability rollout

This runbook describes the private implementation delivered by OMN-17876 and
the separately authorized live application owned by OMN-17878. It contains no
secret values, endpoint values, IP/port literals, runner identities, or
environment-file contents.

## Source contract

The source of truth is `config/runner_fleet.yaml` and the typed models under
`src/omnibase_infra/observability/runner_health/`. The capability is inactive
in source control. Its required scheduling boundary is the `model-review`
label in the `omnibase-ci` runner group.

The three logical references are opaque IDs only:

| Purpose | Reference ID |
| --- | --- |
| Credential reference | `dc9565c8-7f13-46dc-bd89-9694c13e1d2f` |
| Endpoint reference | `b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7` |
| Healthcheck reference | `2672472a-bac9-4344-8c8c-79da6cb604ae` |

The reference values must never be resolved into this repository or an
evidence artifact.

## Preflight evidence

`collect_model_review_capability_observation` is the runner-local probe
boundary. It probes each canonical credential, endpoint, and healthcheck
reference for presence and health, probes reviewer CLI availability, and
derives a correlation UUID. UUIDv5 is correlation only: this slice ships no
live attestation verifier, so preflight always reports
`live_attestation_unavailable` and remains not-ready. A future sanctioned
verifier/receipt contract must be supplied by the operator rollout.
`ModelModelReviewCapabilityObservation` records only labels, groups, opaque
reference IDs, reference health, reviewer CLI availability, attestation
identity, UTC observation time, and the fixed provenance marker
`runner-local-model-review-preflight`. Preflight requires all three references
to be present and healthy, plus verified fresh provenance; missing, extra, or
stale facts fail closed.

The contract canary emits an allowlisted JSON projection containing only:

* workflow run ID;
* not-ready contract status and unobserved selection;
* runner group and allowlisted labels;
* opaque reference IDs;
* unverified attestation state; and
* not-run verdict status.

It is a contract fixture canary, not proof of live runner execution. OMN-17878
owns the later bounded RSD canary that proves runner selection and complete
no-fallback verdicts.

## Operator application (OMN-17878 only)

After the implementation PR is merged and independently reviewed, the
operator must perform these serial steps under explicit authorization:

1. Resolve the three references through the sanctioned private configuration
   system and verify presence/health only.
2. Register exactly one candidate in group `omnibase-ci` with `model-review`
   plus the existing required labels.
3. Activate the candidate overlay only for that runner; do not recreate the
   generic fleet. Keep the overlay in `docker/compose-overrides.list` while
   the candidate is active so monitor auto-repair preserves it.
4. Verify a fresh, provenance-bearing preflight and container health result.
   The fixed sanctioned verifier command/receipt must be present and pass;
   this implementation intentionally provides neither.
5. Run the bounded same-repository RSD canary and prove both review legs return
   a complete verdict with no fallback.
6. Only after the canary passes, set the non-secret RSD selector to require
   `self-hosted`, `omnibase-ci`, and `model-review`.

No fork workflow may be routed to this candidate, and no RSD source change is
part of this rollout.

## Rollback

1. Restore the prior generic selector `["self-hosted","omnibase-ci"]`.
2. Disable capability and overlay selection.
3. Drain the candidate, remove its `model-review` scheduling advertisement,
   and unregister it if required by the operator procedure.
4. Verify generic-pool availability and fail-closed behavior at review.

Rollback evidence must use the same sanitized enum/boolean/opaque-ID fields,
report `not_ready`, `not_observed`, `unverified`, and `not_run`, and contain no
secret, endpoint, topology, or runner-identity values.
