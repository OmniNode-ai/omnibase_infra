# Design-review verdict: generic lease (OMN-21418)

Verdict: **PASS with conditions.** The lease design in `GenericLease.tla` may
be built as the generic lease node, under the conditions below.

Reviewed against the committed TLC output in this directory, bound to
`model.sha256` (the sha256 of the spec, then each cfg in sorted order). TLC2
Version 2026.09.23.154203 (rev: 4260e47), `-workers 4`.

| Config | Expected | TLC result | Distinct states |
| --- | --- | --- | --- |
| Model.cfg | pass TypeOK, FenceMonotonicPerResource, NoEffectWithStaleFence, ExpiredLeaseFreesResource | No error has been found, 0 states left on queue | 7108859 |
| mut_renew_changes_fence.cfg | counterexample | FenceMonotonicPerResource violated | 33 |
| mut_effect_skips_check.cfg | counterexample | NoEffectWithStaleFence violated | 14540 |
| mut_regrant_same_fence.cfg | counterexample | FenceMonotonicPerResource violated | 146 |
| mut_regrant_same_fence_effect.cfg | counterexample | NoEffectWithStaleFence violated | 13984 |
| mut_expiry_holds.cfg | counterexample | ExpiredLeaseFreesResource violated | 137 |
| obs_renew_changes_fence_effect.cfg | recorded observation | No error has been found, 0 states left on queue | 12454029 |
| wit_stale_refused.cfg | witness violated | NotStaleRefused violated | 5358 |

## Findings

1. All three safety properties hold over the bounded model, and each mutation
   breaks the property it targets: a renewal that issues a new fence or a
   regrant that reuses the last fence breaks the one-fence-one-grant identity,
   a target that accepts any fence lets an older grant's effect land after a
   newer one's, and a store that keeps an expired lease blocks a new grant.
2. Reusing a fence on regrant also breaks effect ordering
   (`mut_regrant_same_fence_effect`): the target cannot tell the two grants
   apart. Fresh fences from the shared counter are load-bearing.
3. A renewal that changes the fence breaks only the identity property
   (`obs_renew_changes_fence_effect` passes). Keeping the fence on renewal is
   still required, because the fence must name exactly one grant.
4. The witness is violated, so a stale effect reaching a target that has
   accepted a newer fence is reachable and is refused; the passing result is
   not vacuous.
5. The guarantee is ordering, not exclusion in time: a client whose lease has
   expired, and which has not yet learned it, can still have an effect
   accepted until an effect of a newer grant reaches that target.

## Conditions on the build

- Every effect target must refuse an effect whose fence is below the highest
  fence it has accepted for that resource.
- A grant must be one atomic store transaction that takes a fresh fence from
  the store revision; a renewal must keep the lease's fence.
- The store must treat a lease whose TTL has run out as free for a new grant.
- A caller that needs exclusion in time, not just ordering, must add it
  itself (finding 5).

## Not covered

- Liveness: no fairness is stated and no temporal property is checked.
- Clock drift between clients and the store, store failover, and partitions
  beyond unbounded delay.
- Bounds: 2 clients, 2 resources, TTL 2, time 0..4, fence counter 0..4,
  3 effects. `ExpiredLeaseFreesResource` uses `ENABLED Grant`, which ignores
  the `RevBound` state constraint.
- The models follow the OMN-21418 ticket text. A reader of the runtime cluster
  and generic lease plan under OMN-21279 should confirm they match its lease
  section.

Recorded by the landing worker for omnibase_infra#5358, 2026-10-26, from the
committed spec, cfgs and TLC output. The verdict the PR described was not
committed: the repository's `DESIGN_*.md` ignore rule matched this file name on
a case-insensitive filesystem.
