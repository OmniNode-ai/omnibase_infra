# Design-review verdict: runtime primary election (OMN-21418)

Verdict: **PASS with conditions.** The election design in
`RuntimePrimaryElection.tla` may be built as the runtime primary election node,
under the conditions below.

Reviewed against the committed TLC output in this directory, bound to
`model.sha256` (the sha256 of the spec, then each cfg in sorted order). TLC2
Version 2026.09.23.154203 (rev: 4260e47), `-workers 4`.

| Config | Expected | TLC result | Distinct states |
| --- | --- | --- | --- |
| Model.cfg | pass TypeOK, AtMostOneActingPrimary, EpochMonotonic | No error has been found, 0 states left on queue | 29872 |
| mut_no_self_fence.cfg | counterexample | AtMostOneActingPrimary violated | 339 |
| mut_unguarded_write.cfg | counterexample | EpochMonotonic violated | 3607 |
| wit_two_epochs_applied.cfg | witness violated | NotTwoEpochsApplied violated | 3309 |

## Findings

1. Both safety properties hold over the bounded model, and each depends on the
   mechanism the design names: dropping the self-fence (`SelfFence = FALSE`)
   lets two runtimes act at once, and dropping the target's epoch check
   (`WriteGuard = FALSE`) lets a delayed stale write land after a newer one.
2. The witness is violated, so re-election with writes on both sides of it is
   reachable and the passing result is not vacuous.
3. `AtMostOneActingPrimary` holds because the runtime's local deadline
   (`TTL - Margin` after its grant or renewal) is never later than the store's
   expiry, on one shared clock. Self-fencing alone does not stop a paused
   runtime's in-flight write from landing late; the target's epoch check does.
   Both are required.
4. Of the three conjuncts of `EpochMonotonic`, only the ordering of `applied`
   is broken by a mutation. The other two (no runtime or in-flight write
   carries an epoch the store has not granted) hold by construction in this
   spec.

## Conditions on the build

- The write target must refuse a write whose epoch is below the highest epoch
  it has accepted. Self-fencing on the runtime is not a substitute.
- `Margin` must be at least the clock drift the implementation allows between
  a runtime and the store. Drift is not modelled; the model assumes one clock.
- A campaign must be one atomic store transaction that creates the free leader
  key and takes the epoch from the store revision.

## Not covered

- Liveness (a primary is eventually elected): no fairness is stated and no
  temporal property is checked.
- Clock drift, store failover and network partitions beyond unbounded delay.
- Bounds: 3 runtimes, TTL 3, Margin 1, time 0..6, 3 writes.
- The models follow the OMN-21418 ticket text. A reader of the runtime cluster
  plan under OMN-21279 should confirm they match its election section.

Recorded by the landing worker for omnibase_infra#5358, 2026-10-26, from the
committed spec, cfgs and TLC output. The verdict the PR described was not
committed: the repository's `DESIGN_*.md` ignore rule matched this file name on
a case-insensitive filesystem.
