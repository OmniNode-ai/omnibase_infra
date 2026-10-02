# Review verdicts: DelegationReaper model (OMN-19441)

Two independent read-only Claude Opus reviews, each in its own session and with no part in writing the model. A
reviewer only saw the files under this directory and the design summary given to it; neither reviewed the
implementation.

## Review 1, on the first draft: CHANGE

Blocking findings, and what the committed model does about each:

1. Terminal identity was `<<command, kind>>`, so two different real terminals from two workers of one command
   (the DLQ replay case) collapsed into one tuple and the trace the requirement forbids was invisible.
   Fixed: a terminal is `<<kind, writer>>` (writer is a worker or `reaper`); `AtMostOneTerminal` counts distinct
   identities. New mutant `mut_ungated_handlers` (two handlers, no reaper) breaks it. New witness
   `wit_two_real_terminals` reaches the case.
2. `wit_healed` was vacuous: its trace never fired the reaper's heal. Fixed: `WitHealedOrphan` requires the
   worker that wrote the slot to be dead, the replay never to have delivered, and the terminal to be on the wire,
   so only the reaper can have published it. New mutant `mut_no_heal` (the reaper does not heal a handler-written
   slot) breaks `EveryClaimedAnswered`.

Non-blocking findings taken: the claim is two steps (reap row first, claim row second) with a crash allowed
between them, and `mut_context_last` (claim row first) breaks `EveryClaimedAnswered`; a fourth worker replays c2;
the header states that the model counts distinct terminal identities and not envelopes, and that the correlation
is a stress input read only by `mut_corr_key`.

## Review 2, on the revised model: no blocking defect in the model; CHANGE to this file's wording only

The second reviewer ran the full state space of `Model.cfg` (1,961,564 distinct states, depth 24, no error),
confirmed each of the eleven mutants fails the property it names and each of the seven witnesses is reached,
confirmed `results/model.sha256` matches the files, and confirmed both blocking findings of review 1 are resolved
(`mut_ungated_handlers.out` state 11 has two real terminals from `w1` and `w2` on the wire;
`wit_healed_orphan.out` states 7-9 show `w1` recording, crashing, and `ReapHeal` publishing while `w2` is idle).
Its only blocking item was an earlier draft of this file that claimed an approval the first reviewer never gave;
that wording is removed. The model files needed no edit.

## Build conditions the implementation must honour

Drawn from review 1 and checked against the code by the author, not by a reviewer:

- B1. The slot compare-and-set discriminator is unique to the caller (a per-attempt token), never a bare timestamp.
- B2. The reap row is written strictly before the claim row, insert-only, first writer wins; the deadline is
  stored, never recomputed from a replay.
- B3. A heal republishes the stored terminal; it never re-derives one.
- B4. A worker that loses the slot publishes nothing on every path, the timeout path included; each late
  result is its own evidence row.
- B5. The command id is the original envelope's message id and survives a DLQ replay (omnibase_infra#4289).
- B6. The consumer commits only after the terminal is handed to the wire.
- B7. Downstream consumers dedupe on the command id, never on the correlation.
- B8. A row that fails to reap does not block the others; a batch cannot starve a row.

## Stated limits (not closed here)

- A crash between the claim-row copy and the in-process publish leaves a copy and nothing on the wire. The model
  merges the copy and the hand-off into one step, so it does not cover this; recovery is the consumer's redelivery
  (B6), and the reaper skips a row that has a copy.
- The deadline counts from the first reap row. A worker that dies after writing the reap row but before the claim
  row, followed by a replay after the deadline, is reaped at the next tick and the replay's real result is kept as
  evidence only. This agrees with `NoEarlyReap`.
- A claim written before the reaper existed has no reap row and is never reaped.
- The deadline is written by the claiming handler and compared by the reaper, each on its own host clock. Skew
  larger than the grace could reap early; the slot still keeps the command to one terminal.
- `ClaimTimeFixed` restates an implementation choice: a replay that restarted the clock could only postpone the
  reaper. `wit_two_real_terminals` and `wit_replay_served` show the cases that arise, not a second terminal on the
  wire. The bounds are small (2 commands, 4 deliveries, clock 0..3).
