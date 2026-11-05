RebuildAdmission: M0 of the subsystem rebuild plan (OMN-22478), round 2
======================================================================

The stop gate, WIP admission and the holder as one state machine, against plan sections 6.2, 7.1,
7.3, 7.5 and W28. It implements and wires nothing, and no node exists in this shape. Results are in
results/SUMMARY.txt. A verdict from this directory is its author's; plan section 12 requires a
second reader, and second_read/ is the earlier reader's evidence against round 1, left unchanged.

What changed from round 1
-------------------------

Round 1 set each design choice with a constant, so "the protocol" was one setting of 22 knobs and a
second reader could show most invariants equal the guard of the action that could break them. Round
2 has no design knobs. Mut = {} is the protocol. Each name in Mut is one break, and scope constants
bound the environment only.

What is one action, and why
---------------------------

The gate store (7.5) is one serializable store, so a transaction on it is one action: Reserve,
Consume, Begin, SafetyOpens, SafetyCloses, Publish, Breaks, Clears, StepPut, Handover. A proof about
those actions assumes the store's isolation: that is the obligation the store's implementation
carries, and a split inside a transaction stays a mutation (late_bind, fence_split, fence_read,
key_read, effect_then_key, gate_lag, expired_token).

Everything outside the store is its own action, and those races are in the protocol's own runs:

  a source going bad before the store records it   SrcBad, then SafetyOpens
  time passing with no write                       GoesStale; TimeOK in Consume re-reads it
  a process exiting before anyone reads it         Ends, then ReadGone
  a begun step acting after a failure opens        Begin, then Act
  an engine acting on what it read before          EngRead, then StepPut or StepDone
  a handover, and replaying after a crash
  a probe read before the store came back          ProbeRead, then EmStart
  a duplicate start the bus still holds            Consume, then Redeliver

Each of those has a witness below, so the race is reachable and not only drawn.

The properties, and what each one rests on
------------------------------------------

"Table" compares the decision with an order written apart from it, in every reachable state.
"Carried" is derived across actions: the guard is not the invariant, and interleavings can break it.
"Atomic" equals the guard of one transaction: it restates that the store gives that transaction, and
says nothing about the race inside it, which is why its mutation splits the transaction.

P1 DecideFollowsOrder            table    includes predecessor, join and next-piece, so StrongOrderTable
   SafetyCauseAdmitted           table    through step 0b, a stale table, stops and no credit
   RepairNeverRefusedForCredit   table
   ReadOnlyAdmitted              table
   KeptPassesStop                table
   CauseOfKeptRuns               table
   EveryStopHasAWayIn            table    no wedge, in every reachable state
   NoStartAgainstOrder           carried  by the epoch and the re-read, from the reservation to the start
   OneRepairLane, OneCauseLane   carried  slots held through a dead process
   NoStepBeginsWhileParked       atomic   the fence in the write
   AtMostOneStepAfterPark        carried  the in-flight bound
P2 AliveWithinLimit              carried  counts processes, never rows
   CreditNotOverdrawn            carried
P3 StopClearsOnlyWithEvidence    carried  ghosts dated apart from the rows the guard reads
   StopsOnlyOnStarted            atomic
P4 RestartRepeatsNoStep          carried  stale reads and replays reach the put, and the compare refuses them
   NoWriteByLostOwner            atomic   the fence in the put, with a stale engine racing it
   ConsumedAtMostOnce            carried  redelivery after a new reservation exists
   NeverConsumedAfterEpoch       atomic
   NoStartWhileStoreDown         atomic
   OneEmergencyPerOutage         carried  across hosts, with no store to serialize on

Round 1's ExactlyOneOwner was deleted: it read one variable, so it could not fail. Both engines
acting on the order at once is reachable (wit_WitOldEngineStillActing), and what must hold is that
only one write lands. Round 1's OwnerComesBack and RunningContinuationSuspended were one action's
fairness restated; deleted. In their place ReadyStepAdvances, and the sweep has no liveness property.

What the model says about the sweep: with the fence in every write, the sweep adds nothing to any
safety property here. It does keep a parked lane parked until a read brings it back, and
wit_WitParkedHoldsPlace and mut_p2_parked_frees_place say what a parked lane holds. A lane that never
reaches a fence is not modelled. The plan keeps the sweep for read-back, and no result here proves it.

Liveness, each a chain another writer can interrupt:

   SafetyCauseStarts      through a dead lane holding the slot, a stale table, stops, dead engines, no credit
   WriterRepairStarts     retried while each repair lane fails
   StopCauseStarts        the lowest stop's cause lane; safety failures and staleness are named releases
   AdmissibleStarts       for every class, against every other class's kills
   FixedSafetyCloses      on its source, with the writer dead
   BadSourceRecorded      through an outage
   StopClearsOrEscalates, SafetyStopClearsOrEscalates   through the timer's own death and an outage
   ReadyStepAdvances      through crashes and a handover
   EmergencyStarts        with the store never coming back

What AdmissibleStarts does not say. Strong fairness on Reserve and Consume says the request is asked
again until it starts, and the protocol leaves Consume enabled right after each Reserve. The model
has no clock, so it cannot say the reservation's lifetime beats the real write rate; 7.1's live drill
owns that. ctl_weak_retry_* show weak fairness is not enough.

Fairness
--------

The module's Fairness comments name each clause. Each has a ctl_drop_<clause> config showing a
property needs it. One round-2 clause, on Expire, was deleted when ctl_drop_expire held. Also deleted
from round 1: WF on Reserve, Consume and SafetyFix for every task, Suspend and Resume, and SF on
Publish. No clause assumes that a lane fixes anything, that an operator acts, or that the reading
writer stops dying. The clauses do assume: an outage ends (not for live_emergency_outage), an
escalation timer's supervisor works, an engine's supervisor works (not for live_holder_dead), a dead
process is read.

Why classes are split into parts, and what holds them together
---------------------------------------------------------------

Round 1 said another class adds no behaviour because "every gate fact can change in every state".
That was false: joined needs a join lane, pubUp a repair lane, a clean source a safety cause lane,
broken a cause lane. Round 2 makes no such argument. Base_union and live_union_* put every class
together, and the parts exist for two other reasons: a larger bound on one mechanism than the union
can afford (nine racers, three stops, two safety failures, both engine steps), and every kind in
the table. What a part does not prove about the union is what Base_union proves, at its own bound.

Scopes are in the cfgs; results/SUMMARY.txt gives workers, memory and state counts per run.

  Base_union   every class at once: ordinary, join, next-piece, 2 cause, kept, continuation, repair,
               safety cause, emergency; crash, handover, store down, writer death, source lag, credit,
               sweep. One ordinary of piece 1, WIP limit 1, 1 safety failure, stops on 2 pieces.
  Base_mini    two ordinaries, the same classes, no outage, crash or sweep
  Base_table   every kind against 2 safety failures, stops on 3 pieces
  Base_wip2    3 ordinaries + join + next-piece, limit 2, with outage, parking, dead processes
  Base_wip8    9 ordinaries + join, limit 8. AliveWithinLimit is inductive under one guard, so nine
               racers add no behaviour two would not; kept because the plan's writers are nine
  Base_holder  2 steps, EngRace on, crash, restart, handover, outage, redelivery
  Base_stop    typed rows, stops on 3 pieces, cause lanes and kept
  Base_fence   the fence with every class's step as two actions
  Base_emergency  outage, emergency lane on two hosts, sweep

Pieces are 1..3: piece 2 is the plan's piece 7, piece 3 is 8 or 9.

Not modelled
------------

Class shares, the third-lease and five-whys variants beyond one typed row, an accepted release coming
back at expiry, wall-clock values, debits taken afterwards. Starts that bypass admission: the model
has no such path, and 7.5's reconcile detects them; it does not prevent them. A lane that never
reaches a fence. WORKING against WORKING 24 h is one bit (3.5). Conformance of anything.

Mutations and findings
----------------------

mut_*  one named break per property; each must report its named invariant. ctl_*  the liveness
twin: a mutation or a dropped fairness clause; each must report its named property. wit_*  must be
violated. Two mutations are the plan's text as read, not breaks of a design:

  mut_p1_plan_cause_needs_credit   7.1 applies the freed-place guard to "every admission that can open
                                   a pull request" and exempts safety and repair lanes only. With no
                                   credit a stop's cause lane is refused, and EveryStopHasAWayIn breaks.
  mut_p1_plan_cloud_lane           7.5 says a cloud lane never reaches the store; 7.1 says every
                                   running mutating lane checks before each outward action.

Rerun, for each cfg:
  java -XX:+UseParallelGC -Xmx24g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 16 \
    -lncheck final -checkpoint 60 -gzip -metadir <fresh dir> RebuildAdmission.tla
Exit 0 is required of Base_* and live_*, 12 of mut_* and wit_*, 13 of ctl_*.

results/model.sha256 is the sha256 of RebuildAdmission.tla followed by every *.cfg in sorted name
order, as bytes with nothing between them; test_rebuild_admission_model.py recomputes it.
