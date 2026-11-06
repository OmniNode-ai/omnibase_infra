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
Consume, Begin, SafetyOpens, ProbeSource, SafetyCloses, Publish, Breaks, Clears, JoinPasses, StepPut,
Handover. A proof about
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
  a source read clean, then going bad again        ProbeSource, then SrcBad, then SafetyCloses
  a duplicate start the bus still holds            Consume, then Redeliver

Each of those has a witness below, so the race is reachable and not only drawn.

The properties, and what each one rests on
------------------------------------------

"Table" compares the decision with an order written apart from it, in every reachable state.
"Carried" is derived across actions: the guard is not the invariant, and interleavings can break it.
"Atomic" equals the guard of one transaction: it restates that the store gives that transaction, and
says nothing about the race inside it, which is why its mutation splits or removes the compare.

P1 DecideFollowsOrder            table    includes predecessor, join and next-piece, so StrongOrderTable
   SafetyCauseAdmitted           table    through step 0b, a stale table, stops and no credit
   RepairNeverRefusedForCredit   table    a private repair lane too, during a failure
   ReadOnlyAdmitted              table
   KeptPassesStop                table
   CauseOfKeptRuns               table
   EveryStopHasAWayIn            table    no wedge, in every reachable state
   ContinuationPasses, B0Passes, table    (extension) the other way: DecideFollowsOrder alone lets a table
   B0DrawsCredit,                         refuse too much
   KeptAtActiveOnStale,
   KeptAboveSevenRefusedOnStale
   NoStartAgainstOrder           carried  by the epoch and the re-read, from the reservation to the start
   OneRepairLane, OneCauseLane,  carried  slots held through a dead process
   OneJoinLane
   NoStepBeginsWhileParked       atomic   the fence in the write
   OutageHoldsSteps              atomic   no store, no step, the emergency lane's excepted
   AtMostOneStepAfterPark        carried  the in-flight bound
P2 AliveWithinLimit              carried  counts processes, never rows
   PlacesHeldUntilGone,          carried  (extension) a dead process keeps its place and slot
   SlotsHeldUntilGone
   CreditNotOverdrawn            atomic   one compare in the consume
P3 StopClearsOnlyWithEvidence    carried  ghosts dated apart from the rows the guard reads
   StopsOnlyPastBuilding         atomic
   OrderOnlyOverWorkingPiece     carried  a start with nothing WORKING: built, read WORKING and joined
   SafetyClosesOnCleanProbe      carried  a ghost dated apart from the read the guard takes
P4 RestartRepeatsNoStep          carried  stale reads and replays reach the put, and the compare refuses them
   NoWriteByLostOwner            atomic   the fence in the put, with a stale engine racing it
   ConsumedAtMostOnce            atomic   Redeliver goes through a compare on identity and use, changes no
                                          lane, and its flag cannot rise under Mut = {} by logic; each
                                          removed compare reports. At most once rests on 7.5's contract
                                          suite ("a duplicate delivery"), and not on this model
   NeverConsumedAfterEpoch       carried  a ghost per write site no guard reads, so a missing Kill shows
   NoStartWhileStoreDown         atomic
   OneEmergencyPerOutage         atomic   one credential on one host: across hosts there is no store to
                                          serialize on, so its mutation is a second credential

Round 1's ExactlyOneOwner was deleted: it read one variable, so it could not fail. Both engines
acting on the order at once is reachable (wit_WitOldEngineStillActing), and what must hold is that
only one write lands. Round 1's OwnerComesBack and RunningContinuationSuspended were one action's
fairness restated; deleted.

What the model says about the sweep: with the fence in every write, the sweep adds nothing to any
safety property here. It keeps a parked lane parked until a read brings it back, and
wit_WitParkedHoldsPlace and mut_p2_parked_frees_place say what a parked lane holds. A lane that never
reaches a fence is not modelled. The plan keeps the sweep for read-back, and no result here proves it.

Liveness, a chain another writer can interrupt unless said otherwise:

   SafetyCauseStarts      through a dead lane holding the slot, a stale table, stops, dead engines, no credit
   WriterRepairStarts     retried while each repair lane fails
   StopCauseStarts        the lowest stop's cause lane; safety failures and staleness are named releases
   AdmissibleStarts       one per class
   FixedSafetyCloses      on its source's dated read, with the writer dead; a release if the source goes bad again
   BadSourceRecorded      through an outage
   StopClearsOrEscalates, SafetyStopClearsOrEscalates, BuildingEscalates
                          through the timer's own death and an outage
   EmergencyStarts        with the store never coming back
   ParkedLaneComesBack    one action's fairness once its premise holds; it says the premise is enough
   ReadyStepAdvances      <>[]Ready. []<>Ready fails: a safety failure and a crash can take turns so
                          Ready and an up owner never meet. So without EngRace this restates the owner's
                          fairness and the supervisor's; under EngRace (live_engine_race) the put can be
                          refused and the engine must read again

What AdmissibleStarts does not say. Strong fairness on Reserve and Consume says the request is asked
again until it starts, and the protocol leaves Consume enabled right after each Reserve. The model
has no clock, so it cannot say the reservation's lifetime beats the real write rate; 7.1's live drill
owns that. ctl_weak_retry_* show weak fairness is not enough.

Fairness
--------

The module's Fairness comments name each clause. Each has a ctl_drop_<clause> config showing a
property needs it. One round-2 clause, on Expire, was deleted when its control held. Also deleted
from round 1: WF on Reserve, Consume and SafetyFix for every task, Suspend and Resume, and SF on
Publish. No clause assumes that a lane fixes anything, that an operator acts, or that the reading
writer stops dying. The clauses do assume: an outage ends (not for live_emergency_outage), an
escalation timer's supervisor works, an engine's supervisor works (not for live_holder_dead), a dead
process is read, a source is read.

Why classes are split into parts, and what holds them together
---------------------------------------------------------------

Round 1 said another class adds no behaviour because "every gate fact can change in every state".
That was false: joined needs a join lane, pubUp a repair lane, a clean source a safety cause lane,
broken a cause lane. Round 2 makes no such argument. Base_union and Base_union_core put every class
of request together, and the parts exist for two other reasons: a larger bound on one mechanism than
the union can afford, and every kind in the table.

Base_union has no result, and nothing here rests on it. Its run was stopped at 19,578,120 distinct
states after 48 minutes on 16 workers, at BFS depth 24 with 5.8 million states on the queue and the
queue growing, to keep its host under its core count. Base_union_core, which drops five things from
it, needed about 150 million distinct states and depth 57, so Base_union does not fit a lab unit's
three-hour bound or the hosts' memory. Every class at once is therefore Base_union_core plus the
reading under its scope below, which is an argument and not a run.

Scopes, every one in its cfg; results/SUMMARY.txt gives workers, memory and state counts per run.

  Base_union       every class at once: ordinary, join, next-piece, 2 cause, kept, continuation,
                   repair, safety cause, emergency; crash, handover, store down, writer death, source
                   lag, credit, sweep, the continuation lane's step in two actions, dead processes.
                   One ordinary of piece 1, WIP limit 1, 1 safety failure, stops on 2 pieces.
                   No result: see above.
  Base_union_core  the same requests and environment, without sweep, crash, handover, dead processes
                   and the split step. Why that loses no safety property of Mut = {}, by reading and
                   not by a run: parked and a dead lane's place only disable actions, up only disables
                   the engine, and a handover's Kill sends each reservation where Expire can; no
                   invariant reads parked, up or handed, and inflight is read by Begin and Act alone.
                   It ran in parts, each resuming TLC's own checkpoint of the one before with -recover,
                   at the same module and cfg bytes. The chain is part 1, then part 2 from part 1's
                   checkpoint of 04:40:29Z, then part 4 from part 2's checkpoint of 07:34:41Z;
                   results/Base_union_core.part<n>.out holds each, and Base_union_core.out is the last.
                   TLC does not keep the fingerprint polynomial across -recover unless -fp is given:
                   part 1 ran with fp 60 and part 2 with fp 74 (each file's header), so the
                   fingerprints part 2 inherited matched nothing it computed and it explored again
                   everything reachable from the queue it inherited. Part 2 ran 2 h 52 min without
                   finishing, and the distinct counts from part 2 on are not this scope's size. No
                   state is lost: every reachable state was either generated and checked by an
                   earlier part or is reachable from the queue the next part starts with, since a
                   checkpoint is the seen set and its unexpanded queue together. That is an argument about TLC, for the reader to check.
                   Part 3 resumed part 2's checkpoint with fp 0, repeated the same waste, and was
                   stopped by its writer; it is not in the chain. Part 4 passes -fp 74, and its queue
                   then shrank from the first minute as part 2's had, where part 3's grew.
  Base_mini        two ordinaries, the same classes, no outage, crash or sweep
  Base_table       every kind against 2 safety failures, stops on 3 pieces (u builds piece 2)
  Base_wip2        3 ordinaries + join + next-piece, limit 2, with outage, parking, dead processes
  Base_wip8        9 ordinaries + join, limit 8. AliveWithinLimit is inductive under one guard, so nine
                   racers add no behaviour two would not; kept because the plan's writers are nine
  Base_holder      2 steps, EngRace on, crash, restart, handover, outage, redelivery
  Base_stop        typed rows, stops on 3 pieces (u builds piece 2), cause lanes and kept
  Base_fence_*     one per class, that class's step as two actions; two lanes' steps do not interact
  Base_emergency   outage, emergency lane on two hosts, sweep, steps of the emergency lane too
  Base_ext_table   Base_mini's scope plus the table the other way: over-refusal fails (reader's N3)
  Base_ext_second_lanes  second join, cause and repair lanes present, dead processes (N2, N5)
  Base_ext_above_seven   four pieces: a join and a stop above the line; no ordinary lane on a non-kept
                   piece above 7 (N6)

Liveness scopes. No liveness run has every class and every environment: TLC's liveness check is the
bound, and live_union at about 10 million states did not finish in 80 minutes on the earlier model.

  live_union_admit_*  o1, j, u, c3, kf, g, r1, s1; stop on 3; writer death, source lag, credit.
                      No c2, no e1, no store down, no crash, no sweep, no split step
  live_union_safety, _repair, _stop, _esc   that scope with the lane's own dead process or the timers
  live_union_crash    that scope with crash and handover
  live_union_low      a stop at the piece-7 line: o1, j, u, c2, g, r1, s1
  live_union_store    store down with e1: o1, g, r1, s1, e1, c3
  live_engine_race    EngRace on; live_park sweep on; live_emergency, live_stop and the rest are small

Pieces are 1..3: piece 2 is the plan's piece 7, piece 3 is 8 or 9.

Not modelled
------------

Class shares, the third-lease and five-whys variants beyond one typed row, an accepted release coming
back at expiry, wall-clock values, debits taken afterwards, dependency repair (its admission is the
cause lane of N, and M is no fact the decision reads), shadow until decision 5 (it only removes
refusals), a read-only principal. Starts that bypass admission: the model has no such path, and 7.5's
reconcile detects them; it does not prevent them. A lane that never reaches a fence. WORKING against
WORKING 24 h is one bit: 3.5 says pull needs the window and a clear one dated probe. Redelivery runs
in Base_holder and its mutations only. Conformance of anything.

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

results/SUMMARY.txt binds each result to the sha of the module and of its own cfg. A TLC run reads only
those, so a run started before a later cfg-only or extension commit stands at its own bytes. The
*_ext_* configs run on RebuildAdmissionExt.tla, which extends the module and adds properties and the
overrides that break them; it exists so the module stayed byte identical under the long runs.

Rerun, for each cfg:
  java -XX:+UseParallelGC -Xmx24g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 16 \
    -lncheck final -checkpoint 30 -gzip -metadir <fresh dir> <module>
where <module> is RebuildAdmissionExt.tla for the *_ext_* configs and RebuildAdmission.tla for every
other. To resume a run its time bound killed, replace -metadir with -recover <its checkpoint dir>
and add -fp <n>, n the polynomial in the killed run's header line "with fp n".
Exit 0 is required of Base_* and live_*, 12 of mut_* and wit_*, 13 of ctl_*.

results/model.sha256 is the sha256 of RebuildAdmission.tla, then RebuildAdmissionExt.tla, then every
*.cfg in sorted name order, as bytes with nothing between them; test_rebuild_admission_model.py
recomputes it. It is the digest of these files now. The results were not all made at this commit:
each lab unit printed the digest of its own checkout, and results ran at three commits whose digests
differ. A result stands here because the bytes it read, its module and its own cfg, are equal to these;
results/SUMMARY.txt names, for each, the commit it ran at, the host, workers, heap and states.
