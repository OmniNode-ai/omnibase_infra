RebuildAdmission: M0 of the subsystem rebuild plan (OMN-22478), round 4
======================================================================

Admission, the stop and the holder as one state machine, written to the plan's text of 11-06
(sections 6.2, 7.1 and 7.5 at plan head 10c08fa2, text hash 52faf70168ffd3f1 by section 12's
recipe). It implements and wires nothing. Results are in results/SUMMARY.txt. A verdict from this
directory is its author's; plan section 12 requires a separate reader. second_read/ is an earlier
reader's evidence against round 1, left unchanged.

Why round 4
-----------

Round 4 is round 3 with the findings of reader m0-reader-1650 (NOT PASS at digest aaad6c7dde757b31)
fixed, by the operator's ruling of 2026-11-06T17:52:26Z ("let's fix M0", on the reader's D1: a lane
on a stopped piece waits for the gate store before any outward step while the store is down; starts
still go through, logged).

- D1, the fence fails closed. Begin is a store write with the fence in it and needs the store's
  answer; with no answer the step is not taken and the lane waits. A lane cannot tell without an
  answer whether its piece was stopped, so every fenced step waits, not only those of stopped
  pieces. A step begun before the outage may still land (Act needs no store), which is the in-flight
  bound. StepFence states 7.1 O3 and section 12 P1 as an action property read off the step itself,
  with no ghost and no outage exception; NoStepBeginsAfterStop's ghost now records a fenced begin
  whatever the guard read. down_proceeds restores round 3's logged step and breaks both, StepFence
  with the reader's 310-state trace shape (reserve, consume, stop, store down, begin).
  wit_WitLoggedStepUnderStop is retired: the state it reached is now refused, and
  mut_p1_down_proceeds reaches it. wit_WitInFlightLandsInOutage shows a step begun before the stop
  landing while the store is down.
- F2, a reservation does not stay stuck. The consume is an admission request with a deadline (7.1
  "No answer", 7.5), so NoAnswer fires from a held reservation as well as from idle: the start
  proceeds, logged. ResvNotStuck ((pc = "resv") ~> (pc # "resv")) is checked in both live configs,
  and resv_waits (NoAnswer from idle only, round 3) breaks it.
- F3, the kept pieces. KeptPieces names them (8 and 9 in the text): MaxPiece and, optionally,
  MaxPiece - 1. Base_kept has MaxPiece = 4, KeptPieces = {3, 4}, stops on 1, 3 and 4, the cause lane
  of each (c1, c8, c3), a kept fix on 4 (kf), outage, credit, the holder's step and escalation; its
  witnesses show a kept fix running under its own stop, a kept piece's cause lane running under its
  stop, and all three stops open at once with each cause lane running. Base_kept does not fit a
  lab unit (declared INFEASIBLE below, with what it reached); Base_kept_core is the same scope
  without the escalation timer and the holder's step, and holds. Every other config has
  KeptPieces = {MaxPiece}, which is round 3's table.
- F4, the epoch writes not modelled, listed under Not modelled below.

Why round 3
-----------

The operator's rulings of 2026-11-06T12:05:56Z and 12:10:28Z removed refusals round 2 encoded: a
safety failure starts its own cause lane and decides no other start; a stale table starts one
repair lane and refuses nothing; a reading decides no start; the pull rule (a piece waits for the
one below) is gone; a stop holds only its own piece; admission that cannot answer lets the start
through and logs it, so the emergency start and its credential are gone; the holder records state
and holds no start. Round 2's module was written to the earlier table, so its runs say nothing about
this text. Round 3 is a new, smaller module. Each dropped refusal is now a mutation that must report
the over-refusal property, so the model shows it does not refuse for it.

Dropped, and why:
  Step 1's refusal under a safety failure, step 2's above-seven refusal on a stale table, the pull
  rule, the stop holding pieces above it, the holder in admission: dropped by the text; each is a
  mutation (safety_refuses, stale_refuses, stop_holds_above, holder_in_engine) of
  NoRefusalOutsideRules.
  EmStart, OneEmergencyPerOutage, the probe and NoStartWhileStoreDown: the text has no emergency
  start; NoAnswer replaces them, and down_refuses breaks it.
  The sweep and parking: round 2 found the sweep adds nothing to any safety property once the fence
  is in every write; the text keeps it for read-back, which no model here proves.
  BUILDING, StopsOnlyPastBuilding, OrderOnlyOverWorkingPiece, BuildingEscalates: when a stop may
  open is 3.5's guard on one action; no property of section 12 reads it. Stops open on any piece in
  BreakPieces, which only adds behaviour.
  Safety close on a dated probe, safety escalation: not in section 12's four properties. A failure
  closes on one clean read of its source (an abstraction) and decides no start but its cause lane's.
  The order table (cur as the lowest unfinished piece, joins as facts): the order is now moved by
  hand or by the holder and is information to admission; the model keeps "started" as the fact a
  task of a piece needs, written by the holder's step.
  Typed release rows, dead-but-unread processes as a state: a process is read gone when it ends;
  "counted until read gone" is checked by close_frees, a row freeing a place while the process lives.

What is one action, and why
---------------------------

The gate store is one serializable store (7.5), so a transaction on it is one action: Reserve,
Consume, Begin, Breaks, SeeRow, Probe, Clears, Escalates, SafetyOpens, SafetyCloses, Publish, StepPut,
Handover. Everything outside it is its own action and its races are in the protocol's runs: a source
going bad (SrcBad), time passing (GoesStale, Expire), a process ending (Ends), a begun step landing
after a stop (Begin, then Act), an engine acting on what it read (EngRead, then StepPut), a kill
(Crash), a duplicate start delivery (two messages per start for DupTasks). A split inside a
transaction is a mutation: late_bind, fence_split, consume_split, effect_then_key, key_then_effect.

The properties, what each rests on, and the break that must report it
-----------------------------------------------------------------------

"Table": two writings of 7.1 compared in every reachable state over every request kind (AllKinds):
Decide, the step table in the text's order, and Why, the flat list of the reasons the text allows a
refusal for (private repository, stop, scope, WIP, credit). "Carried": derived across actions; the
guard is not the invariant. "Atomic": equals the guard of one transaction, so it restates what the
store gives; its mutation splits or removes the compare.

P1  DecideFollowsRules       table    nothing admitted against a reason
                                      mut: stop_lets_own, private_open, no_scope
    NoRefusalOutsideRules    table    nothing refused without one: no reading, safety failure, stale
                                      table or holder refuses (the 11-06 rulings)
                                      mut: safety_refuses, stale_refuses, stop_holds_above,
                                      holder_in_engine, ro_after_private, kept_refused_under_stop
    NoStartAgainstRules      carried  a start consumed later has no reason against it then: the
                                      epoch releases every live reservation at a write to a fact
                                      mut: no_advance_stop, no_advance_clear, no_advance_safety,
                                      no_advance_publish, late_bind
    NoStepBeginsAfterStop    atomic   the fence in the store's write, failing closed: with no
                                      answer no step begins
                                      mut: fence_off, fence_split, down_proceeds (_ghost)
    StepFence                action   7.1 O3 as written, read off the step with no outage
                                      exception: no fenced lane's outward step begins while its
                                      piece is stopped; checked in every Base config
                                      mut: down_proceeds
    AtMostOneStepAfterStop   carried  the in-flight bound: one step begun before the stop may land
                                      mut: fence_once
    AdmissibleStarts         liveness an admissible request starts, and with no answer it starts
                                      logged (Ext)
                                      mut: down_refuses; ctl: weak_retry
    ResvNotStuck             liveness a lane holding a reservation leaves it: it starts through
                                      the store, starts logged with no answer, or loses it to an
                                      advance (Ext, both live configs)
                                      mut: resv_waits
P2  AliveWithinLimit         carried  processes per piece, counted by the store's admissions, within
                                      the limit; live reservations count in their own transaction;
                                      a join's first task counts in the next piece; a row frees
                                      nothing while its process lives
                                      mut: rows_only, close_frees, join_own_piece, no_wip
    OneCauseLane             carried  one cause lane per stopped piece, a second refused
                                      mut: cause_no_slot
    CreditNotOverdrawn       carried  the reserve's check races; the consume's compare holds it
                                      mut: credit_no_compare
P3  StopClearsOnlyWithEvidence carried a go-and-see row and a WORKING probe, both after the stop
                                      opened and after the last lease; ghosts dated apart from what
                                      the guard reads
                                      mut: rows_survive, see_before_lease, reading_survives,
                                      clear_no_rows, clear_no_reading
    StopClearsOrEscalates    liveness the escalation writer on its own timer, independent of the
                                      reading writer (Ext)
                                      mut: escalate_on_reader; ctl: drop_esc
P4  RestartRepeatsNoStep     carried  one put holds fence compare, key compare, key and effect; a
                                      kill anywhere, a replay or the fallback repeats nothing
                                      mut: unkeyed, effect_then_key, keys_per_engine
    NoLostStep               carried  a step the order moved past has its effect
                                      mut: key_then_effect
    NoWriteByLostOwner       atomic   the fence in the put; a stale engine may still act
                                      (wit_WitOldEngineStillActing) and its write does not land
                                      mut: no_fence, fence_read
    ConsumedAtMostOnce       atomic   a start delivered twice starts once: one compare on use in the
                                      consume. It rests on the store's isolation, which 7.5's
                                      contract suite ("a duplicate delivery") must show
                                      mut: no_used_check, consume_split
    NeverConsumedAfterEpoch  atomic   restates the epoch abstraction (an advance releases every live
                                      reservation); its consequence that matters is
                                      NoStartAgainstRules
                                      mut: no_epoch

Witnesses (each must be violated, so what it names is reachable): an ordinary lane runs during a
safety failure and on a stale table, a lane of piece 2 runs under a stop on piece 1, a start with no
answer, admitted plus logged starts above the limit in an outage, the limit reached, a stop cleared
and escalated, the holder starting piece 2, a step landing after the stop and one landing during an
outage, the old engine still acting after a handover, the fallback advancing, a duplicate delivery,
a kept fix and a kept piece's cause lane each running under its own stop, and stops on 1, 8 and 9
open at once with each cause lane running.

Fairness (Ext): a request is asked again until it starts (strong: through the store or with no
answer); an outage ends ("storeback", dropped in live_ext_outage so it may last for ever); the
escalation timer fires (strong). No clause assumes a lane fixes anything, a reading writer comes
back or an operator acts. ctl_ext_drop_esc and ctl_ext_weak_retry show two of the clauses are needed.

Choices the text leaves open, stated for the reader
---------------------------------------------------

- The fence under an outage is no longer open: it fails closed (RULING 2026-11-06T17:52:26Z, D1
  above). A start with no answer still proceeds, logged; an outward step with no answer does not.
  Every fenced step waits during an outage, whether or not its piece is stopped, since the lane
  cannot know. No liveness property says a waiting step is taken once the store is back.
- An unanswered start is not counted by the gate, so AliveWithinLimit counts admitted processes;
  wit_WitOutageOverLimit shows admitted plus logged starts above the limit. 7.1 counts them on the
  floor and 7.5's hourly reconcile catches them; neither is modelled.
- A stop's escalation row is written in the store, so a stop under an outage that never ends neither
  clears nor escalates; live_ext_outage checks AdmissibleStarts only.
- A join's first task is the next piece's work for its stop as well as for its count.
- The bounded repair lane's "a second is refused" is counted as a WIP reason.
- The holder's step writes "started"; with no rollback in the model, an advance at a step is never
  needed by a property (it only enables), and neither is the handover's.

Not modelled
------------

Class shares and the fix-forward share, the freed-place guard's per-class pools (one credit), the
third-lease and five-whys variants, wall-clock values and the reservation lifetime against the write
rate (7.1's live drill owns it), debits taken afterwards, starts that bypass admission (7.5's
reconcile detects them), a lane that never reaches a fence, the sweep and read-back, when a stop may
open (3.5), BUILDING, the 24-hour window, the 7-day comparison and decision 5's observe mode (it only
removes refusals), dependency repair, the read-only principal, cloud lanes, stop groups by cause key,
the M0-PASS condition of step 4a, and conformance of anything.

Epoch writes not modelled (reader F4). 7.1's epoch names every write that changes a fact the
decision reads. The model has a write for a stop opened (Breaks) and cleared (Clears), a piece
started (StepPut), a handover (Handover), a safety failure opened or closed and the table
republished. It has none for: a piece made live, a piece rolled back, a change of a task's approved
scope (sc is a constant of each kind), and a change of a path's enforcement (observe or enforce,
decision 5, not modelled). "started" only grows. So no no_advance_* mutation exists for these
writes, and NoStartAgainstRules covers them only by analogy with the four advances it does break
(no_advance_stop, no_advance_clear, no_advance_safety, no_advance_publish), not by a run.

Scopes, every one in its cfg; results/SUMMARY.txt gives workers and states per run. Pieces are
1..MaxPiece; MaxPiece is 8 or 9, the kept pieces.

  Base_admit    o1, o2, join, piece-2 ordinary, two cause lanes of piece 1, a safety cause lane, two
                repair lanes; stops on 1 and 2; safety, stale table, credit, escalation, the holder's
                step; the store always answers
  Base_outage   o1, o2, join, piece-2 ordinary, two cause lanes; the store goes down and back
  Base_fence    o1 and the join with their outward steps as two actions, outage on
  Base_holder   a continuation lane delivered twice, piece-2 ordinary, two steps, crash, restart,
                handover, outage
  Base_wip2     three ordinaries and the join, limit 2
  Base_kept     o1 and the cause lanes of 1, 8 and 9 (c1, c8, c3) and a kept fix on 9 (kf); pieces
                1..4 with 3 and 4 kept; stops on 1, 8 and 9; outage, credit, the holder's step,
                escalation; declared infeasible below
  Base_kept_core Base_kept without the escalation timer and the holder's step (both are in
                Base_admit with the stops on 1 and 2)
  Base_union    every class and environment at once; declared infeasible below, kept so the
                digest names it
  live_ext_*    o1, a cause lane, a safety cause lane, a repair lane, a stop on 1, outage

INFEASIBLE declarations, read by the test (one line each; such a config is not proven and has no
result that says it holds):
INFEASIBLE Base_kept: it does not fit a lab unit. At ae740dc42's bytes on h053 with 8 workers it ran 44 minutes and was stopped at BFS depth 20 with 13,000,684 distinct states, 4,073,970 still queued and the queue still growing by about 80,000 a minute, with no violation found up to there; a bus lab unit is capped at 2 hours. Base_kept_core, the same tasks, pieces and stops without the escalation timer and the holder's step, holds. What Base_kept adds over it, escalation and the holder's step beside stops on the kept pieces, is not covered by a run.
INFEASIBLE Base_union: not rerun at round 4's bytes; what follows was measured at round 3's. It does not fit a lab unit. Every behaviour of Base_admit is one of its behaviours with the extra tasks idle, so it has at least Base_admit's 8,512,000 distinct states (38 min on 8 workers), and it adds three tasks (c3, kf, g), a third stoppable piece, two outward steps of two actions, a duplicate delivery, outages, crashes and the handover on top. At 6403d05a5's bytes on h053 it reached 720,548 distinct states with 574,187 still queued after 7 min on 4 workers (stopped to keep h053 under 85C), and 5,606,682 with 4,302,254 queued and growing after 22 min on 8 workers in an earlier run whose bytes are unconfirmed (its run directory is gone); its size is estimated, not measured. Each environment holds in its own config (Base_admit, Base_outage, Base_fence, Base_holder, Base_wip2); their interleaving all at once is what no run covers.

Rerun, for each cfg:
  java -XX:+UseParallelGC -Xmx24g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 16 \
    -lncheck final -checkpoint 30 -gzip -metadir <fresh dir> -fp <n> <module>
Give -fp: TLC picks a new fingerprint polynomial on -recover unless -fp is given, and a resumed run
then explores everything again (found in round 2).
where <module> is RebuildAdmissionExt.tla for the *_ext_* configs and RebuildAdmission.tla for every
other. Exit 0 is required of Base_* and live_*, 12 of mut_* and wit_*, 13 of the *_ext_* breaks and of
mut_p1_down_proceeds, whose StepFence is an action property.

results/model.sha256 is the sha256 of RebuildAdmission.tla, then RebuildAdmissionExt.tla, then every
*.cfg in sorted name order, as bytes with nothing between them (section 12's recipe);
test_rebuild_admission_model.py recomputes it. results/SUMMARY.txt binds each result to the commit
it ran at and the bytes of its module and cfg.
