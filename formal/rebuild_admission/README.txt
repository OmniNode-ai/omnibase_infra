RebuildAdmission: M0 of the subsystem rebuild plan (OMN-22478), round 3
======================================================================

Admission, the stop and the holder as one state machine, written to the plan's text of 11-06
(sections 6.2, 7.1 and 7.5 at plan head 10c08fa2, text hash 52faf70168ffd3f1 by section 12's
recipe). It implements and wires nothing. Results are in results/SUMMARY.txt. A verdict from this
directory is its author's; plan section 12 requires a separate reader. second_read/ is an earlier
reader's evidence against round 1, left unchanged.

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
    NoStepBeginsAfterStop    atomic   the fence in the store's write
                                      mut: fence_off, fence_split
    AtMostOneStepAfterStop   carried  the in-flight bound: one step begun before the stop may land
                                      mut: fence_once
    AdmissibleStarts         liveness an admissible request starts, and with no answer it starts
                                      logged (Ext)
                                      mut: down_refuses; ctl: weak_retry
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
and escalated, the holder starting piece 2, a step landing after the stop and a logged step under a
stop, the old engine still acting after a handover, the fallback advancing, a duplicate delivery.

Fairness (Ext): a request is asked again until it starts (strong: through the store or with no
answer); an outage ends ("storeback", dropped in live_ext_outage so it may last for ever); the
escalation timer fires (strong). No clause assumes a lane fixes anything, a reading writer comes
back or an operator acts. ctl_ext_drop_esc and ctl_ext_weak_retry show two of the clauses are needed.

Choices the text leaves open, stated for the reader
---------------------------------------------------

- The fence under an outage: 7.1 puts the fence in the store's write and says a start with no answer
  proceeds, logged. The model treats a step the same way: with no answer it proceeds and is logged
  (fl = "log"), so NoStepBeginsAfterStop and the in-flight bound hold while the store answers, and
  wit_WitLoggedStepUnderStop shows a logged step under a stop.
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
  Base_union    every class and environment at once; declared infeasible below, kept so the
                digest names it
  live_ext_*    o1, a cause lane, a safety cause lane, a repair lane, a stop on 1, outage

INFEASIBLE declarations, read by the test (one line each; such a config is not proven and has no
result that says it holds):
INFEASIBLE Base_union: it does not fit a lab unit. Every behaviour of Base_admit is one of its behaviours with the extra tasks idle, so it has at least Base_admit's 8,512,000 distinct states (38 min on 8 workers), and it adds three tasks (c3, kf, g), a third stoppable piece, two outward steps of two actions, a duplicate delivery, outages, crashes and the handover on top. At 6403d05a5's bytes on h053 it reached 720,548 distinct states with 574,187 still queued after 7 min on 4 workers (stopped to keep h053 under 85C), and 5,606,682 with 4,302,254 queued and growing after 22 min on 8 workers in an earlier run whose bytes are unconfirmed (its run directory is gone); its size is estimated, not measured. Each environment holds in its own config (Base_admit, Base_outage, Base_fence, Base_holder, Base_wip2); their interleaving all at once is what no run covers.

Rerun, for each cfg:
  java -XX:+UseParallelGC -Xmx24g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 16 \
    -lncheck final -checkpoint 30 -gzip -metadir <fresh dir> -fp <n> <module>
Give -fp: TLC picks a new fingerprint polynomial on -recover unless -fp is given, and a resumed run
then explores everything again (found in round 2).
where <module> is RebuildAdmissionExt.tla for the *_ext_* configs and RebuildAdmission.tla for every
other. Exit 0 is required of Base_* and live_*, 12 of mut_* and wit_*, 13 of the *_ext_* breaks.

results/model.sha256 is the sha256 of RebuildAdmission.tla, then RebuildAdmissionExt.tla, then every
*.cfg in sorted name order, as bytes with nothing between them (section 12's recipe);
test_rebuild_admission_model.py recomputes it. results/SUMMARY.txt binds each result to the commit
it ran at and the bytes of its module and cfg.
