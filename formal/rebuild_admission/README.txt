RebuildAdmission: M0 of the subsystem rebuild plan (OMN-22478)
==============================================================

The stop gate, WIP admission and the holder as one state machine: the admission table, the holder
and fallback, the single-use start reservation. It implements and wires nothing. The plan's rule
is that no stop, admission or holder implementation starts before this model has a PASS verdict
from a second reader. The verdict and the state counts are in results/SUMMARY.txt.

The four properties, exactly
----------------------------

P1 Admission follows the order (safety failure, stale table, stop, order); no send is admitted
   against it, including a stale table with a safety-failure cause lane; no continuation runs
   during a safety failure.
     DecideFollowsOrder, NoStartAgainstOrder, SafetyCauseAdmitted, ReadOnlyAdmitted, OneRepairLane,
     NoContinuationAdmittedInSafety, NoContinuationStepInSafety, RunningContinuationSuspended (live)
P2 Claims per piece stay within the limit under concurrent admission.
     ClaimsWithinLimit, on what an independent reading counts, not on what the gate counted.
P3 A stop clears only with its evidence rows, and eventually clears or escalates.
     StopClearsOnlyWithEvidence, StopClearsOrEscalates (live)
P4 A holder restart repeats no step; exactly one of holder and fallback owns the order; a start
   reservation is consumed at most once and never after the epoch advanced.
     RestartRepeatsNoStep, ExactlyOneOwner, NoWriteByLostOwner, ConsumedAtMostOnce,
     NeverConsumedAfterEpoch, OwnerComesBack (live)

Beyond the four: FixedSafetyCloses (live), a safety failure whose cause is fixed closes.

The writers, which fix the bounds
---------------------------------

Stop state. One store under one file lock, so each write is atomic (the model's actions). Roles in
the reducer's contract: detector, swarm lane, readback, operator. Plan adds the reading writer, a
read-only go-and-see lane and a typed release. Model: Breaks, Fix, Clears, EvidenceRow, Escalates.

WIP state. Claim and terminal rows on the ledger, from the lane runner, the landing controller's
workers and hand-written lane claims, on nine start paths in the plan: the lane runner, the brief
check, three lab-fill paths, the Agent and Workflow hook, the landing fix workers, scheduled ticks,
cloud lanes. Bound: nine racing start requests against limit 2 and limit 8 (wip2, wip8), and
every class of request at once (order).

Holder state. No holder record exists yet. Plan writers: the engine holding the order, its
fallback, and the admission effect on all nine paths. Bound: two engines, one fenced handover,
crashes without bound, two order steps.

Safety state. Four sources read independently of the table, one or two failures open at once.

Parts: one spec, one model, scoped by constants
-----------------------------------------------

part        checks                       scope
order       the ghost invariants         3 ordinary + join, next-piece ordinary, 2 cause, kept,
                                         continuation, repair, safety cause lanes (10 requests);
                                         1 safety failure; stops on 2 pieces; writer can die
table       the table for every kind     all kinds over all reachable facts; 2 safety failures;
            in every reachable state     stops on 3 pieces
wip2, wip8  P2 under the race            9 ordinary requests + 2 joins + next-piece ordinary; limit
                                         2 with crashes and handover, limit 8
holder      P4                           3 ordinary, 2 joins, next-piece ordinary; crash, restart,
                                         fenced handover, 2 steps
stop        P3 evidence                  3 cause lanes, 1 ordinary; stops on 3 pieces; typed rows
live_stop   P3 liveness                  2 cause lanes and a repair lane; stops on 2 pieces; typed rows
live_safety P1 suspend, FixedSafetyCloses continuation, safety cause and repair lanes; writer can die
live_owner  P4 owner comes back          1 ordinary, 1 join; crash, restart, handover

Why classes can be split: a request's decision reads the gate's facts and the occupancy of its own
class (WIP room, cause slot, repair slot). Every gate fact can change in every state whatever the
other classes do, so another class adds no behaviour a part does not already have, except a fix
or a repair, which only enables. The order part holds them together as a check on that argument.

Pieces are 1..3 with piece 2 as "piece 7": below, at and above the step-2 line, a lower and a higher
stop, and an unfinished piece below both.

Abstractions: no epoch number (an advance kills live reservations, which is what comparing decides);
no history counters (recurrence is unbounded, the state space is finite); WORKING 24 h is one
published reading; time is actions plus fairness.

Fairness, stated in the module's Fairness definition. Nothing assumes a cause lane fixes anything,
that an operator acts, or that the reading writer stops dying (a writer that dies before every
publish forever is excluded by strong fairness on Publish only for FixedSafetyCloses' sake; such a
writer's stop still escalates). Escalation is assumed to run off a timer independent of the table,
the holder and the reading writer: the plan does not yet name its writer.

The obligations: what the passing configuration needs that the plan's text does not state
-------------------------------------------------------------------------------------------

O1 The WIP check counts live reservations under the same transaction as the reservation. (The
   text counts claim rows, and writes the claim after the check.)
O2 Every write to the gate's state (safety failure opened or closed, stop opened or cleared, table
   stale or republished, a reading) advances the epoch, not only a piece advance, or the consuming
   step re-decides. (The text names the epoch at consumption and does not say what advances it.)
O3 Each step of a running continuation lane checks for an open safety failure, and a sweep suspends
   running ones. (The text says a safety failure "suspends them" and names no mechanism.)
O4 While a safety failure is open and the table is stale, the bounded repair lane is admitted as
   a cause lane of that failure. (Step 1 refuses it today; see finding_f4_wedge.)
O5 Step keys live in one store both engines read, and a handover advances the epoch. (The text
   keys steps per engine's own store: Temporal history, or ledger rows.)
O6 The gate and the independent reading count a join's first task in the same piece. (Step 4b says
   the next piece; the W3 caption says the upstream piece.)
O7 A stop's escalation has a named writer and a time. (The plan names one escalation only, for the
   fallback's failure; the stop reducer's contract has a stale decision row, not yet wired.)

Mutations: each must produce a counterexample; results/<name>.out keeps it
---------------------------------------------------------------------------

P1 mut_p1_stale_before_safety         stale check before safety: the cause lane is refused
   mut_p1_stop_before_stale           stop check before stale
   mut_p1_cont_in_safety              continuation admitted during a safety failure
   mut_p1_cont_no_step_fence          the plan as written, O3
   mut_p1_gate_no_epoch               the plan as written, O2: reserve, safety failure opens, start
   mut_p1_repair_unbounded            two repair lanes
   mut_p1_no_suspend                  a running continuation is never suspended
P2 mut_p2_wip_counts_rows_only        the plan as written, O1: three reserve under limit 2
   mut_p2_wip_counts_rows_only_limit8 the same at limit 8 with nine racers
   mut_p2_join_counted_elsewhere      the plan as written, O6
P3 mut_p3_clear_no_rows               a WORKING reading alone clears
   mut_p3_clear_no_reading            rows alone clear
   mut_p3_see_before_lease            a go-and-see row older than the last lease counts
   mut_p3_no_escalation               the plan as written, O7: a stop whose cause is not fixed stays
P4 mut_p4_unkeyed_step                a restart repeats the step
   mut_p4_keys_per_engine             the plan as written, O5: the fallback repeats the holder's step
   mut_p4_no_fence                    both engines own the order
   mut_p4_reservation_reused          a redelivered start consumes twice
   mut_p4_consume_after_epoch         consumed after an advance
   mut_p4_handover_keeps_reservation  the plan as written, O5: consumed across a handover
   mut_p4_no_restart                  a dead owner never comes back
Finding, not one of the four:
   finding_f4_wedge                   the plan as written, O4: the reading writer dies, a safety failure
                                      is open and its cause fixed; step 1 refuses the repair lane; the
                                      failure needs a WORKING reading to close; a stale table reads
                                      none; only the operator gets out

wit_*.cfg are reachability witnesses: each must be violated, so no pass is vacuous.

Limits
------

The bounds above are the check; a larger plan (more pieces, more classes) is not proved by it.
Liveness runs on smaller scopes than safety. The model is of the plan's protocol, not of any node:
the reducers, the holder and the admission effect do not exist in this shape yet, and their
conformance is theirs to show.

Rerun, for each cfg (no runner script lives in this repository, by rule):
  java -XX:+UseParallelGC -Xmx32g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 16 \
    -lncheck final -metadir <fresh dir> RebuildAdmission.tla
Exit 0 is required of Base_*, exit 12 or 13 of mut_*, wit_* and finding_*.
