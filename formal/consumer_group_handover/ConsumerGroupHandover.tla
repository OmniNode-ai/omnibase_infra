------------------------ MODULE ConsumerGroupHandover ------------------------
(* The consumer-group offset handover of the one-runtime fold (task T8 of    *)
(* the one-runtime plan, OMN-21288), across a fold, a rollback and a second  *)
(* fold. T9 (the worker's five consumers) reuses it.                         *)
(*                                                                            *)
(* Two sides consume one topic partition. "old" is the retired side: the     *)
(* standalone container and its consumer group. "new" is the runtime side:   *)
(* the main runtime's consumer group, which exists only after a fold. A      *)
(* group with no committed offset starts at the bus default, `latest`, for   *)
(* "new"; the retired container started at `earliest` (the handover's        *)
(* from_auto_offset_reset). A producer publishes at any time, including      *)
(* during a handover. A consumer processes the next event of its group and   *)
(* commits it in one step: it runs the effect (or the UPSERT of a            *)
(* projection) and moves its group's committed offset.                       *)
(*                                                                            *)
(* The operator asks for three changes in turn: fold (cur old -> new),       *)
(* rollback (new -> old) and a second fold (old -> new). Each is one         *)
(* cutover, a procedure of steps that a crash can interrupt between any two: *)
(*   Fence   stop BOTH sides' consumers and mark both fenced                 *)
(*   Copy    write the destination group's commit                            *)
(*   Start   join the destination group (an uncommitted group starts at its  *)
(*           reset policy) and unfence                                       *)
(*   Record  the deployed side becomes the destination                       *)
(* A crash of the procedure loses its place (phase back to idle), leaves the *)
(* consumers and the commits as they are, and the retry begins at Fence. A   *)
(* consumer can also crash and is restarted by the container policy.         *)
(*                                                                            *)
(* Two kinds of consumer share the model through the constant Kind:          *)
(*   effect  a consumer whose handler cannot tolerate a repeat (T8's effects *)
(*           and worker). The rule is the plan's T8 rule: every cutover       *)
(*           fences both sides and copies the commit of the side that LAST   *)
(*           RAN onto the destination, over whatever the destination holds.  *)
(*           A rollback is a cutover to "old", so it carries the reverse     *)
(*           handover.                                                       *)
(*   writer  a projection writer that UPSERTs on a deterministic key (T5,    *)
(*           omnibase_infra#5523, plan_offset_handover): copy "old" onto     *)
(*           "new" only when "new" has no commit; no reverse handover. A      *)
(*           rollback or second fold replays events, and the UPSERT absorbs  *)
(*           them.                                                           *)
(*                                                                            *)
(* Property                  Kind    Config that must break it               *)
(* NoEventLost               effect  mut_no_copy (the cutover copies         *)
(*                                   nothing, so "new" joins at latest)      *)
(* NoDoubleEffect            effect  mut_no_reverse (a rollback copies       *)
(*                                   nothing back onto "old"),               *)
(*                                   mut_recopy_fixed (a retry copies from   *)
(*                                   the side deployed before, not the side  *)
(*                                   that last ran)                          *)
(* NoDoubleProjection        writer  mut_insert_projection (the projection   *)
(*                                   inserts instead of upserting)           *)
(* FencedNoCommit            effect  mut_no_stop (the fence marks the sides  *)
(*                                   but stops no consumer)                  *)
(* SecondFoldFresh           effect  mut_stale_trust (the writers' rule on   *)
(*                                   the forward copy: a committed "new"     *)
(*                                   keeps its stale commit)                 *)
(* FoldWitness, RollbackWitness, SecondFoldWitness, WriterReplayWitness      *)
(*                           witness cfgs wit_*.cfg; each violation shows    *)
(*                           the path completes                              *)
(*                                                                            *)
(* NoDoubleEffect holds only for Kind = effect, because a writer's replay is *)
(* by design; for writers NoDoubleProjection takes its place. SecondFoldFresh*)
(* is stated for Kind = effect only. Each is vacuous in the other Kind's     *)
(* configs and the cfgs list all invariants regardless.                      *)
(*                                                                            *)
(* Bounds: one partition, 6 events, 3 operator changes (fold, rollback,      *)
(* second fold), 4 crashes of the procedure, 3 consumer crashes. results/    *)
(* holds TLC output for every cfg and model.sha256.                          *)
(*                                                                            *)
(* Not modelled:                                                              *)
(*  - partitions beyond the one modelled: the real handover is per partition *)
(*    and plan_offset_handover treats each alone, so the safety argument     *)
(*    here is per partition; a cutover that copies some partitions and       *)
(*    crashes before the rest is covered only by the retry re-copying all.   *)
(*  - rebalance timing and group membership: a group is either running or    *)
(*    not; the broker accepts a commit for a group with no member.           *)
(*  - the window between a handler's effect and its commit: process and      *)
(*    commit are one step, so a crash never repeats an event. A handler that *)
(*    cannot tolerate the Kafka at-least-once redelivery needs an idempotency*)
(*    key, which the plan's T8 text already requires.                        *)
(*  - retention: committed offsets below the first retained offset (the      *)
(*    handler's rule 3) and offsets of a deleted group.                      *)
(*  - OMN-21932: a retired group named by a derived name that cannot         *)
(*    express a literal id. The model takes the retired group as given.      *)
(*  - the two sides running at once on purpose: the fence removes that case. *)
(*  - image and overlay rolling back together; only the offsets are modelled.*)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS N,               \* events the producer publishes
          MaxOps,          \* operator changes: fold, rollback, second fold
          MaxOrchCrash,    \* crashes of the cutover procedure
          MaxConsCrash,    \* crashes of a consumer
          Kind,            \* "effect" or "writer"
          MUT              \* "none" or the name of the mutation

ASSUME Kind \in {"effect", "writer"}
ASSUME MaxOps = 3

Sides == {"old", "new"}
Other(s) == IF s = "old" THEN "new" ELSE "old"
Min(a, b) == IF a < b THEN a ELSE b

VARIABLES pub,           \* events published; event e sits at offset e - 1
          comm,          \* comm[s]: events 1..comm[s] committed by group s, -1 = no commit
          eff,           \* eff[e]: times event e's effect ran (capped at 2)
          rows,          \* rows[e]: projection rows keyed on event e (capped at 2)
          stamp,         \* ghost: operator change count when event e first ran, -1 = not yet
          up,            \* sides whose consumer runs
          fenced,        \* sides fenced by the cutover
          cur,           \* the deployed side
          want,          \* the side the operator asked for
          phase,         \* cutover phase: idle fenced copied started
          ops,           \* operator changes made
          lastRan,       \* the side whose consumer committed last
          fencedCommit,  \* ghost: a fenced side committed
          orchCrashes, consCrashes

vars == <<pub, comm, eff, rows, stamp, up, fenced, cur, want, phase, ops,
          lastRan, fencedCommit, orchCrashes, consCrashes>>

Phases == {"idle", "fenced", "copied", "started"}

TypeOK ==
  /\ pub \in 0..N
  /\ comm \in [Sides -> -1..N]
  /\ eff \in [1..N -> 0..2]
  /\ rows \in [1..N -> 0..2]
  /\ stamp \in [1..N -> -1..MaxOps]
  /\ up \subseteq Sides
  /\ fenced \subseteq Sides
  /\ cur \in Sides
  /\ want \in Sides
  /\ phase \in Phases
  /\ ops \in 0..MaxOps
  /\ lastRan \in Sides
  /\ fencedCommit \in BOOLEAN
  /\ orchCrashes \in 0..MaxOrchCrash
  /\ consCrashes \in 0..MaxConsCrash

Init ==
  /\ pub = 0
  /\ comm = [s \in Sides |-> IF s = "old" THEN 0 ELSE -1]
  /\ eff = [e \in 1..N |-> 0]
  /\ rows = [e \in 1..N |-> 0]
  /\ stamp = [e \in 1..N |-> -1]
  /\ up = {"old"}
  /\ fenced = {}
  /\ cur = "old"
  /\ want = "old"
  /\ phase = "idle"
  /\ ops = 0
  /\ lastRan = "old"
  /\ fencedCommit = FALSE
  /\ orchCrashes = 0
  /\ consCrashes = 0

\* ---- the producer
Publish ==
  /\ pub < N
  /\ pub' = pub + 1
  /\ UNCHANGED <<comm, eff, rows, stamp, up, fenced, cur, want, phase, ops,
                 lastRan, fencedCommit, orchCrashes, consCrashes>>

\* ---- a consumer: run the effect and commit, in one step
Process(s) ==
  /\ s \in up
  /\ comm[s] < pub
  /\ LET e == comm[s] + 1
     IN /\ eff' = [eff EXCEPT ![e] = Min(@ + 1, 2)]
        /\ rows' = IF MUT = "insert_projection"
                     THEN [rows EXCEPT ![e] = Min(@ + 1, 2)]
                     ELSE [rows EXCEPT ![e] = 1]
        /\ stamp' = IF stamp[e] < 0 THEN [stamp EXCEPT ![e] = ops] ELSE stamp
        /\ comm' = [comm EXCEPT ![s] = e]
  /\ lastRan' = s
  /\ fencedCommit' = (fencedCommit \/ s \in fenced)
  /\ UNCHANGED <<pub, up, fenced, cur, want, phase, ops, orchCrashes, consCrashes>>

ConsumerCrash(s) ==
  /\ s \in up
  /\ consCrashes < MaxConsCrash
  /\ up' = up \ {s}
  /\ consCrashes' = consCrashes + 1
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, fenced, cur, want, phase, ops,
                 lastRan, fencedCommit, orchCrashes>>

\* The container policy restarts the deployed side's consumer when no
\* cutover is pending; it rejoins its group at the group's commit.
Restart(s) ==
  /\ phase = "idle"
  /\ cur = want
  /\ s = cur
  /\ s \notin up
  /\ up' = up \cup {s}
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, fenced, cur, want, phase, ops,
                 lastRan, fencedCommit, orchCrashes, consCrashes>>

\* ---- the operator asks for the other side: fold, rollback, second fold
Ask ==
  /\ phase = "idle"
  /\ cur = want
  /\ ops < MaxOps
  /\ want' = Other(cur)
  /\ ops' = ops + 1
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, up, fenced, cur, phase, lastRan,
                 fencedCommit, orchCrashes, consCrashes>>

\* ---- the cutover
\* Fence both sides: no consumer runs, both are marked. mut_no_stop marks
\* them and stops nothing.
Fence ==
  /\ phase = "idle"
  /\ cur # want
  /\ phase' = "fenced"
  /\ fenced' = Sides
  /\ up' = IF MUT = "no_stop" THEN up ELSE {}
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, cur, want, ops, lastRan,
                 fencedCommit, orchCrashes, consCrashes>>

\* The side whose commit is copied: the side that last ran. mut_recopy_fixed
\* copies from the side deployed before the cutover instead.
Src == IF MUT = "recopy_fixed" THEN cur ELSE lastRan

\* The destination group's commit after the copy.
CopyValue(src, dst) ==
  IF Kind = "writer"
    THEN IF dst = "new" /\ comm["new"] < 0 THEN comm[src] ELSE comm[dst]
  ELSE IF MUT = "no_copy" THEN comm[dst]
  ELSE IF MUT = "no_reverse" /\ dst = "old" THEN comm[dst]
  ELSE IF MUT = "stale_trust" /\ dst = "new" /\ comm["new"] >= 0 THEN comm[dst]
  ELSE comm[src]

Copy ==
  /\ phase = "fenced"
  /\ phase' = "copied"
  /\ comm' = [comm EXCEPT ![want] = CopyValue(Src, want)]
  /\ UNCHANGED <<pub, eff, rows, stamp, up, fenced, cur, want, ops, lastRan,
                 fencedCommit, orchCrashes, consCrashes>>

\* Join the destination group: an uncommitted group starts at its reset
\* policy, "old" at earliest (0) and "new" at latest (the end).
Start ==
  /\ phase = "copied"
  /\ phase' = "started"
  /\ up' = {want}
  /\ fenced' = {}
  /\ comm' = IF comm[want] >= 0 THEN comm
             ELSE [comm EXCEPT ![want] = IF want = "old" THEN 0 ELSE pub]
  /\ UNCHANGED <<pub, eff, rows, stamp, cur, want, ops, lastRan, fencedCommit,
                 orchCrashes, consCrashes>>

Record ==
  /\ phase = "started"
  /\ phase' = "idle"
  /\ cur' = want
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, up, fenced, want, ops, lastRan,
                 fencedCommit, orchCrashes, consCrashes>>

\* The procedure dies between two steps and loses its place. The consumers,
\* the fences and the commits stay as they are.
OrchCrash ==
  /\ phase # "idle"
  /\ orchCrashes < MaxOrchCrash
  /\ phase' = "idle"
  /\ orchCrashes' = orchCrashes + 1
  /\ UNCHANGED <<pub, comm, eff, rows, stamp, up, fenced, cur, want, ops,
                 lastRan, fencedCommit, consCrashes>>

Next ==
  \/ Publish
  \/ \E s \in Sides : Process(s) \/ ConsumerCrash(s) \/ Restart(s)
  \/ Ask \/ Fence \/ Copy \/ Start \/ Record \/ OrchCrash

Spec == Init /\ [][Next]_vars

\* ---- invariants
\* The sides whose commit a consumer reads next: the running ones, and the
\* destination once its commit is written.
Live == up \cup (IF phase \in {"copied", "started"} THEN {want} ELSE {})

NoEventLost == \A s \in Live : \A e \in 1..comm[s] : eff[e] >= 1

NoDoubleEffect == Kind # "effect" \/ \A e \in 1..N : eff[e] <= 1

NoDoubleProjection == \A e \in 1..N : rows[e] <= 1

FencedNoCommit == ~fencedCommit

SecondFoldFresh ==
  \/ Kind # "effect"
  \/ ~(ops = 3 /\ phase = "copied")
  \/ comm["new"] >= comm[lastRan]

\* ---- witnesses; a violation shows the path completing
FoldWitness ==
  ~(ops >= 1 /\ cur = "new" /\ phase = "idle" /\ \E e \in 1..N : stamp[e] = 1)

RollbackWitness ==
  ~(ops >= 2 /\ cur = "old" /\ phase = "idle" /\ \E e \in 1..N : stamp[e] = 2)

SecondFoldWitness ==
  ~(ops = 3 /\ cur = "new" /\ phase = "idle" /\ pub = N
    /\ (\A e \in 1..N : eff[e] = 1)
    /\ (\A k \in 0..3 : \E e \in 1..N : stamp[e] = k))

WriterReplayWitness == \A e \in 1..N : eff[e] <= 1
=============================================================================
