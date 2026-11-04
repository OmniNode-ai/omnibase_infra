------------------------ MODULE MirrorConditionalUpload ------------------------
(* The mirror's conditional upload (OMN-22258, task T0 of the transcript      *)
(* intent-to-outcome graph plan, task T0a). One transcript file grows on the  *)
(* local machine (its length is a count of records, 1..3). Two mirror runs, A *)
(* and B, each read the local file at some moment (a snapshot of n records)   *)
(* and upload that snapshot as the object's copy. The object store records    *)
(* the revision (record count) of the copy it holds. A run may take up to two *)
(* snapshots, so a run can read 1 record, be paused, and upload after the     *)
(* other run has uploaded 3.                                                  *)
(*                                                                            *)
(* The correct put is conditional and atomic on the store: it stores the      *)
(* copy only when the recorded revision is below the copy's, and is refused   *)
(* otherwise (the store's `if revision <` precondition, one atomic step).     *)
(*                                                                            *)
(* Property                  Kind       Config that must break it             *)
(* RecordedNeverDecreases    invariant  mut_put_unconditional (run A reads 1, *)
(*                                      run B puts 2, run A puts 1 after),    *)
(*                                      mut_check_then_put (the run reads the *)
(*                                      recorded revision, then puts          *)
(*                                      unconditionally)                      *)
(* RecordedIsLongestPut      invariant  mut_put_unconditional_longest (the    *)
(*                                      copy held is not the longest one put) *)
(* EverySnapshotCovered      property   mut_no_fairness (a snapshot is never  *)
(*                                      put or refused)                       *)
(* RefusedWitness            witness    wit_refused (Model.cfg values; a      *)
(*                                      shorter copy is refused)              *)
(* FullWitness               witness    wit_full (the longest copy, 3 records, *)
(*                                      is the one the store ends up holding) *)
(*                                                                            *)
(* Bounds: 2 runs, 2 snapshots per run, local length 1..3. results/ holds TLC *)
(* output for every cfg and model.sha256 (sha256 of this file then the cfg    *)
(* files in sorted order).                                                    *)
(*                                                                            *)
(* Not modelled: a crash of a run mid-upload (a partial object is the        *)
(* store's concern; its put is atomic), the object's content (the revision    *)
(* stands for it), and a delete of the object.                                *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANT MUT      \* "none" | "put_unconditional" | "check_then_put"

Runs   == {"A", "B"}
MaxLen == 3
MaxSnaps == 2

VARIABLES local,     \* records in the local file now
          store,     \* revision the object store records (0: no object)
          hiStore,   \* ghost: highest revision the store ever recorded
          maxPut,    \* ghost: longest copy any put has tried to store
          pc,        \* [Runs -> "idle" | "read" | "checked"]
          snap,      \* [Runs -> records in the run's snapshot]
          head,      \* [Runs -> recorded revision the run read] (check_then_put)
          snaps,     \* [Runs -> snapshots taken]
          taken,     \* ghost: every snapshot length any run took
          refusedShort  \* ghost: a put of a copy shorter than the recorded one was refused

vars == <<local, store, hiStore, maxPut, pc, snap, head, snaps, taken, refusedShort>>

Init == /\ local = 1
        /\ store = 0
        /\ hiStore = 0
        /\ maxPut = 0
        /\ pc = [r \in Runs |-> "idle"]
        /\ snap = [r \in Runs |-> 0]
        /\ head = [r \in Runs |-> 0]
        /\ snaps = [r \in Runs |-> 0]
        /\ taken = {}
        /\ refusedShort = FALSE

Max2(a, b) == IF a > b THEN a ELSE b

Grow == /\ local < MaxLen
        /\ local' = local + 1
        /\ UNCHANGED <<store, hiStore, maxPut, pc, snap, head, snaps, taken, refusedShort>>

Read(r) == /\ pc[r] = "idle"
           /\ snaps[r] < MaxSnaps
           /\ pc' = [pc EXCEPT ![r] = "read"]
           /\ snap' = [snap EXCEPT ![r] = local]
           /\ snaps' = [snaps EXCEPT ![r] = @ + 1]
           /\ taken' = taken \cup {local}
           /\ UNCHANGED <<local, store, hiStore, maxPut, head, refusedShort>>

\* Control step: the run reads the recorded revision before it puts.
Head(r) == /\ MUT = "check_then_put"
           /\ pc[r] = "read"
           /\ pc' = [pc EXCEPT ![r] = "checked"]
           /\ head' = [head EXCEPT ![r] = store]
           /\ UNCHANGED <<local, store, hiStore, maxPut, snap, snaps, taken, refusedShort>>

\* The put of the snapshot. Correct: stored only when the recorded revision is
\* below it, decided on the store in the same step.
Put(r) ==
  /\ \/ /\ MUT = "none" /\ pc[r] = "read"
     \/ /\ MUT = "put_unconditional" /\ pc[r] = "read"
     \/ /\ MUT = "check_then_put" /\ pc[r] = "checked"
  /\ pc' = [pc EXCEPT ![r] = "idle"]
  /\ maxPut' = Max2(maxPut, snap[r])
  /\ LET ok == CASE MUT = "none"           -> store < snap[r]
                 [] MUT = "put_unconditional" -> TRUE
                 [] MUT = "check_then_put" -> head[r] < snap[r]
     IN /\ store' = IF ok THEN snap[r] ELSE store
        /\ hiStore' = IF ok THEN Max2(hiStore, snap[r]) ELSE hiStore
        /\ refusedShort' = (refusedShort \/ (~ok /\ snap[r] < store))
  /\ UNCHANGED <<local, snap, head, snaps, taken>>

Next == Grow \/ \E r \in Runs : Read(r) \/ Head(r) \/ Put(r)

Spec       == Init /\ [][Next]_vars /\ WF_vars(Next)
SpecNoFair == Init /\ [][Next]_vars

TypeOK == /\ local \in 1..MaxLen
          /\ store \in 0..MaxLen
          /\ hiStore \in 0..MaxLen
          /\ maxPut \in 0..MaxLen
          /\ pc \in [Runs -> {"idle", "read", "checked"}]
          /\ snap \in [Runs -> 0..MaxLen]
          /\ head \in [Runs -> 0..MaxLen]
          /\ snaps \in [Runs -> 0..MaxSnaps]
          /\ taken \subseteq 1..MaxLen
          /\ refusedShort \in BOOLEAN

\* The recorded revision is never below one it has held.
RecordedNeverDecreases == store >= hiStore

\* The copy held is the longest one any run has tried to put.
RecordedIsLongestPut == store = maxPut

\* Every snapshot a run took is eventually covered: put, or superseded by a
\* longer copy the store already holds.
EverySnapshotCovered == \A n \in 1..MaxLen : (n \in taken) ~> (store >= n)

\* Witnesses: each is an invariant that the model must violate.
RefusedWitness == ~refusedShort
FullWitness == store # MaxLen

=============================================================================
