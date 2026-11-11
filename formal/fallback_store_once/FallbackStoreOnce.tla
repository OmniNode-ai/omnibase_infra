---------------------------- MODULE FallbackStoreOnce ----------------------------
(* Store-once of the fallback result (OMN-22258, task T0 of the transcript    *)
(* intent-to-outcome graph plan, task T4). Three calls arrive at the same     *)
(* time for ONE fallback key. c1 computed the result "a", c2 computed "b",    *)
(* c3 computed "a" again. The store keeps one result per key. A call whose    *)
(* result is the stored one is answered `same` (idempotent), the first call   *)
(* to reach the store is answered `stored`, and a call with a different       *)
(* result is `refused` and leaves the stored one alone.                       *)
(*                                                                            *)
(* The write is one atomic put-if-absent on the store (the model's Put). The  *)
(* controls remove that: write on arrival overwrites, and lookup-then-put     *)
(* reads the key and writes later, so two calls that both saw it empty both   *)
(* write.                                                                     *)
(*                                                                            *)
(* Property                  Kind       Config that must break it             *)
(* OneStoredResult           invariant  mut_write_on_arrival,                  *)
(*                                      mut_lookup_then_put (two different    *)
(*                                      results were written)                 *)
(* OneWrite                  invariant  mut_lookup_then_put_writes (two calls   *)
(*                                      that saw the key empty both write,    *)
(*                                      even the same value)                  *)
(* DifferentRefused          invariant  mut_write_on_arrival_answered (a call  *)
(*                                      stored or same whose result is not   *)
(*                                      the one now stored)                   *)
(* SomeResultStored          property   mut_no_fairness (SpecNoFair)          *)
(* EveryCallAnswered         property   mut_no_fairness (SpecNoFair: a call    *)
(*                                      may never be served)                  *)
(* RefusedWitness            witness    wit_refused (Model.cfg values; a       *)
(*                                      different second result is refused)   *)
(* SameWitness               witness    wit_same (an equal second result is    *)
(*                                      answered same)                        *)
(*                                                                            *)
(* Bounds: 3 calls, 2 distinct results, one key. results/ holds TLC output    *)
(* for every cfg and model.sha256 (sha256 of this file then the cfg files in  *)
(* sorted order).                                                             *)
(*                                                                            *)
(* Not modelled: the key's lifetime (a key never expires here), a crash of a  *)
(* call between its compute and its put (the put is its first effect), and    *)
(* the store's own failure.                                                   *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANT MUT      \* "none" | "write_on_arrival" | "lookup_then_put"

Calls == {"c1", "c2", "c3"}
Val   == [c \in Calls |-> IF c = "c2" THEN "b" ELSE "a"]
Empty == "none"

VARIABLES stored,   \* the value now stored for the key, or Empty
          wrote,    \* ghost: every value ever written to the key
          writes,   \* ghost: number of writes to the key
          pc,       \* [Calls -> "new" | "looked" | "done"]
          seen,     \* [Calls -> value the lookup returned] (lookup_then_put only)
          out       \* [Calls -> "wait" | "stored" | "same" | "refused"]

vars == <<stored, wrote, writes, pc, seen, out>>

Init == /\ stored = Empty
        /\ wrote = {}
        /\ writes = 0
        /\ pc = [c \in Calls |-> "new"]
        /\ seen = [c \in Calls |-> Empty]
        /\ out = [c \in Calls |-> "wait"]

\* The correct put: one atomic put-if-absent.
Put(c) ==
  /\ MUT = "none"
  /\ pc[c] = "new"
  /\ pc' = [pc EXCEPT ![c] = "done"]
  /\ IF stored = Empty
       THEN /\ stored' = Val[c]
            /\ wrote' = wrote \cup {Val[c]}
            /\ writes' = writes + 1
            /\ out' = [out EXCEPT ![c] = "stored"]
       ELSE /\ UNCHANGED <<stored, wrote, writes>>
            /\ out' = [out EXCEPT ![c] = IF stored = Val[c] THEN "same" ELSE "refused"]
  /\ UNCHANGED seen

\* Control: the put overwrites whatever is there.
WriteOnArrival(c) ==
  /\ MUT = "write_on_arrival"
  /\ pc[c] = "new"
  /\ pc' = [pc EXCEPT ![c] = "done"]
  /\ stored' = Val[c]
  /\ wrote' = wrote \cup {Val[c]}
  /\ writes' = writes + 1
  /\ out' = [out EXCEPT ![c] = "stored"]
  /\ UNCHANGED seen

\* Control: look the key up, then, in a later step, write if the lookup was empty.
Lookup(c) ==
  /\ MUT = "lookup_then_put"
  /\ pc[c] = "new"
  /\ pc' = [pc EXCEPT ![c] = "looked"]
  /\ seen' = [seen EXCEPT ![c] = stored]
  /\ UNCHANGED <<stored, wrote, writes, out>>

WriteAfterLookup(c) ==
  /\ MUT = "lookup_then_put"
  /\ pc[c] = "looked"
  /\ pc' = [pc EXCEPT ![c] = "done"]
  /\ IF seen[c] = Empty
       THEN /\ stored' = Val[c]
            /\ wrote' = wrote \cup {Val[c]}
            /\ writes' = writes + 1
            /\ out' = [out EXCEPT ![c] = "stored"]
       ELSE /\ UNCHANGED <<stored, wrote, writes>>
            /\ out' = [out EXCEPT ![c] = IF seen[c] = Val[c] THEN "same" ELSE "refused"]
  /\ UNCHANGED seen

Next == \E c \in Calls : Put(c) \/ WriteOnArrival(c) \/ Lookup(c) \/ WriteAfterLookup(c)

Spec       == Init /\ [][Next]_vars /\ WF_vars(Next)
SpecNoFair == Init /\ [][Next]_vars

TypeOK == /\ stored \in {Empty, "a", "b"}
          /\ wrote \subseteq {"a", "b"}
          /\ writes \in 0..3
          /\ pc \in [Calls -> {"new", "looked", "done"}]
          /\ seen \in [Calls -> {Empty, "a", "b"}]
          /\ out \in [Calls -> {"wait", "stored", "same", "refused"}]

OneStoredResult == Cardinality(wrote) <= 1

\* One write reaches the key, whatever its value.
OneWrite == writes <= 1

DifferentRefused == \A c \in Calls :
    out[c] \in {"stored", "same"} => Val[c] = stored

EveryCallAnswered == \A c \in Calls : <>(out[c] # "wait")

\* A result is stored: a store that refuses every call does not pass.
SomeResultStored == <>(stored # Empty)

\* Witnesses: each is an invariant that the model must violate.
RefusedWitness == \A c \in Calls : out[c] # "refused"
SameWitness    == \A c \in Calls : out[c] # "same"

=============================================================================
