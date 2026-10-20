---------------------------- MODULE PreprSlotLease ----------------------------
(* The pre-PR slot lease of scripts/runtime_build/prepr_runtime_pool.py      *)
(* (OMN-20085). A lease is a directory on the slot host holding lease.json.  *)
(* Every shell step of acquire_lease and release_lease is one action here;   *)
(* nothing between two steps is atomic.                                      *)
(*                                                                            *)
(* acquire_lease: mkdir the lease; on success write lease.json. Otherwise    *)
(* read lease.json; a lease judged expired is broken under the guard         *)
(* directory <lease>.break: re-read, and only when lease.json is still the   *)
(* one judged expired, rm -rf, mkdir, write; then rmdir the guard.           *)
(* release_lease: under the same guard, read lease.json and rm -rf it when   *)
(* it is this holder's.                                                      *)
(*                                                                            *)
(* The lease starts held by Old, already expired and still running, so it    *)
(* can release while acquirers race to break it. A token is the holder that  *)
(* wrote it (an acquirer writes at most once). Expire is time passing.       *)
(*                                                                            *)
(* Property     Kind       Mutation that must break it                       *)
(* OneHolder    invariant  mut_two_step_break, mut_no_reread,                *)
(*                         mut_unguarded_release (the driver before          *)
(*                         OMN-20085's release fix), mut_break_live          *)
(* Terminates   temporal   mut_leak_guard (a refused break keeps the guard)  *)
(*                                                                            *)
(* Bounds: 3 acquirers and the expired holder. results/ holds TLC output for *)
(* Model.cfg and each mut_*.cfg, host paths cut to basenames, and           *)
(* model.sha256, the sha256 of this file then the cfg files in sorted       *)
(* order. Each run, from this directory:                                     *)
(*   java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 2 \             *)
(*     -metadir "$(mktemp -d)" PreprSlotLease.tla                            *)
EXTENDS Naturals, FiniteSets

CONSTANTS Procs,         \* acquirers
          Old,           \* the holder of the expired lease at the start
          BreakGuard,    \* the break runs under the guard directory
          ReRead,        \* the break re-reads lease.json inside the guard
          ExpiryCheck,   \* only a lease judged expired is broken
          ReleaseGuard,  \* release runs under the guard directory
          GuardFreed     \* a refused break removes the guard again

All == Procs \cup {Old}
Absent == "absent"   \* no lease directory
Empty == "empty"     \* the directory exists, lease.json not yet written

VARIABLES lease,     \* Absent, Empty or the token (holder) in lease.json
          expired,   \* tokens whose until has passed
          guard,     \* the guard directory exists
          pc,
          seen       \* what an acquirer read from lease.json

vars == <<lease, expired, guard, pc, seen>>

HoldPcs == {"bunguard_ok", "hold", "rguard", "rread", "rrm"}
Holding == {p \in All : pc[p] \in HoldPcs}

TypeOK ==
  /\ lease \in All \cup {Absent, Empty}
  /\ expired \subseteq All
  /\ guard \in BOOLEAN
  /\ seen \in [All -> All \cup {Absent, Empty}]

Init ==
  /\ lease = Old
  /\ expired = {Old}
  /\ guard = FALSE
  /\ pc = [p \in All |-> IF p = Old THEN "hold" ELSE "try"]
  /\ seen = [p \in All |-> Absent]

Goto(p, s) == pc' = [pc EXCEPT ![p] = s]

\* ---- acquire: `if mkdir lease; then printf ... > lease.json; else cat lease.json`
Try(p) ==
  /\ pc[p] = "try"
  /\ IF lease = Absent
       THEN /\ lease' = Empty /\ Goto(p, "write")
       ELSE /\ UNCHANGED lease /\ Goto(p, "read")
  /\ UNCHANGED <<expired, guard, seen>>

Write(p) ==
  /\ pc[p] = "write"
  /\ lease' = p
  /\ Goto(p, "hold")
  /\ UNCHANGED <<expired, guard, seen>>

Read(p) ==
  /\ pc[p] = "read"
  /\ seen' = [seen EXCEPT ![p] = lease]
  /\ Goto(p, "decide")
  /\ UNCHANGED <<lease, expired, guard>>

\* an empty read parses to no lease and is refused; a live lease is refused
Decide(p) ==
  /\ pc[p] = "decide"
  /\ IF seen[p] \in All /\ (seen[p] \in expired \/ ~ExpiryCheck)
       THEN Goto(p, IF BreakGuard THEN "bguard" ELSE "brm")
       ELSE Goto(p, "done")
  /\ UNCHANGED <<lease, expired, guard, seen>>

BGuard(p) ==
  /\ pc[p] = "bguard"
  /\ IF ~guard
       THEN /\ guard' = TRUE /\ Goto(p, IF ReRead THEN "breread" ELSE "brm")
       ELSE /\ UNCHANGED guard /\ Goto(p, "done")
  /\ UNCHANGED <<lease, expired, seen>>

BReRead(p) ==
  /\ pc[p] = "breread"
  /\ Goto(p, IF lease = seen[p] THEN "brm" ELSE "bunguard_refuse")
  /\ UNCHANGED <<lease, expired, guard, seen>>

BRm(p) ==
  /\ pc[p] = "brm"
  /\ lease' = Absent
  /\ Goto(p, "bmkdir")
  /\ UNCHANGED <<expired, guard, seen>>

BMkdir(p) ==
  /\ pc[p] = "bmkdir"
  /\ IF lease = Absent
       THEN /\ lease' = Empty /\ Goto(p, "bwrite")
       ELSE /\ UNCHANGED lease
            /\ Goto(p, IF BreakGuard THEN "bunguard_refuse" ELSE "done")
  /\ UNCHANGED <<expired, guard, seen>>

BWrite(p) ==
  /\ pc[p] = "bwrite"
  /\ lease' = p
  /\ Goto(p, IF BreakGuard THEN "bunguard_ok" ELSE "hold")
  /\ UNCHANGED <<expired, guard, seen>>

BUnguardOk(p) ==
  /\ pc[p] = "bunguard_ok"
  /\ guard' = FALSE
  /\ Goto(p, "hold")
  /\ UNCHANGED <<lease, expired, seen>>

BUnguardRefuse(p) ==
  /\ pc[p] = "bunguard_refuse"
  /\ guard' = (guard /\ ~GuardFreed)
  /\ Goto(p, "done")
  /\ UNCHANGED <<lease, expired, seen>>

\* ---- release: under the guard, `case $(cat lease.json) in *holder*) rm -rf lease`
Release(p) ==
  /\ pc[p] = "hold"
  /\ Goto(p, IF ReleaseGuard THEN "rguard" ELSE "rread")
  /\ UNCHANGED <<lease, expired, guard, seen>>

\* release waits for the guard; the driver's wait is bounded and, when it runs
\* out, gives up without touching the lease, which is safe and is not modelled
RGuard(p) ==
  /\ pc[p] = "rguard"
  /\ ~guard
  /\ guard' = TRUE
  /\ Goto(p, "rread")
  /\ UNCHANGED <<lease, expired, seen>>

RRead(p) ==
  /\ pc[p] = "rread"
  /\ Goto(p, IF lease = p THEN "rrm" ELSE IF ReleaseGuard THEN "runguard" ELSE "done")
  /\ UNCHANGED <<lease, expired, guard, seen>>

RRm(p) ==
  /\ pc[p] = "rrm"
  /\ lease' = Absent
  /\ Goto(p, IF ReleaseGuard THEN "runguard" ELSE "done")
  /\ UNCHANGED <<expired, guard, seen>>

RUnguard(p) ==
  /\ pc[p] = "runguard"
  /\ guard' = FALSE
  /\ Goto(p, "done")
  /\ UNCHANGED <<lease, expired, seen>>

\* time passes: the lease on disk reaches its until
Expire ==
  /\ lease \in All \ expired
  /\ expired' = expired \cup {lease}
  /\ UNCHANGED <<lease, guard, pc, seen>>

Step(p) ==
  \/ Try(p) \/ Write(p) \/ Read(p) \/ Decide(p)
  \/ BGuard(p) \/ BReRead(p) \/ BRm(p) \/ BMkdir(p) \/ BWrite(p)
  \/ BUnguardOk(p) \/ BUnguardRefuse(p)
  \/ Release(p) \/ RGuard(p) \/ RRead(p) \/ RRm(p) \/ RUnguard(p)

Next == Expire \/ \E p \in All : Step(p)

Spec == Init /\ [][Next]_vars /\ \A p \in All : WF_vars(Step(p))

\* ---- properties
\* No two processes hold the slot on leases that are both still live.
OneHolder ==
  \A p, q \in Holding : p # q => (p \in expired \/ q \in expired)

\* Every acquirer and the old holder finish; none waits on the guard forever.
Terminates == <>(\A p \in All : pc[p] = "done")
=============================================================================
