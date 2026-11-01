---------------------------- MODULE HostBringup ----------------------------
(* One run of the host bring-up (OMN-21882, task T0 of the host bring-up    *)
(* plan, OMN-19130). Up to three launchers (two operators and a lane) start *)
(* a run against one host. A run takes the per-host lease, closes the runs  *)
(* the lease still owes a terminal, then loops: probe every target, plan    *)
(* one action, run it, probe again. It ends with one terminal event.        *)
(*                                                                            *)
(* The lease is the root-owned file of the plan's section 4.5: it holds the *)
(* holder, a fence, an expiry and the run ids owed an `abandoned` terminal. *)
(* A grant is one atomic step that issues a fence from a counter, as the    *)
(* generic lease does (formal/generic_lease/GenericLease.tla: one fence     *)
(* names one grant, a renewal keeps it, a lease past its expiry is free).   *)
(* This model reuses that design and adds what generic_lease does not have: *)
(* the flock a host action runs under, an external action with a hard       *)
(* timeout, a terminal event per run id, and a per-pass plan.               *)
(*                                                                            *)
(* Targets a, b, c with needs a <- b <- c. a is a host action (its whole    *)
(* script under flock, the fence compared inside the lock). b is an         *)
(* external action (broker, store, repository), outside the lock, with a    *)
(* hard timeout. c is a wait: its action is a person, and the run waits for *)
(* its check to read true. The world can regress one target (a hand change, *)
(* a reboot), so a satisfied target can read unsatisfied in a later pass.   *)
(* A run can pause for any number of ticks between two steps; a paused run  *)
(* whose lease expired is stale, and a successor takes the lease. A run can *)
(* be killed; a kill releases the flock (the kernel does) and leaves the    *)
(* lease and any external action in flight.                                  *)
(*                                                                            *)
(* Property                    Kind       Config that must break it         *)
(* OneActivePerFence           invariant  mut_regrant_same_fence (a regrant *)
(*                                        reuses the last fence)            *)
(* NeedsReadSatisfied          invariant  mut_sticky_reads (the plan trusts *)
(*                                        reads of an earlier pass)         *)
(* WaitNeverPassesUnchecked    invariant  mut_wait_timeout_passes (a wait   *)
(*                                        that times out carries on)        *)
(* AtMostOneTerminal           invariant  mut_terminal_unguarded (a         *)
(*                                        terminal is emitted without       *)
(*                                        looking for one on the run id)    *)
(* AbandonedClosed             invariant  mut_no_abandon (the next holder   *)
(*                                        skips the abandoned terminal),    *)
(*                                        mut_owed_not_chained (a takeover  *)
(*                                        forgets runs the lease still      *)
(*                                        owed), mut_release_before_terminal*)
(*                                        (a run frees the lease, then      *)
(*                                        emits its terminal)               *)
(* NoLandAfterSuccessorVerify  invariant  mut_grace_le_timeout (the lease   *)
(*                                        grace is not above the action     *)
(*                                        timeout), mut_ext_no_renew (an    *)
(*                                        external action does not extend   *)
(*                                        the lease when it starts)         *)
(* StaleHolderRefused          invariant  mut_cmp_before_lock (the fence is *)
(*                                        compared before the lock is taken)*)
(* NoActionOutsideLock         invariant  mut_no_lock (a host action runs   *)
(*                                        without the flock)                *)
(* StaleWitness                witness    wit_stale_refused (Model.cfg      *)
(*                                        values; its violation shows a     *)
(*                                        stale holder refused at the       *)
(*                                        comparison inside the lock)       *)
(* AbandonedWitness            witness    wit_abandoned_emitted (a run is   *)
(*                                        closed by another run)            *)
(* ExtLandsWitness             witness    wit_ext_lands (an external action *)
(*                                        lands)                            *)
(* WaitPassesWitness           witness    wit_wait_passes (a wait passes    *)
(*                                        on its check reading true)        *)
(*                                                                            *)
(* Bounds: 3 runs, 3 targets, time 0..4, lease grace 2, action timeout 1,   *)
(* 2 kills, 1 regression. results/ holds TLC output for every cfg and       *)
(* model.sha256, the sha256 of this file then the cfg files in sorted       *)
(* order. tests/formal/test_host_bringup_model.py runs and checks them.     *)
(*                                                                            *)
(* Not modelled: the lease is taken before root-access in the plan's        *)
(* words, here it is taken at the start of a run; the pre-root targets are  *)
(* safe without it. A multi-step host action is one atomic step under the   *)
(* lock. The rows an external action writes are one boolean.                *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Runs,           \* launchers: two operators and a lane
          None,           \* no holder
          Grace,          \* lease_grace_s: ticks a takeover waits after the last renewal
          ActTimeout,     \* the hard timeout of an external action, in ticks
          MaxTime,        \* time bound
          MaxKills,       \* kills allowed in a behavior
          MaxRegress,     \* world regressions allowed in a behavior
          MUT             \* "none" or the name of the mutation

Targets == {"a", "b", "c"}
Needs == [t \in Targets |->
            IF t = "a" THEN {} ELSE IF t = "b" THEN {"a"} ELSE {"b"}]
Kind == [t \in Targets |->
           IF t = "a" THEN "host" ELSE IF t = "b" THEN "ext" ELSE "wait"]

VARIABLES now,        \* global time
          rev,        \* last fence issued
          lease,      \* the lease file: [holder, fence, exp, owed]
          lock,       \* the flock holder on the host
          world,      \* world[t]: the check of t reads true
          ext,        \* external actions in flight: [run, t, fence, deadline]
          term,       \* term[r]: terminal events emitted for run id r
          pc,         \* pc[r]: phase of run r
          fence,      \* fence[r]: the fence r was granted, 0 before
          tgt,        \* tgt[r]: the target r planned an action on
          passRead,   \* passRead[r]: targets r read satisfied in its current pass
          credit,     \* credit[r]: targets r read satisfied in any pass (mut_sticky_reads)
          waitRead,   \* waitRead[r]: what r's last wait check read
          cmpOk,      \* cmpOk[r]: r's fence comparison, when made outside the lock
          verified,   \* runs that have probed since their grant
          killsLeft, regressLeft,
          hostStale,  \* ghost: a host action landed for a holder whose fence was not current
          hostUnlocked, \* ghost: a host action landed without its run holding the flock
          lateLanded, \* ghost: an external action landed after a successor's verify
          staleRefused, \* ghost: a host action of a stale holder was refused
          abandonEmitted \* ghost: a run was closed abandoned by another run

vars == <<now, rev, lease, lock, world, ext, term, pc, fence, tgt, passRead,
          credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
          hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

Phases == {"new", "abandon", "probe", "plan", "cmp", "hostlock", "hostapply",
           "extstart", "extwait", "waiting", "waitpassed", "fin1", "fin2",
           "done", "stopped", "dead"}
Idle == {"new", "done", "stopped", "dead"}
PostAbandon == {"probe", "plan", "cmp", "hostlock", "hostapply", "extstart",
                "extwait", "waiting", "waitpassed", "fin1", "fin2"}
Acting == {"cmp", "hostlock", "hostapply", "extstart", "waiting"}

\* Runs are interchangeable.
RunSym == Permutations(Runs)

Free == [holder |-> None, fence |-> 0, exp |-> 0, owed |-> {}]

TypeOK ==
  /\ now \in 0..MaxTime
  /\ rev \in Nat
  /\ lease.holder \in Runs \cup {None}
  /\ lease.fence \in Nat
  /\ lease.exp \in Nat
  /\ lease.owed \subseteq Runs
  /\ lock \in Runs \cup {None}
  /\ world \in [Targets -> BOOLEAN]
  /\ term \in [Runs -> 0..3]
  /\ pc \in [Runs -> Phases]
  /\ fence \in [Runs -> Nat]
  /\ passRead \in [Runs -> SUBSET Targets]
  /\ credit \in [Runs -> SUBSET Targets]
  /\ waitRead \in [Runs -> BOOLEAN]
  /\ cmpOk \in [Runs -> BOOLEAN]
  /\ verified \subseteq Runs
  /\ hostStale \in BOOLEAN
  /\ hostUnlocked \in BOOLEAN
  /\ lateLanded \in BOOLEAN
  /\ staleRefused \in BOOLEAN
  /\ abandonEmitted \in BOOLEAN

Init ==
  /\ now = 0
  /\ rev = 0
  /\ lease = Free
  /\ lock = None
  /\ world = [t \in Targets |-> FALSE]
  /\ ext = {}
  /\ term = [r \in Runs |-> 0]
  /\ pc = [r \in Runs |-> "new"]
  /\ fence = [r \in Runs |-> 0]
  /\ tgt = [r \in Runs |-> "a"]
  /\ passRead = [r \in Runs |-> {}]
  /\ credit = [r \in Runs |-> {}]
  /\ waitRead = [r \in Runs |-> FALSE]
  /\ cmpOk = [r \in Runs |-> FALSE]
  /\ verified = {}
  /\ killsLeft = MaxKills
  /\ regressLeft = MaxRegress
  /\ hostStale = FALSE
  /\ hostUnlocked = FALSE
  /\ lateLanded = FALSE
  /\ staleRefused = FALSE
  /\ abandonEmitted = FALSE

\* ---- helpers
Holds(r) == lease.holder = r /\ lease.fence = fence[r]
Expired == lease.holder # None /\ now >= lease.exp
Grantable == lease.holder = None \/ Expired

\* A terminal event on run id x. The correct emit is keyed on the run id: it
\* adds nothing when one is there.
EmitTerm(x) ==
  IF MUT = "terminal_unguarded" \/ term[x] = 0 THEN [term EXCEPT ![x] = @ + 1]
  ELSE term

Released(r) == IF Holds(r) THEN [lease EXCEPT !.holder = None, !.exp = 0] ELSE lease

ReleaseFirst == MUT = "release_before_terminal"

\* ---- time
Tick ==
  /\ now < MaxTime
  /\ \A e \in ext : now < e.deadline
  /\ now' = now + 1
  /\ UNCHANGED <<rev, lease, lock, world, ext, term, pc, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- the world changes under the run
Regress(t) ==
  /\ regressLeft > 0
  /\ world[t]
  /\ world' = [world EXCEPT ![t] = FALSE]
  /\ regressLeft' = regressLeft - 1
  /\ UNCHANGED <<now, rev, lease, lock, ext, term, pc, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* The person does what the wait target asks.
PersonDone ==
  /\ ~world["c"]
  /\ world' = [world EXCEPT !["c"] = TRUE]
  /\ UNCHANGED <<now, rev, lease, lock, ext, term, pc, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

Kill(r) ==
  /\ killsLeft > 0
  /\ pc[r] \notin Idle
  /\ pc' = [pc EXCEPT ![r] = "dead"]
  /\ lock' = IF lock = r THEN None ELSE lock
  /\ killsLeft' = killsLeft - 1
  /\ UNCHANGED <<now, rev, lease, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- the lease
\* One atomic step under the flock: take the free or expired lease with a
\* fresh fence and add the previous holder to the runs the lease owes.
Acquire(r) ==
  /\ pc[r] = "new"
  /\ Grantable
  /\ lock = None
  /\ LET f == IF MUT = "regrant_same_fence" /\ rev > 0 THEN rev ELSE rev + 1
         prior == IF lease.holder = None THEN {} ELSE {lease.holder}
         owed == IF MUT = "owed_not_chained" THEN prior ELSE lease.owed \cup prior
     IN /\ rev' = IF f > rev THEN f ELSE rev
        /\ lease' = [holder |-> r, fence |-> f, exp |-> now + Grace, owed |-> owed]
        /\ fence' = [fence EXCEPT ![r] = f]
  /\ pc' = [pc EXCEPT ![r] = "abandon"]
  /\ UNCHANGED <<now, lock, world, ext, term, tgt, passRead, credit, waitRead,
                 cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* The holder renews on its own fence; the fence stays.
Renew(r) ==
  /\ pc[r] \in {"probe", "plan", "waiting", "waitpassed"}
  /\ Holds(r)
  /\ lease' = [lease EXCEPT !.exp = IF now + Grace > @ THEN now + Grace ELSE @]
  /\ UNCHANGED <<now, rev, lock, world, ext, term, pc, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* A run that stops emits a terminal of its own and frees the lease if it holds it.
Stop(r) ==
  /\ term' = EmitTerm(r)
  /\ lease' = Released(r)
  /\ pc' = [pc EXCEPT ![r] = "stopped"]
  /\ lock' = IF lock = r THEN None ELSE lock

RenewRefused(r) ==
  /\ pc[r] \in {"abandon", "probe", "plan", "waiting", "waitpassed"}
  /\ ~Holds(r)
  /\ Stop(r)
  /\ UNCHANGED <<now, rev, world, ext, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- the abandoned terminals
\* The new holder closes, once and keyed on the run id, each run the lease owes.
EmitAbandoned(r, x) ==
  /\ pc[r] = "abandon"
  /\ Holds(r)
  /\ MUT # "no_abandon"
  /\ x \in lease.owed
  /\ term' = EmitTerm(x)
  /\ abandonEmitted' = (abandonEmitted \/ term[x] = 0)
  /\ lease' = [lease EXCEPT !.owed = @ \ {x}]
  /\ UNCHANGED <<now, rev, lock, world, ext, pc, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused>>

EndAbandon(r) ==
  /\ pc[r] = "abandon"
  /\ Holds(r)
  /\ \/ lease.owed = {}
     \/ MUT = "no_abandon"
  /\ lease' = [lease EXCEPT !.owed = {}]
  /\ pc' = [pc EXCEPT ![r] = "probe"]
  /\ UNCHANGED <<now, rev, lock, world, ext, term, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- one pass: probe, plan, one action
Probe(r) ==
  /\ pc[r] = "probe"
  /\ passRead' = [passRead EXCEPT ![r] = {t \in Targets : world[t]}]
  /\ credit' = IF MUT = "sticky_reads"
                 THEN [credit EXCEPT ![r] = @ \cup {t \in Targets : world[t]}]
                 ELSE credit
  /\ verified' = verified \cup {r}
  /\ pc' = [pc EXCEPT ![r] = "plan"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, waitRead,
                 cmpOk, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* The plan picks a target that read unsatisfied whose needs read satisfied.
Seen(r) == IF MUT = "sticky_reads" THEN credit[r] ELSE passRead[r]

NextPhase(t) ==
  IF Kind[t] = "host"
    THEN IF MUT = "cmp_before_lock" THEN "cmp"
         ELSE IF MUT = "no_lock" THEN "hostapply" ELSE "hostlock"
    ELSE IF Kind[t] = "ext" THEN "extstart" ELSE "waiting"

Plan(r) ==
  /\ pc[r] = "plan"
  /\ \E t \in Targets :
       /\ t \notin passRead[r]
       /\ Needs[t] \subseteq Seen(r)
       /\ tgt' = [tgt EXCEPT ![r] = t]
       /\ pc' = [pc EXCEPT ![r] = NextPhase(t)]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

AllSatisfied(r) ==
  /\ pc[r] = "plan"
  /\ passRead[r] = Targets
  /\ pc' = [pc EXCEPT ![r] = "fin1"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- a host action: the whole script under the flock, the fence compared inside
\* mut_cmp_before_lock compares first, outside the lock, and trusts it later.
Cmp(r) ==
  /\ pc[r] = "cmp"
  /\ cmpOk' = [cmpOk EXCEPT ![r] = Holds(r)]
  /\ pc' = [pc EXCEPT ![r] = "hostlock"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

TakeLock(r) ==
  /\ pc[r] = "hostlock"
  /\ lock = None
  /\ lock' = r
  /\ pc' = [pc EXCEPT ![r] = "hostapply"]
  /\ UNCHANGED <<now, rev, lease, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

HostPasses(r) == IF MUT = "cmp_before_lock" THEN cmpOk[r] ELSE Holds(r)

HostApply(r) ==
  /\ pc[r] = "hostapply"
  /\ IF HostPasses(r)
       THEN /\ world' = [world EXCEPT ![tgt[r]] = TRUE]
            /\ pc' = [pc EXCEPT ![r] = "probe"]
            /\ lock' = IF lock = r THEN None ELSE lock
            /\ term' = term
            /\ lease' = lease
            /\ hostStale' = (hostStale \/ ~Holds(r))
            /\ hostUnlocked' = (hostUnlocked \/ lock # r)
            /\ staleRefused' = staleRefused
       ELSE /\ Stop(r)
            /\ staleRefused' = (staleRefused \/ ~Holds(r))
            /\ UNCHANGED <<world, hostStale, hostUnlocked>>
  /\ UNCHANGED <<now, rev, ext, fence, tgt, passRead, credit, waitRead,
                 cmpOk, verified, killsLeft, regressLeft, lateLanded, abandonEmitted>>

\* ---- an external action: outside the lock, a hard timeout
ExtStart(r) ==
  /\ pc[r] = "extstart"
  /\ IF Holds(r)
       THEN /\ lease' = IF MUT = "ext_no_renew" THEN lease
                          ELSE [lease EXCEPT !.exp = IF now + Grace > @ THEN now + Grace ELSE @]
            /\ ext' = ext \cup {[run |-> r, t |-> tgt[r], fence |-> fence[r],
                                 deadline |-> now + ActTimeout]}
            /\ pc' = [pc EXCEPT ![r] = "extwait"]
            /\ UNCHANGED <<term, lock>>
       ELSE /\ Stop(r)
            /\ UNCHANGED ext
  /\ UNCHANGED <<now, rev, world, fence, tgt, passRead, credit, waitRead, cmpOk,
                 verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* The action lands at the service, or the timeout kills it. A run that was
\* killed or paused does not stop its action.
ExtLand(e) ==
  /\ e \in ext
  /\ now <= e.deadline
  /\ world' = [world EXCEPT ![e.t] = TRUE]
  /\ ext' = ext \ {e}
  /\ lateLanded' = (lateLanded \/ \E y \in verified : fence[y] > e.fence)
  /\ UNCHANGED <<now, rev, lease, lock, term, pc, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, staleRefused, abandonEmitted>>

ExtExpire(e) ==
  /\ e \in ext
  /\ now >= e.deadline
  /\ ext' = ext \ {e}
  /\ UNCHANGED <<now, rev, lease, lock, world, term, pc, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

ExtDone(r) ==
  /\ pc[r] = "extwait"
  /\ \A e \in ext : e.run # r
  /\ pc' = [pc EXCEPT ![r] = "probe"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- a wait: passes only when its check reads true
WaitCheck(r) ==
  /\ pc[r] = "waiting"
  /\ waitRead' = [waitRead EXCEPT ![r] = world[tgt[r]]]
  /\ pc' = [pc EXCEPT ![r] = IF world[tgt[r]] THEN "waitpassed" ELSE "waiting"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, passRead,
                 credit, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* wait_timeout_s: the run stops and prints the person line. The mutation
\* carries on as if the check had read true.
WaitTimeout(r) ==
  /\ pc[r] = "waiting"
  /\ IF MUT = "wait_timeout_passes"
       THEN /\ waitRead' = [waitRead EXCEPT ![r] = world[tgt[r]]]
            /\ pc' = [pc EXCEPT ![r] = "waitpassed"]
            /\ UNCHANGED <<term, lock, lease>>
       ELSE /\ Stop(r)
            /\ UNCHANGED waitRead
  /\ UNCHANGED <<now, rev, world, ext, fence, tgt, passRead, credit,
                 cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

WaitPassed(r) ==
  /\ pc[r] = "waitpassed"
  /\ pc' = [pc EXCEPT ![r] = "probe"]
  /\ UNCHANGED <<now, rev, lease, lock, world, ext, term, fence, tgt, passRead,
                 credit, waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

\* ---- the end: one terminal for the run, and the lease freed
\* The correct order is terminal first, then release. mut_release_before_terminal
\* frees the lease first, so a kill between the two leaves a run no lease owes.
Fin1(r) ==
  /\ pc[r] = "fin1"
  /\ IF ReleaseFirst
       THEN /\ lease' = Released(r)
            /\ term' = term
       ELSE /\ term' = EmitTerm(r)
            /\ lease' = lease
  /\ pc' = [pc EXCEPT ![r] = "fin2"]
  /\ UNCHANGED <<now, rev, lock, world, ext, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

Fin2(r) ==
  /\ pc[r] = "fin2"
  /\ IF ReleaseFirst
       THEN /\ term' = EmitTerm(r)
            /\ lease' = lease
       ELSE /\ lease' = Released(r)
            /\ term' = term
  /\ pc' = [pc EXCEPT ![r] = "done"]
  /\ UNCHANGED <<now, rev, lock, world, ext, fence, tgt, passRead, credit,
                 waitRead, cmpOk, verified, killsLeft, regressLeft,
                 hostStale, hostUnlocked, lateLanded, staleRefused, abandonEmitted>>

Next ==
  \/ Tick
  \/ PersonDone
  \/ \E t \in Targets : Regress(t)
  \/ \E e \in ext : (ExtLand(e) \/ ExtExpire(e))
  \/ \E r \in Runs :
       \/ Kill(r) \/ Acquire(r) \/ Renew(r) \/ RenewRefused(r)
       \/ EndAbandon(r) \/ Probe(r) \/ Plan(r) \/ AllSatisfied(r)
       \/ Cmp(r) \/ TakeLock(r) \/ HostApply(r)
       \/ ExtStart(r) \/ ExtDone(r)
       \/ WaitCheck(r) \/ WaitTimeout(r) \/ WaitPassed(r)
       \/ Fin1(r) \/ Fin2(r)
       \/ \E x \in Runs : EmitAbandoned(r, x)

Spec == Init /\ [][Next]_vars

\* ---- properties
\* At most one run is active at a fence: a fence names one grant, so a stale
\* holder and its successor never share one.
OneActivePerFence ==
  \A x, y \in Runs :
    (x # y /\ pc[x] \notin Idle /\ pc[y] \notin Idle) => fence[x] # fence[y]

\* A run acts on a target only when each of its needs read satisfied in the
\* run's own current pass.
NeedsReadSatisfied ==
  \A r \in Runs : pc[r] \in Acting => Needs[tgt[r]] \subseteq passRead[r]

\* A wait carries on only when its check read true.
WaitNeverPassesUnchecked ==
  \A r \in Runs : pc[r] = "waitpassed" => waitRead[r]

\* A run id has at most one terminal event.
AtMostOneTerminal ==
  \A r \in Runs : term[r] <= 1

\* Once a run has closed the runs its lease owed and gone on to probe, every
\* run that held an earlier fence has its terminal: the killed holder's run
\* was closed abandoned by a later holder.
AbandonedClosed ==
  \A y, x \in Runs :
    (pc[y] \in PostAbandon /\ fence[x] > 0 /\ fence[x] < fence[y]) => term[x] >= 1

\* No external action of an old holder lands after a successor has verified.
NoLandAfterSuccessorVerify == ~lateLanded

\* A host action lands only for the holder whose fence is current.
StaleHolderRefused == ~hostStale

\* A host action lands only while its run holds the flock.
NoActionOutsideLock == ~hostUnlocked

\* ---- witness (Model.cfg values; TLC must report it violated)
\* A stale holder's host action reaches the comparison inside the lock and is
\* refused: the race the fence and the lock exist for occurs.
StaleWitness == ~staleRefused

\* The next holder emits the abandoned terminal of a run.
AbandonedWitness == ~abandonEmitted

\* An external action lands at its service.
ExtLandsWitness == ~world["b"]

\* A wait passes because its check read true.
WaitPassesWitness == ~(\E r \in Runs : pc[r] = "waitpassed")
=============================================================================
