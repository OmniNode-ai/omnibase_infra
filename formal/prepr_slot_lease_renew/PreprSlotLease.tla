-------------------------- MODULE PreprSlotLease --------------------------
(*
  The host lease of the pre-PR verify slots and the one-call `verify` driver
  (OMN-20085, under OMN-18893).

  What this models, and the code it stands for
  --------------------------------------------
  Several runners (one driver process each, on the operator Mac or on a peer
  lane's machine) race for a small ordered set of slots. Each run has a unique
  identity (`<lane>~<run id>`), never just the lane name: two runs of one lane
  are two runners here and in the code. A slot is leased for a bounded time: a
  lease names its holder and an absolute expiry. The driver in
  scripts/runtime_build/prepr_runtime_pool.py:

    Attempt   survey the slots in order and try the first; when a peer holds it
              try the next; when every slot is held report "nofree".
    Begin     before EVERY phase, teardown and snap-post included, one remote
              command on the host renews the lease and then starts the phase
              (renew_lease_command, prefixed to the phase by _run_stack). The
              renewal is refused when the lease is no longer this runner's: the
              phase is not started (exit 97, LEASE-LOST), and the runner stops
              and does not tear down, because what is on the slot is a
              successor's. Nothing on the driver's side can fall between the
              renewal and the start, so Begin is one step.
    Step      one phase, at most StepLen long, inside the lease the renewal
              just set (until = now + the phase budget + RENEW_MARGIN_S, so
              TTL >= StepLen). The host kills a phase when its budget ends
              (`timeout -k 30 <budget>` in the same command), so StepLen holds on
              the host and not only in the driver's ssh call. The driver may be delayed for any time
              between two phases (a sleeping laptop); the next Begin then finds
              the lease lost or renews it, and either is safe.
    Release   remove the lease, but only if it is still this runner's.

  A lease whose expiry has passed is free, so a runner that died without
  releasing cannot hold a slot for ever. That frees the LEASE. A hard-killed run
  leaves its containers on the slot, and the driver's survey reads a slot with
  containers as busy, so the residue is cleaned by a reaper (OMN-18896, not
  built by this change): the action Reap below is that assumption, stated as a
  rule and not as code, and the mutation "no_reaper" shows what fails without it.

  The faults are a hard kill (SIGKILL, power loss: nothing runs after it and the
  lease and the residue are left in place) and a soft kill (SIGTERM, SIGHUP,
  SIGINT: the driver's handler goes straight to the teardown phase and then the
  release).

  Time is discrete. The acquire, the renewal and the release are each one
  critical section on the host. Taking a free lease is one `mkdir` of the lease
  directory, which the kernel admits once. Breaking an expired lease, renewing
  and releasing each hold the guard directory `<lease>.break` (also a `mkdir`)
  and re-read lease.json inside it: the break replaces only the exact lease the
  runner judged expired, the renewal and the release act only on a lease that
  still names this runner. So each is atomic with respect to the others, and
  Attempt and Begin are one step each. The mutation "racy_break" is what the
  acquire did before OMN-20085: read the lease, later `rm -rf` it if the read
  was expired, later `mkdir`. A runner that read a lease before it was released
  and retaken acts on the stale read and deletes the new holder's lease: two
  runners operate on one slot. The real-shell race tests in
  tests/unit/scripts/test_prepr_slot_lease_omn18893.py check the same
  critical sections against the driver's own commands.

  Assumed and not modelled: the lease and guard directories sit on a local
  filesystem on each host, and the driver-supplied clock is the one clock every
  expiry is judged by. A guard directory left by a process killed inside its
  few-millisecond critical section is not cleaned by anything: renewals and
  breaks then refuse (safe, but the slot stays held until an operator removes
  the guard).
  An ssh command that times out after it ran on the host is answered by a
  release of this run's own identity before the next slot is tried; if that
  release ran before the timed-out acquire did, the acquire leaves a lease of
  this run's own identity that no operation runs under and that lapses within
  the first phase's budget plus the margin (seven minutes). A phase command sent
  twice because no reply came back starts the phase once: it creates a marker
  with mkdir, which refuses a second creation, before it renews anything.

  Properties (each has a mutation that must break it; the mut_*.cfg files)
  ---------------------------------------------------------------------
  MutualExclusion          two runners never operate on one slot at once
  OperatingOwnsLiveLease   a runner in a phase, teardown included, holds an
                           unexpired lease of that slot
  NoSoftKillAbandon        a soft-killed runner never leaves its lease behind
  AbandonedReusable        a dead runner's lease becomes free again (temporal)
  DeadSlotCleaned          a dead runner's residue is eventually cleaned (temporal)
*)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Runners, TTL, StepLen, Steps, MaxTime, MaxKills, MUT

\* The slots in the order the driver tries them: slot 1, then slot 2.
Order == <<"s1", "s2">>
Slots == {Order[i] : i \in 1..Len(Order)}
None == "none"

ASSUME /\ TTL >= 1 /\ StepLen >= 1 /\ Steps >= 2 /\ MaxTime >= TTL /\ MaxKills >= 0

VARIABLES now, lease, pc, idx, slotOf, stepStart, stepsLeft, seen,
          softKilled, killsLeft, dirty

vars == <<now, lease, pc, idx, slotOf, stepStart, stepsLeft, seen,
          softKilled, killsLeft, dirty>>

NoLease == [h |-> None, until |-> 0]

\* A lease is free when nobody holds it or its expiry has passed. The mutation
\* "no_expiry" is the rule that reclaimed nothing: only a released lease is free.
Free(l) == IF MUT = "no_expiry" THEN l.h = None ELSE (l.h = None \/ l.until <= now)

Phases == {"idle", "saw", "broke", "between", "run", "done", "nofree", "dead", "lost"}

TypeOK ==
    /\ now \in 0..MaxTime
    /\ lease \in [Slots -> [h : Runners \cup {None}, until : 0..(MaxTime + TTL)]]
    /\ pc \in [Runners -> Phases]
    /\ idx \in [Runners -> 1..Len(Order)]
    /\ slotOf \in [Runners -> Slots \cup {None}]
    /\ stepStart \in [Runners -> 0..MaxTime]
    /\ stepsLeft \in [Runners -> 0..Steps]
    /\ seen \in [Runners -> [h : Runners \cup {None}, until : 0..(MaxTime + TTL)]]
    /\ softKilled \in [Runners -> BOOLEAN]
    /\ killsLeft \in 0..MaxKills
    /\ dirty \in [Slots -> BOOLEAN]

Init ==
    /\ now = 0
    /\ lease = [s \in Slots |-> NoLease]
    /\ pc = [r \in Runners |-> "idle"]
    /\ idx = [r \in Runners |-> 1]
    /\ slotOf = [r \in Runners |-> None]
    /\ stepStart = [r \in Runners |-> 0]
    /\ stepsLeft = [r \in Runners |-> 0]
    /\ seen = [r \in Runners |-> NoLease]
    /\ softKilled = [r \in Runners |-> FALSE]
    /\ killsLeft = MaxKills
    /\ dirty = [s \in Slots |-> FALSE]

Cur(r) == Order[idx[r]]

\* A lease or a renewal is only made while the whole lease fits inside the
\* model's horizon, so every lease this model grants can expire inside it.
CanTry == now + TTL <= MaxTime

Operating(r) == pc[r] = "run"

\* The runner takes slot s: the lease names it and the run is ready for its
\* first phase.
Take(r, s) ==
    /\ lease' = [lease EXCEPT ![s] = [h |-> r, until |-> now + TTL]]
    /\ slotOf' = [slotOf EXCEPT ![r] = s]
    /\ stepsLeft' = [stepsLeft EXCEPT ![r] = Steps]
    /\ pc' = [pc EXCEPT ![r] = "between"]

\* The runner moves to the next slot, or reports that none is free.
Advance(r) ==
    /\ IF idx[r] < Len(Order)
          THEN /\ idx' = [idx EXCEPT ![r] = idx[r] + 1]
               /\ pc' = [pc EXCEPT ![r] = "idle"]
          ELSE /\ pc' = [pc EXCEPT ![r] = "nofree"]
               /\ UNCHANGED idx

\* ---------------------------------------------------------------- acquire, as built
Attempt(r) ==
    /\ pc[r] = "idle"
    /\ MUT # "racy_break"
    /\ CanTry
    /\ LET s == Cur(r) IN
       IF Free(lease[s])
          THEN /\ Take(r, s)
               /\ UNCHANGED <<idx, seen, stepStart>>
          ELSE /\ Advance(r)
               /\ UNCHANGED <<lease, slotOf, stepStart, stepsLeft, seen>>
    /\ UNCHANGED <<now, softKilled, killsLeft, dirty>>

\* ---------------------------------------------------------------- acquire, as it was
\* Three separate remote commands: mkdir-or-cat, rm -rf, mkdir.
RacyRead(r) ==
    /\ pc[r] = "idle"
    /\ MUT = "racy_break"
    /\ CanTry
    /\ LET s == Cur(r) IN
       IF lease[s].h = None
          THEN /\ Take(r, s)
               /\ UNCHANGED <<idx, seen, stepStart>>
          ELSE /\ seen' = [seen EXCEPT ![r] = lease[s]]
               /\ pc' = [pc EXCEPT ![r] = "saw"]
               /\ UNCHANGED <<lease, idx, slotOf, stepStart, stepsLeft>>
    /\ UNCHANGED <<now, softKilled, killsLeft, dirty>>

RacyBreak(r) ==
    /\ pc[r] = "saw"
    /\ LET s == Cur(r) IN
       IF seen[r].until <= now
          THEN /\ lease' = [lease EXCEPT ![s] = NoLease]     \* rm -rf, unconditional
               /\ pc' = [pc EXCEPT ![r] = "broke"]
               /\ UNCHANGED <<idx, slotOf, stepStart, stepsLeft, seen>>
          ELSE /\ Advance(r)
               /\ UNCHANGED <<lease, slotOf, stepStart, stepsLeft, seen>>
    /\ UNCHANGED <<now, softKilled, killsLeft, dirty>>

RacyMkdir(r) ==
    /\ pc[r] = "broke"
    /\ LET s == Cur(r) IN
       IF lease[s].h = None
          THEN /\ Take(r, s)
               /\ UNCHANGED <<idx, seen, stepStart>>
          ELSE /\ Advance(r)
               /\ UNCHANGED <<lease, slotOf, stepStart, stepsLeft, seen>>
    /\ UNCHANGED <<now, softKilled, killsLeft, dirty>>

\* ---------------------------------------------------------------- phases
\* Begin renews the lease and starts the next phase, in one critical section.
\* When the lease is no longer this runner's the runner stops. The mutation
\* "no_renew_check" is a driver that goes on regardless, as the code did before
\* this renewal was built.
Begin(r) ==
    /\ pc[r] = "between"
    /\ CanTry
    /\ LET s == slotOf[r] IN
       IF lease[s].h = r \/ MUT = "no_renew_check"
          THEN /\ lease' = [lease EXCEPT ![s] =
                                IF lease[s].h = r
                                   THEN [h |-> r, until |-> now + TTL]
                                   ELSE lease[s]]
               /\ stepStart' = [stepStart EXCEPT ![r] = now]
               /\ pc' = [pc EXCEPT ![r] = "run"]
               \* the first phase puts the driver's containers on the slot
               /\ dirty' = [dirty EXCEPT ![s] = TRUE]
               /\ UNCHANGED <<stepsLeft>>
          ELSE /\ pc' = [pc EXCEPT ![r] = "lost"]
               /\ UNCHANGED <<lease, stepStart, stepsLeft, dirty>>
    /\ UNCHANGED <<now, idx, slotOf, seen, softKilled, killsLeft>>

\* A phase ends. The last one is the teardown, which leaves the slot clean.
EndStep(r) ==
    /\ pc[r] = "run"
    /\ stepsLeft' = [stepsLeft EXCEPT ![r] = stepsLeft[r] - 1]
    /\ IF stepsLeft[r] = 1
          THEN /\ dirty' = [dirty EXCEPT ![slotOf[r]] = FALSE]
               /\ pc' = [pc EXCEPT ![r] = "done"]
               \* Release removes the lease only when it is still this runner's
               /\ lease' = [lease EXCEPT ![slotOf[r]] =
                                IF lease[slotOf[r]].h = r
                                   THEN NoLease ELSE lease[slotOf[r]]]
          ELSE /\ pc' = [pc EXCEPT ![r] = "between"]
               /\ UNCHANGED <<dirty, lease>>
    /\ UNCHANGED <<now, idx, slotOf, stepStart, seen, softKilled, killsLeft>>

\* ---------------------------------------------------------------- faults
HardKill(r) ==
    /\ pc[r] \in {"run", "between"}
    /\ killsLeft > 0
    /\ killsLeft' = killsLeft - 1
    /\ pc' = [pc EXCEPT ![r] = "dead"]
    /\ UNCHANGED <<now, lease, idx, slotOf, stepStart, stepsLeft, seen,
                   softKilled, dirty>>

\* The signal handler goes straight to the teardown phase and the release. The
\* mutation "no_signal_teardown" is the default handler, which does neither.
SoftKill(r) ==
    /\ pc[r] \in {"run", "between"}
    /\ stepsLeft[r] > 1
    /\ killsLeft > 0
    /\ killsLeft' = killsLeft - 1
    /\ softKilled' = [softKilled EXCEPT ![r] = TRUE]
    /\ IF MUT = "no_signal_teardown"
          THEN /\ pc' = [pc EXCEPT ![r] = "dead"]
               /\ UNCHANGED stepsLeft
          ELSE /\ pc' = [pc EXCEPT ![r] = "between"]
               /\ stepsLeft' = [stepsLeft EXCEPT ![r] = 1]
    /\ UNCHANGED <<now, lease, idx, slotOf, stepStart, seen, dirty>>

\* ---------------------------------------------------------------- the reaper
\* ASSUMED, not built here (OMN-18896): a slot whose lease is free is cleaned.
Reap(s) ==
    /\ MUT # "no_reaper"
    /\ dirty[s]
    /\ Free(lease[s])
    /\ \A r \in Runners : ~(pc[r] = "run" /\ slotOf[r] = s)
    /\ dirty' = [dirty EXCEPT ![s] = FALSE]
    /\ UNCHANGED <<now, lease, pc, idx, slotOf, stepStart, stepsLeft, seen,
                   softKilled, killsLeft>>

\* ---------------------------------------------------------------- time
\* Time may pass unless it would carry a runner past the end of its own phase.
Tick ==
    /\ now < MaxTime
    /\ \A r \in Runners : Operating(r) => (now + 1) - stepStart[r] < StepLen
    /\ now' = now + 1
    /\ UNCHANGED <<lease, pc, idx, slotOf, stepStart, stepsLeft, seen,
                   softKilled, killsLeft, dirty>>

Live(r) ==
    \/ Attempt(r) \/ RacyRead(r) \/ RacyBreak(r) \/ RacyMkdir(r)
    \/ Begin(r) \/ EndStep(r)

Next ==
    \/ Tick
    \/ \E s \in Slots : Reap(s)
    \/ \E r \in Runners : Live(r) \/ HardKill(r) \/ SoftKill(r)

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(Tick)
    /\ \A s \in Slots : WF_vars(Reap(s))
    /\ \A r \in Runners : WF_vars(Live(r))

\* ---------------------------------------------------------------- properties
MutualExclusion ==
    \A r1, r2 \in Runners :
        (r1 # r2 /\ Operating(r1) /\ Operating(r2)) => slotOf[r1] # slotOf[r2]

OperatingOwnsLiveLease ==
    \A r \in Runners :
        Operating(r) => (lease[slotOf[r]].h = r /\ lease[slotOf[r]].until > now)

NoSoftKillAbandon ==
    \A r \in Runners : ~(softKilled[r] /\ pc[r] = "dead")

AbandonedReusable ==
    \A r \in Runners : \A s \in Slots :
        (pc[r] = "dead" /\ lease[s].h = r) ~> Free(lease[s])

DeadSlotCleaned ==
    \A r \in Runners : \A s \in Slots :
        (pc[r] = "dead" /\ slotOf[r] = s /\ dirty[s]) ~> ~dirty[s]

=============================================================================
