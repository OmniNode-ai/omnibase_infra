------------------------------ MODULE LabReconcile ------------------------------
(***************************************************************************)
(* OMN-19421 (T0.4 of the lab release-sync plan, seam L0.4).               *)
(*                                                                         *)
(* Writers of a .201 compose project, and the one lock that serialises     *)
(* them:                                                                   *)
(*   - the reconcile dispatcher (T2.3): stability-test only, one slot per  *)
(*     release event, grants against the window file (T0.3);              *)
(*   - runtime-train (T2.4): merges a runtime PR, which queues a deploy    *)
(*     agent job on the dev lane; it reads the lock and the windows at     *)
(*     merge time;                                                         *)
(*   - the deploy agent: starts a queued dev-lane job some time later;     *)
(*   - hand refreshes (refresh_dev_lane.sh, refresh_stability_lane.sh,     *)
(*     deploy-runtime.sh), which take the same lane lock.                  *)
(* The lane lock is lane_lock.py: one fcntl lock per compose project,      *)
(* never stolen, released by the kernel when the holder dies.              *)
(* A mutation ends in PASS, FAILED_ROLLED_BACK or FAILED_STRANDED, written *)
(* by the holder; a holder that dies writes nothing, so a reaper writes    *)
(* FAILED_STRANDED for it once its lock is free.                           *)
(*                                                                         *)
(* Time is discrete: one tick is 5 minutes. Every mutation is killed at    *)
(* its declared ceiling (the job timeout), so it never runs past it.       *)
(*                                                                         *)
(* Design switches (the constants) exist so each invariant can be shown to *)
(* be non-vacuous: every switch set to its unsafe value must make exactly  *)
(* the invariant it protects fail.                                         *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    MaxT,          \* time horizon, in ticks
    Windows,       \* probe start ticks
    Guard,         \* ticks before a probe start that are blacked out (10 min = 2)
    ProbeDur,      \* ticks after a probe start that are blacked out
    Ceiling,       \* declared ceiling of every mutation, in ticks (forward + verify + rollback)
    MaxMut,        \* bound on mutations started (reconcile ids 1..MaxMut)
    MaxRel,        \* bound on release events (each asks the dispatcher for one stability slot)
    MaxMerge,      \* bound on runtime merges (each queues one deploy agent job)
    MaxHand,       \* bound on hand refreshes per project
    HandProjects,  \* compose projects a hand refresh may target
    UseLock,       \* TRUE: every writer takes lane_lock for its compose project
    GuardAtLock,   \* TRUE: the window check sits in lock acquisition (every writer);
                   \* FALSE: only the dispatcher and runtime-train (at merge time) check
    Reaper,        \* TRUE: a reaper writes FAILED_STRANDED for a dead holder
    ReaperChecksReceipt, \* TRUE: the reaper skips a holder that already wrote its receipt
    NoOpShortCircuit     \* TRUE: a writer whose target equals the running composition does not mutate

Projects == {"dev", "stab"}
Ids      == 1..MaxMut
Kinds    == {"PASS", "FAILED_ROLLED_BACK", "FAILED_STRANDED"}
BAD      == 99   \* running composition is neither the prior nor the target (partial or broken)

Live     == {"forward", "rollback", "closing"}   \* holder process alive, lock held
Mutating == {"forward", "rollback"}              \* containers being changed
NONE == [proj |-> "none", actor |-> "none", target |-> 0, prior |-> 0,
         start |-> 0, deadline |-> 0, status |-> "none"]

VARIABLES
    now,        \* current tick
    desired,    \* [Projects -> Nat]  rendered from the release (stab) or dev head (dev)
    running,    \* [Projects -> Nat \cup {BAD}]
    rolledBack, \* [Projects -> BOOLEAN]  last terminal on this lane was a rollback
    mut,        \* [Ids -> record]; NONE marks an unused id
    receipts,   \* [Ids -> Seq(Kinds)]
    nextId,
    pendingStab,\* dispatcher slot requests not yet granted
    agentQueue, \* deploy agent jobs not yet started
    nRel, nMerge, nHand

vars == <<now, desired, running, rolledBack, mut, receipts, nextId,
          pendingStab, agentQueue, nRel, nMerge, nHand>>

-----------------------------------------------------------------------------
Blackout(a, b) == \E w \in Windows : a <= w + ProbeDur /\ b + Guard >= w
WindowOK(a, b) == ~Blackout(a, b)

Locked(p) == \E i \in Ids : mut[i].status # "none" /\ mut[i].proj = p /\ mut[i].status \in Live

\* Preconditions every writer shares.
CanStart(p) ==
    /\ nextId <= MaxMut
    /\ now + Ceiling <= MaxT
    /\ UseLock => ~Locked(p)
    /\ GuardAtLock => WindowOK(now, now + Ceiling)

Begin(p, actor) ==
    /\ mut' = [mut EXCEPT ![nextId] =
                 [proj |-> p, actor |-> actor, target |-> desired[p],
                  prior |-> running[p], start |-> now,
                  deadline |-> now + Ceiling, status |-> "forward"]]
    /\ nextId' = nextId + 1

-----------------------------------------------------------------------------
Init ==
    /\ now = 0
    /\ desired = [p \in Projects |-> 0]
    /\ running = [p \in Projects |-> 0]
    /\ rolledBack = [p \in Projects |-> FALSE]
    /\ mut = [i \in Ids |-> NONE]
    /\ receipts = [i \in Ids |-> <<>>]
    /\ nextId = 1
    /\ pendingStab = 0
    /\ agentQueue = 0
    /\ nRel = 0 /\ nMerge = 0
    /\ nHand = [p \in Projects |-> 0]

\* Time advances only when no live holder is at its deadline.
Tick ==
    /\ now < MaxT
    /\ \A i \in Ids : mut[i].status # "none" /\ mut[i].status \in Live => mut[i].deadline > now
    /\ now' = now + 1
    /\ UNCHANGED <<desired, running, rolledBack, mut, receipts, nextId,
                   pendingStab, agentQueue, nRel, nMerge, nHand>>

\* An omnibase_infra release: desired state for stability-test moves forward
\* and the dispatcher is asked for one slot.
ReleaseEvent ==
    /\ nRel < MaxRel
    /\ nRel' = nRel + 1
    /\ desired' = [desired EXCEPT !["stab"] = @ + 1]
    /\ pendingStab' = pendingStab + 1
    /\ UNCHANGED <<now, running, rolledBack, mut, receipts, nextId, agentQueue, nMerge, nHand>>

\* runtime-train merges a runtime PR (T2.4): it reads the dev lock and the
\* window for a job that would start now. The merge moves dev's desired state
\* and queues one deploy agent job, which starts later.
RuntimeMerge ==
    /\ nMerge < MaxMerge
    /\ UseLock => ~Locked("dev")
    /\ WindowOK(now, now + Ceiling)
    /\ nMerge' = nMerge + 1
    /\ desired' = [desired EXCEPT !["dev"] = @ + 1]
    /\ agentQueue' = agentQueue + 1
    /\ UNCHANGED <<now, running, rolledBack, mut, receipts, nextId, pendingStab, nRel, nHand>>

\* The dispatcher (T2.3) always checks the window itself, in both designs.
DispatcherGrant ==
    /\ pendingStab > 0
    /\ CanStart("stab")
    /\ WindowOK(now, now + Ceiling)
    /\ NoOpShortCircuit => desired["stab"] # running["stab"]
    /\ pendingStab' = pendingStab - 1
    /\ Begin("stab", "dispatcher")
    /\ UNCHANGED <<now, desired, running, rolledBack, receipts, agentQueue, nRel, nMerge, nHand>>

DispatcherNoOp ==
    /\ NoOpShortCircuit
    /\ pendingStab > 0
    /\ desired["stab"] = running["stab"]
    /\ pendingStab' = pendingStab - 1
    /\ UNCHANGED <<now, desired, running, rolledBack, mut, receipts, nextId, agentQueue, nRel, nMerge, nHand>>

AgentStart ==
    /\ agentQueue > 0
    /\ CanStart("dev")
    /\ NoOpShortCircuit => desired["dev"] # running["dev"]
    /\ agentQueue' = agentQueue - 1
    /\ Begin("dev", "agent")
    /\ UNCHANGED <<now, desired, running, rolledBack, receipts, pendingStab, nRel, nMerge, nHand>>

AgentNoOp ==
    /\ NoOpShortCircuit
    /\ agentQueue > 0
    /\ desired["dev"] = running["dev"]
    /\ agentQueue' = agentQueue - 1
    /\ UNCHANGED <<now, desired, running, rolledBack, mut, receipts, nextId, pendingStab, nRel, nMerge, nHand>>

HandStart(p) ==
    /\ nHand[p] < MaxHand
    /\ CanStart(p)
    /\ NoOpShortCircuit => desired[p] # running[p]
    /\ nHand' = [nHand EXCEPT ![p] = @ + 1]
    /\ Begin(p, "hand")
    /\ UNCHANGED <<now, desired, running, rolledBack, receipts, pendingStab, agentQueue, nRel, nMerge>>

-----------------------------------------------------------------------------
\* Holder steps. The receipt is written by the holder before it releases the lock.
SetStatus(i, s) == mut' = [mut EXCEPT ![i].status = s]
Emit(i, k)      == receipts' = [receipts EXCEPT ![i] = Append(@, k)]
Unch == UNCHANGED <<now, desired, nextId, pendingStab, agentQueue, nRel, nMerge, nHand>>

ForwardPass(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "forward"
    /\ running' = [running EXCEPT ![mut[i].proj] = mut[i].target]
    /\ rolledBack' = [rolledBack EXCEPT ![mut[i].proj] = FALSE]
    /\ Emit(i, "PASS") /\ SetStatus(i, "closing") /\ Unch

\* Lane-health failure: containers are half replaced, rollback begins.
ForwardHealthFail(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "forward"
    /\ running' = [running EXCEPT ![mut[i].proj] = BAD]
    /\ SetStatus(i, "rollback")
    /\ UNCHANGED <<rolledBack, receipts>> /\ Unch

\* Provenance-only failure, no rollback target, or an incomplete anchor:
\* the lane is left as it is, which is neither prior nor target.
ForwardNoRollback(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "forward"
    /\ running' = [running EXCEPT ![mut[i].proj] = BAD]
    /\ Emit(i, "FAILED_STRANDED") /\ SetStatus(i, "closing")
    /\ UNCHANGED rolledBack /\ Unch

RollbackOK(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "rollback"
    /\ running' = [running EXCEPT ![mut[i].proj] = mut[i].prior]
    /\ rolledBack' = [rolledBack EXCEPT ![mut[i].proj] = TRUE]
    /\ Emit(i, "FAILED_ROLLED_BACK") /\ SetStatus(i, "closing") /\ Unch

RollbackFail(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "rollback"
    /\ running' = [running EXCEPT ![mut[i].proj] = BAD]
    /\ Emit(i, "FAILED_STRANDED") /\ SetStatus(i, "closing")
    /\ UNCHANGED rolledBack /\ Unch

ReleaseLock(i) ==
    /\ mut[i].status # "none" /\ mut[i].status = "closing"
    /\ SetStatus(i, "done")
    /\ UNCHANGED <<running, rolledBack, receipts>> /\ Unch

\* The holder dies (OOM, reboot, runner cancel), including after it wrote its
\* receipt but before its exit trap ran. The kernel releases the lock.
Die(i) ==
    /\ mut[i].status # "none" /\ mut[i].status \in Live
    /\ running' = IF mut[i].status \in Mutating
                  THEN [running EXCEPT ![mut[i].proj] = BAD] ELSE running
    /\ SetStatus(i, "dead")
    /\ UNCHANGED <<rolledBack, receipts>> /\ Unch

\* The job timeout kills the holder at its declared ceiling.
Kill(i) ==
    /\ mut[i].status # "none" /\ mut[i].status \in Live
    /\ now = mut[i].deadline
    /\ Die(i)

Reap(i) ==
    /\ Reaper
    /\ mut[i].status # "none" /\ mut[i].status = "dead"
    /\ ReaperChecksReceipt => receipts[i] = <<>>
    /\ Emit(i, "FAILED_STRANDED") /\ SetStatus(i, "reaped")
    /\ UNCHANGED <<running, rolledBack>> /\ Unch

-----------------------------------------------------------------------------
Next ==
    \/ Tick \/ ReleaseEvent \/ RuntimeMerge
    \/ DispatcherGrant \/ DispatcherNoOp \/ AgentStart \/ AgentNoOp
    \/ \E p \in HandProjects : HandStart(p)
    \/ \E i \in Ids : \/ ForwardPass(i) \/ ForwardHealthFail(i) \/ ForwardNoRollback(i)
                      \/ RollbackOK(i) \/ RollbackFail(i) \/ ReleaseLock(i)
                      \/ Die(i) \/ Reap(i)

Fairness ==
    /\ WF_vars(Tick)
    /\ \A i \in Ids : WF_vars(Kill(i)) /\ WF_vars(Reap(i)) /\ WF_vars(ReleaseLock(i))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
TypeOK ==
    /\ now \in 0..MaxT
    /\ \A i \in Ids : receipts[i] \in Seq(Kinds)
    /\ \A i \in Ids : mut[i].status = "none" \/ mut[i].status \in Live \cup {"none", "done", "dead", "reaped"}

\* I1: no two mutations of one compose project overlap.
NoOverlap ==
    \A i, j \in Ids : (i # j /\ mut[i].status # "none" /\ mut[j].status # "none"
                       /\ mut[i].status \in Mutating /\ mut[j].status \in Mutating)
                      => mut[i].proj # mut[j].proj

\* I2: no live mutation's [start, ceiling] comes within Guard ticks of a
\* probe window (stronger than "its ceiling does not end within 10 minutes of
\* a window": it also refuses a start inside a window).
NoWindowCollision ==
    \A i \in Ids : (mut[i].status # "none" /\ mut[i].status \in Live)
                   => WindowOK(mut[i].start, mut[i].deadline)

\* I3a (safety half): no reconcile ever carries more than one terminal receipt.
AtMostOneReceipt == \A i \in Ids : Len(receipts[i]) <= 1

\* I3b (liveness half): every started reconcile eventually carries one.
EveryStartedTerminates ==
    \A i \in Ids : [](mut[i].status # "none" => <>(Len(receipts[i]) = 1))

\* I4: a rolled-back lane never reports lane_sync PASS. lane_sync compares the
\* running composition with the desired state rendered from the release (T1.1),
\* never with the last receipt.
LaneSyncPass(p) == running[p] = desired[p]
RolledBackNeverGreen == \A p \in Projects : rolledBack[p] => ~LaneSyncPass(p)

\* Terminal states are genuine: the model stops at the horizon.
=============================================================================
