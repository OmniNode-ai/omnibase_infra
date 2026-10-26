------------------------ MODULE RuntimePrimaryElection ------------------------
(* Runtime primary election over the coordination store (OMN-21418, the     *)
(* runtime cluster and generic lease plan under OMN-21279). The store holds *)
(* one leader key bound to a TTL lease; a grant bumps the election epoch,   *)
(* which is the store revision of the grant. A runtime that wins keeps its  *)
(* own local deadline, TTL - Margin after the grant or renewal, and         *)
(* self-fences: it acts (issues a write) only before that deadline, and it  *)
(* steps down when the deadline passes or when it sees another holder. The  *)
(* write target accepts a write only when its epoch is at least the highest *)
(* epoch it has accepted.                                                   *)
(*                                                                            *)
(* Time is one global integer clock; the store expires the key at its own   *)
(* TTL, possibly late. A runtime may pause for any number of ticks between  *)
(* two steps, and a write may sit in the network for any number of ticks.   *)
(* Clock drift between a runtime and the store is not modelled: Margin is   *)
(* the budget the implementation must hold it under.                        *)
(*                                                                            *)
(* Property                Kind       Mutation that must break it            *)
(* AtMostOneActingPrimary  invariant  mut_no_self_fence (a runtime acts on  *)
(*                                    its belief alone, past its deadline)  *)
(* EpochMonotonic          invariant  mut_unguarded_write (the target       *)
(*                                    accepts a write without the epoch     *)
(*                                    check, so a delayed stale write lands)*)
(* NotTwoEpochsApplied     witness    wit_two_epochs_applied (Model.cfg     *)
(*                                    values; its violation shows writes of *)
(*                                    two epochs reach the target)          *)
(*                                                                            *)
(* Bounds: 3 runtimes, TTL 3, Margin 1, time 0..6, 3 writes. results/ holds *)
(* TLC output for Model.cfg and each mut_*.cfg and wit_*.cfg, and           *)
(* model.sha256, the sha256 of this file then the cfg files in sorted       *)
(* order. Each run, from this directory:                                    *)
(*   java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 \             *)
(*     -metadir "$(mktemp -d)" RuntimePrimaryElection.tla                    *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Nodes,        \* runtimes
          None,         \* no holder of the leader key
          TTL,          \* lease TTL at the store
          Margin,       \* the runtime's local deadline is TTL - Margin
          MaxTime,      \* time bound
          MaxWrites,    \* write bound
          SelfFence,    \* a runtime acts only before its local deadline
          WriteGuard    \* the target checks the write's epoch

ASSUME TTL > Margin /\ Margin \in Nat

VARIABLES now,       \* global time
          holder,    \* the leader key's holder at the store, or None
          storeExp,  \* when the store expires the leader key's lease
          epoch,     \* the store's election epoch (revision of the last grant)
          role,      \* "primary" or "follower", each runtime's belief
          myEpoch,   \* the epoch a runtime won with
          deadline,  \* a runtime's local self-fence deadline
          net,       \* writes in flight: <<id, epoch>>
          issued,    \* writes issued so far
          applied,   \* epochs of the writes the target accepted, in order
          tgtEpoch   \* the highest epoch the target has accepted

vars == <<now, holder, storeExp, epoch, role, myEpoch, deadline,
          net, issued, applied, tgtEpoch>>

TypeOK ==
  /\ now \in 0..MaxTime
  /\ holder \in Nodes \cup {None}
  /\ storeExp \in Nat
  /\ epoch \in Nat
  /\ role \in [Nodes -> {"primary", "follower"}]
  /\ myEpoch \in [Nodes -> Nat]
  /\ deadline \in [Nodes -> Nat]
  /\ net \subseteq (1..MaxWrites) \X Nat
  /\ issued \in 0..MaxWrites
  /\ applied \in Seq(Nat)
  /\ tgtEpoch \in Nat

Init ==
  /\ now = 0
  /\ holder = None
  /\ storeExp = 0
  /\ epoch = 0
  /\ role = [n \in Nodes |-> "follower"]
  /\ myEpoch = [n \in Nodes |-> 0]
  /\ deadline = [n \in Nodes |-> 0]
  /\ net = {}
  /\ issued = 0
  /\ applied = << >>
  /\ tgtEpoch = 0

\* A runtime acts when it believes it is primary and, under self-fencing,
\* its local deadline has not passed.
Acting(n) == role[n] = "primary" /\ (SelfFence => now < deadline[n])

Tick ==
  /\ now < MaxTime
  /\ now' = now + 1
  /\ UNCHANGED <<holder, storeExp, epoch, role, myEpoch, deadline,
                 net, issued, applied, tgtEpoch>>

\* The store expires the leader key once its TTL has run out (maybe late).
Expire ==
  /\ holder # None
  /\ now >= storeExp
  /\ holder' = None
  /\ UNCHANGED <<now, storeExp, epoch, role, myEpoch, deadline,
                 net, issued, applied, tgtEpoch>>

\* A follower creates the free leader key: one atomic store transaction.
Campaign(n) ==
  /\ role[n] = "follower"
  /\ holder = None
  /\ holder' = n
  /\ epoch' = epoch + 1
  /\ storeExp' = now + TTL
  /\ role' = [role EXCEPT ![n] = "primary"]
  /\ myEpoch' = [myEpoch EXCEPT ![n] = epoch + 1]
  /\ deadline' = [deadline EXCEPT ![n] = now + TTL - Margin]
  /\ UNCHANGED <<now, net, issued, applied, tgtEpoch>>

\* A keepalive succeeds only on the runtime's own unexpired lease.
Renew(n) ==
  /\ Acting(n)
  /\ holder = n
  /\ now < storeExp
  /\ storeExp' = now + TTL
  /\ deadline' = [deadline EXCEPT ![n] = now + TTL - Margin]
  /\ UNCHANGED <<now, holder, epoch, role, myEpoch, net, issued, applied, tgtEpoch>>

\* Step down: the local deadline passed (self-fence), or the runtime read the
\* store and found the key is not its own.
StepDown(n) ==
  /\ role[n] = "primary"
  /\ \/ SelfFence /\ now >= deadline[n]
     \/ holder # n
  /\ role' = [role EXCEPT ![n] = "follower"]
  /\ UNCHANGED <<now, holder, storeExp, epoch, myEpoch, deadline,
                 net, issued, applied, tgtEpoch>>

\* An acting primary sends a write stamped with its epoch.
Issue(n) ==
  /\ Acting(n)
  /\ issued < MaxWrites
  /\ issued' = issued + 1
  /\ net' = net \cup {<<issued + 1, myEpoch[n]>>}
  /\ UNCHANGED <<now, holder, storeExp, epoch, role, myEpoch, deadline,
                 applied, tgtEpoch>>

\* The target receives a write; with the guard it refuses a stale epoch.
Deliver(w) ==
  /\ w \in net
  /\ net' = net \ {w}
  /\ IF ~WriteGuard \/ w[2] >= tgtEpoch
       THEN /\ applied' = Append(applied, w[2])
            /\ tgtEpoch' = w[2]
       ELSE UNCHANGED <<applied, tgtEpoch>>
  /\ UNCHANGED <<now, holder, storeExp, epoch, role, myEpoch, deadline, issued>>

Next ==
  \/ Tick \/ Expire
  \/ \E n \in Nodes : Campaign(n) \/ Renew(n) \/ StepDown(n) \/ Issue(n)
  \/ \E w \in net : Deliver(w)

Spec == Init /\ [][Next]_vars

\* ---- properties
\* No two runtimes act as primary at the same moment.
AtMostOneActingPrimary ==
  Cardinality({n \in Nodes : Acting(n)}) <= 1

\* The epochs the target accepts never go down, and no runtime or write
\* carries an epoch the store has not granted.
EpochMonotonic ==
  /\ \A i \in 1..(Len(applied) - 1) : applied[i] <= applied[i + 1]
  /\ \A n \in Nodes : myEpoch[n] <= epoch
  /\ \A w \in net : w[2] <= epoch

\* ---- witness (Model.cfg values; TLC must report it violated)
\* The target accepts writes from two different epochs: re-election and
\* writes on both sides of it are reachable, so the properties are not vacuous.
NotTwoEpochsApplied ==
  ~(\E i, j \in 1..Len(applied) : applied[i] # applied[j])
=============================================================================
