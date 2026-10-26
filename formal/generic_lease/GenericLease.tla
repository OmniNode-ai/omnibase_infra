----------------------------- MODULE GenericLease -----------------------------
(* The generic lease (OMN-21418, the runtime cluster and generic lease plan *)
(* under OMN-21279). A client leases a named resource from the coordination *)
(* store; a grant is one atomic store transaction that issues a fence, the  *)
(* store revision of the grant, from one counter shared by every resource. *)
(* A renewal (keepalive) extends the lease's TTL and keeps its fence. The   *)
(* store treats a lease whose TTL has run out as free. A client stamps each *)
(* effect with the fence it holds; the effect target for the resource       *)
(* accepts an effect only when its fence is at least the highest fence it   *)
(* has accepted for that resource.                                          *)
(*                                                                            *)
(* A client may pause for any number of ticks between two steps and keeps   *)
(* believing it holds a lease until a renewal fails or it releases, so a    *)
(* client whose lease expired can still send effects. An effect may sit in  *)
(* the network for any number of ticks. The grant a fence was issued by is  *)
(* tracked as a ghost: grants[r] is the sequence of fences r's grants       *)
(* issued, and an effect carries the index of its grant.                    *)
(*                                                                            *)
(* Property                   Kind       Config that must break it          *)
(* FenceMonotonicPerResource  invariant  mut_regrant_same_fence (a new grant *)
(*                                       reuses the resource's last fence), *)
(*                                       mut_renew_changes_fence (a renewal *)
(*                                       issues a new fence, so a fence no  *)
(*                                       longer names one grant)            *)
(* NoEffectWithStaleFence     invariant  mut_effect_skips_check (the target *)
(*                                       accepts any fence),                *)
(*                                       mut_regrant_same_fence_effect (the *)
(*                                       same-fence mutation checked on     *)
(*                                       this invariant alone: the target   *)
(*                                       cannot tell the two grants apart)  *)
(* ExpiredLeaseFreesResource  invariant  mut_expiry_holds (an expired lease *)
(*                                       blocks a grant until released)     *)
(* NotStaleRefused            witness    wit_stale_refused (Model.cfg       *)
(*                                       values; its violation shows the    *)
(*                                       target refusing a stale effect)    *)
(*                                                                            *)
(* obs_renew_changes_fence_effect.cfg runs mut_renew_changes_fence's values *)
(* with FenceMonotonicPerResource left out, and passes: in this model a     *)
(* renewal that changes the fence breaks only the one-fence-one-grant       *)
(* identity, not the effect ordering or expiry.                             *)
(*                                                                            *)
(* Bounds: 2 clients, 2 resources, TTL 2, time 0..4, fence counter 0..4,    *)
(* 3 effects. results/ holds TLC output for every cfg, and model.sha256,    *)
(* the sha256 of this file then the cfg files in sorted order. Each run,    *)
(* from this directory:                                                     *)
(*   java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 \             *)
(*     -metadir "$(mktemp -d)" GenericLease.tla                              *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Clients,          \* lease holders
          Resources,        \* named resources
          None,             \* no holder
          TTL,              \* lease TTL at the store
          MaxTime,          \* time bound
          MaxRev,           \* fence counter bound (a state constraint)
          MaxEffects,       \* effect bound
          RenewKeepsFence,  \* a renewal keeps the lease's fence
          FreshFence,       \* every grant takes a new fence from the counter
          EffectCheck,      \* the target checks the effect's fence
          ExpiryFrees       \* the store treats an expired lease as free

VARIABLES now,      \* global time
          rev,      \* the store's revision counter: the last fence issued
          lease,    \* lease[r]: [holder, fence, exp, g] at the store
          grants,   \* ghost: grants[r][g] is the fence r's g-th grant issued
          belief,   \* belief[c][r]: [fence, g] the client thinks it holds, or Nil
          net,      \* effects in flight: [id, r, fence, g]
          sent,     \* effects sent so far
          highest,  \* highest[r]: the highest fence r's target accepted
          applied   \* ghost: applied[r] is the grant index of each accepted effect

vars == <<now, rev, lease, grants, belief, net, sent, highest, applied>>

Nil == [fence |-> 0, g |-> 0]
Free == [holder |-> None, fence |-> 0, exp |-> 0, g |-> 0]

TypeOK ==
  /\ now \in 0..MaxTime
  /\ rev \in Nat
  /\ lease \in [Resources -> [holder : Clients \cup {None}, fence : Nat,
                               exp : Nat, g : Nat]]
  /\ grants \in [Resources -> Seq(Nat)]
  /\ belief \in [Clients -> [Resources -> [fence : Nat, g : Nat]]]
  /\ sent \in 0..MaxEffects
  /\ highest \in [Resources -> Nat]
  /\ applied \in [Resources -> Seq(Nat)]

Init ==
  /\ now = 0
  /\ rev = 0
  /\ lease = [r \in Resources |-> Free]
  /\ grants = [r \in Resources |-> << >>]
  /\ belief = [c \in Clients |-> [r \in Resources |-> Nil]]
  /\ net = {}
  /\ sent = 0
  /\ highest = [r \in Resources |-> 0]
  /\ applied = [r \in Resources |-> << >>]

Expired(r) == lease[r].holder # None /\ now >= lease[r].exp
Live(r) == lease[r].holder # None /\ now < lease[r].exp

\* The store's own test that a grant may proceed.
Grantable(r) == lease[r].holder = None \/ (ExpiryFrees /\ Expired(r))

Tick ==
  /\ now < MaxTime
  /\ now' = now + 1
  /\ UNCHANGED <<rev, lease, grants, belief, net, sent, highest, applied>>

\* One atomic store transaction: take the free (or expired) resource and a fence.
Grant(c, r) ==
  /\ Grantable(r)
  /\ LET g == Len(grants[r]) + 1
         f == IF ~FreshFence /\ Len(grants[r]) > 0
                THEN grants[r][Len(grants[r])]
                ELSE rev + 1
     IN /\ rev' = IF f > rev THEN f ELSE rev
        /\ lease' = [lease EXCEPT ![r] = [holder |-> c, fence |-> f,
                                          exp |-> now + TTL, g |-> g]]
        /\ grants' = [grants EXCEPT ![r] = Append(@, f)]
        /\ belief' = [belief EXCEPT ![c][r] = [fence |-> f, g |-> g]]
  /\ UNCHANGED <<now, net, sent, highest, applied>>

\* A keepalive succeeds only on the client's own unexpired grant.
Renew(c, r) ==
  /\ belief[c][r] # Nil
  /\ IF lease[r].holder = c /\ lease[r].g = belief[c][r].g /\ Live(r)
       THEN IF RenewKeepsFence
              THEN /\ lease' = [lease EXCEPT ![r].exp = now + TTL]
                   /\ UNCHANGED <<rev, belief>>
              ELSE /\ rev' = rev + 1
                   /\ lease' = [lease EXCEPT ![r].exp = now + TTL,
                                             ![r].fence = rev + 1]
                   /\ belief' = [belief EXCEPT ![c][r].fence = rev + 1]
       ELSE \* the renewal fails: the client learns it lost the lease
            /\ belief' = [belief EXCEPT ![c][r] = Nil]
            /\ UNCHANGED <<rev, lease>>
  /\ UNCHANGED <<now, grants, net, sent, highest, applied>>

\* A client releases its own grant (a release of a lost grant is a no-op).
Release(c, r) ==
  /\ belief[c][r] # Nil
  /\ lease' = IF lease[r].holder = c /\ lease[r].g = belief[c][r].g
                THEN [lease EXCEPT ![r] = Free]
                ELSE lease
  /\ belief' = [belief EXCEPT ![c][r] = Nil]
  /\ UNCHANGED <<now, rev, grants, net, sent, highest, applied>>

\* A client that believes it holds r sends an effect stamped with its fence.
Effect(c, r) ==
  /\ belief[c][r] # Nil
  /\ sent < MaxEffects
  /\ sent' = sent + 1
  /\ net' = net \cup {[id |-> sent + 1, r |-> r, fence |-> belief[c][r].fence,
                       g |-> belief[c][r].g]}
  /\ UNCHANGED <<now, rev, lease, grants, belief, highest, applied>>

\* The target for e.r receives an effect.
Deliver(e) ==
  /\ e \in net
  /\ net' = net \ {e}
  /\ IF ~EffectCheck \/ e.fence >= highest[e.r]
       THEN /\ highest' = [highest EXCEPT ![e.r] = IF e.fence > @ THEN e.fence ELSE @]
            /\ applied' = [applied EXCEPT ![e.r] = Append(@, e.g)]
       ELSE UNCHANGED <<highest, applied>>
  /\ UNCHANGED <<now, rev, lease, grants, belief, sent>>

Next ==
  \/ Tick
  \/ \E c \in Clients, r \in Resources :
       Grant(c, r) \/ Renew(c, r) \/ Release(c, r) \/ Effect(c, r)
  \/ \E e \in net : Deliver(e)

Spec == Init /\ [][Next]_vars

\* Bound the fence counter; ENABLED below ignores this constraint.
RevBound == rev <= MaxRev

\* ---- properties
\* Each grant of a resource issues a fence above the one before, and the
\* fence on the store's lease is the one its grant issued: a fence names
\* exactly one grant.
FenceMonotonicPerResource ==
  \A r \in Resources :
    /\ \A i \in 1..(Len(grants[r]) - 1) : grants[r][i] < grants[r][i + 1]
    /\ lease[r].holder # None => lease[r].fence = grants[r][lease[r].g]

\* No target accepts an effect of an older grant after an effect of a newer
\* one: once the new holder's effect lands, the old holder's are refused.
NoEffectWithStaleFence ==
  \A r \in Resources :
    \A i \in 1..(Len(applied[r]) - 1) : applied[r][i] <= applied[r][i + 1]

\* A lease whose TTL has run out never keeps the resource from a new grant.
ExpiredLeaseFreesResource ==
  \A r \in Resources : Expired(r) => \A c \in Clients : ENABLED Grant(c, r)

\* ---- witness (Model.cfg values; TLC must report it violated)
\* An effect stamped with an older grant's fence reaches a target that has
\* accepted a newer one, and is refused: the race the fence exists for occurs.
NotStaleRefused ==
  ~(\E e \in net : e.fence < highest[e.r])
=============================================================================
