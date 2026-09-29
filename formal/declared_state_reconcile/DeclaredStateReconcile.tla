-------------------- MODULE DeclaredStateReconcile --------------------
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS Grants, Declared, Reconcilers, MaxEdits, MaxFlips, MUT

ASSUME Declared \subseteq Grants

VARIABLES grants, ver, observable, lock, pc,
          obsVer, obsGrants, toAdd, toRemove, expect,
          editsLeft, flipsLeft,
          staleApplied, unobservableApplied, removedNeeded

vars == <<grants, ver, observable, lock, pc,
          obsVer, obsGrants, toAdd, toRemove, expect,
          editsLeft, flipsLeft,
          staleApplied, unobservableApplied, removedNeeded>>

Phases == {"idle", "observed", "planned", "applying", "halted"}

TypeOK ==
    /\ grants \in SUBSET Grants
    /\ ver \in Nat
    /\ observable \in BOOLEAN
    /\ lock \in Reconcilers \cup {"none"}
    /\ pc \in [Reconcilers -> Phases]
    /\ obsVer \in [Reconcilers -> Nat]
    /\ obsGrants \in [Reconcilers -> SUBSET Grants]
    /\ toAdd \in [Reconcilers -> SUBSET Grants]
    /\ toRemove \in [Reconcilers -> SUBSET Grants]
    /\ expect \in [Reconcilers -> Nat]
    /\ editsLeft \in 0..MaxEdits
    /\ flipsLeft \in 0..MaxFlips
    /\ staleApplied \in BOOLEAN
    /\ unobservableApplied \in BOOLEAN
    /\ removedNeeded \in BOOLEAN

Init ==
    /\ grants \in SUBSET Grants
    /\ ver = 0
    /\ observable = TRUE
    /\ lock = "none"
    /\ pc = [r \in Reconcilers |-> "idle"]
    /\ obsVer = [r \in Reconcilers |-> 0]
    /\ obsGrants = [r \in Reconcilers |-> {}]
    /\ toAdd = [r \in Reconcilers |-> {}]
    /\ toRemove = [r \in Reconcilers |-> {}]
    /\ expect = [r \in Reconcilers |-> 0]
    /\ editsLeft = MaxEdits
    /\ flipsLeft = MaxFlips
    /\ staleApplied = FALSE
    /\ unobservableApplied = FALSE
    /\ removedNeeded = FALSE

\* Toggle exactly one grant, so every hand edit changes the surface.
HandEdit ==
    /\ editsLeft > 0
    /\ \E g \in Grants:
           grants' = IF g \in grants
                     THEN grants \ {g}
                     ELSE grants \cup {g}
    /\ ver' = ver + 1
    /\ editsLeft' = editsLeft - 1
    /\ UNCHANGED <<observable, lock, pc, obsVer, obsGrants,
                   toAdd, toRemove, expect, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Flip ==
    /\ flipsLeft > 0
    /\ observable
    /\ observable' = FALSE
    /\ flipsLeft' = flipsLeft - 1
    /\ UNCHANGED <<grants, ver, lock, pc, obsVer, obsGrants,
                   toAdd, toRemove, expect, editsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Recover ==
    /\ ~observable
    /\ observable' = TRUE
    /\ UNCHANGED <<grants, ver, lock, pc, obsVer, obsGrants,
                   toAdd, toRemove, expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Observe(r) ==
    /\ pc[r] = "idle"
    /\ IF observable
       THEN /\ obsVer' = [obsVer EXCEPT ![r] = ver]
            /\ obsGrants' = [obsGrants EXCEPT ![r] = grants]
            /\ expect' = [expect EXCEPT ![r] = ver]
            /\ pc' = [pc EXCEPT ![r] = "observed"]
       ELSE /\ pc' = [pc EXCEPT ![r] = "halted"]
            /\ UNCHANGED <<obsVer, obsGrants, expect>>
    /\ UNCHANGED <<grants, ver, observable, lock, toAdd, toRemove,
                   editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Restart(r) ==
    /\ pc[r] = "halted"
    /\ observable
    /\ pc' = [pc EXCEPT ![r] = "idle"]
    /\ UNCHANGED <<grants, ver, observable, lock, obsVer, obsGrants,
                   toAdd, toRemove, expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Plan(r) ==
    /\ pc[r] = "observed"
    /\ toAdd' = [toAdd EXCEPT ![r] = Declared \ obsGrants[r]]
    /\ toRemove' =
           [toRemove EXCEPT
               ![r] = IF MUT = "no_filter"
                      THEN obsGrants[r]
                      ELSE obsGrants[r] \ Declared]
    /\ pc' = [pc EXCEPT ![r] = "planned"]
    /\ UNCHANGED <<grants, ver, observable, lock, obsVer, obsGrants,
                   expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Acquire(r) ==
    /\ pc[r] = "planned"
    /\ IF MUT = "no_lock"
       THEN UNCHANGED lock
       ELSE /\ lock = "none"
            /\ lock' = r
    /\ pc' = [pc EXCEPT ![r] = "applying"]
    /\ UNCHANGED <<grants, ver, observable, obsVer, obsGrants,
                   toAdd, toRemove, expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

FreshOK(r) == (MUT = "no_fresh") \/ (ver = expect[r])
ObservableOK == (MUT = "no_observable_guard") \/ observable

ApplyAdd(r) ==
    /\ pc[r] = "applying"
    /\ toAdd[r] # {}
    /\ FreshOK(r)
    /\ ObservableOK
    /\ \E g \in toAdd[r]:
           /\ grants' = grants \cup {g}
           /\ toAdd' = [toAdd EXCEPT ![r] = @ \ {g}]
    /\ ver' = ver + 1
    /\ expect' = [expect EXCEPT ![r] = @ + 1]
    /\ staleApplied' = (staleApplied \/ (ver # expect[r]))
    /\ unobservableApplied' = (unobservableApplied \/ ~observable)
    /\ UNCHANGED <<observable, lock, pc, obsVer, obsGrants, toRemove,
                   editsLeft, flipsLeft, removedNeeded>>

ApplyRemove(r) ==
    /\ pc[r] = "applying"
    /\ toAdd[r] = {}
    /\ toRemove[r] # {}
    /\ FreshOK(r)
    /\ ObservableOK
    /\ \E g \in toRemove[r]:
           /\ grants' = grants \ {g}
           /\ toRemove' = [toRemove EXCEPT ![r] = @ \ {g}]
           /\ removedNeeded' = (removedNeeded \/ (g \in Declared))
    /\ ver' = ver + 1
    /\ expect' = [expect EXCEPT ![r] = @ + 1]
    /\ staleApplied' = (staleApplied \/ (ver # expect[r]))
    /\ unobservableApplied' = (unobservableApplied \/ ~observable)
    /\ UNCHANGED <<observable, lock, pc, obsVer, obsGrants, toAdd,
                   editsLeft, flipsLeft>>

Finish(r) ==
    /\ pc[r] = "applying"
    /\ toAdd[r] = {}
    /\ toRemove[r] = {}
    /\ pc' = [pc EXCEPT ![r] = "idle"]
    /\ lock' = IF lock = r THEN "none" ELSE lock
    /\ UNCHANGED <<grants, ver, observable, obsVer, obsGrants,
                   toAdd, toRemove, expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Refuse(r) ==
    /\ pc[r] = "applying"
    /\ (~FreshOK(r) \/ ~ObservableOK)
    /\ pc' = [pc EXCEPT ![r] = "idle"]
    /\ lock' = IF MUT = "leak_lock"
               THEN lock
               ELSE IF lock = r THEN "none" ELSE lock
    /\ toAdd' = [toAdd EXCEPT ![r] = {}]
    /\ toRemove' = [toRemove EXCEPT ![r] = {}]
    /\ UNCHANGED <<grants, ver, observable, obsVer, obsGrants,
                   expect, editsLeft, flipsLeft,
                   staleApplied, unobservableApplied, removedNeeded>>

Next ==
    \/ HandEdit
    \/ Flip
    \/ Recover
    \/ \E r \in Reconcilers:
           \/ Observe(r)
           \/ Plan(r)
           \/ Acquire(r)
           \/ ApplyAdd(r)
           \/ ApplyRemove(r)
           \/ Finish(r)
           \/ Refuse(r)
           \/ Restart(r)

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(Recover)
    /\ \A r \in Reconcilers:
           /\ WF_vars(Observe(r))
           /\ WF_vars(Plan(r))
           /\ WF_vars(Acquire(r))
           /\ WF_vars(ApplyAdd(r))
           /\ WF_vars(ApplyRemove(r))
           /\ WF_vars(Finish(r))
           /\ WF_vars(Refuse(r))
           /\ WF_vars(Restart(r))

NoInterleave ==
    Cardinality({r \in Reconcilers : pc[r] = "applying"}) <= 1

NoStaleApply == ~staleApplied
NoNeededRemoved == ~removedNeeded
NoApplyOnUnobservable == ~unobservableApplied

\* Finite budgets imply finitely many disturbances, even if unused
\* budget remains forever: HandEdit and Flip deliberately have no fairness.
Converges == <>[](grants = Declared)

=======================================================================
