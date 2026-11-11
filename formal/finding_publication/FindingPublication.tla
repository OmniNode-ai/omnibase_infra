---------------------------- MODULE FindingPublication ----------------------------
(* Finding publication protocol (OMN-22258, task T0 of the transcript         *)
(* intent-to-outcome graph plan, task T9). One finding id. Two runs may both  *)
(* work on it at the same time; a run can die at any point (one crash in      *)
(* all). A run does, in order, against the outside forge and the reducer:     *)
(*   B   create-or-adopt the branch (its name is derived from the finding id, *)
(*       so two runs name the same branch)                                    *)
(*   E1  emit the stage event "branch" to the reducer                         *)
(*   P   create-or-adopt the pull request for that branch                     *)
(*   E2  emit the stage event "pr" to the reducer                             *)
(* B and P are two separate forge calls, so a run that dies between them      *)
(* leaves a branch with no pull request. The reducer records the finding's    *)
(* stage (0 none, 1 branch, 2 pr) from stage events that arrive in any order. *)
(* A recorded finding never moves to an earlier stage: the reducer keeps the  *)
(* higher of the recorded stage and the event's.                              *)
(*                                                                            *)
(* Create-or-adopt is one atomic call on the forge: it creates the object if  *)
(* none exists for the name, otherwise returns the existing one and the run   *)
(* carries on with it (the forge refuses a second pull request for the same   *)
(* head branch, the run reads that refusal as `adopt`).                       *)
(*                                                                            *)
(* The forge contract the model assumes, and the control removes: a create-   *)
(* or-adopt call on the forge is atomic and keyed by the branch name, as the  *)
(* hosted forge's refusal of a second pull request for one head branch is.    *)
(* The control `lookup_then_create` is the client algorithm that lets the     *)
(* forge's create call run after a lookup, and models a forge that accepts a  *)
(* second pull request; T9 must either rely on the refusal (and prove it on   *)
(* the forge adapter) or keep the create-or-adopt call atomic.                *)
(*                                                                            *)
(* Property                  Kind       Config that must break it             *)
(* AtMostOnePr               invariant  mut_lookup_then_create (the run looks *)
(*                                      up the pull request, and creates one  *)
(*                                      later without adopting: two runs     *)
(*                                      create two)                           *)
(* BranchGetsPr              invariant  mut_branch_is_done (a run that finds  *)
(*                                      the branch already there treats the   *)
(*                                      finding as finished: after the other  *)
(*                                      run died between B and P no pull     *)
(*                                      request is ever opened)               *)
(* RecordedNeverBack         invariant  mut_stage_by_arrival (the reducer     *)
(*                                      takes the stage of the last event)    *)
(* RecordedBackedByForge     invariant  Model.cfg (a recorded stage is        *)
(*                                      backed by the forge object it names)  *)
(* PrGetsRecorded            property   mut_no_fairness (SpecNoFair)          *)
(* AdoptBranchWitness        witness    wit_adopt_branch (a run died after    *)
(*                                      the branch, before the pull request;  *)
(*                                      the other run opened it and it is     *)
(*                                      recorded)                             *)
(* AdoptPrWitness            witness    wit_adopt_pr (a run died after the    *)
(*                                      pull request, before recording it;    *)
(*                                      the other run adopted it and          *)
(*                                      recorded it)                          *)
(* ReorderWitness            witness    wit_reorder (the stage "branch" event *)
(*                                      is applied after "pr" and ignored)    *)
(*                                                                            *)
(* Bounds: 2 runs, 1 finding, 1 crash. results/ holds TLC output for every    *)
(* cfg and model.sha256 (sha256 of this file then the cfg files in sorted     *)
(* order).                                                                    *)
(*                                                                            *)
(* Not modelled: a second crash, a closed or merged pull request (the         *)
(* finding stays one pull request), forge outages and rate limits (a failed   *)
(* call is a crash), and the branch's content.                                *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS MUT,        \* "none" | "lookup_then_create" | "branch_is_done" | "stage_by_arrival"
          MaxCrash    \* crashes of a run in one behaviour

Runs == {"r1", "r2"}

VARIABLES branch,    \* the forge has the branch
          prs,       \* pull requests the forge holds for the finding
          pc,        \* [Runs -> "new" | "b_done" | "b_rec" | "looked" | "p_done" | "done" | "dead"]
          looked,    \* [Runs -> TRUE if the lookup found a pull request] (lookup_then_create)
          pend,      \* stage events emitted and not yet applied: a set of <<run, stage>>
          rec,       \* the reducer's recorded stage for the finding, 0..2
          hiRec,     \* ghost: highest stage ever recorded
          crashes,
          crashedAt,       \* ghost: where the crashed run stood ("none" before a crash)
          ignored          \* ghost: the reducer ignored an older stage event

vars == <<branch, prs, pc, looked, pend, rec, hiRec, crashes, crashedAt, ignored>>

Init == /\ branch = FALSE
        /\ prs = 0
        /\ pc = [r \in Runs |-> "new"]
        /\ looked = [r \in Runs |-> FALSE]
        /\ pend = {}
        /\ rec = 0
        /\ hiRec = 0
        /\ crashes = 0
        /\ crashedAt = "none"
        /\ ignored = FALSE

Active(r) == pc[r] \notin {"done", "dead"}

\* B: create-or-adopt the branch.
StepB(r) ==
  /\ pc[r] = "new"
  /\ branch' = TRUE
  /\ IF branch /\ MUT = "branch_is_done"
        THEN pc' = [pc EXCEPT ![r] = "done"]
        ELSE pc' = [pc EXCEPT ![r] = "b_done"]
  /\ UNCHANGED <<prs, looked, pend, rec, hiRec, crashes, crashedAt, ignored>>

\* E1: emit the stage event "branch".
Emit1(r) ==
  /\ pc[r] = "b_done"
  /\ pend' = pend \cup {<<r, 1>>}
  /\ pc' = [pc EXCEPT ![r] = "b_rec"]
  /\ UNCHANGED <<branch, prs, looked, rec, hiRec, crashes, crashedAt, ignored>>

\* P: create-or-adopt the pull request, one atomic call.
StepP(r) ==
  /\ MUT # "lookup_then_create"
  /\ pc[r] = "b_rec"
  /\ prs' = IF prs = 0 THEN 1 ELSE prs
  /\ pc' = [pc EXCEPT ![r] = "p_done"]
  /\ UNCHANGED <<branch, looked, pend, rec, hiRec, crashes, crashedAt, ignored>>

\* Control: look the pull request up, and in a later step create one when the
\* lookup found none. The forge does not refuse the second pull request.
Lookup(r) ==
  /\ MUT = "lookup_then_create"
  /\ pc[r] = "b_rec"
  /\ looked' = [looked EXCEPT ![r] = prs > 0]
  /\ pc' = [pc EXCEPT ![r] = "looked"]
  /\ UNCHANGED <<branch, prs, pend, rec, hiRec, crashes, crashedAt, ignored>>

CreateAfterLookup(r) ==
  /\ MUT = "lookup_then_create"
  /\ pc[r] = "looked"
  /\ prs' = IF looked[r] THEN prs ELSE prs + 1
  /\ pc' = [pc EXCEPT ![r] = "p_done"]
  /\ UNCHANGED <<branch, looked, pend, rec, hiRec, crashes, crashedAt, ignored>>

\* E2: emit the stage event "pr".
Emit2(r) ==
  /\ pc[r] = "p_done"
  /\ pend' = pend \cup {<<r, 2>>}
  /\ pc' = [pc EXCEPT ![r] = "done"]
  /\ UNCHANGED <<branch, prs, looked, rec, hiRec, crashes, crashedAt, ignored>>

\* A run dies where it stands; its forge objects and its emitted events remain.
Crash(r) ==
  /\ crashes < MaxCrash
  /\ pc[r] \notin {"new", "done", "dead"}
  /\ pc' = [pc EXCEPT ![r] = "dead"]
  /\ crashes' = crashes + 1
  /\ crashedAt' = pc[r]
  /\ UNCHANGED <<branch, prs, looked, pend, rec, hiRec, ignored>>

\* The reducer applies one stage event; events arrive in any order.
Apply(e) ==
  /\ e \in pend
  /\ pend' = pend \ {e}
  /\ LET s == e[2]
         new == IF MUT = "stage_by_arrival" THEN s ELSE IF s > rec THEN s ELSE rec
     IN /\ rec' = new
        /\ hiRec' = IF new > hiRec THEN new ELSE hiRec
        /\ ignored' = (ignored \/ (s < rec /\ MUT # "stage_by_arrival"))
  /\ UNCHANGED <<branch, prs, pc, looked, crashes, crashedAt>>

Next == \/ \E r \in Runs : StepB(r) \/ Emit1(r) \/ StepP(r) \/ Lookup(r)
                           \/ CreateAfterLookup(r) \/ Emit2(r) \/ Crash(r)
        \/ \E e \in pend : Apply(e)

Spec       == Init /\ [][Next]_vars /\ WF_vars(Next)
SpecNoFair == Init /\ [][Next]_vars

TypeOK == /\ branch \in BOOLEAN
          /\ prs \in 0..2
          /\ pc \in [Runs -> {"new", "b_done", "b_rec", "looked", "p_done", "done", "dead"}]
          /\ looked \in [Runs -> BOOLEAN]
          /\ pend \subseteq (Runs \X {1, 2})
          /\ rec \in 0..2
          /\ hiRec \in 0..2
          /\ crashes \in 0..MaxCrash
          /\ crashedAt \in {"none", "b_done", "b_rec", "looked", "p_done"}
          /\ ignored \in BOOLEAN

\* At most one pull request exists for the finding id.
AtMostOnePr == prs <= 1

\* A branch that exists with no pull request is completed: when no run is
\* working any more, a branch has a pull request.
BranchGetsPr == (\A r \in Runs : ~Active(r)) => (branch => prs >= 1)

\* A recorded finding is never moved back to an earlier stage.
RecordedNeverBack == rec >= hiRec

\* The recorded stage names something the forge has.
RecordedBackedByForge == /\ rec >= 1 => branch
                         /\ rec >= 2 => prs >= 1

\* A pull request that exists is eventually recorded (stage 2), and a finding
\* whose branch exists is eventually given a pull request.
PrGetsRecorded == (prs >= 1) ~> (rec = 2)
BranchGetsPrEventually == branch ~> (prs >= 1)

\* Witnesses: each is an invariant that the model must violate.
\* A run died between the branch and the pull request; the other run made the
\* pull request and it is recorded.
AdoptBranchWitness == ~(crashedAt \in {"b_done", "b_rec"} /\ prs >= 1 /\ rec = 2)
\* A run died after the pull request, before recording it; the other run adopted
\* the pull request and recorded it.
AdoptPrWitness == ~(crashedAt = "p_done" /\ prs = 1 /\ rec = 2)
ReorderWitness == ~ignored

=============================================================================
