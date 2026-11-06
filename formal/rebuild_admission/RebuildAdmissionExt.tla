------------------------ MODULE RebuildAdmissionExt -------------------------
(***************************************************************************)
(* Liveness for RebuildAdmission: the fairness each property needs, and    *)
(* the two eventual properties the 11-06 text keeps. The *_ext_* configs   *)
(* run on this module; every other config runs on RebuildAdmission.tla.    *)
(***************************************************************************)
EXTENDS RebuildAdmission

CONSTANTS
  LiveTask,   \* the request AdmissibleStarts is about
  SFTasks,    \* requests asked again until they start (strong fairness)
  WFTasks,    \* the same with weak fairness only: a control
  Drop        \* fairness clauses a config removes: "storeback", "esc"

On(x) == IF x \in Drop THEN {} ELSE {"on"}
ConsumeAny(t) == Consume(t) \/ \E d \in {1, 2} : ConsumeD(t, d)

\* No clause assumes a lane fixes anything, a reading writer comes back or an operator acts.
Fairness ==
  \* a request is asked again until it starts, through the store or with no answer
  /\ \A t \in SFTasks : SF_vars(Reserve(t)) /\ SF_vars(ConsumeAny(t)) /\ SF_vars(NoAnswer(t))
  /\ \A t \in WFTasks : WF_vars(Reserve(t)) /\ WF_vars(ConsumeAny(t)) /\ WF_vars(NoAnswer(t))
  \* an outage ends; "storeback" in Drop lets it last for ever
  /\ \A o \in On("storeback") : WF_vars(StoreBack)
  \* the escalation timer fires; strong, since the store it writes to can flap
  /\ \A o \in On("esc"), p \in BreakPieces : SF_vars(Escalates(p))

LiveSpec == Spec /\ Fairness

\* P1: no unanswered request refuses a start, and an admissible one starts.
AdmissibleStarts ==
  []<>(pc[LiveTask] = "idle" /\ (~storeUp \/ Decide(K(LiveTask), LiveTask) = "admit"))
    => []<>(pc[LiveTask] = "run")
\* P3: a stop clears or escalates.
StopClearsOrEscalates == \A p \in BreakPieces : (stop[p] = "open") ~> (stop[p] # "open")
=============================================================================
