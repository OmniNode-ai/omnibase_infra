------------------------ MODULE RebuildAdmissionExt -------------------------
(***************************************************************************)
(* Properties added after the long runs started, kept apart so the module  *)
(* they ran on stays byte identical (reader m0-reader-0215's condition).   *)
(* Nothing here changes the protocol. The *Break operators replace a       *)
(* definition of the module from a cfg, with <-, to show a property can    *)
(* fail: the module's Mut names cannot reach these without editing it.     *)
(***************************************************************************)
EXTENDS RebuildAdmission

\* DecideFollowsOrder is one way: a table refusing too much passes it. These are the other way.
Pub(k, q) == [k |-> k, p |-> q, s |-> "-", r |-> "pub"]
\* 7.1: continuation classes and B0 pass a stop and a stale table; only a safety failure or no credit refuses.
ContinuationPasses == (safety = {} /\ credit > 0) => Decide(Pub("cont", 0)) = "admit"
B0Passes == (safety = {} /\ credit > 0) => Decide(Pub("b0", 0)) = "admit"
\* 7.3: B0 and the cutover draw credit, and it refuses with none.
B0DrawsCredit == credit = 0 => Decide(Pub("b0", 0)) = "refuse"
\* Step 2 refuses kept fixes above piece 7 only. Step 5 holds a kept entry of another piece until it
\* is the lowest unfinished, so a stale table must not refuse one at the active piece at or below 7.
KeptAtActiveOnStale ==
  (safety = {} /\ ~fresh /\ Stops = {} /\ credit > 0 /\ cur <= Seven) => Decide(Pub("kept", cur)) = "admit"
KeptAboveSevenRefusedOnStale == \A q \in Pieces :
  (safety = {} /\ ~fresh /\ q > Seven) => Decide(Pub("kept", q)) = "refuse"

\* Counted until read as gone (section 12 P2, 7.1 opus2-M3): a dead process keeps what it held.
PlacesHeldUntilGone == \A p \in Pieces :
  Cardinality({t \in Tasks : Wip(K(t)) /\ K(t).p = p /\ pc[t] \in {"run", "dead"}}) <= Limit
SlotsHeldUntilGone ==
  /\ \A p \in Pieces : Cardinality({t \in Tasks : K(t).k = "cause" /\ K(t).p = p /\ pc[t] \in {"resv", "run", "dead"}}) <= 1
  /\ Cardinality({t \in Tasks : K(t).k = "repair" /\ pc[t] \in {"resv", "run", "dead"}}) <= 1

\* ---- the breaks, each replacing one definition from its cfg ----
\* The protocol's Decide, repeated: an override may not call what it replaces.
Protocol(k) ==
  IF k.k = "ro" THEN "admit" ELSE First(Step0b(k), Step1(k), Step2(k), Step3(k), StepOrder(k))
DecideContRefused(k) ==
  IF k.k = "cont" /\ Stops # {} THEN "refuse" ELSE Protocol(k)
DecideB0Exempt(k) ==
  IF k.k = "b0" /\ safety = {} THEN "admit" ELSE Protocol(k)
DecideKeptAllStale(k) ==
  IF k.k = "kept" /\ ~fresh THEN "refuse" ELSE Protocol(k)
DecideKeptStaleAbove(k) ==
  IF k.k = "kept" /\ safety = {} /\ credit > 0 /\ (k.p \in KeptPieces) /\ Stops = {} THEN "admit" ELSE Protocol(k)
HoldingFreesAtExit == {"resv", "run"}
SlotFreesAtExit(k) ==
  \A t \in Tasks : (K(t).k = k.k /\ K(t).p = k.p /\ K(t).s = k.s) => pc[t] \in {"idle", "dead"}
RepairFreeFreesAtExit == \A t \in Tasks : K(t).k = "repair" => pc[t] \in {"idle", "dead"}
=============================================================================
