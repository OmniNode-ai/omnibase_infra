---------------------------- MODULE HourlySend ----------------------------
(***************************************************************************)
(* OMN-23732: the send decisions and the dedupe of the hourly check.        *)
(*                                                                         *)
(* What this models. Each hourly run (a "tick") picks a send key, checks    *)
(* the shared sent record, sends one lane for the key through the runner,  *)
(* and records the key. The sent record is one file (lab-fill.sent.json    *)
(* today) that more than one writer replaces: overlapping ticks (a         *)
(* detached worker outlives its tick) and another writer (the ruling       *)
(* recovery's records, an operator edit). A key that is live in the record *)
(* is skipped; a key older than Window may be sent again (window_hours).   *)
(*                                                                         *)
(* Time is the integer now, advanced by Advance up to MaxNow. The record   *)
(* maps each slot to Nil or [by, at]: a claim stamped with the claim's     *)
(* now, which the send refreshes to the send's now, so the window runs     *)
(* from the send. A tick's token is (its name, its claim now).             *)
(*                                                                         *)
(* The design (MUT = "none"):                                              *)
(*   1. check-and-claim is one atomic step on the fresh record (a lock or  *)
(*      compare-and-set), and it writes before the runner is called       *)
(*   2. a claim names its owner and stamp (a token)                        *)
(*   3. the send is fenced: it happens only inside the claim's own window  *)
(*      and while the record still holds the tick's own token, checked in *)
(*      the same step as the runner call; the same step refreshes the     *)
(*      record's stamp to the send's now                                  *)
(*   4. a refused send rolls back only its own token                       *)
(*   5. every writer merges into the fresh record, never replaces it from *)
(*      a snapshot                                                         *)
(*   6. an unreadable record fails closed: no send, no write              *)
(*                                                                         *)
(* One mutant per design point (MUT):                                      *)
(*   nonatomic_claim    snapshot, decide on the snapshot, replace whole    *)
(*   record_after_send  decide, send, then record: a crash between loses   *)
(*                      the key (one tick is enough)                       *)
(*   as_built           the code today: snapshot at the start, decide on   *)
(*                      it, send with no fence, write the whole snapshot   *)
(*                      plus the sent keys at the end                      *)
(*   unfenced_send      the send checks neither the token nor the window   *)
(*   unguarded_rollback a refused send clears whatever the record holds   *)
(*   w2_replace         the other writer replaces the record from a       *)
(*                      snapshot                                           *)
(*   unreadable_empty   an unreadable record is read as empty             *)
(*                                                                         *)
(* Left out: what the runner does with a lane after the call (the lane's  *)
(* own dedupe), the ledger rows, slot capacity, the choice among keys (a  *)
(* tick may pick any key), clock skew between hosts (one now), and         *)
(* liveness (a claimed key whose send never happens stays blocked until    *)
(* its window ends: a cost the design accepts, not a property checked).    *)
(* The runner call and the token check are one step, as the build must     *)
(* make them (or the runner must be idempotent on the claim token).        *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Ticks, Keys, Nil, Window, MaxNow, MaxSends, MUT

Ow == "ow"          \* the key only the other writer touches
W2 == "w2"          \* the other writer
Slots == Keys \cup {Ow}

VARIABLES now, rec, readable, faulted, pc, cur, tok, snap, sends, w2pc, w2snap

vars == <<now, rec, readable, faulted, pc, cur, tok, snap, sends, w2pc, w2snap>>

EmptyRec == [s \in Slots |-> Nil]

Entry == [by : Ticks \cup {W2}, at : 0..MaxNow]

\* What a reader sees: the record, or (a mutant) an empty one when the file is unreadable.
FailClosed == MUT /= "unreadable_empty"
CanRead == readable \/ ~FailClosed
View == IF readable THEN rec ELSE EmptyRec

Live(e) == e /= Nil /\ now - e.at < Window
Mine(i, e) == e /= Nil /\ e.by = i /\ e.at = tok[i]

Init ==
    /\ now = 0
    /\ rec = EmptyRec
    /\ readable = TRUE
    /\ faulted = FALSE
    /\ pc = [i \in Ticks |-> "idle"]
    /\ cur = [i \in Ticks |-> Nil]
    /\ tok = [i \in Ticks |-> 0]
    /\ snap = [i \in Ticks |-> EmptyRec]
    /\ sends = <<>>
    /\ w2pc = "idle"
    /\ w2snap = EmptyRec

TypeOK ==
    /\ now \in 0..MaxNow
    /\ readable \in BOOLEAN
    /\ \A s \in Slots : rec[s] = Nil \/ rec[s] \in Entry
    /\ pc \in [Ticks -> {"idle", "snapped", "decided", "claimed", "sent"}]
    /\ w2pc \in {"idle", "snapped", "done"}
    /\ \A n \in 1..Len(sends) : sends[n].key \in Keys

\* A write replaces the file, so it leaves it readable.
Write(r) == /\ rec' = r /\ readable' = TRUE

Advance ==
    /\ now < MaxNow
    /\ now' = now + 1
    /\ UNCHANGED <<rec, readable, faulted, pc, cur, tok, snap, sends, w2pc, w2snap>>

Fault ==
    /\ ~faulted /\ readable
    /\ readable' = FALSE /\ faulted' = TRUE
    /\ UNCHANGED <<now, rec, pc, cur, tok, snap, sends, w2pc, w2snap>>

Repair ==
    /\ ~readable
    /\ readable' = TRUE
    /\ UNCHANGED <<now, rec, faulted, pc, cur, tok, snap, sends, w2pc, w2snap>>

\* ---- the tick ------------------------------------------------------------

\* The design: check the fresh record and write the claim in one step.
Claim(i, k) ==
    /\ MUT \notin {"nonatomic_claim", "record_after_send", "as_built"}
    /\ pc[i] = "idle" /\ CanRead
    /\ ~Live(View[k])
    /\ Write([View EXCEPT ![k] = [by |-> i, at |-> now]])
    /\ pc' = [pc EXCEPT ![i] = "claimed"]
    /\ cur' = [cur EXCEPT ![i] = k]
    /\ tok' = [tok EXCEPT ![i] = now]
    /\ UNCHANGED <<now, faulted, snap, sends, w2pc, w2snap>>

\* nonatomic_claim: read the record, later decide on that copy and replace the file with it.
Snap(i) ==
    /\ MUT \in {"nonatomic_claim", "as_built"}
    /\ pc[i] = "idle" /\ CanRead
    /\ snap' = [snap EXCEPT ![i] = View]
    /\ pc' = [pc EXCEPT ![i] = "snapped"]
    /\ UNCHANGED <<now, rec, readable, faulted, cur, tok, sends, w2pc, w2snap>>

ClaimSnap(i, k) ==
    /\ MUT = "nonatomic_claim"
    /\ pc[i] = "snapped"
    /\ ~Live(snap[i][k])
    /\ Write([snap[i] EXCEPT ![k] = [by |-> i, at |-> now]])
    /\ pc' = [pc EXCEPT ![i] = "claimed"]
    /\ cur' = [cur EXCEPT ![i] = k]
    /\ tok' = [tok EXCEPT ![i] = now]
    /\ UNCHANGED <<now, faulted, snap, sends, w2pc, w2snap>>

\* as_built: decide on the snapshot, write nothing yet.
DecideSnap(i, k) ==
    /\ MUT = "as_built"
    /\ pc[i] = "snapped"
    /\ ~Live(snap[i][k])
    /\ pc' = [pc EXCEPT ![i] = "decided"]
    /\ cur' = [cur EXCEPT ![i] = k]
    /\ tok' = [tok EXCEPT ![i] = now]
    /\ UNCHANGED <<now, rec, readable, faulted, snap, sends, w2pc, w2snap>>

\* record_after_send: decide on the record now, send, and write the key afterwards.
Decide(i, k) ==
    /\ MUT = "record_after_send"
    /\ pc[i] = "idle" /\ CanRead
    /\ ~Live(View[k])
    /\ pc' = [pc EXCEPT ![i] = "decided"]
    /\ cur' = [cur EXCEPT ![i] = k]
    /\ tok' = [tok EXCEPT ![i] = now]
    /\ UNCHANGED <<now, rec, readable, faulted, snap, sends, w2pc, w2snap>>

\* The fence: the tick's own window is not over and, once it claimed, its own token is still
\* in the record (a later claim by another tick replaced it).
Fenced(i) ==
    IF MUT \in {"unfenced_send", "as_built"}
    THEN TRUE
    ELSE /\ now - tok[i] < Window
         /\ pc[i] = "claimed" => (readable /\ Mine(i, rec[cur[i]]))

LateRecord == MUT \in {"record_after_send", "as_built"}

\* The runner accepts: one lane is sent for the key, stamped with the send's now. The design
\* refreshes the record's stamp in the same step, so the window runs from the send.
Send(i) ==
    /\ pc[i] \in {"claimed", "decided"}
    /\ Fenced(i)
    /\ sends' = Append(sends, [key |-> cur[i], st |-> now, by |-> i])
    /\ IF pc[i] = "claimed" /\ MUT /= "unfenced_send"
       THEN Write([rec EXCEPT ![cur[i]] = [by |-> i, at |-> now]])
       ELSE UNCHANGED <<rec, readable>>
    /\ pc' = [pc EXCEPT ![i] = IF LateRecord THEN "sent" ELSE "idle"]
    /\ UNCHANGED <<now, faulted, cur, tok, snap, w2pc, w2snap>>

\* The key reaches the record only now. record_after_send merges into the fresh record;
\* as_built replaces the file with its snapshot plus the key.
Record(i) ==
    /\ LateRecord
    /\ pc[i] = "sent" /\ CanRead
    /\ Write(IF MUT = "as_built"
             THEN [snap[i] EXCEPT ![cur[i]] = [by |-> i, at |-> tok[i]]]
             ELSE [View EXCEPT ![cur[i]] = [by |-> i, at |-> tok[i]]])
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED <<now, faulted, cur, tok, snap, sends, w2pc, w2snap>>

\* The runner refuses (no lane is sent): the claim is rolled back.
Refuse(i) ==
    /\ pc[i] \in {"claimed", "decided"}
    /\ IF pc[i] = "claimed"
       THEN /\ CanRead
            /\ IF MUT = "unguarded_rollback"
               THEN Write([View EXCEPT ![cur[i]] = Nil])
               ELSE IF Mine(i, View[cur[i]])
                    THEN Write([View EXCEPT ![cur[i]] = Nil])
                    ELSE UNCHANGED <<rec, readable>>
       ELSE UNCHANGED <<rec, readable>>
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED <<now, faulted, cur, tok, snap, sends, w2pc, w2snap>>

\* The tick dies at any point: its local state is gone, the record stays as written.
Crash(i) ==
    /\ pc[i] /= "idle"
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED <<now, rec, readable, faulted, cur, tok, snap, sends, w2pc, w2snap>>

\* ---- the other writer ----------------------------------------------------

W2Merge ==
    /\ MUT /= "w2_replace"
    /\ w2pc = "idle" /\ CanRead
    /\ Write([View EXCEPT ![Ow] = [by |-> W2, at |-> now]])
    /\ w2pc' = "done"
    /\ UNCHANGED <<now, faulted, pc, cur, tok, snap, sends, w2snap>>

W2Snap ==
    /\ MUT = "w2_replace"
    /\ w2pc = "idle" /\ CanRead
    /\ w2snap' = View
    /\ w2pc' = "snapped"
    /\ UNCHANGED <<now, rec, readable, faulted, pc, cur, tok, snap, sends>>

W2Write ==
    /\ MUT = "w2_replace"
    /\ w2pc = "snapped"
    /\ Write([w2snap EXCEPT ![Ow] = [by |-> W2, at |-> now]])
    /\ w2pc' = "done"
    /\ UNCHANGED <<now, faulted, pc, cur, tok, snap, sends, w2snap>>

Next ==
    \/ Advance \/ Fault \/ Repair
    \/ \E i \in Ticks : Snap(i) \/ Send(i) \/ Record(i) \/ Refuse(i) \/ Crash(i)
    \/ \E i \in Ticks, k \in Keys : Claim(i, k) \/ ClaimSnap(i, k) \/ Decide(i, k) \/ DecideSnap(i, k)
    \/ W2Merge \/ W2Snap \/ W2Write

Spec == Init /\ [][Next]_vars

SendBound == Len(sends) <= MaxSends

\* ---- properties ----------------------------------------------------------

\* No key is sent twice inside its dedupe window: two lanes sent for one key are at least
\* Window apart, by the clock of the sends themselves.
NoKeyDoubleSend ==
    \A a, b \in 1..Len(sends) :
        (a < b /\ sends[a].key = sends[b].key)
            => (sends[b].st - sends[a].st >= Window \/ sends[a].st - sends[b].st >= Window)

\* The literal form: no key is sent twice. It holds only when the window outlasts the
\* horizon (Window > MaxNow), the config once.cfg.
NoKeySentTwice ==
    \A a, b \in 1..Len(sends) : a < b => sends[a].key /= sends[b].key

\* Witnesses: both are violated, so a send and a send again after the window are reachable.
NothingSent == Len(sends) = 0
NoResend == NoKeySentTwice

=============================================================================
