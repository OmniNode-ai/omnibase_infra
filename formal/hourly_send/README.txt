HourlySend: the send decisions and the dedupe of the hourly check (OMN-23732)
===========================================================================

Files
-----
HourlySend.tla   the model; its header says what it models and what it leaves out
Model.cfg        the design: NoKeyDoubleSend (no key is sent twice inside its dedupe window)
once.cfg         the literal form, NoKeySentTwice, with a window that outlasts the horizon
mut_*.cfg        one mutant per design point; each flips MUT and must violate NoKeyDoubleSend
                 (mut_as_built is the code today)
wit_*.cfg        witnesses: a send, and a send of one key again after its window, are reachable
results/         one TLC output per config (the last line is exit=<code>), model.sha256 (the
                 sha256 of the spec then the cfg files in sorted order) and
                 design_review_verdict.txt

Run
---
TLC 2.19 (tla2tools.jar, sha256 936a2620...50e88) on lab host h201 in the
eclipse-temurin:17-jre-alpine image, one process per config:

  java -XX:+UseParallelGC -Xmx2g -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 2 \
       -metadir <scratch>/states-<cfg> HourlySend.tla

TLC exits 0 when it finds no error and 12 on an invariant violation. tests/formal/
test_hourly_send_model.py reads results/ (or runs TLC where TLA2TOOLS_JAR and java exist) and
binds them to the digest; `python tests/formal/test_hourly_send_model.py --regenerate` rewrites
results/ on a host with Java.

The model
---------
Ticks (two overlapping runs) pick a key, check the shared sent record, send one lane through the
runner and record the key. One other writer touches the same file. A key live in the record is
skipped; a key older than Window may be sent again. The record may become unreadable and be
repaired. A tick may die at any step. The design is the proposed orchestrator's: an atomic
check-and-claim that writes before the runner call, a token on each claim, a send fenced on the
token and the claim's own window that refreshes the stamp, a rollback of only its own token,
merge-writes for every writer, and a fail-closed read.

Mapping to the running throughput tick (the code today)
-------------------------------------------------------
Read from node_idle_slot_fill_effect (omnimarket) and the tick's ledger rows; the sent record
itself, lab-fill.sent.json in ~/.local/state/omni/throughput-tick on the orchestrator host, is
not on this lab host and is not read here.

Model                          Tick today
-----------------------------  ------------------------------------------------------------
a send key k in Keys           item["key"]: repair:<loop>, dev-red:<repo>, <repo>#<n>,
                               repo-status:<repo>, ticket:<OMN-n> (cloud backlog),
                               exp:<name>, ruling-relaunch:<ticket>:<terminal at>, and a
                               board item's <kind>:<key>
rec[k] = [by, at]              sent[key] = {lane, at, kind, model} in lab-fill.sent.json;
                               by is lane, at is request.now (the tick's decision clock)
Window                         request.window_hours (3.0 by default; the overlay's 24)
Live(e): now - e.at < Window   now - at < window in handle(): the skip "sent as <lane> at <at>"
Send(i) (the runner accepts)   _send(fill_argv(...)) returns ok; the tick then calls
                               request.record("lab-fill", lane, "<kind> <key>, <model>"), the
                               STATUS row "throughput tick lab-fill: <lane>: <kind> <key>, <model>"
                               (st is that row's time; tok is request.now, in the lane's name)
Refuse(i)                      "UNKNOWN lab-fill-dispatch:<lane>": the runner refused, the key
                               is not recorded and the next tick sends it again
the whole-record replace       write_sent(): one replace of the snapshot plus the sent keys,
(nonatomic_claim, as_built)    once at the end of the run, only when something was sent
Fault with unreadable_empty    read_sent(): an unreadable record is read as empty
Crash(i)                       a tick killed between _send and write_sent loses the key
the other writer W2            the ruling recovery and any other writer of the state directory

One live tick, read from the ledger: the throughput tick's STATUS row of 2026-11-14T05:24:00Z
reads "throughput tick lab-fill: fill-ticket-OMN-23732-20261114T0509Z: ticket ticket:OMN-23732,
sonnet". The key is ticket:OMN-23732 (cloud_backlog.py: "ticket:" + the ticket id), the lane
name carries the decision clock 20261114T0509Z, and the row is stamped 15 minutes later: the
decision clock and the send clock differ, which is why the design refreshes the stamp at the
send.
