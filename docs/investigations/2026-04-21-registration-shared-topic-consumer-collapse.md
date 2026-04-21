# Ticket Draft: Registration Shared-Topic Consumer Collapse

- Date: 2026-04-21
- Related ticket: `OMN-9420`
- Runtime target: `192.168.86.201`
- Scope: `omnibase_infra` runtime / Kafka event bus subscription semantics

## Summary

`OMN-9420` started as a registration verifier false negative, but live triage on `.201`
found a second issue with the runtime itself.

The registration orchestrator contract declares 7 `event_bus.subscribe_topics`, and the
live contract in the runtime container matches that declaration. The live runtime also
loads the registration plugin code that calls `EventBusSubcontractWiring.wire_subscriptions()`
with all 7 topics.

However, Kafka only shows registration-specific consumer groups for 5 of the 7 topics.
The two missing topics are:

- `onex.evt.platform.node-introspection.v1`
- `onex.evt.platform.node-heartbeat.v1`

The reason is in `EventBusKafka.subscribe()`: subscriptions are registered with distinct
consumer identities, but consumer startup is keyed only by `topic`, not by
`(topic, consumer_group)`. If another subscriber in the same runtime process subscribes
to the topic first, later subscribers are appended to the in-process fanout list and no
new Kafka consumer group is created for them.

That means the registration orchestrator can be logically subscribed in-process while its
own canonical Kafka consumer group does not exist for that topic.

## Live Evidence

### Contract and plugin path

Inside `omninode-runtime-effects` on `.201`:

- `/app/src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml`
  declares all 7 subscribe topics, including:
  - `onex.evt.platform.node-introspection.v1`
  - `onex.evt.platform.node-heartbeat.v1`
- `/app/src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py`
  loads that contract and calls:
  - `self._wiring.wire_subscriptions(subcontract=subcontract, node_name="registration-orchestrator")`

### Registration consumer groups observed in Kafka

Observed on `.201` via Redpanda `rpk group list`:

- Present under `local.runtime_config.registration-orchestrator.consume.1.0.0.__t.*`
  - `onex.cmd.platform.node-registration-acked.v1`
  - `onex.cmd.platform.request-introspection.v1`
  - `onex.cmd.platform.topic-catalog-query.v1`
  - `onex.evt.platform.registry-request-introspection.v1`
  - `onex.intent.platform.runtime-tick.v1`
- Not present under that registration prefix
  - `onex.evt.platform.node-introspection.v1`
  - `onex.evt.platform.node-heartbeat.v1`

### Other consumers already exist for the missing topics

Observed on `.201`:

- `local.omnibase_infra.node_ledger_projection_compute.consume.1.0.0.__t.onex.evt.platform.node-introspection.v1`
- `local.omnibase_infra.node_ledger_projection_compute.consume.1.0.0.__t.onex.evt.platform.node-heartbeat.v1`
- `local.omnimarket.projection_registration.consume.1.0.0.__t.onex.evt.platform.node-introspection.v1`
- `local.omnimarket.projection_registration.consume.1.0.0.__t.onex.evt.platform.node-heartbeat.v1`

This matches the code path below: the first subscriber for a topic starts the Kafka
consumer; later subscribers on the same topic reuse the in-process fanout and never get
their own Kafka group.

## Root Cause

`src/omnibase_infra/event_bus/event_bus_kafka.py`

Current behavior:

1. `subscribe()` derives `effective_group_id` from the caller identity.
2. It appends `(group_id, subscription_id, callback)` to `self._subscribers[topic]`.
3. It only starts a consumer when `topic not in self._consumers`.
4. `_consume_loop()` fans every message on that topic to all callbacks in
   `self._subscribers[topic]`, regardless of which subscriber identity requested it.

Relevant shape:

```python
self._subscribers[topic].append((effective_group_id, subscription_id, on_message))

if topic not in self._consumers and self._started:
    await self._start_consumer_for_topic(topic, effective_group_id)
```

And later:

```python
subscribers = list(self._subscribers.get(topic, []))
for group_id, subscription_id, callback in subscribers:
    await callback(event_message)
```

So the runtime currently has these semantics:

- one Kafka consumer per topic
- many in-process callbacks per topic
- the first subscriber's consumer group becomes the Kafka-visible owner
- later subscribers lose consumer-group attribution for that topic

## Why This Matters

This breaks the invariant implied by the subscription API and consumer group naming:

- callers provide distinct identities
- verification expects a node's declared subscriptions to be reflected in that node's
  canonical consumer groups
- Kafka admin output cannot reliably answer "is node X subscribed to topic Y?" when
  another in-process subscriber reached the topic first

This is especially visible for shared platform topics like:

- `node-introspection`
- `node-heartbeat`

## Recommended Ticket Shape

Title:

`fix(event_bus): preserve per-subscriber consumer-group identity for shared Kafka topics`

Problem statement:

- `EventBusKafka.subscribe()` collapses multiple subscribers on the same topic behind a
  single Kafka consumer keyed only by topic.
- Distinct subscribers with different canonical consumer groups do not get distinct
  Kafka consumer ownership.
- Verification and operational tooling lose truthful attribution for shared topics.

Expected behavior:

- Different `(topic, consumer_group)` pairs should have distinct Kafka consumers.
- Multiple callbacks sharing the same `(topic, consumer_group)` may still fan out
  in-process.
- Runtime readiness should remain topic-based, but internal bookkeeping must not erase
  per-group ownership.

## Implementation Direction

Preferred direction:

1. Change consumer/task bookkeeping from `topic` to `(topic, consumer_group)`.
2. Scope subscriber registries by `(topic, consumer_group)` so consume loops only fan out
   to callbacks registered for that exact consumer group.
3. Keep topic-level readiness aggregation by unioning per-group consumer state back to
   the topic level.
4. Add regression tests proving:
   - same topic + different groups => distinct Kafka consumers
   - same topic + same group => one Kafka consumer, multiple callbacks
   - unsubscribe of one group does not tear down another group's consumer
   - readiness still reports correctly for required topics with per-group consumers

Fallback direction if runtime change is deferred:

- downgrade verifier semantics for shared-topic attribution gaps to `QUARANTINE`
  with explicit evidence that another consumer group already owns the topic, but this
  should be treated as a verifier mitigation, not the full runtime fix.

## Acceptance Criteria

- Registration orchestrator on live runtime shows canonical consumer groups for all 7
  declared subscribe topics, or the runtime explicitly exposes a truthful replacement
  ownership model that verification can consume.
- A regression test reproduces the current collapse bug and passes with the fix.
- `OMN-9420` live verification on `.201` no longer fails due to missing
  `node-introspection` / `node-heartbeat` ownership.

## Notes

- This finding should not be lost: the verifier grounding fix in `OMN-9420` was still
  necessary. It removed one false negative. This investigation uncovered a second,
  runtime-level attribution bug that only became visible after grounding was corrected.
