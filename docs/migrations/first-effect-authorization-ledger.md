# First-effect observation ledger verification

Migration `104_create_first_effect_authorization_ledger.sql` creates an
explicitly `public`, value-redacted observation/recording scaffold. Its
authorization-named relation and identifiers are schema facts only: it confers
no effect permission, has no publisher, and cannot guarantee no-republish. The
canonical outbox is the payload-owning in-row state-IO outbox from migration
093. The adapter has no `begin_publishing` API. The non-authorizing
`require_transactional_first_effect_outbox()` guard rejects the payload-free
observation adapter where an external implementation must atomically write the
canonical outbox intent and its ledger binding.

The real ephemeral PostgreSQL acceptance is:

```bash
uv run pytest tests/integration/migrations/test_104_first_effect_authorization_ledger.py -m integration -q
```

It starts an isolated Unix-socket cluster with `initdb`/`pg_ctl`, applies the
real forward and rollback files through `psql -f`, proves an incompatible
pre-existing relation fails, attests two isolated backend sessions and races
their synchronized CAS updates, and verifies a lost terminal-response retry
returns the committed row only for identical hashes. It also verifies the
`PUBLISHED_UNKNOWN` terminal and blocked outcomes, conflicting publish and
terminal evidence, and direct illegal trigger transitions. When the local
PostgreSQL binaries are unavailable, pytest reports a skip; this is an explicit
deployment gate, not substituted by the connection-free query-shape tests.

No payload, secret, prompt, or model-output value is valid test data for this
ledger. The tests use only synthetic lowercase SHA-256-shaped values and a
generated UUID.
