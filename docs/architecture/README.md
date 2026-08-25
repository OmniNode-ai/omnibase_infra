# Architecture Documentation

Most of this directory's content has moved to the OmniNode knowledge base — the
canonical public home for platform architecture docs. Follow a doc's own stub
file for its exact new location, or browse the
[knowledge base's architecture section](https://github.com/OmniNode-ai/knowledge-base/tree/main/architecture)
directly.

Full documentation → https://github.com/OmniNode-ai/knowledge-base

## Still in this repo

Three docs are deliberately **not yet migrated** — each names a real class,
script, or config identifier that spells out the org's secrets-manager
vendor name, which the knowledge base's sanitization guard forbids, and a
mechanical rename would misrepresent the actual API. They need a deliberate
rewrite (or classification into the knowledge base's private/restricted
tier) before they can move:

| Document | Why it stays |
|----------|--------------|
| [Config Discovery](CONFIG_DISCOVERY.md) | Entire document is the config-fetch-from-secrets-backend mechanism; real kwargs/class/env-var names throughout |
| [Handler Protocol-Driven Architecture](HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) | Reference table of real handler class/module names, including the secrets handler |
| [Circuit Breaker Thread Safety](CIRCUIT_BREAKER_THREAD_SAFETY.md) | Worked example is the real secrets-handler class/module/enum/test names |

## Related documentation

- [Pattern Documentation](../patterns/README.md) — implementation patterns (not yet migrated)
- [Operations Runbooks](../operations/README.md) — production operations
- [ADRs](../decisions/README.md) — why things work this way
