# Security Policy

## Reporting a Vulnerability

Do not report security vulnerabilities via public GitHub issues.

Email **contact@omninode.ai** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 5 business days.

## Sensitive Material

`omnibase_infra` is the org's Docker/deployment repo — it carries real secret-shaped
configuration far more than most repos in this org. Do not commit:

- `.env` files, secrets, API keys, or credentials (see `.gitignore` / `docker/.env.example`)
- Infisical machine identity secrets, encryption keys, or auth secrets
- private hostnames, internal IPs, or operator machine paths
- production data exports or credential-bearing logs

## Required Local Checks

Run the relevant validators before changing security-sensitive code, docs, or configuration:

```bash
uv run python scripts/validate.py all
pre-commit run --all-files
```

See `CLAUDE.md` for the full validator and pre-commit hook map, including the
`no-consul-references` and secret-pattern guards.
