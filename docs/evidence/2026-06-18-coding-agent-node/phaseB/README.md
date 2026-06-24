# Phase B — Ship claude+codex in the effects image + DEV cred mount

**Date:** 2026-06-18
**Lane:** DEV (project `omnibase-infra`) on the runtime host — pre-authorized; stability/prod/judge untouched.
**Phase 0 dependency:** omni_home PR #182 (MERGED) — proved headless codex/claude auth in a Linux container on the runtime host.

## Verdict: PASS

| Surface | Result | Proof file |
|---------|--------|-----------|
| Node + codex + claude baked into effects image | YES (node v20.20.2, codex 0.141.0, claude 2.1.181) | `which.txt` |
| Read-only cred bind-mount attached on DEV effects | YES (codex `auth.json` file-`:ro`, claude dir-`:ro`) | `cred-mounts.txt` |
| codex read-only agent call authenticates via mounted creds | PASS (`agent_message "hello"`, exit 0) | `codex-readonly.txt` |
| claude read-only agent call authenticates via mounted creds | PASS (`is_error:false`, `result:"...hello"`, `permission_denials:[]`, exit 0) | `claude-readonly.txt` |
| Canary: effects healthy, RestartCount ≤ 5 over ~2 min | PASS (running, restarts=0, healthy throughout) | `canary-watch.txt` |

## Deployed image

```
sha256:9ac86cebe5c458f4991166353f2d4ac1c4d5085f4578ba41286f366cd4584229
rev=6d6acf802178  ver=0.38.3  src=workspace
```

Built workspace-mode from the canonical runtime host `/data/omninode/omni_home` clones (intentional dev forward-rebuild: omnibase-infra clone HEAD `6d6acf80` and onex-change-control track ahead of the omnimarket lock pin; recorded via `ALLOW_SIBLING_PIN_DRIFT=1`, never silent). Non-blank identity satisfies the workspace guard.

Rollback baseline (the prior healthy image) was `sha256:2a3ea84f52df5fb04aeb116d6e521fa0a60928638d07551073088a900322cada` (rev `6d6acf80`, ver `0.38.3`). Canary passed, so no rollback was performed.

## Credential source (DEV only — NOT baked, NOT committed)

Both creds are operator-supplied host paths bound read-only:
- codex: host `<home>/.onex-agent-creds/codex-auth.json` → `/home/omniinfra/.codex/auth.json:ro`
- claude: host `<home>/.onex-agent-creds/.claude` → `/home/omniinfra/.claude:ro`

Source: copied via `docker cp` out of the still-running Phase 0 container `onex-phase0-codingagent-auth` (`/home/omniinfra/.codex`, `/home/omniinfra/.claude`). Per Phase 0, claude 2.x on macOS keeps its credential in the Keychain, so the Linux container login — not a Mac file — is the source. Both creds carry refresh tokens.

## Two load-bearing decisions made in Phase B (documented, not silent)

1. **codex credential mounts as a FILE (`auth.json`), not the whole `~/.codex` dir.**
   codex 0.141.0 writes app-server/session state into `CODEX_HOME` (`~/.codex`) at
   runtime; a fully read-only `~/.codex` dir mount makes `codex exec` fail with
   "Read-only file system" / "Permission denied" even though the creds are valid
   (verified in-container twice). Binding only `auth.json:ro` keeps the credential
   VALUE read-only while leaving the rest of `~/.codex` writable. The Dockerfile
   pre-creates `/home/omniinfra/.codex` owned by `omniinfra` so the file-mount lands
   inside a writable, runtime-user-owned dir (Docker otherwise auto-creates it
   root-owned and codex can't write). claude's whole `~/.claude` dir mounts `:ro`
   fine (`claude -p` needs no write to that dir).

2. **codex write mode = `-s danger-full-access`, NOT `workspace-write`.**
   Phase 0 proved codex's internal `-s workspace-write` bwrap sandbox cannot create
   a user namespace inside the cap-dropped effects container (`bwrap: No permissions
   to create a new namespace`). The coding-agent effect runs codex with
   `-s danger-full-access` — the container IS the sandbox boundary — and we do NOT
   grant `CAP_SYS_ADMIN` / userns to the effects container (least-privilege). The
   argv itself lands in Phase A; this is the supporting image/runtime decision.

## Files

- `which.txt` — `which claude && which codex` + versions, run as `omniinfra` in the running DEV effects container
- `codex-readonly.txt` — `codex exec -s read-only --json "print the word hello"` transcript + exit
- `claude-readonly.txt` — `claude -p --output-format json --permission-mode plan "print the word hello"` transcript + exit
- `cred-mounts.txt` — `docker inspect` of the read-only cred bind-mounts on the running effects container
- `image-digest.txt` — deployed effects image digest + identity labels
- `canary-watch.txt` — ~2-min RestartCount / health watch after recreate
