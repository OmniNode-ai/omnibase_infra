# Headless Pipeline Stage Templates

> **Ticket**: OMN-6984 | **Parent**: OMN-6936
>
> Per-stage command templates for the headless close-out pipeline.
> Each template defines the exact `claude -p` invocation for one pipeline stage,
> including tool allowlists, prompts, and expected output schema.

## Usage

The headless close-out script (`scripts/headless-close-out.sh`) sources these templates
via the `get_stage_template()` function. Each template returns a structured object with:

- `prompt` -- the exact prompt text for `claude -p`
- `allowed_tools` -- minimal tool set for this stage
- `timeout_seconds` -- per-invocation timeout
- `output_schema` -- expected JSON output keys

## Template Catalog

### Stage: merge-sweep

**Purpose**: Merge all open PRs with passing CI in a single repo.

```bash
STAGE="merge-sweep"
ALLOWED_TOOLS="Bash(git:*,gh:*) Read Glob Grep"
TIMEOUT=900

PROMPT="Merge-sweep for \${REPO}: \
List all open PRs in OmniNode-ai/\${REPO} with passing CI checks. \
For each green PR, merge it using 'gh pr merge --squash --auto'. \
Skip PRs with: failing CI, unresolved review comments, draft status, or merge conflicts. \
Working directory: \${REPO_PATH}

Report JSON:
{
  \"schema_version\": \"1.0\",
  \"stage\": \"merge-sweep\",
  \"repo\": \"\${REPO}\",
  \"status\": \"success|partial|failed\",
  \"prs_merged\": [{\"number\": N, \"title\": \"...\"}],
  \"prs_skipped\": [{\"number\": N, \"reason\": \"...\"}],
  \"prs_failed\": [{\"number\": N, \"error\": \"...\"}]
}"
```

**Output schema**:
| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | `success` (all merged), `partial` (some skipped), `failed` (errors) |
| `prs_merged` | array | PRs successfully merged |
| `prs_skipped` | array | PRs skipped with reason |
| `prs_failed` | array | PRs that errored during merge |

---

### Stage: release

**Purpose**: Bump version, tag, and push for a repo with unreleased commits.

```bash
STAGE="release"
ALLOWED_TOOLS="Bash(git:*,gh:*,uv:*) Read Edit Write Glob Grep"
TIMEOUT=900

PROMPT="Release \${REPO}: \
Check if there are unreleased commits since the last git tag. \
If yes: bump the version in pyproject.toml, update CHANGELOG, create a git tag, and push. \
Follow the existing release conventions in the repo. \
Read merge-sweep results for context: \${MERGE_SWEEP_RESULT} \
Working directory: \${REPO_PATH}

Report JSON:
{
  \"schema_version\": \"1.0\",
  \"stage\": \"release\",
  \"repo\": \"\${REPO}\",
  \"status\": \"success|skipped|failed\",
  \"version\": \"X.Y.Z\",
  \"tag\": \"vX.Y.Z\",
  \"commits_since_last_tag\": N
}"
```

**Output schema**:
| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | `success` (released), `skipped` (no unreleased commits), `failed` |
| `version` | string | New version number (empty if skipped) |
| `tag` | string | Git tag created (empty if skipped) |
| `commits_since_last_tag` | int | Number of commits included in release |

---

### Stage: redeploy

**Purpose**: Rebuild Docker runtime with new package versions.

```bash
STAGE="redeploy"
ALLOWED_TOOLS="Bash(git:*,docker:*,uv:*) Read Edit Write Glob Grep"
TIMEOUT=1200

PROMPT="Redeploy runtime for \${REPO}: \
Sync bare clones (pull-all.sh), rebuild Docker runtime images, \
seed Infisical if contracts changed, and verify health endpoints. \
Read release results for version context: \${RELEASE_RESULT} \
Working directory: \${REPO_PATH}

Report JSON:
{
  \"schema_version\": \"1.0\",
  \"stage\": \"redeploy\",
  \"repo\": \"\${REPO}\",
  \"status\": \"success|skipped|failed\",
  \"images_rebuilt\": [\"image:tag\"],
  \"health_checks\": {\"service\": \"healthy|unhealthy\"}
}"
```

**Output schema**:
| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | `success`, `skipped` (no new releases), `failed` |
| `images_rebuilt` | array | Docker images rebuilt |
| `health_checks` | object | Service health check results |

---

### Stage: ticket-close

**Purpose**: Close completed Linear tickets and update status.

```bash
STAGE="ticket-close"
ALLOWED_TOOLS="Bash(gh:*) Read Grep mcp__linear-server__*"
TIMEOUT=600

PROMPT="Ticket close for \${REPO}: \
Find all Linear tickets referencing merged PRs in OmniNode-ai/\${REPO}. \
For each ticket with all associated PRs merged: update status to Done, \
add a completion comment with PR links. \
Working directory: \${REPO_PATH}

Report JSON:
{
  \"schema_version\": \"1.0\",
  \"stage\": \"ticket-close\",
  \"repo\": \"\${REPO}\",
  \"status\": \"success|partial|failed\",
  \"tickets_closed\": [{\"id\": \"OMN-XXXX\", \"pr\": \"#N\"}],
  \"tickets_skipped\": [{\"id\": \"OMN-XXXX\", \"reason\": \"...\"}]
}"
```

---

### Stage: integration-sweep

**Purpose**: Verify integration health after merges and releases.

```bash
STAGE="integration-sweep"
ALLOWED_TOOLS="Bash(git:*,uv:*,docker:*) Read Glob Grep"
TIMEOUT=900

PROMPT="Integration sweep for \${REPO}: \
Run contract compliance checks and integration tests for recently merged work. \
Verify: import health, contract parity, topic alignment, CI green. \
Working directory: \${REPO_PATH}

Report JSON:
{
  \"schema_version\": \"1.0\",
  \"stage\": \"integration-sweep\",
  \"repo\": \"\${REPO}\",
  \"status\": \"pass|warn|fail\",
  \"checks\": [
    {\"name\": \"...\", \"status\": \"pass|warn|fail\", \"detail\": \"...\"}
  ]
}"
```

---

## Adding New Templates

1. Define the stage name, tool allowlist, timeout, and prompt
2. Include explicit JSON output schema in the prompt
3. Add the template to the `STAGE_TEMPLATES` associative array in `headless-close-out.sh`
4. Document the template in this file

## Template Variables

All templates support these shell variables (expanded at runtime):

| Variable | Description |
|----------|-------------|
| `${REPO}` | Repository name (e.g., `omnibase_core`) |
| `${REPO_PATH}` | Full path to repo (e.g., `/Volumes/PRO-G40/Code/omni_home/omnibase_core`) |
| `${RUN_ID}` | Unique run identifier |
| `${STATE_DIR}` | State directory for this run |
| `${MERGE_SWEEP_RESULT}` | JSON output from merge-sweep stage (stages 2+) |
| `${RELEASE_RESULT}` | JSON output from release stage (stages 3+) |
