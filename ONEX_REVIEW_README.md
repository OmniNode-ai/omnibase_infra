# ONEX Opus Nightly Review System

Deterministic, diff-driven Opus 4.1 agents for preventing code drift while migrating away from archived code.

## Overview

The ONEX Review System provides two modes of operation:
1. **Baseline Review**: One-time comprehensive review of current `main` branch
2. **Nightly Review**: Incremental daily reviews of changes since last successful run

## Quick Start

```bash
# First time setup - run baseline review
./scripts/onex_review_orchestrator.sh baseline

# Daily runs - incremental review
./scripts/onex_review_orchestrator.sh nightly

# Process findings from previous run
./scripts/onex_review_orchestrator.sh process .onex_nightly/*/review_output/findings.ndjson
```

## Components

### 1. Producer Scripts
- `onex_baseline_producer.sh`: Generates sharded diffs for baseline review
- `onex_nightly_producer.sh`: Generates incremental diffs for daily review

### 2. Agent Runner
- `onex_agent_runner.py`: Invokes Opus agents with formatted prompts
- Handles both baseline (sharded) and nightly (single) reviews
- Parses agent responses into NDJSON findings and Markdown summaries

### 3. Findings Processor
- `onex_findings_processor.py`: Analyzes and reports on review findings
- Generates fix scripts, GitHub issues, and detailed reports

### 4. Orchestrator
- `onex_review_orchestrator.sh`: Main entry point coordinating all components

## Configuration

### Policy File (`config/policy.yaml`)
Defines:
- Repository-specific import restrictions
- Severity levels for different rule categories
- Token and byte limits for processing

### Rules (v0.1)

#### Naming Rules
- `ONEX.NAMING.PROTOCOL_001`: Protocol classes must start with "Protocol"
- `ONEX.NAMING.MODEL_001`: Model classes must start with "Model"
- `ONEX.NAMING.ENUM_001`: Enum classes must start with "Enum"
- `ONEX.NAMING.NODE_001`: Node classes must start with "Node"

#### Boundary Rules
- `ONEX.BOUNDARY.FORBIDDEN_IMPORT_001`: Forbidden cross-repository imports

#### SPI Purity Rules
- `ONEX.SPI.RUNTIMECHECKABLE_001`: Protocols need @runtime_checkable
- `ONEX.SPI.FORBIDDEN_LIB_001`: No filesystem/network access in SPI

#### Typing Hygiene
- `ONEX.TYPE.UNANNOTATED_DEF_001`: Functions need type annotations
- `ONEX.TYPE.ANY_001`: Avoid using Any type
- `ONEX.TYPE.OPTIONAL_ASSERT_001`: Optional immediately asserted non-null

#### Waiver Hygiene
- `ONEX.WAIVER.MALFORMED_001`: Waivers need reason and expiry
- `ONEX.WAIVER.EXPIRED_001`: Expired waivers are errors

## Output Format

### NDJSON Findings
```json
{
  "ruleset_version": "0.1",
  "rule_id": "ONEX.NAMING.PROTOCOL_001",
  "severity": "error",
  "repo": "omnibase-core",
  "path": "src/omnibase_core/protocols/protocol_event.py",
  "line": 12,
  "message": "Protocol class does not start with 'Protocol'",
  "evidence": {"class_name": "EventHandler"},
  "suggested_fix": "Rename to ProtocolEventHandler",
  "fingerprint": "a1b2c3d4"
}
```

### Markdown Summary
- Executive summary with risk score (0-100)
- Top violations
- Waiver issues
- Next actions
- Coverage notes

## Waiver Format

Add waivers inline in code:
```python
# onex:ignore ONEX.NAMING.PROTOCOL_001 reason=Temporary rename in flight expires=2025-10-15
class EventHandler(Protocol):  # Will be fixed
    pass
```

## Directory Structure

```
.onex_baseline/
  └── {repo_name}/
      └── {timestamp}/
          ├── files.list
          ├── nightly.stats
          ├── nightly.names
          ├── nightly.diff
          ├── metadata.json
          ├── shards/
          │   └── diff_shard_*.diff
          └── review_output/
              ├── findings.ndjson
              ├── summary.md
              └── combined_output.txt

.onex_nightly/
  └── {repo_name}/
      └── {timestamp}/
          ├── changed_files.list
          ├── nightly.stats
          ├── nightly.names
          ├── nightly.diff
          ├── metadata.json
          ├── commits.log
          └── review_output/
              └── (same as baseline)

.onex_nightly_prev  # Marker file tracking last successful SHA
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: ONEX Nightly Review

on:
  schedule:
    - cron: '0 2 * * *'  # 22:00 America/New_York
  workflow_dispatch:

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history needed

      - name: Run ONEX Nightly Review
        run: ./scripts/onex_review_orchestrator.sh nightly

      - name: Upload Findings
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: onex-findings
          path: .onex_nightly/*/review_output/
```

## Operational Notes

1. **Baseline Review**: Run once per repository to establish current state
2. **Nightly Review**: Run at 22:00 America/New_York for consistency
3. **Marker Management**: Only update `.onex_nightly_prev` after successful agent run
4. **Size Limits**:
   - Baseline: 200KB per shard
   - Nightly: 500KB total diff
5. **Coverage**: Truncated diffs are noted in summaries

## Troubleshooting

### No changes detected
- Check `.onex_nightly_prev` contains correct SHA
- Verify `git fetch origin` succeeded
- Ensure you're on the correct branch

### Large diffs
- Diffs are automatically truncated at size limits
- Consider running baseline if many changes accumulated
- Check for mass refactoring that might need waiver

### Missing dependencies
- Ensure `git`, `python3`, `csplit`, `awk` are installed
- Policy file must exist at `config/policy.yaml`

## Future Enhancements

- [ ] Actual Opus API integration (currently using mock)
- [ ] Parallel shard processing for baseline
- [ ] Incremental baseline updates
- [ ] Integration with issue tracking systems
- [ ] Custom rule plugins
- [ ] Performance metrics tracking

## Support

For issues or questions about the ONEX Review System:
- Check agent outputs in review_output directories
- Review policy.yaml for configuration issues
- Examine metadata.json for processing details