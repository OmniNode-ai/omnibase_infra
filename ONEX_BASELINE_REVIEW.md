# ONEX Baseline Review System

## Overview

This repository includes the ONEX Baseline Review system for continuous code quality monitoring against ONEX architectural standards.

## Components

### 1. Baseline Producer (`onex_baseline_producer.sh`)
Generates baseline review inputs by:
- Scanning repository for non-archived files
- Creating git diffs against empty tree (baseline)
- Sharding large diffs into 200KB chunks
- Generating stats and file lists

### 2. Baseline Reviewer (`onex_baseline_reviewer.py`)
Analyzes code against ONEX ruleset v0.1:
- Processes diff shards efficiently
- Checks naming conventions, boundaries, typing, and SPI purity
- Generates NDJSON findings and Markdown summaries
- Calculates risk scores

### 3. Policy Configuration (`policy.yaml`)
Defines repository-specific rules:
- Forbidden import patterns
- Module boundary enforcement
- SPI purity requirements

## Usage

### Running a Baseline Review

```bash
# 1. Generate baseline inputs
./onex_baseline_producer.sh

# 2. Run the reviewer
python3 onex_baseline_reviewer.py \
  --repo omnibase_infra \
  --input-dir .onex_baseline/omnibase_infra/[timestamp] \
  --output baseline_review.out
```

### Output Format

The review generates two outputs separated by `---ONEX-SEP---`:

1. **NDJSON Findings**: Machine-readable violation records
2. **Markdown Summary**: Human-readable report with:
   - Risk score (0-100)
   - Violation counts by category
   - Top issues requiring attention
   - Recommended next actions

## Current Status

**Baseline Review Results (2025-09-19)**
- **Risk Score**: 100/100 (due to critical errors)
- **Total Findings**: 247
  - 3 critical errors (Protocol decorator issues)
  - 244 warnings (typing and naming conventions)

### Key Issues Identified

1. **Critical Errors**:
   - 3 Protocol classes missing `@runtime_checkable` decorator

2. **Type Safety**:
   - 215 uses of `Any` type (should use specific types)
   - 5 functions lacking type annotations

3. **Naming Conventions**:
   - 15 Node classes not prefixed with "Node"
   - 9 Model classes not prefixed with "Model"

## Ruleset v0.1

### A. Naming Rules
- `ONEX.NAMING.PROTOCOL_001`: Protocol classes must start with "Protocol"
- `ONEX.NAMING.MODEL_001`: Model classes must start with "Model"
- `ONEX.NAMING.ENUM_001`: Enum classes must start with "Enum"
- `ONEX.NAMING.NODE_001`: Node classes must start with "Node"

### B. Boundary Rules
- `ONEX.BOUNDARY.FORBIDDEN_IMPORT_001`: Forbidden cross-module imports

### C. SPI Purity Rules
- `ONEX.SPI.RUNTIMECHECKABLE_001`: Protocol classes need `@runtime_checkable`
- `ONEX.SPI.FORBIDDEN_LIB_001`: No I/O operations in SPI layer

### D. Typing Hygiene
- `ONEX.TYPE.UNANNOTATED_DEF_001`: Functions need type annotations
- `ONEX.TYPE.ANY_001`: Avoid using `Any` type
- `ONEX.TYPE.OPTIONAL_ASSERT_001`: Don't immediately assert Optional types

### E. Waiver Hygiene
- `ONEX.WAIVER.MALFORMED_001`: Waivers need reason and expiry
- `ONEX.WAIVER.EXPIRED_001`: Expired waivers must be removed

## Integration with CI/CD

The baseline review system can be integrated into nightly builds:

1. **Initial Baseline**: Run once to establish current state
2. **Nightly Reviews**: Check only new commits since last review
3. **PR Reviews**: Run on feature branches before merge

## Next Steps

1. **Immediate Actions**:
   - Fix 3 critical Protocol decorator issues
   - Begin replacing `Any` types with specific types

2. **Short-term Goals**:
   - Achieve 0 critical errors
   - Reduce warnings by 50%
   - Establish waiver process for legitimate exceptions

3. **Long-term Goals**:
   - Maintain risk score below 20
   - Automate fixes for common violations
   - Integrate with PR approval workflow

## Files Generated (Not Committed)

The following files are generated during review but excluded from version control:
- `.onex_baseline/` - Directory containing diff shards and analysis
- `baseline_review.out` - Full review output with findings
- `*.ndjson` - Machine-readable finding records

These artifacts are listed in `.gitignore` to prevent repository bloat.