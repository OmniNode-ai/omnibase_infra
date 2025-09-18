# Task Completion Workflow for ONEX Infrastructure

## MANDATORY: Agent-First Approach
**NEVER complete tasks directly - Always delegate to agents**

## Pre-Task Setup
1. **Use agent-onex-coordinator** for task analysis and routing
2. **Set up TodoWrite tracking** for multi-step operations
3. **Identify specialist agents** needed for implementation

## During Task Execution
1. **Route through orchestration agents** for complex workflows
2. **Coordinate with specialist agents** for domain-specific work
3. **Track progress** via TodoWrite for transparency
4. **Monitor quality gates** throughout execution

## Task Completion Checklist
```bash
# Code Quality Validation
ruff check                 # Zero linting errors
ruff format               # Consistent formatting  
mypy src/                 # Zero type errors
pytest                    # All tests passing
pytest --cov             # Coverage requirements met

# ONEX Compliance
# - Zero `Any` types used
# - All models are Pydantic with proper typing
# - Contract-driven configuration
# - Protocol-based resolution (no isinstance)
# - OnexError exception chaining

# Pre-commit validation
pre-commit run --all-files

# Agent validation
agent-contract-validator   # Contract compliance
agent-testing             # Comprehensive test strategy
agent-security-audit     # Security compliance (if applicable)
```

## Final Validation Steps
1. **Integration tests passing** (if infrastructure changes)
2. **Documentation updated** (if architectural changes)
3. **Performance benchmarks met** (if performance-critical)
4. **Security audit passed** (if security-relevant)

## Post-Task Requirements
1. **Clean up TodoWrite** list if tasks are complete
2. **Update agent knowledge** via agent-rag-update
3. **Commit with semantic messages** via agent-commit
4. **Create PR** via agent-pr-create (if ready for review)

## Emergency Protocols
- **Production issues**: Use agent-production-monitor
- **Security incidents**: Use agent-security-audit
- **Performance degradation**: Use agent-performance
- **Critical bugs**: Use agent-debug-intelligence
