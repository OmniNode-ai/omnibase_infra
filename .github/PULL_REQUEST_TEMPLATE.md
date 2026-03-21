## Summary

<!-- Brief description of changes -->

## Changes

<!-- List of changes made -->

## Test plan

<!-- How were these changes tested? -->

## Type safety checklist
- [ ] No new `metadata["key"]` or `metadata.get("key")` string literal access on Pydantic model fields
- [ ] No new `metadata: dict[str, Any]` fields without TypedDict or `# ONEX_EXCLUDE:` comment
- [ ] No new bare `except Exception` — must use narrowed type, or minimal-scope boundary with `logger.exception(...)` + degrade comment, or typed wrap/re-raise
- [ ] If adding a key to a metadata dict, the key is defined in the relevant TypedDict
- [ ] If adding a service to `docker-compose.infra.yml` with required (`:?`) env vars, update `tests/integration/docker/test_docker_integration.py` fixture dict in `test_compose_config_valid`

## Related issues

<!-- OMN-XXXX -->
