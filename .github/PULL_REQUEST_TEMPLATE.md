## Summary

<!-- Brief description of changes -->

## Changes

<!-- List of changes made -->

## Test plan

<!-- How were these changes tested? -->

## Runtime startup gate (OMN-9126)

If this PR touches `auto_wiring/`, `service_kernel.py`, handler `__init__` signatures, or kernel-level registration:
- [ ] Test loads the **real** contract manifest from disk (no fake/stub handlers)
- [ ] Test calls `wire_from_manifest` with the same args the kernel uses in production
- [ ] Test asserts zero wiring failures
- [ ] CI boots `omninode-runtime` in a compose sandbox and asserts `RestartCount == 0` after 45s
- [ ] N/A — this PR does not touch any of the above

## Type safety checklist
- [ ] No new `metadata["key"]` or `metadata.get("key")` string literal access on Pydantic model fields
- [ ] No new `metadata: dict[str, Any]` fields without TypedDict or `# ONEX_EXCLUDE:` comment
- [ ] No new bare `except Exception` — must use narrowed type, or minimal-scope boundary with `logger.exception(...)` + degrade comment, or typed wrap/re-raise
- [ ] If adding a key to a metadata dict, the key is defined in the relevant TypedDict
- [ ] If adding a service to `docker-compose.infra.yml` with required (`:?`) env vars, update `tests/integration/docker/test_docker_integration.py` fixture dict in `test_compose_config_valid`

## Related issues

<!-- OMN-XXXX -->
