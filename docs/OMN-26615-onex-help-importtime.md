# OMN-26615 AC3: onex --help importtime profile

Repeatable profiling of `onex --help` with stdin and stdout not a terminal
(stdin redirected from /dev/null, stdout captured to a file).

## Builds

- Before: omninode-onex 0.0.1.post1, Python 3.12, a fresh venv.
- After: dev heads - omnibase-core 0.47.82.dev24+g552386277, omnibase-infra 0.38.87 (worktree), omnimarket 0.4.329 (worktree), Python 3.12, a fresh venv.

## Measurement command

```sh
python -X importtime <venv>/bin/onex --help </dev/null ><out>.stdout 2><out>.importtime
```

Each build profiled 5 times; wall times are seconds.

## Wall times

| run | before (s) | after (s) |
| --- | --- | --- |
| 1 | 4.010 | 0.310 |
| 2 | 4.018 | 0.309 |
| 3 | 4.034 | 0.303 |
| 4 | 4.044 | 0.311 |
| 5 | 4.049 | 0.303 |
| median | 4.034 | 0.309 |

Median wall time before: 4.034 s
Median wall time after:  0.309 s

## Slowest imports before (self time, us)

import time:  192374 | 192374 | omnibase_infra.enums.generated.enum_omnibase_infra_topic
import time:  114082 | 114082 | omnibase_core.models.core.model_node_action_type
import time:   81467 | 81467 | omnibase_infra.enums.generated.enum_platform_topic
import time:   77729 | 77729 | omnimarket.events.model_ledger_row_event
import time:   51024 | 51024 | redis.maint_notifications
import time:   38590 | 38590 | omnimarket.nodes.node_push_validation_effect.models.model_push_validation_request
import time:   35544 | 35544 | omnimarket.models.floor_17a.model_floor_17a
import time:   32100 | 32100 | fastapi.openapi.models
import time:   26463 | 26463 | omnimarket.nodes.node_llm_delegation_call_effect.handlers.handler_llm_delegation_call
import time:   25090 | 25090 | omnimarket.models.model_github_gateway_io
import time:   23540 | 23540 | omnibase_infra.models.mcp.model_mcp_server_config
import time:   23155 | 23155 | omnimarket.events.runtime_deployment
import time:   21729 | 21729 | omnimarket.nodes.node_onex_setup_effect.handlers.handler_onex_setup
import time:   17698 | 17698 | omnibase_infra.enums.generated.enum_occ_topic
import time:   15172 | 15172 | omnimarket.nodes.node_pr_landing_decision_compute.models.model_landing_facts
import time:   14932 | 14932 | omnimarket.nodes.node_model_router.handlers.handler_model_router
import time:   14054 | 14054 | omnimarket.nodes.node_llm_delegation_call_effect.handlers.review_cost_recorder
import time:   13488 | 13488 | omnimarket.models.evidence_pipeline.wire.topics
import time:   13341 | 13341 | omnibase_infra.enums.generated.enum_artifact_topic
import time:   12314 | 12314 | omnimarket.models.delegation.wire.model_delegate_skill_response
import time:   11711 | 11711 | omnibase_infra.cli.model_lane_credential
import time:   11005 | 11005 | omnibase_core.cli.cli_commands
import time:   10677 | 10677 | omnimarket.inference.task_class_authority
import time:   10523 | 10523 | omnimarket.models.model_repo_write_io
import time:   10438 | 10438 | omnimarket.models.model_pr_duplicate_guard

## Slowest imports after (self time, us)

import time:   19183 | 19183 | omnibase_core.cli.cli_commands
import time:   11275 | 11275 | omnibase_core.models.effects.model_effect_classification
import time:    6245 | 6245 | pydantic_core.core_schema
import time:    4974 | 4974 | annotated_types
import time:    4456 | 4456 | pydantic.types
import time:    2619 | 2619 | yaml.reader
import time:    2473 | 2473 | pydantic._internal._decorators
import time:    2356 | 2356 | pydantic.functional_validators
import time:    2334 | 2334 | omnibase_core.models.cli.model_cli_execution
import time:    2239 | 2239 | omnibase_core.models.cli.model_cli_action
import time:    2206 | 2206 | omnibase_core.constants.constants_event_types
import time:    1922 | 1922 | omnibase_core.models.cli.model_cli_result_metadata
import time:    1791 | 1791 | typing
import time:    1708 | 1708 | logging
import time:    1538 | 1538 | omnibase_core
import time:    1480 | 1480 | omnibase_core.models.cli.model_cli_advanced_params
import time:    1475 | 1475 | pydantic.json_schema
import time:    1462 | 1462 | omnibase_core.models.cli.model_cli_result
import time:    1460 | 1460 | inspect
import time:    1377 | 1377 | omnibase_core.models.cli.model_cli_debug_info
import time:    1363 | 1363 | typing_extensions
import time:    1358 | 1358 | yaml.resolver
import time:    1348 | 1348 | ssl
import time:    1289 | 1289 | click.types
import time:    1283 | 1283 | omnibase_core.models.primitives.model_semver

The slow imports named above before (`omnibase_infra`, `omnimarket`, `redis`,
`fastapi`, `aiohttp`, generated enum modules) are gone from the after profile;
`onex --help` after loads only `omnibase_core` plus entry-point metadata.

## Raw `python -X importtime` excerpt (run 1, last 5 lines)

### before
```text
import time:        79 |         79 |       omnimarket.nodes.node_operator_capture_effect.handlers.ledger_read
import time:       144 |       2874 |     omnimarket.nodes.node_operator_capture_effect.handlers.handler_capture_process
import time:       137 |       3010 |   omnimarket.nodes.node_operator_capture_effect.handlers.handler_operator_capture_effect
import time:       121 |      12735 | omnimarket.nodes.node_operator_capture_effect.handlers.capture_ledger_host
import time:        89 |         89 | omnimarket.nodes.node_work_ledger_append_effect.local_serve
```

### after
```text
import time:        79 |       1211 | omnibase_core.doctor.checks.check_postgres
import time:       100 |        100 | omnibase_core.doctor.checks.check_python_version
import time:        97 |         97 | omnibase_core.doctor.checks.check_repos_synced
import time:        73 |         73 | omnibase_core.doctor.checks.check_stale_worktrees
import time:       113 |        113 | omnibase_core.cli.cli_user_config
```
