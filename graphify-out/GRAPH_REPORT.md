# Graph Report - omnibase_infra  (2026-11-16)

## Corpus Check
- 6478 files · ~4,761,491 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 139 file(s) not represented in the graph (top: .service 32, (none) 24, .timer 22)

## Summary
- 109395 nodes · 160032 edges · 5800 communities (4414 shown, 1386 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 9048 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `48600366`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- typing
- pydantic
- ModelInfraErrorContext
- .with_correlation
- migrations
- pathlib
- json
- cli_delegate.py
- deploy_agent/executor.py
- gateway_forwarder.py
- ModelInjectedRequest
- lab_pass_receipt.py
- urllib_parse
- agent.py
- re
- runtime_host_process.py
- importlib
- handler_wiring.py
- ModelRegistrationProjection
- sys
- collections_abc
- os
- dataclasses
- argparse
- ModelAutoWiringManifest
- ModelBackendResult
- RuntimeHostProcess
- service_kernel.py
- ast
- event_bus_kafka.py
- RepositoryExecutionError
- ProtocolEventBusLike
- lab_alarm.py
- EventBusKafka
- rsd_live_delegation_overlay.py
- cli_skill.py
- subprocess
- runtime/models/__init__.py
- tools/contract_topic_extractor.py
- RegistrationReducerService
- application_database_acl.py
- prepr_runtime_pool.py
- receipt_mode.py
- sanitize_error_message
- ServiceRegistration
- HandlerHttpRest
- SecretResolver
- release_train.py
- ModelSessionStateResult
- models/validation/__init__.py
- CHANGELOG.md
- _parse_contract
- JobStore
- ContractRegistryReducer
- check_dev_lane_staleness.py
- post_merge/consumer.py
- VerifiedProjectionTenantAuthority
- backend_probe.py
- handler_contract_subscribe_preflight.py
- prove_application_database_domain_enforcement.py
- deploy_agent/consumer.py
- cli_auth.py
- TopicProvisioner
- handler_evidence_autoclose_sweep.py
- handler_lab_proof_plan.py
- MixinNodeIntrospection
- enum
- _prepare_handler_wiring
- EnumCheckSeverity
- InfraUnavailableError
- check_lane_sibling_revision.py
- KafkaContractSource
- ModelCheckpoint
- EnumHandlerType
- HandlerQdrant
- ModelUpdatePlan
- HandlerGraph
- build_budget.py
- urllib_request
- handler_runtime_deploy.sh
- node_chain_orchestrator/models/__init__.py
- ServiceLlmEndpointHealth
- math
- handler_savings_correlation.py
- ._publish_raw_to_dlq
- run_suite.py
- EnumHandlerTypeCategory
- bootstrap
- prove_application_database_acl.py
- MixinLlmHttpTransport
- omnimarket_drift_guard.py
- generated/__init__.py
- ModelCheckResult
- ConsumerRejoinSupervisor
- StatusApp
- EventBusSubcontractWiring
- prove.py
- ModelHandlerValidationError
- ModelRegistryApiRequest
- HandlerLedgerQuery
- x-runtime-env
- ModelEventMessage
- HandlerGatewayHeartbeat
- StoreEffectIdempotencyInmemory
- ModelEventHeaders
- MessageDispatchEngine
- registration_reducer_service.py
- HandlerDlqDepthMonitor
- application_database_domain_enforcement.py
- node_contract_registry_reducer/reducer.py
- omnibase_infra/protocols/__init__.py
- handler_a2a_task.py
- ModelSnapshotTopicConfig
- sanitize_error_string
- HandlerSetupOrchestrator
- prepush_override_grant.py
- environment
- HandlerContractSubscribePreflight
- environment
- lint_topic_names.py
- seed-keycloak-clients.py
- application_relation_ownership.py
- AdapterLlmProviderOpenai
- environment
- handler_reward_binder.py
- kafka_contract_source.py
- EnumInfraTransportType
- ModelDevLaneCodeRefPin
- ModelRuleCheckResult
- apply_application_database_acl.py
- HandlerLlmOpenaiCompatible
- handler_coding_agent_invoke.py
- AdapterLinearGraphQLProjectTracker
- DispatchResultApplier
- ModelKafkaEventBusConfig
- HandlerBifrostGateway
- x-dogfood-runtime-env
- x-judge-runtime-env
- x-lakshman-runtime-env
- AdapterKafkaAdminLag
- nonrequired_check_failure_rate.py
- ModelDelegationDiagnosis
- HandlerManifestPersistence
- audit-runner-routing.py
- worktree_prune_timer.py
- ContractLinter
- ModelNodeIdentity
- handler_reconcile_workspace_venvs.py
- main
- c29_customer_byo_key_probe.py
- bulk_pr_throttle.py
- deploy-runners.sh
- HandlerRuntimeErrorTriage
- recreate_supervisor.py
- detect_test_paths.py
- deploy_source_ref.py
- scoped_effects_deploy.py
- ModelChainViolation
- HandlerDb
- validator_declarative_node.py
- routing.py
- ModelTopicCatalogResponse
- ContractFileWatcher
- HandlerMCP
- handler_onboarding.py
- handler_standalone_compose_recreate.py
- ModelLocalProvisionEffectOutput
- lane_census_inventory.py
- apply_topic_namespace
- runtime_cost_estimator_build.py
- handler_dev_lane_pin.py
- MixinAsyncCircuitBreaker
- ModelRuntimeAttachReadiness
- ModelContractDiscoveryResult
- service_gateway_forwarder.py
- infra_validators.py
- instance_lanes.py
- ProtocolMessageDispatcher
- service_runtime_health_monitor.py
- health_payload.py
- verify_migration_checksum_adoption.py
- CLAUDE.md - Omnibase Infrastructure
- .run
- TopicPartition
- AgentActionsConsumer
- c13_customer_local_probe.py
- runtime_migration_freeze_check.py
- handler_chain_canary.py
- ModelSecretSeedRequest
- InjectionEffectivenessConsumer
- reconcile_verify_movement.py
- handler_api_fixture_record.py
- EnumPolicyType
- ModelExecutionShapeViolationResult
- check_lab_probe_windows.py
- resolve_embedded_runtime_config
- action_verify.py
- AdapterTicketLinear
- managed_staging_canary_catalog.py
- ProtocolSnapshotStore
- TransitionNotificationOutbox
- validate_naming.py
- grammar.py
- TransitionReducer
- ReaderInjectionEffectivenessPostgres
- validate_markdown_links.py
- ModelLabProofProfileVariant
- ProtocolPostgresAdapter
- bus_backed_exposure_readers.py
- environment
- services
- NodeInvocationAdapter
- ServiceFreshnessMonitor
- HandlerGmailApi
- HandlerVectorSearch
- ModelBrokerAclBinding
- commands.py
- HandlerActionArchiveCacheFill
- handler_consumer_flow.py
- generate_application_database_table_grants.py
- application_database.py
- ModelAdjudicatorState
- ModelContextSection
- ModelSecurityValidationResult
- _consumer_flow_collection.py
- janitor_docker_network.py
- HandlerLabProofVerdict
- ModelCutoverFamilyContract
- HandlerRequestAdjudication
- HandlerChainCanary
- HandlerMetricsPrometheus
- ServiceLlmCostAggregator
- with
- in_progress_demotion.py
- ModelBoardProbeResultEvent
- replay_hook_history.py
- EventRegistry
- ModelBatchLifecycleResult
- checker_first_party_sources.py
- services
- generate_ticket_plan.py
- agent_learning_extraction/consumer.py
- HandlerDockerForwarderStateReader
- ModelRunnerFleetConfig
- build_effects_candidate.py
- declared_consumer_groups.py
- validate.py
- check_topic_completeness.py
- LongLivedTerminalCorrelator
- consumers
- with
- product_reason_graph.py
- rollback_lane_to_known_good.py
- verify_stability_refresh.py
- ServiceTopicRegistry
- RegistryCompute
- ConsumerFlowCounters
- node_dev_roll_events_effect/contract.yaml
- ServiceArtifactStore
- RegistryInfraLlmInferenceEffect
- OMN-19817.yaml
- OMN-19958.yaml
- preflight_lane_deploy_attribution.py
- MixinDictLikeAccessors
- ApplicationDatabaseFunctionSignature
- PostgresTransformationEvidenceCollector
- RepositoryPostgresCutoverJournal
- ModelLogContext
- ModelDiscoveryResult
- node_c11_delegation_legs_effect/contract.yaml
- node_c16_pinned_backend_effect/contract.yaml
- node_chain_retrieval_effect/contract.yaml
- node_ledger_projection_compute/handlers/contract.yaml
- node_standalone_compose_recreate_effect/contract.yaml
- registration_projector.yaml
- InfraRoutingDecisionsConsumer
- SkillLifecycleConsumer
- ModelPricingTable
- service_eval_runner.py
- with
- validate_no_direct_env.py
- publisher.py
- HandlerAuthGate
- lane_settle_budget.py
- deploy-gateway.sh
- heavy_lock.py
- reconcile-workspace-venvs.sh
- venv_readback.py
- HandlerBaselinesBatchCompute
- node_chain_verify_reducer/contract.yaml
- node_declared_state_apply_effect/contract.yaml
- node_declared_state_reconcile_orchestrator/contract.yaml
- HandlerSavingsEstimation
- variants
- with
- catalog/cli.py
- CollectorRunnerHealth
- check_dep_provenance.py
- ProtocolContainerAware
- check_hostile_review_threads.py
- prepr_teardown_slot.py
- HandlerSyncRevertWatchdog
- CloudLedgerReader
- SessionEventConsumer
- node_chain_store_effect/contract.yaml
- node_ci_live_contact_effect/contract.yaml
- node_hotpatch_ledger_reconcile_effect/contract.yaml
- HandlerSavingsCorrelation
- node_setup_orchestrator/contract.yaml
- node_ticket_contract_compose_effect/contract.yaml
- application_database_red_control_registry.py
- verify
- RuntimeDelegationDispatchPort
- services
- with
- with
- with
- with
- handler_invariant_evaluate.py
- validate_timezone_aware_datetime
- c16_receipt_identity_probe.py
- NodeGatewayDelivery
- _build_link_verdicts
- node_rrh_emit_effect/contract.yaml
- profiles
- agent-actions-consumer.yaml
- consumer-health-projection.yaml
- worktree_reaper.py
- DefaultEnvelopeChunker
- handlers/models/__init__.py
- handler_c16_pinned_backend.py
- handler_deploy_job_reading.py
- host_slot.py
- compare_environments.py
- ModelDurabilityConfirmation
- ProtocolDoneWriteReceiptGate
- node_deploy_runners/contract.yaml
- node_row_count_diagnostic_effect/contract.yaml
- TopicCategoryASTVisitor
- compute_schema_fingerprint
- Changelog
- OMN-18613.yaml
- OMN-19521.yaml
- OMN-20359.yaml
- OMN-20715.yaml
- OMN-21050.yaml
- atomic_venv_install.py
- context-audit-consumer.yaml
- infra-routing-decisions-consumer.yaml
- llm-cost-aggregation-consumer.yaml
- skill-lifecycle-consumer.yaml
- render_bifrost_delegation_contract.py
- env
- with
- ModelDispatchContext
- check_delegation_tenant_conversion_readiness.py
- ConsumerHealthEmitter
- HandlerLabProofRun
- ModelDispatchMetrics
- HandlerPluginLoader
- HandlerGatewayLinkHealthUpsert
- node_merge_provenance_compute/contract.yaml
- node_reward_binder_effect/contract.yaml
- RuntimeContractConfigLoader
- ServiceRetryWorker
- with
- with
- with
- env
- steps
- boot_gate.py
- node_registration_orchestrator/models/__init__.py
- HandlerSlackWebhook
- generator.py
- ModelLaneClientTransport
- entrypoint_preflight.py
- check_runner_host_artifact_freshness.py
- unit_drift.py
- LabOverlayApplier
- _fail
- AdapterProtocolEventPublisherKafka
- HandlerCatalogRequest
- ModelChainCanaryRequest
- node_release_identity_compute/contract.yaml
- node_release_notes_apply_effect/contract.yaml
- WriterInjectionEffectivenessPostgres
- relation_evidence
- decision-store-consumer.yaml
- environment
- with
- env
- env
- with
- env
- validator_any_type.py
- build_sanitized_evidence
- ContractTopicExtractor
- lineage_fence.py
- prepush_dispatch.sh
- omninode-ci-required-context-probe.py
- ServiceAutoEvalRunner
- ModelComponentHealth
- HandlerDockerConsumerFlowTarget
- validate_canonical_topic
- node_contract_subscribe_preflight_effect/contract.yaml
- node_service_discovery_effect/models/__init__.py
- ProjectorPluginLoader
- omnibase_infra/errors/__init__.py
- ModelSubscriptionHandle
- infisical.yaml
- injection-effectiveness-consumer.yaml
- ConsumerTopicVisitor
- prove_tenant_uuid_conversion.py
- boundary_failure_terminal.py
- handler_developer_principal_plan/handler.py
- main
- governed_helper_gate.py
- lane_container_memory_event.py
- check_topology_grant_delivery.py
- ValidationAggregator
- .run_once
- GatewayTokenMinter
- HandlerWriteDecision
- node_dev_boot_guard_effect/contract.yaml
- node_dev_lane_code_ref_pin_effect/contract.yaml
- node_dev_lane_pin_compose_guard_effect/contract.yaml
- _LinearClient
- node_runtime_start_first_effect/contract.yaml
- RuntimeLogEventBridge
- ConsumerHealthProjectionConsumer
- registry_api/routes.py
- OMN-17292.yaml
- OMN-19383.yaml
- OMN-19496.yaml
- ci-relay.yaml
- linear-relay.yaml
- projection-api.yaml
- waitlist-signup-notifier.yaml
- environment
- omnipc2-ci-runner-5
- shlex
- with
- dependency-cascade-dispatch.yml
- steps
- dlq_replay.py
- handler_hotpatch_ledger_reconcile.py
- check_application_database_sql.py
- register
- EnumNodeArchetype
- RemoteTaskStateRepository
- ModelDiscoveredContract
- HandlerGmailIntentPoll
- quote_identifier
- ModelRuntimeContractConfig
- node_auth_gate_compute/contract.yaml
- ModelGatewayEnvelope
- node_migration_append_only_check_compute/contract.yaml
- node_migration_freeze_check_compute/contract.yaml
- node_migration_sequence_check_compute/contract.yaml
- node_setup_validate_effect/contract.yaml
- node_sibling_sha_pin_ratchet_compute/contract.yaml
- WiringHealthChecker
- routing_decision_v1.yaml
- OMN-18508.yaml
- OMN-19339.yaml
- OMN-19565.yaml
- OMN-19623.yaml
- OMN-19812.yaml
- OMN-19957.yaml
- OMN-20764.yaml
- OMN-20959.yaml
- OMN-21052.yaml
- OMN-21206.yaml
- inject_env
- steps
- CatalogManifest
- handler_c11_delegation_legs.py
- run-forward-migrations.sh
- HandlerWorkLedgerQuery
- HandlerConsumerHealthTriage
- node_context_audit_dlq_effect/contract.yaml
- HandlerDlqReplay
- node_kafka_replay_compute/contract.yaml
- node_rrh_storage_effect/contract.yaml
- OMN-17882.yaml
- OMN-19314.yaml
- candidate-matrix.yaml
- migration-gate.yaml
- omnimarket-projection-delegation.yaml
- omnimarket-projection-llm-cost.yaml
- omnimarket-projection-run-trace.yaml
- omnimemory-qdrant.yaml
- principal-issuer
- omnipc2-customer-plane-runner-1
- arm64-verify-proof
- shared-env-runner-parity.yml
- HandlerDevLaneBranchInstall
- ModelConsumerSyncStatus
- manifest_fetch.py
- expand_contract_env_refs
- model_application_relation_evidence_inventory.py
- runtime_llm_endpoint_env.py
- handler_git_fix_verifier.py
- shutil
- omninode-fleet-failure-probe.py
- HandlerLabProofPlan
- ModelResolvedDependencies
- ModelGatewaySession
- handler_score_models.py
- node_onboarding_orchestrator/contract.yaml
- node_rrh_validate_compute/contract.yaml
- node_rsd_state_reducer/contract.yaml
- node_scope_extract_compute/contract.yaml
- node_scope_workflow_reducer/contract.yaml
- HandlerServiceValidate
- connect_cloud.yaml
- OMN-21288.yaml
- OMN-21386.yaml
- environment
- with
- with
- with
- with
- with
- with
- with
- steps
- steps
- ContractAttachReadinessGate
- ConsumerMetrics
- scanner_orchestration_monolith_ratchet.py
- e2e_cloud_workflow_harness.py
- ModelMCPToolDefinition
- DevRollEventsPublisher
- node_setup_infisical_effect/contract.yaml
- ServiceContractPublisher
- OMN-21923.yaml
- lane-mirror-forwarder.yaml
- environment
- omnipc2-verify-runner-1
- deployed_readback
- HandlerTopicMigrationExecutor
- steps
- with
- with
- AgentRegistry
- .compute_savings
- validator_localhandler.py
- HandlerLLMCompletion
- audit_required_context_parity_cli.py
- c11_negative_paths_probe.py
- check_pin_reachability.py
- check_migration_class.py
- closer_receipt_residual.py
- HandlerWorkspaceRuntimeConfigMaterializer
- SharedListener
- ServiceMCPToolRegistry
- ModelProjectorSchema
- node_model_router_compute/contract.yaml
- node_scope_file_read_effect/contract.yaml
- HandlerInfisicalFullSetup
- node_setup_preflight_effect/contract.yaml
- TopicResolver
- mixin_any_type_exemption.py
- typing_extensions
- OMN-20275.yaml
- runtime_config.yaml
- forward-migration.yaml
- keycloak.yaml
- omnimemory-memgraph.yaml
- factory-floor-web
- environment
- omninode-studio-runner-1
- prepr_pool_group.py
- compose-config-lint
- steps
- with
- ProtocolArchitectureRule
- ModelRelationEvidence
- provider_rung_canary_observe.py
- .__init__
- check_prod_promotion_lineage.py
- preflight_hotpatch_ledger.py
- register-repo.py
- ServiceEnvelopeValidator
- ModelFileValidationResult
- node_emit_daemon_runtime/contract.yaml
- _handle_interactive
- BatchResponsePublisher
- ContextAuditConsumer
- validation
- OMN-21753.yaml
- io
- generate_entry_points.py
- postgres.yaml
- volumes
- volumes
- services
- handler_runner_fleet_health_evaluate.py
- dispatch-parity-gate.yml
- with
- with
- s6-runtime-seam-gate.yml
- with
- url-authority-gate.yml
- JsonPayload
- RequestResponseWiring
- HandlerBootFailureLedger
- handlers.py
- application_database_source_tenant_authority.py
- prepr_verify_lane.sh
- GatewayTransportHttpx
- HandlerOnexApiImageImportCheck
- node_baselines_batch_compute/contract.yaml
- HandlerBrokerDiskWatermark
- RuntimeLocalIngressServer
- set_statement_timeout
- AnyTypeDetector
- suites
- OMN-15629.yaml
- OMN-18087.yaml
- OMN-19215.yaml
- OMN-19988.yaml
- OMN-20117.yaml
- OMN-20915.yaml
- OMN-21031.yaml
- OMN-21041.yaml
- OMN-21053.yaml
- OMN-21056.yaml
- OMN-21071.yaml
- OMN-21271.yaml
- OMN-21371.yaml
- OMN-21403.yaml
- OMN-21481.yaml
- OMN-21536.yaml
- OMN-21587.yaml
- OMN-21623.yaml
- OMN-21690.yaml
- OMN-21696.yaml
- OMN-21697.yaml
- OMN-21698.yaml
- OMN-21699.yaml
- OMN-21700.yaml
- OMN-21703.yaml
- OMN-21704.yaml
- OMN-21705.yaml
- OMN-21706.yaml
- OMN-21719.yaml
- OMN-21792.yaml
- OMN-21802.yaml
- OMN-21920.yaml
- OMN-21934.yaml
- OMN-22211.yaml
- OMN-22530.yaml
- OMN-22532.yaml
- OMN-22534.yaml
- OMN-22535.yaml
- OMN-22543.yaml
- OMN-22555.yaml
- OMN-22561.yaml
- OMN-22622.yaml
- OMN-22634.yaml
- OMN-22635.yaml
- OMN-22644.yaml
- OMN-22650.yaml
- OMN-22668.yaml
- OMN-22691.yaml
- OMN-22711.yaml
- OMN-22724.yaml
- OMN-22756.yaml
- OMN-22808.yaml
- OMN-22815.yaml
- OMN-22870.yaml
- OMN-23009.yaml
- OMN-23027.yaml
- OMN-23039.yaml
- OMN-23043.yaml
- OMN-23055.yaml
- OMN-23062.yaml
- OMN-23066.yaml
- OMN-23072.yaml
- OMN-23106.yaml
- OMN-23143.yaml
- OMN-23304.yaml
- OMN-23332.yaml
- OMN-23333.yaml
- OMN-23342.yaml
- OMN-23431.yaml
- OMN-23459.yaml
- OMN-23513.yaml
- OMN-23521.yaml
- OMN-23524.yaml
- OMN-23525.yaml
- OMN-23540.yaml
- OMN-23562.yaml
- OMN-23579.yaml
- OMN-23624.yaml
- OMN-23659.yaml
- OMN-23715.yaml
- OMN-23858.yaml
- OMN-23884.yaml
- OMN-23896.yaml
- OMN-23947.yaml
- OMN-23951.yaml
- OMN-23981.yaml
- OMN-23988.yaml
- OMN-24009.yaml
- OMN-24013.yaml
- OMN-24014.yaml
- OMN-24017.yaml
- OMN-24032.yaml
- OMN-24048.yaml
- OMN-24067.yaml
- OMN-24077.yaml
- OMN-24081.yaml
- OMN-24119.yaml
- OMN-24127.yaml
- OMN-24134.yaml
- OMN-24153.yaml
- OMN-24186.yaml
- OMN-24202.yaml
- OMN-24218.yaml
- OMN-24223.yaml
- OMN-24260.yaml
- OMN-24280.yaml
- OMN-24299.yaml
- OMN-24306.yaml
- OMN-24308.yaml
- OMN-24323.yaml
- OMN-24446.yaml
- OMN-24463.yaml
- OMN-24465.yaml
- OMN-24466.yaml
- OMN-24471.yaml
- OMN-24480.yaml
- OMN-24496.yaml
- OMN-24524.yaml
- OMN-24526.yaml
- OMN-24562.yaml
- OMN-24617.yaml
- environment
- local-tenant-registry-seed.yaml
- omnidash.yaml
- omnimemory-valkey.yaml
- docker-compose.dev-lane.yml
- services
- services
- dev
- with
- steps
- runtime_change_classifier.py
- validator_registry.py
- handler_architecture_validation.py
- expr
- runtime_sibling_sha_pin_ratchet.py
- ServiceResolutionEventPublisher
- RunStepSequence
- check_architecture.sh
- check_no_credential_in_log.py
- post_hostile_review_threads.py
- load_bearing_gates
- queue-runner.py
- trigger_rebuild_on_merge.py
- verify_deployed_versions.py
- ModelDispatchLogContext
- node_chain_replay_compute/contract.yaml
- node_checkpoint_effect/contract.yaml
- .from_contracts_dir
- node_rsd_data_fetch_effect/contract.yaml
- node_scope_manifest_write_effect/contract.yaml
- MixinProjectorNotificationPublishing
- ModelWarning
- DemoLoopGate
- HandlerServiceDiscoveryMock
- OMN-11187.yaml
- OMN-13902.yaml
- OMN-13912.yaml
- OMN-13966.yaml
- OMN-14108.yaml
- OMN-14186.yaml
- OMN-15361.yaml
- OMN-15645.yaml
- OMN-16106.yaml
- OMN-16553.yaml
- OMN-16632.yaml
- OMN-16812.yaml
- OMN-17105.yaml
- OMN-17232.yaml
- OMN-17305.yaml
- OMN-17353.yaml
- OMN-17991.yaml
- OMN-18143.yaml
- OMN-18144.yaml
- OMN-18475.yaml
- OMN-18489.yaml
- OMN-18534.yaml
- OMN-18550.yaml
- OMN-18594.yaml
- OMN-18595.yaml
- OMN-18596.yaml
- OMN-18630.yaml
- OMN-18632.yaml
- OMN-18648.yaml
- OMN-18713.yaml
- OMN-18843.yaml
- OMN-18866.yaml
- OMN-18893.yaml
- OMN-18894.yaml
- OMN-18896.yaml
- OMN-19043.yaml
- OMN-19064.yaml
- OMN-19077.yaml
- OMN-19091.yaml
- OMN-19092.yaml
- OMN-19124.yaml
- OMN-19127.yaml
- OMN-19131.yaml
- OMN-19155.yaml
- OMN-19175.yaml
- OMN-19211.yaml
- OMN-19214.yaml
- OMN-19414.yaml
- OMN-19416.yaml
- OMN-19417.yaml
- OMN-19424.yaml
- OMN-19427.yaml
- OMN-19437.yaml
- OMN-19438.yaml
- OMN-19445.yaml
- OMN-19448.yaml
- OMN-19452.yaml
- OMN-19455.yaml
- OMN-19507.yaml
- OMN-19513.yaml
- OMN-19558.yaml
- OMN-19561.yaml
- OMN-19566.yaml
- OMN-19655.yaml
- OMN-19747.yaml
- OMN-19795.yaml
- OMN-19807.yaml
- OMN-19854.yaml
- OMN-19911.yaml
- OMN-19944.yaml
- OMN-19972.yaml
- OMN-19975.yaml
- OMN-19983.yaml
- OMN-19986.yaml
- OMN-19987.yaml
- OMN-19991.yaml
- OMN-19992.yaml
- OMN-19996.yaml
- OMN-19998.yaml
- OMN-20008.yaml
- OMN-20032.yaml
- OMN-20068.yaml
- OMN-20074.yaml
- OMN-20079.yaml
- OMN-20112.yaml
- OMN-20113.yaml
- OMN-20174.yaml
- OMN-20232.yaml
- OMN-20266.yaml
- OMN-20298.yaml
- OMN-20376.yaml
- OMN-20381.yaml
- OMN-20385.yaml
- OMN-20387.yaml
- OMN-20410.yaml
- OMN-20413.yaml
- OMN-20422.yaml
- OMN-20489.yaml
- OMN-20528.yaml
- OMN-20597.yaml
- OMN-20604.yaml
- OMN-20637.yaml
- OMN-20696.yaml
- OMN-20717.yaml
- OMN-20737.yaml
- OMN-20740.yaml
- OMN-20743.yaml
- OMN-20767.yaml
- OMN-20772.yaml
- OMN-20782.yaml
- OMN-20794.yaml
- OMN-20798.yaml
- OMN-20802.yaml
- OMN-20805.yaml
- OMN-20820.yaml
- OMN-20821.yaml
- OMN-20822.yaml
- OMN-20827.yaml
- OMN-20834.yaml
- OMN-20837.yaml
- OMN-20844.yaml
- OMN-20864.yaml
- OMN-20867.yaml
- OMN-20870.yaml
- OMN-20871.yaml
- OMN-20876.yaml
- OMN-20910.yaml
- OMN-20919.yaml
- OMN-20922.yaml
- OMN-20923.yaml
- OMN-20931.yaml
- OMN-20932.yaml
- OMN-20933.yaml
- OMN-20934.yaml
- OMN-20935.yaml
- OMN-20940.yaml
- OMN-20942.yaml
- OMN-20945.yaml
- OMN-20947.yaml
- OMN-20949.yaml
- OMN-20963.yaml
- OMN-20991.yaml
- OMN-20999.yaml
- OMN-21000.yaml
- OMN-21015.yaml
- OMN-21029.yaml
- OMN-21034.yaml
- OMN-21045.yaml
- OMN-21048.yaml
- OMN-21084.yaml
- OMN-21087.yaml
- OMN-21088.yaml
- OMN-21090.yaml
- OMN-21099.yaml
- OMN-21110.yaml
- OMN-21146.yaml
- OMN-21147.yaml
- OMN-21164.yaml
- OMN-21217.yaml
- OMN-21226.yaml
- OMN-21256.yaml
- OMN-21266.yaml
- OMN-21268.yaml
- OMN-21272.yaml
- OMN-21279.yaml
- OMN-21298.yaml
- OMN-21334.yaml
- OMN-21336.yaml
- OMN-21339.yaml
- OMN-21341.yaml
- OMN-21350.yaml
- OMN-21351.yaml
- OMN-21396.yaml
- OMN-21400.yaml
- OMN-21401.yaml
- OMN-21402.yaml
- OMN-21418.yaml
- OMN-21421.yaml
- OMN-21423.yaml
- OMN-21430.yaml
- OMN-21431.yaml
- OMN-21433.yaml
- OMN-21436.yaml
- OMN-21487.yaml
- OMN-21497.yaml
- OMN-21523.yaml
- OMN-21529.yaml
- OMN-21538.yaml
- OMN-21551.yaml
- OMN-21590.yaml
- OMN-21604.yaml
- OMN-21611.yaml
- OMN-21622.yaml
- OMN-21645.yaml
- OMN-21689.yaml
- OMN-21708.yaml
- OMN-21722.yaml
- OMN-21731.yaml
- OMN-21799.yaml
- OMN-21864.yaml
- OMN-21882.yaml
- OMN-21891.yaml
- OMN-21906.yaml
- OMN-21909.yaml
- OMN-21924.yaml
- OMN-21928.yaml
- OMN-21933.yaml
- OMN-21935.yaml
- OMN-21939.yaml
- OMN-21950.yaml
- OMN-21968.yaml
- OMN-21970.yaml
- OMN-21996.yaml
- OMN-22006.yaml
- OMN-22052.yaml
- OMN-22103.yaml
- OMN-22111.yaml
- OMN-22123.yaml
- OMN-22130.yaml
- OMN-22151.yaml
- OMN-22153.yaml
- OMN-22162.yaml
- OMN-22166.yaml
- OMN-22172.yaml
- OMN-22177.yaml
- OMN-22216.yaml
- OMN-22258.yaml
- OMN-22275.yaml
- OMN-22291.yaml
- OMN-22294.yaml
- OMN-22302.yaml
- OMN-22320.yaml
- OMN-22356.yaml
- OMN-22358.yaml
- OMN-22359.yaml
- OMN-22362.yaml
- OMN-22364.yaml
- OMN-22376.yaml
- OMN-22377.yaml
- OMN-22386.yaml
- OMN-22393.yaml
- OMN-22430.yaml
- OMN-22547.yaml
- OMN-22552.yaml
- OMN-22574.yaml
- OMN-22580.yaml
- OMN-22600.yaml
- OMN-22604.yaml
- OMN-22607.yaml
- OMN-22617.yaml
- OMN-22638.yaml
- OMN-22648.yaml
- OMN-22664.yaml
- OMN-22665.yaml
- OMN-22675.yaml
- OMN-22683.yaml
- OMN-22685.yaml
- OMN-22704.yaml
- OMN-22712.yaml
- OMN-22714.yaml
- OMN-22715.yaml
- OMN-22716.yaml
- OMN-22812.yaml
- OMN-22844.yaml
- OMN-22845.yaml
- OMN-22863.yaml
- OMN-22867.yaml
- OMN-22877.yaml
- OMN-22896.yaml
- OMN-22930.yaml
- OMN-22946.yaml
- OMN-23002.yaml
- OMN-23024.yaml
- OMN-23029.yaml
- OMN-23050.yaml
- OMN-23058.yaml
- OMN-23063.yaml
- OMN-23075.yaml
- OMN-23156.yaml
- OMN-23163.yaml
- OMN-23164.yaml
- OMN-23169.yaml
- OMN-23176.yaml
- OMN-23177.yaml
- OMN-23235.yaml
- OMN-23241.yaml
- OMN-23324.yaml
- OMN-23341.yaml
- OMN-23356.yaml
- OMN-23359.yaml
- OMN-23362.yaml
- OMN-23387.yaml
- OMN-23395.yaml
- OMN-23402.yaml
- OMN-23410.yaml
- OMN-23418.yaml
- OMN-23466.yaml
- OMN-23468.yaml
- OMN-23471.yaml
- OMN-23508.yaml
- OMN-23518.yaml
- OMN-23535.yaml
- OMN-23539.yaml
- OMN-23544.yaml
- OMN-23613.yaml
- OMN-23732.yaml
- OMN-23750.yaml
- OMN-23838.yaml
- OMN-23841.yaml
- OMN-23843.yaml
- OMN-23888.yaml
- OMN-23903.yaml
- OMN-23907.yaml
- OMN-23932.yaml
- OMN-23966.yaml
- OMN-23967.yaml
- OMN-23973.yaml
- OMN-24004.yaml
- OMN-24037.yaml
- OMN-24063.yaml
- OMN-24118.yaml
- OMN-24179.yaml
- OMN-24206.yaml
- OMN-24214.yaml
- OMN-24215.yaml
- OMN-24216.yaml
- OMN-24341.yaml
- OMN-24397.yaml
- OMN-24426.yaml
- OMN-24427.yaml
- OMN-24428.yaml
- OMN-24429.yaml
- OMN-24447.yaml
- OMN-24474.yaml
- OMN-24476.yaml
- OMN-24491.yaml
- OMN-24508.yaml
- OMN-24509.yaml
- OMN-24544.yaml
- OMN-24674.yaml
- OMN-24695.yaml
- OMN-24716.yaml
- OMN-24780.yaml
- OMN-9050.yaml
- cloud-migration.yaml
- retry-worker.yaml
- services
- redpanda
- options
- omninode-pypi-cache
- labels
- with
- steps
- steps
- ModelLlmEndpointEntry
- apply_governed_compose_config.py
- prepush_smart_tests.sh
- ledger_watermark.py
- ONEXToMCPAdapter
- HandlerBuildLoopAppend
- node_rsd_score_compute/contract.yaml
- OMN-14936.yaml
- OMN-17101.yaml
- OMN-20073.yaml
- OMN-20085.yaml
- OMN-20134.yaml
- OMN-20589.yaml
- OMN-20769.yaml
- OMN-20967.yaml
- OMN-21004.yaml
- OMN-21009.yaml
- OMN-21028.yaml
- OMN-21093.yaml
- OMN-21228.yaml
- OMN-21335.yaml
- OMN-21337.yaml
- OMN-21666.yaml
- OMN-21932.yaml
- OMN-21947.yaml
- OMN-22144.yaml
- OMN-22152.yaml
- OMN-22183.yaml
- OMN-22536.yaml
- OMN-22545.yaml
- OMN-22713.yaml
- OMN-22821.yaml
- OMN-22858.yaml
- OMN-22899.yaml
- OMN-23003.yaml
- OMN-23152.yaml
- OMN-23705.yaml
- OMN-23996.yaml
- object_type
- object_type
- object_type
- object_type
- object_type
- object_type
- object_type
- object_type
- object_type
- intelligence-migration.yaml
- omnimarket-projection-baselines.yaml
- omnimarket-projection-cost-by-repo.yaml
- omnimarket-projection-live-events.yaml
- omnimarket-projection-registration.yaml
- omnimarket-projection-savings.yaml
- omnimarket-projection-session-outcome.yaml
- hardcoded_env
- redpanda.yaml
- redpanda
- runtime-effects
- x-prepr-slot-db-env
- with
- with
- playwright-launch
- env
- delegate_request_payload.py
- .discover_handlers
- ModelInteractiveStep
- ModelDockerDatabaseConsumerCatalog
- deploy_lane_verify_route.py
- provider_rung_canary_probe.py
- runner_saturation_record.py
- provision-keycloak.py
- prepr_verify_rendered_slot.py
- HandlerRunnerHookMountDrift
- ModelUpsertResult
- ModelArchitectureViolation
- node_checkpoint_validate_compute/contract.yaml
- HandlerLlmInferenceCommand
- node_model_health_effect/contract.yaml
- node_registry_effect/contract.yaml
- HandlerRunnerFleetSnapshot
- provenance
- omnibase_infra
- OMN-20439.yaml
- performance_benchmarks
- omninode-system-slack-report.sh
- labels
- environment
- services
- preflight_required_compose_env.py
- name
- HandlerEventProvenanceStamp
- ProtocolDiscoveryOperations
- cutover/models/__init__.py
- node_evidence_autoclose_sweep_effect/models/__init__.py
- c12_provider_catalogue_probe.py
- c28_consumer_flow_probe.py
- merge_queue_enqueue.py
- validate_runner_image.py
- queue-contract-validator.py
- _Teardown
- ServicePolicyEngine
- ProtocolBrokerPrincipalAdmin
- ModelStorageResult
- ModelTopicParser
- scanner_imperative_orchestrator_ratchet.py
- ProtocolChainVectorClient
- performance_benchmarks
- HookObservability
- RegistryPayload
- UUID
- ModelApplicationDatabaseAclPolicy
- OMN-21485.yaml
- OMN-21683.yaml
- OMN-21688.yaml
- OMN-21916.yaml
- OMN-22054.yaml
- OMN-22158.yaml
- OMN-22391.yaml
- OMN-24191.yaml
- OMN-24290.yaml
- omnibase-infra-memgraph.yaml
- environment
- environment
- runner-monitor.sh
- socket
- steps
- with
- steps
- steps
- steps
- validate_application_migration_manifest.py
- ModelPayloadHttp
- handler_dev_lane_branch_install.py
- check_dockerfile_pins.py
- lockfile_cve_autobump.py
- skip_count_ratchet.py
- prove_migration_revision_equivalence.py
- onex-setup.py
- repoint_dev_lane_onex_api.py
- resolve_workspace_pins.py
- validate_test_root_collection.py
- TransportHolderLlmHttp
- .graphql_transport
- render_settings
- DelegateLaneSelectionError
- DevBootFailedError
- node_baseline_capture/contract.yaml
- offer_companion_sources.py
- HandlerReleaseIdentity
- HandlerValidationLedgerProjection
- ConsumerMetrics
- ConsumerMetrics
- EffectMockRegistry
- ValidatorSecurity
- OMN-20320.yaml
- OMN-20388.yaml
- OMN-21115.yaml
- OMN-21324.yaml
- OMN-21733.yaml
- OMN-21757.yaml
- OMN-21863.yaml
- OMN-22027.yaml
- OMN-22119.yaml
- OMN-22132.yaml
- OMN-22233.yaml
- OMN-22311.yaml
- OMN-22360.yaml
- OMN-22392.yaml
- OMN-22692.yaml
- OMN-22958.yaml
- OMN-23011.yaml
- OMN-23067.yaml
- OMN-23368.yaml
- omnimemory-kreuzberg.yaml
- phoenix.yaml
- environment
- runtime-effects
- services
- docker-compose.runners-hp051-ci-mq-runner.yml
- docker-compose.runners-hp053-ci-runner.yml
- labels
- fcntl
- resolve_lab_lane.py
- with
- steps
- steps
- direct-model-call.yml
- env
- steps
- runner-disk-preflight
- DevLaneBranchInstallBlobs
- cli_identity.py
- ProtocolCutoverJournalRepository
- HandlerOnexApiImageBuild
- lane_transport.py
- routing_api/routes.py
- _run_registration_only
- disk_gc_plan.py
- delegation_channel.py
- generate_feature_inventory.py
- lane_census_plan.py
- refresh_stability_lane.sh
- lane_lock.py
- prepr_pool_prove.sh
- refresh_dev_lane.sh
- stage_pinned_proof_root.py
- note
- GatewaySessionKeeper
- HandlerCodingAgentInvoke
- timeouts
- ModelCheckpointEffectOutput
- cascade_supersession.py
- ProtocolRuntimeScheduler
- ServiceMCPToolSync
- DecisionProjector
- OMN-18202.yaml
- OMN-20557.yaml
- OMN-21229.yaml
- OMN-21629.yaml
- OMN-21642.yaml
- OMN-21702.yaml
- OMN-21714.yaml
- OMN-21717.yaml
- OMN-21736.yaml
- OMN-21758.yaml
- OMN-21761.yaml
- OMN-21765.yaml
- OMN-21784.yaml
- OMN-21789.yaml
- OMN-21807.yaml
- OMN-21853.yaml
- OMN-21894.yaml
- OMN-21913.yaml
- OMN-21981.yaml
- OMN-22002.yaml
- OMN-22097.yaml
- OMN-22128.yaml
- OMN-23246.yaml
- valkey.yaml
- migration-gate
- postgres
- docker-compose.runners-hp053-ci-mq-runner.yml
- rebuilt-postgres16-proof
- integration-test-check.yml
- steps
- steps
- shellcheck-gate.yml
- HandlerCostApiDependencies
- handler_update_scores.py
- HandlerRunnerFleetHealthEvaluate
- runner_health/__init__.py
- InteractiveExecutor
- omnibase_infra/types/__init__.py
- AdapterModelRouter
- ModelLabPassReceipt
- check_sibling_lock_pins.py
- StoreGatewayCredential
- attempt
- ModelPlacementServingReport
- HandlerPlacementRuntime
- RegistryInfraPlacementRuntime
- OperationBindingResolver
- ModelDispatchError
- ModelPostMergeLabPassReceiptEvent
- ProjectionApplyCounters
- HandlerBaselineCapture
- HandlerInProgressProbeHygiene
- ModelReducerExecutionResult
- properties
- node_runtime_orchestrator/contract.yaml
- ModelLifecycleHooks
- ServiceNodeIntrospection
- validation
- OMN-15092.yaml
- OMN-15484.yaml
- OMN-15642.yaml
- OMN-16343.yaml
- OMN-18353.yaml
- OMN-18638.yaml
- OMN-18656.yaml
- OMN-19312.yaml
- OMN-19802.yaml
- OMN-19925.yaml
- OMN-20339.yaml
- OMN-22478.yaml
- OMN-8468.yaml
- redpanda-local-sasl.yaml
- hardcoded_env
- omninode-runtime
- environment
- environment
- runtime_lane_env_overlay.py
- steps
- steps
- steps
- adapter_project_tracker_linear.py
- qdrant/__init__.py
- check_dev_candidate_delivery_liveness.py
- JobState
- VerificationResult
- gateway_control_plane.py
- recompute_measured_pricing.py
- release_drift_monitor_lib.py
- run_baselines_batch_compute.py
- validate_clean_root.py
- StoreLaneCredential
- HandlerBrokerPrincipalVerify
- HandlerDeclaredStateDiff
- build_shared_listener
- HandlerRegistrationStorageMock
- ModelBindingResolutionMetrics
- ModelRuntimeFoldPolicy
- node_architecture_validator/contract.yaml
- PublishingConstraintVisitor
- DLQConsumer
- HandlerMergeProvenanceObserve
- correlation_id
- FactoryObservabilitySink
- registry_infra_ledger_write.py
- build_core_runtime
- OMN-10278.yaml
- OMN-13419.yaml
- OMN-13445.yaml
- OMN-13554.yaml
- OMN-13713.yaml
- OMN-15310.yaml
- OMN-15358.yaml
- OMN-15741.yaml
- OMN-16171.yaml
- OMN-16367.yaml
- OMN-16769.yaml
- OMN-16788.yaml
- OMN-16821.yaml
- OMN-16926.yaml
- OMN-16947.yaml
- OMN-16961.yaml
- OMN-17079.yaml
- OMN-17106.yaml
- OMN-17135.yaml
- OMN-17137.yaml
- OMN-17152.yaml
- OMN-17166.yaml
- OMN-17221.yaml
- OMN-17277.yaml
- OMN-17296.yaml
- OMN-17317.yaml
- OMN-17379.yaml
- OMN-17395.yaml
- OMN-17398.yaml
- OMN-17432.yaml
- OMN-17443.yaml
- OMN-17447.yaml
- OMN-17485.yaml
- OMN-17648.yaml
- OMN-17664.yaml
- OMN-17678.yaml
- OMN-17792.yaml
- OMN-17855.yaml
- OMN-17862.yaml
- OMN-17886.yaml
- OMN-17888.yaml
- OMN-17893.yaml
- OMN-17949.yaml
- OMN-17981.yaml
- OMN-18124.yaml
- OMN-18134.yaml
- OMN-18266.yaml
- OMN-18272.yaml
- OMN-18296.yaml
- OMN-18411.yaml
- OMN-18693.yaml
- OMN-19315.yaml
- cloud-migration-files.yaml
- hardcoded_env
- environment
- gateway-attach-test-lane
- x-runner-base
- environment
- healthcheck
- env
- steps
- env
- steps
- StoreKeyedMemory
- run_k1_k6_delegation_evidence.py
- CutoverCoordinator
- .get_widget_mapping
- check_lockfile_cve.py
- run_validators_in_process.py
- pr_body.py
- EnumReplayStatus
- _run_errors
- ModelTracingContext
- ModelValidationOutcome
- node_node_graph_reducer/contract.yaml
- handlers
- validations
- ModelUpdatePlanState
- ModelNetworkPoolStatus
- LifecycleHookExecutor
- ConsumerMetrics
- StoreSnapshotInMemory
- findings_for_module
- pr_ci_zombie_detector.py
- OMN-10089.yaml
- OMN-10123.yaml
- OMN-10207.yaml
- OMN-10281.yaml
- OMN-10332.yaml
- OMN-10334.yaml
- OMN-10338.yaml
- OMN-10340.yaml
- OMN-10341.yaml
- OMN-10342.yaml
- OMN-10343.yaml
- OMN-10345.yaml
- OMN-10348.yaml
- OMN-10382.yaml
- OMN-10387.yaml
- OMN-10388.yaml
- OMN-10392.yaml
- OMN-10445.yaml
- OMN-10447.yaml
- OMN-10448.yaml
- OMN-10449.yaml
- OMN-10450.yaml
- OMN-10453.yaml
- OMN-10489.yaml
- OMN-10500.yaml
- OMN-10518.yaml
- OMN-10519.yaml
- OMN-10530.yaml
- OMN-10532.yaml
- OMN-10534.yaml
- OMN-10563.yaml
- OMN-10586.yaml
- OMN-10587.yaml
- OMN-10635.yaml
- OMN-10782.yaml
- OMN-10784.yaml
- OMN-10786.yaml
- OMN-10790.yaml
- OMN-10796.yaml
- OMN-10864.yaml
- OMN-10868.yaml
- OMN-10943.yaml
- OMN-11050.yaml
- OMN-11067.yaml
- OMN-11068.yaml
- OMN-11069.yaml
- OMN-11073.yaml
- OMN-11095.yaml
- OMN-11104.yaml
- OMN-11108.yaml
- OMN-11130.yaml
- OMN-11197.yaml
- OMN-11241.yaml
- OMN-11246.yaml
- OMN-11248.yaml
- OMN-11315.yaml
- OMN-11318.yaml
- OMN-11319.yaml
- OMN-11320.yaml
- OMN-11350.yaml
- OMN-11673.yaml
- OMN-12184.yaml
- OMN-12503.yaml
- OMN-12513.yaml
- OMN-12522.yaml
- OMN-12536.yaml
- OMN-12559.yaml
- OMN-12597.yaml
- OMN-12627.yaml
- OMN-12631.yaml
- OMN-12633.yaml
- OMN-12682.yaml
- OMN-12700.yaml
- OMN-12723.yaml
- OMN-12755.yaml
- OMN-12827.yaml
- OMN-12832.yaml
- OMN-12841.yaml
- OMN-12848.yaml
- OMN-12860.yaml
- OMN-12912.yaml
- OMN-12940.yaml
- OMN-13008.yaml
- OMN-13011.yaml
- OMN-13109.yaml
- OMN-13141.yaml
- OMN-13149.yaml
- OMN-13190.yaml
- OMN-13191.yaml
- OMN-13201.yaml
- OMN-13203.yaml
- OMN-13237.yaml
- OMN-13238.yaml
- OMN-13247.yaml
- OMN-13403.yaml
- OMN-13405.yaml
- OMN-13417.yaml
- OMN-13532.yaml
- OMN-13533.yaml
- OMN-13546.yaml
- OMN-13581.yaml
- OMN-13654.yaml
- OMN-13656.yaml
- OMN-13664.yaml
- OMN-13666.yaml
- OMN-13712.yaml
- OMN-13719.yaml
- OMN-13725.yaml
- OMN-13726.yaml
- OMN-14115.yaml
- OMN-14953.yaml
- OMN-14993.yaml
- OMN-15009.yaml
- OMN-15330.yaml
- OMN-15384.yaml
- OMN-15807.yaml
- OMN-15837.yaml
- OMN-15838.yaml
- OMN-16056.yaml
- OMN-16107.yaml
- OMN-16418.yaml
- OMN-16420.yaml
- OMN-16661.yaml
- OMN-19532.yaml
- OMN-19741.yaml
- OMN-20568.yaml
- OMN-20600.yaml
- OMN-20602.yaml
- OMN-7841.yaml
- OMN-8065.yaml
- OMN-8701.yaml
- OMN-9159.yaml
- OMN-9211.yaml
- OMN-9471.yaml
- OMN-9532.yaml
- OMN-9552.yaml
- OMN-9553.yaml
- OMN-9554.yaml
- OMN-9648.yaml
- OMN-9877.yaml
- OMN-9905.yaml
- redpanda-partition-cap.yaml
- migration-gate
- x-runner-base
- projection_apply_flow.py
- steps
- canonical-inference-gate.yml
- env
- steps
- legacy-rds-fixture-proof.yml
- pr-title
- precommit-parity-gate.yml
- with
- steps
- validate-validator-requirements.yml
- validator-runtime-profiles.yml
- Any
- CorpusCapture
- ProtocolDiscoveryOperations
- gate/executor.py
- EnumNodeOutputType
- ModelGateBindingFixtures
- deploy_agent_window.py
- declared_supersession.py
- bounded_delegation_routes.py
- delegation_health_check.py
- post_release_dev_bump.py
- resolve_node_migration_source_ref.py
- build_plan
- onex-cli.sh
- sync-topic-registry.py
- system_health_check.sh
- check_tenant_guc_domain_parity.py
- validate_db_quality_gate.py
- venv_install_plan.py
- build_events_at
- ModelDlqMetrics
- ModelFileSystemRequest
- handler_ledger_write_guard.py
- ModelLlmEndpointOverlay
- RowCountProbe
- ModelGeneratedComputeNodeSpec
- ModelExecutionShapeValidationResult
- HandlerDelegationChainLedger
- HandlerMergeProvenance
- node_registration_orchestrator/contract.yaml
- properties
- response_fields
- HandlerScopeCheckInitiate
- InvocationSecurityEnforcer
- ProjectorShellPlaceholder
- RegistryMessageType
- ModelCostApiBase
- ServiceRegistryDiscovery
- util_dlq_credential_redaction.py
- OMN-10081.yaml
- OMN-10137.yaml
- OMN-10168.yaml
- OMN-10169.yaml
- OMN-10204.yaml
- OMN-10237.yaml
- OMN-10238.yaml
- OMN-14060.yaml
- OMN-1419.yaml
- OMN-9215.yaml
- OMN-9410.yaml
- OMN-9420.yaml
- OMN-9550.yaml
- OMN-9556.yaml
- OMN-9557.yaml
- OMN-9600.yaml
- OMN-9727.yaml
- c12_provider_catalogue_observe.py
- ci
- hardcoded_env
- com.omninode.lane
- environment
- com.omninode.lane
- omninode-dns-cache
- environment
- environment
- hp053-ci-runner-2
- omninode-prod-deploy-runner-1
- runtime-worker
- runner-job-started.sh
- contract_resolver/routes.py
- steps
- with
- non-dev-base-guard.yml
- with
- deliver_reports
- capture_delegation_evidence.py
- check-pinned-wheels.py
- generate_claude_lane_block.py
- .run
- runner_fleet_event.py
- ModelGatewayRenewalDirective
- HandlerSourceAttestation
- timeouts
- ServiceFlakeDetector
- node_contract_loader_effect/contract.yaml
- ModelDlqMessage
- ProtocolKafkaReplayConsumer
- ModelRegistryIntent
- error_handling
- node_service_discovery_effect/contract.yaml
- node_slack_alerter_effect/contract.yaml
- steps
- ModelOptionalCorrelationId
- intelligence_hook_event_v1.yaml
- ConsumerMetrics
- stack_shape.yaml
- omnibase_core
- strip_runtime_entry_points.py
- OMN-16322.yaml
- autoheal.yaml
- hardcoded_env
- hardcoded_env
- projection-tenant-registry-writer
- redpanda-sasl-enable
- environment
- keycloak
- volumes
- volumes
- docker-compose.stability-test.yml
- HandlerMergeGroupRunReadGithub
- lane_census_event.py
- router.py
- resolve-arm64-proof-legs/action.yml
- with
- with
- route
- delegation-health-check.yml
- main-target-guard.yml
- steps
- no-bare-compose-teardown
- inputs
- with
- .create_app
- ContractCapabilityExtractor
- build_and_start_core_runtime
- check_dispatcher_route_coverage.py
- check_integration_skips.py
- verify_pypi_pin_resolvability.py
- worker_cycle.py
- prepush_remote_verify.py
- lint_compose_dangling_deps.py
- monitor_logs.py
- detect_foreign_prepush.py
- ModelCiBusLane
- AdapterLlmToolProvider
- HandlerDeclaredStateApply
- HandlerDeclaredStateReconcile
- HandlerDeployRunners
- HandlerVenvAtomicInstall
- MCPServerLifecycle
- properties
- properties
- local_bundle
- properties
- overseer_tick_projector.yaml
- ProtocolDoneWriteReceiptQueries
- ModelConfigRef
- SourceContractComposite
- SessionSnapshotStore
- entries
- ConfigSessionStorage
- update_env_file
- shared
- OMN-12245.yaml
- OMN-13350.yaml
- OMN-13531.yaml
- OMN-16293.yaml
- OMN-18326.yaml
- coordination_rules
- workflow_coordination
- hardcoded_env
- forward-migration
- environment
- gate-runner
- environment
- forward-migration
- runtime-effects
- environment
- docker-compose.runners-hp051-ci-runner.yml
- no-cache
- with
- with
- steps
- with
- with
- steps
- probe-placement-drift.yml
- seed-provenance-check
- wheel-content-parity.yml
- ModelCaptureResult
- http/__init__.py
- handler_rsd_score_calculate.py
- assert_image_config_paths.py
- contract_topic_graph.py
- validate-kafka-schema-handshake.py
- emit_baselines_raw_snapshot.py
- desired_state.py
- Any
- record_down_migration_execution.py
- check_cli_registry_vocabulary.py
- check_topic_literals.py
- integrations
- EnumIntrospectionReason
- HandlerDeclaredStateLeaseLocal
- HandlerCodingAgentOrchestrator
- HandlerRepoStateCollect
- properties
- node_bus_forwarder_effect/contract.yaml
- settings
- ._process_message
- node_event_bus_wiring_effect/contract.yaml
- node_gmail_archive_cleanup_effect/contract.yaml
- properties
- ModelLifecycleState
- coordination_rules
- verification_receipt_projector.yaml
- ProtocolMessageTypeRegistry
- ActionAuthorizationClaimUnixRpc
- MetricCollector
- ServiceBatchComputeEffectivenessMetrics
- FileSpoolLedgerSink
- PublisherContractCorpus
- Added
- StoreIdempotencyInmemory
- console_scripts
- endpoints
- OMN-10079.yaml
- OMN-10615.yaml
- OMN-10616.yaml
- OMN-10637.yaml
- OMN-10638.yaml
- OMN-10657.yaml
- OMN-10712.yaml
- OMN-10728.yaml
- OMN-10741.yaml
- OMN-10755.yaml
- OMN-10780.yaml
- OMN-10792.yaml
- OMN-10794.yaml
- OMN-10797.yaml
- OMN-10814.yaml
- OMN-10867.yaml
- OMN-10972.yaml
- OMN-11061.yaml
- OMN-11131.yaml
- OMN-11161.yaml
- OMN-11162.yaml
- OMN-11163.yaml
- OMN-11185.yaml
- OMN-11196.yaml
- OMN-11200.yaml
- OMN-11201.yaml
- OMN-11242.yaml
- OMN-11244.yaml
- OMN-11247.yaml
- OMN-11299.yaml
- OMN-12815.yaml
- OMN-13037.yaml
- OMN-13362.yaml
- OMN-13404.yaml
- OMN-13406.yaml
- OMN-13507.yaml
- OMN-13530.yaml
- OMN-14974.yaml
- OMN-16994.yaml
- OMN-8131.yaml
- OMN-9120.yaml
- OMN-9139.yaml
- OMN-9266.yaml
- OMN-9529.yaml
- OMN-9713.yaml
- OMN-9769.yaml
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- omninode_runtime_service
- omnibase_infra
- hardcoded_env
- hardcoded_env
- projection-delegation-writer
- forward-migration
- projection-savings-writer
- alerter-injected-red-canary.yml
- hostile-review-thread-gate
- with
- with
- model_lab_proof_match.py
- ModelLabProofProfileRegistry
- ModelProjectorDiscoveryResult
- dogfood_delegation_fault_routes.py
- CountingTransportConsumer
- check_workflow_uv_available.py
- compute_golden.py
- runner_image_identity.py
- adjacency
- lane_census_refresh_decision.py
- lane_scratch.py
- RuntimeErrorEmitter
- ._maybe_warning_alert
- sibling_clone_manifest.sh
- BaseModel
- validator_healthcheck_semantic_probe.py
- BrokerSession
- ConcurrentCommitLedger
- TopicStartupValidator
- ModelWorkspaceValidateCommand
- ModelDispatchOutcome
- ModelLaneCensusRegistry
- ModelMessageTypeEntry
- gateway_forwarder
- effect_operations
- HandlerPostgresTopicUpdate
- node_gmail_intent_poller_effect/contract.yaml
- HandlerKafkaReplay
- ModelBackendResult
- properties
- ModelBufferedLogEntry
- CredentialsWriter
- ProtocolPlacementSession
- ProtocolRegistryMetrics
- util_semver.py
- OverlayFileLoader
- MixinMessageTypeQuery
- ContractSourceNotConfiguredError
- ProtocolSessionAggregator
- principals
- omninode_runtime_service
- omniintelligence
- principals
- application
- principals
- application
- pattern_exemptions
- LifespanDriver
- runner_routing_policy.yaml
- ModelOperationBindingsSubcontract
- OMN-10048.yaml
- OMN-10122.yaml
- OMN-9620.yaml
- OMN-9629.yaml
- OMN-9637.yaml
- OMN-9646.yaml
- OMN-9728.yaml
- redpanda-scram-user
- omninode-runtime
- redpanda-partition-cap
- omninode-runtime
- omninode-runtime
- projection-registration-writer
- runtime-effects
- projection-tenant-credentials-writer
- chain-canary
- sibling-compat
- env
- drill_red_until
- env
- with
- runner-label-exists-caller.yml
- service_keycloak_token_validator.py
- ModelHealthCheckResult
- cli_profile.py
- assemble_delegation_cohort_key.py
- handler_dev_boot_guard.py
- EnumComposeMountType
- HandlerFaultInjectFixture
- node_llm_embedding_effect/__init__.py
- handler_scope_extract.py
- db_table_validator.py
- hooks
- check_reconciler_privilege.py
- check_probe_placement_drift.py
- ci_summary_gate.py
- service_suites_skip_guard.yaml
- lane_lock
- infra-signature-rerun.sh
- provision_db_slot.sh
- release_drift_monitor.py
- check_migration_rls_policy_atomicity.py
- ModelActionAuthorizationClaimRequest
- settings
- processing.py
- ModelEvalFinding
- HandlerBrokerGrantDerive
- HandlerGitFixVerifier
- ModelSecretCache
- fields
- HandlerPlacementEtcd
- StoreIdempotencySqlite
- ModelCodingAgentFsmState
- ModelBifrostLaneBackendBinding
- RuleNoDirectDispatch
- OrchestratorFSMVisitor
- _LinearClient
- AdapterPipelineAlertBridge
- ProtocolEventProjector
- ProtocolPayloadRegistry
- ModelRegistryNodeView
- find_defects
- tenant_scoped_ingress_schema.py
- entries
- overrides
- overrides
- overrides
- contract_registry_reducer.yaml
- node_graph_reducer.yaml
- runtime_orchestrator.yaml
- integration_testing
- local
- projection-registration-writer
- args
- x-lane-overlay-mount
- volumes
- environment
- gates
- call-reject-skip.yml
- with
- env
- deploy-gate
- inputs
- env
- with
- required-check-skip-guard-caller.yml
- env
- ServiceEnvelopeSigner
- delegate_caller.py
- onboarding/loader.py
- workflow_receipt/__init__.py
- check_runner_fleet_image_drift.py
- check_workflow_input_default_types.py
- dependabot_lock_refresh.py
- check_migration_applied_on_lane.py
- Event
- lane_lock.sh
- lane_rollback_decision.py
- integration_config.yaml
- _git_sha_at
- EnumBifrostLaneMediaCapability
- EnumContractType
- HandlerCILiveContact
- ModelDeclaredStateReconcileRequest
- HandlerDevBootGuard
- ModelWorkspaceConfigRootRequest
- HandlerPlacementElection
- HandlerPlacementSession
- RoutingGate
- ModelDispatcherRegistration
- ModelSequenceInfo
- MixinMessageTypeRegistration
- properties
- RuleNoOrchestratorFSM
- DirectDispatchVisitor
- service_gateway_egress_health.py
- HandlerCIRunnerRoute
- HandlerPostgresHeartbeat
- effect
- max_per_label
- node_llm_inference_effect/contract.yaml
- EnumTaskType
- circuit_breaker
- HandlerRsdScoreCalculate
- integration_testing
- event_bus
- workflow_coordination
- EnumServiceDiscoveryOperation
- ModelWasteFinding
- condition_evaluator.py
- BindingExpressionParseError
- ModelConfigProvenance
- ModelPolicyTypeFilter
- venv_purity.py
- ModelDiscoveredContract
- SourceContractPackage
- ServiceEffectivenessInvalidationNotifier
- ConfigLlmCostAggregation
- UUID
- Added
- compose-dev
- repository_overrides
- key_metrics
- metadata
- volumes
- options
- options
- environment
- environment
- runtime-effects
- services
- env
- jobs
- with
- with
- imperative-contract-guard.yml
- ModelCaptureConfig
- ModelDevLaneBranchInstall
- chain_replay.py
- ModelLabComposeContainer
- merge_provenance.py
- registry_infra_registration_storage.py
- store.py
- check_version_pins.py
- check_auto_merge_hold.py
- main
- _Resolver
- merge_queue_janitor.py
- runtime_boot_suites_skip_guard.yaml
- credential_loader.py
- verify_dev_refresh.py
- CapabilityInferenceRules
- artifact_reconcile_cmd
- StoreOnexHomeFiles
- scan.py
- EnumAnyTypeViolation
- EnumHandlerLoaderError
- EnumKafkaPermanentFailure
- ModelVersionKeyedCache
- HandlerPlacementEpochGuard
- HandlerRuntimeTargetCollect
- ModelHandlerContract
- ModelCoverageMetrics
- node_board_probe_effect/contract.yaml
- HandlerQueryDecisions
- DLQQuarantineProducer
- HandlerEventForward
- retry_policy
- input_fields
- handlers
- retry_policy
- retry_policy
- properties
- artifacts
- projection_dispatch_ledger.py
- PublisherTopicScoped
- ModelPublishResult
- first_party_sources.yaml
- EnumAuthSource
- model_bifrost_lane_media_capability_error.py
- contract_loader_effect.yaml
- recommended_actions
- effect
- nodes
- scenarios
- validations
- bundles.yaml
- schemas
- schemas
- schemas
- schemas
- schemas
- schemas
- schemas
- schemas
- schemas
- hardcoded_env
- projection-claude-hook-events-writer.yaml
- environment
- docker-compose.dev-105.yml
- docker-compose.dev-202.yml
- redpanda-partition-cap
- volumes
- volumes
- environment
- interval
- bind
- volumes
- runners/entrypoint.sh
- with
- with
- with
- with
- inputs
- outputs
- env
- verify-sibling-converged
- scrape_configs
- validator_healthcheck_start_period.py
- graph/__init__.py
- EnumProjectionOrderingDirection
- ModelRunnerFleetSnapshot
- ModelRunnerFleetHealthVerdict
- NodeValidationLedgerWriteEffect
- FakeValkeyClient
- check_dashboard_topic_parity.py
- check_lane_census_drift.py
- check_schema_fingerprint.py
- ModelQueueFacts
- SweepExclusion
- loaded_code.py
- emit_repowise_freshness_receipt.py
- PostgresErrorEmitter
- _gh_api
- HealthGateReport
- smoke_delegation.sh
- AdapterTestBoilerplateGeneration
- DemoResetEngine
- refresh_interval_seconds
- filesystem/handler_contract.yaml
- descriptor
- mcp/handler_contract.yaml
- EnumCaptureState
- EnumChainViolationType
- CircuitBreakerEventPublisher
- PayloadRegistryError
- .protect
- .apply
- HandlerReleaseNotesApply
- MixinSharedListener
- ProtocolIdempotencyStore
- ModelContractHotloadWorkloadTopic
- ModelTopicBootBaseline
- ModelLlmUsage
- timeouts
- ModelProjectorColumn
- ModelArchitectureValidationResult
- handlers
- node_coding_agent_orchestrator/contract_topics.py
- node_contract_persistence_effect/contract.yaml
- gateway_attach
- step_timeouts
- handlers
- HandlerProbeHealth
- nodes
- ModelValidationResult
- ModelValidationResult
- nodes
- scenarios
- metadata
- ModelSavingsCounterfactualPolicy
- runtime_overlay_revision.py
- node_validation_adjudicator/contract.yaml
- node_validation_ledger_write_effect/contract.yaml
- node_validation_orchestrator/contract.yaml
- .__exit__
- ProtocolIdempotencyStore
- ProtocolLedgerSink
- ProtocolPlacementElection
- ModelActionAuthorizationClaimResult
- ProtocolProjectorNotificationContext
- ModelMaterializedResources
- DecisionSearchClient
- schemas
- instances/onex-dev.yaml
- schemas
- instances/onex-prod.yaml
- schemas
- OMN-17427.yaml
- response_fields
- effects
- omninode-runtime.yaml
- environment
- gateway-forwarder
- omnibase-infra-memgraph
- redpanda-partition-cap
- options
- omninode-runtime
- cloud_bus
- forwarder
- steps
- with
- chain-canary-c16-receipt-identity.yml
- steps
- inputs
- run_full_tests
- evidence-autoclose-sweep.yml
- env
- with
- env
- public-repo-hygiene.yml
- env
- outputs
- ModelArchitectureValidationRequest
- EnumRunnerReadinessSignal
- analyze_high_output
- ModelWasteDetectionInput
- generate_onboarding_docs.py
- service_llm_usage_normalizer.py
- check_contract_topic_parity.py
- check_deployed_migration_tree_sync.py
- check_wheel_content_parity.py
- run_compliance_sweep.py
- ProtocolDelegationChannel
- run_worker_loop
- main
- transport
- EnumLifecycleTier
- InmemoryReadbackSource
- ModelDeleteResult
- ModelLedgerEventBase
- ModelPreprPoolOverlay
- ModelSnapshotDiff
- RuleNoHandlerPublishing
- node_boot_catalog_effect/contract.yaml
- thresholds
- RegistryInfraContractResolverBridge
- ModelDeclaredChainHop
- handlers
- node_fault_inject_fixture_compute/contract.yaml
- LinearHygieneTransport
- InvariantEvaluator
- handlers
- handlers
- output_fields
- retry_policy
- handlers
- ModelRoutingCommand
- thresholds
- node_registration_storage_effect/contract.yaml
- ModelRegistrationUpdate
- config
- RegistryInfraRegistryApiEffect
- handlers
- ModelRsdState
- node_savings_estimation_compute/contract.yaml
- node_setup_local_provision_effect/contract.yaml
- ConfigWriter
- ModelOptionalUUID
- _merge_lane_overlay
- ModelRoutingDecisionIngest
- LedgerSinkInjectionEffectivenessPostgres
- InMemoryLedgerSink
- llm_registry_validator.py
- verification/contract.yaml
- replay_and_diff
- allowed
- forbidden
- bandit-security
- detect-secrets
- mypy-type-check
- ruff-format-check
- nonrequired_check_alert
- operations
- automatic_recovery
- operations
- event_bus_wiring_effect.yaml
- operations
- runtime_policy.contract.yaml
- keep-list.yaml
- bindings
- local-market
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- omniintelligence
- models
- contract-resolver.yaml
- runtime-effects.yaml
- ci-bus-sasl-enable
- redpanda
- volumes
- redpanda
- infisical
- volumes
- redpanda
- volumes
- redpanda
- interval
- interval
- dogfood_delegation_fault_provider.py
- env
- advisory-job-gate.yml
- env
- inputs
- with
- inputs
- env
- base_url
- steps
- env
- env
- jobs
- kb-doc-gate.yml
- sync_main_to_tag
- jobs
- outputs
- precommit_diff_scope.py
- runtime_skill_client.py
- rules
- contract_capability_extractor.py
- EnumBifrostLaneOverlaySource
- service_discovery/models/model_discovery_result.py
- ModelLabProofNegativeControl
- projection_liveness.py
- handler_lane_entry_point_census.py
- node_kafka_replay_compute/protocols/__init__.py
- _evaluate_alerts
- node_validation_executor/__init__.py
- ModelBifrostLaneBackendCredential
- EnumBifrostLanePlacementMode
- ModelTopicView
- check_context_field_presence.py
- evaluate_external_sweep
- run_on_gate_runner.sh
- default_out_path
- main
- IntentTypeExtractor
- ModelDemoResetReport
- args
- StoreDeveloperProfile
- models
- compute_cost
- widget_mapping.yaml
- capability_mappings
- graph/handler_contract.yaml
- bindings
- security
- EnumRetryErrorCategory
- ProtocolKafkaBroadcastHost
- TopicViolationAlerter
- ModelIdempotencyStoreMetrics
- ModelBifrostLaneEndpointCapacity
- ModelRRHResult
- HandlerToolchainCollect
- ModelCheckCatalogEntry
- ModelExecutorResult
- ModelArchitectureValidationRequest
- validation_rules
- node_artifact_reconciliation_orchestrator/contract.yaml
- state_transitions
- node_ci_runner_route_compute/contract.yaml
- config
- handlers
- version
- handlers
- node_contract_registry_reducer/contract.yaml
- node_db_error_linear_effect/contract.yaml
- node_decision_store_effect/contract.yaml
- handlers
- ModelObservedHop
- node_delegation_chain_ledger_effect/contract.yaml
- node_delta_bundle_effect/contract.yaml
- handlers
- node_delta_metrics_effect/contract.yaml
- handlers
- EnumDlqReplayFilterType
- ModelGmailCleanupConfig
- ModelGmailIntentPollerConfig
- node_lab_proof_plan_compute/contract.yaml
- node_merge_gate_effect/contract.yaml
- operations
- handlers
- step_config
- io_operations
- ModelRsdScoreInput
- ModelDiscoveryResult
- handlers
- ModelSlackAlert
- handlers
- handlers
- handlers
- responses
- ProtocolProjectorContext
- ModelRuntimeNodeGraph
- ModelOptionalString
- bindings
- bindings
- required_validators
- banned-compose-env-vars
- branch-protection-rollout-guard
- enum-governance
- handler-di
- hardcoded-local-paths
- hardcoded-model-config
- hardcoded-topic-compute
- naming-conventions
- no-hardcoded-topics
- no-io-outside-effects
- no-raw-sqlite3
- no-untracked-todos
- pydantic-patterns
- self-gating-workflows
- single-class-per-file
- spdx-headers
- stub-implementations
- tenant-scoped-ingress
- probes
- executions
- version
- extract_fields
- state_transitions
- io_config
- version
- planned_subscribe_topics
- states
- state_transitions
- version
- extra_depends_on
- application
- application
- application
- application
- application
- application
- application
- application
- application
- onex-api.yaml
- services
- ci-bus-labwork-user
- ci-bus-scram-user
- interval
- interval
- services
- interval
- environment
- hp051-ci-mq-runner-1
- hp051-ci-mq-runner-2
- hp051-ci-mq-runner-3
- hp051-ci-mq-runner-4
- hp053-ci-runner-1
- hp053-ci-runner-3
- hp053-ci-runner-4
- hp053-ci-runner-5
- omninode-deploy-runner
- omninode-verify-runner-1
- omninode-verify-runner-2
- omninode-verify-runner-3
- backends
- env
- env
- build-and-push
- outputs
- steps
- env
- env
- compute-matrix
- with
- jobs
- docs-validate.yml
- lockfile-cve-autobump.yml
- with
- post-release-checks
- env
- eval_regression_check.py
- ModelDelegationRetryBounds
- model_node_flow_window.py
- HandlerRsdInitiate
- model_cross_repo_violations_batch_event.py
- RegistryInfraServiceDiscovery
- MCPAuthMiddleware
- venv_reconcile_lock.sh
- xml_etree_elementtree
- evaluate
- gateway_restart_safety_proof.sh
- git-maintenance.sh
- _cooldown_read
- resolve_omnimarket_pins.py
- trigger-deploy.sh
- db/handler_contract.yaml
- descriptor
- http/handler_contract.yaml
- descriptor
- handler_routing
- deep_dive/models.py
- get_github_merged_prs
- EnumLedgerSinkDropPolicy
- build_aiokafka_auth_kwargs_from_env
- ModelReverseDeltaProof
- ModelTopicCatalogChanged
- ModelWastePremiumModelPolicy
- ModelProjectorIndex
- endpoints
- descriptor
- RuleContractDeclaredOrchestratorWorkflow
- handlers
- retry_policy
- handlers
- handlers
- retry_policy
- retry_policy
- retry_policy
- handlers
- io_config
- version
- topics
- retry_policy
- retention_days
- retry_policy
- retry_policy
- node_impact_analyzer_compute/contract.yaml
- node_lane_entry_point_census_effect/contract.yaml
- node_ledger_write_effect/contract.yaml
- ModelLlmEmbeddingResponse
- states
- state_transitions
- node_placement_runtime_effect/contract.yaml
- coordination_rules
- ModelTimeoutCoordinationResult
- state_machine
- node_registry_api_effect/contract.yaml
- metadata
- handlers
- ModelRsdDataFetchResult
- handlers
- HandlerRunnerFleetMaintainStart
- retry_policy
- retry_policy
- handlers
- tags
- retry_policy
- retry_policy
- node_update_plan_reducer/contract.yaml
- handlers
- node_validation_executor/contract.yaml
- handlers
- node_validation_ledger_projection_compute/contract.yaml
- input_fields
- retry_policy
- model_network_pool_alert.py
- MixinConsumptionCounter
- ProtocolAutoWiringManifestLike
- ProtocolProjectorSchemaValidator
- CoreRuntimeHandle
- CoreTransport
- MixinTypedContractEvents
- ModelFailedComponent
- ProtocolContractPublisherSource
- managed_staging_canary_catalog_namespace.yaml
- object_type
- application
- omnibase_infra
- services
- object_type
- omnibase_infra
- omniintelligence
- object_type
- omnibase_infra
- omniintelligence
- build_local_claim_interface
- validator-requirements-baseline.yaml
- git_mirror
- hosts
- effect_operations
- transitions
- major
- wiring_config
- transitions
- major
- workflow_metadata
- foldable_services
- lab/developer-principal-grants.yaml
- local/developer-principal-grants.yaml
- source
- source
- source
- source
- source
- source
- source
- source
- source
- operational_defaults
- runtime-worker.yaml
- tenant-projection-writer.yaml
- redpanda-partition-cap
- volumes
- migration-gate
- valkey
- gateway-runtime
- valkey
- migration-gate
- valkey
- resources
- valkey
- omninode-runtime
- redpanda-partition-cap
- hp051-ci-runner-1
- hp051-ci-runner-2
- hp051-ci-runner-3
- hp051-ci-runner-4
- hp053-ci-mq-runner-1
- hp053-ci-mq-runner-2
- hp053-ci-mq-runner-3
- hp053-ci-mq-runner-4
- interval
- bind
- omnipc2-ci-runner-1
- omnipc2-ci-runner-2
- omnipc2-ci-runner-3
- omnipc2-ci-runner-4
- omnipc2-ci-runner-7
- omnipc2-ci-runner-8
- omnipc2-ci-runner-9
- docker-compose.sim-202.yml
- omninode-runtime
- legacy-rds-fixture/prove.sh
- resolve-ecr-digest/action.yml
- inputs
- no-cache
- chain-canary-c12-provider-catalogue.yml
- jobs
- candidate-boot-gate
- env
- with
- security-scan
- hostile-review-judge.yml
- hostile-review-judge
- steps
- with
- with
- package
- env
- execute
- boot-catalog-local
- inputs
- _execute_shell_check
- ModelFeatureFlagContract
- ModelCIRunnerLabObservation
- EnumBifrostEndpointProbeFailureKind
- check-env-reads.sh
- check_lane_attestation.py
- ProbeInputError
- instance_receipt_lanes.py
- pr_trailers.py
- RestartWatcher
- pull-all.sh
- semantic_mappings
- security
- strategies
- pinned_model_identity
- client
- ModelConsumerCoordinatorRejoinDecision
- ProtocolDevRollEvents
- handler_manifest_persistence.contract.yaml
- optional
- ModelPostgresIdempotencyStoreConfig
- ModelDelegateNamedTests
- ModelDelegationCohortKey
- ModelCreateTopicRefusal
- ModelOperationalTimeouts
- ModelOutboundTimeouts
- ModelThirdPartyEndpoints
- handlers
- handlers
- workflow_metadata
- retry_policy
- board_check_coverage.yaml
- retry_policy
- retry_policy
- node_build_loop_write_effect/contract.yaml
- retry_policy
- liveness
- retry_policy
- .handle
- transitions
- state_machine
- node_contract_resolver_bridge/contract.yaml
- metadata
- retry_policy
- wiring_config
- retry_policy
- node_hostile_review_fix_verifier_effect/contract.yaml
- retry_policy
- ModelKafkaReplayInput
- retry_policy
- retry_policy
- node_ledger_projection_compute/contract.yaml
- transitions
- major
- node_onex_api_image_build_effect/contract.yaml
- retry_policy
- node_onex_api_image_import_check_effect/contract.yaml
- retry_policy
- node_pattern_lifecycle_effect/contract.yaml
- retry_policy
- handlers
- handlers
- ModelIntentExecutionResult
- node_registration_reducer/contract.yaml
- transitions
- columns
- node_runner_fleet_maintain_orchestrator/contract.yaml
- handlers
- published_events
- published_events
- handlers
- handlers
- workflow_metadata
- retry_policy
- EnumScopeCheckStatus
- ModelRegistrationResult
- transitions
- state_machine
- retry_policy
- RegistryInfraSetupLocalProvisionEffect
- handlers
- transitions
- state_machine
- transitions
- state_machine
- retry_policy
- output_fields
- workflow_metadata
- retry_policy
- handlers
- write_config_hybrid
- set_state
- ModelLifecycleHookResult
- ModelLifecycleHookResult
- ModelBindingConfigCacheStats
- ModelBindingConfigResolverConfig
- ModelLocalRuntimeIngressConfig
- overlay_from_env_dict
- ProtocolContractScopedDispatchEngine
- event_bus
- ModelEnforcementScope
- ModelAgentAction
- session_projector/projector.py
- LedgerSinkError
- FakeSecretStore
- ModelRsdPinnedModelIdentity
- c11_delegation_legs.yaml
- evidence_artifacts
- sources
- machine_classes
- runner_fleet.yaml
- activation_gate
- states
- checks
- testing_guidance
- metadata
- main
- bindings
- sources
- local-lane-mirror
- dlq-replay-consumer.yaml
- fault-inject-fixture.yaml
- operational_defaults
- runtime-canary.yaml
- operational_defaults
- operational_defaults
- operational_defaults
- operational_defaults
- options
- environment
- dogfood-delegation-fault-429
- runtime
- depends_on
- redpanda-topic-manager
- healthcheck
- redpanda
- healthcheck
- healthcheck
- healthcheck
- volumes
- healthcheck
- volumes
- healthcheck
- volumes
- omninode-customer-plane-runner-1
- omninode-customer-plane-runner-2
- healthcheck
- omninode-studio-ci-runner-1
- omninode-studio-ci-runner-2
- omninode-studio-ci-runner-3
- omninode-studio-ci-runner-4
- redpanda-partition-cap
- 000_create_multiple_databases.sh
- tables
- outputs
- outputs
- digest
- steps
- chain-canary-c11-negative-paths.yml
- steps
- container
- steps
- chain-canary-c28-consumer-flow.yml
- pr_number
- delegation-health
- steps
- env
- dry_run
- nightly-integration.yml
- jobs
- steps
- verify-onex-api-delivered
- steps
- steps
- cut-tag
- cut
- repos
- dry_run
- route
- force
- jobs
- reemit-queued-receipts
- rerun-refused-deliveries
- steps
- secrets
- handler_placement_rebuild.py
- ModelLabSourceRef
- check_seed_provenance.py
- ci_check_contract_discovery.py
- enable-merge-queue.sh
- gather_repo_facts
- cut-lab-ref.sh
- cut_release_train_tag.sh
- AdapterPostgresConnection
- min_samples
- sample_window_days
- authoritative
- generated_at
- query
- sample_count
- defaults
- descriptor
- path_validation
- version
- http.get
- EnumVerdict
- EnumDeclarativeNodeViolation
- CoordinatorErrorLogGate
- ProtocolDeclaredStateLease
- ModelSubprocessOutcome
- circuit_breaker
- ._publish_and_await
- ModelCostDelta
- ModelOutcomeDelta
- evaluate_projection_liveness
- ModelRuntimeKernelDefaults
- node_artifact_change_detector_effect/contract.yaml
- handlers
- ModelBoardCheckCoverage
- handlers
- node_broker_principal_effect/contract.yaml
- handlers
- topic_sets
- ModelGatewayEgressMetadataScrub
- ModelGatewayEgressRedaction
- handlers
- node_chain_orchestrator/contract.yaml
- circuit_breaker
- error_handling
- handlers
- circuit_breaker
- error_handling
- handlers
- circuit_breaker
- error_handling
- circuit_breaker
- error_handling
- node_developer_principal_plan_compute/contract.yaml
- handlers
- handlers
- ownership
- ModelDlqCommitLedger
- checks
- handlers
- handlers
- node_gateway_attach_effect/contract.yaml
- handlers
- node_gateway_link_health_write_effect/contract.yaml
- handlers
- error_types
- published_events
- circuit_breaker
- error_handling
- handlers
- handlers
- testing_guidance
- node_reconcile_workspace_venvs/contract.yaml
- projection_reader
- published_events
- error_types
- dependencies
- handlers
- node_remote_agent_invoke_effect/contract.yaml
- handlers
- published_events
- node_routing_orchestrator/contract.yaml
- handlers
- handlers
- node_runtime_manifest_reducer/contract.yaml
- metadata
- node_scope_workflow_orchestrator/contract.yaml
- node_session_lifecycle_reducer/contract.yaml
- cost_estimator_build
- node_sibling_clone_pin_effect/contract.yaml
- handlers
- error_types
- nodes
- node_venv_atomic_install/contract.yaml
- ModelWasteCall
- node_workspace_config_root_resolve_effect/contract.yaml
- handlers
- interactive_onboarding.yaml
- ProtocolContractDescriptor
- ModelRegistryDiscoveryResponse
- IsinstanceUnionVisitor
- wire_shape.py
- [0.11.0] - 2026-02-25
- OmniNode Architecture – Constraint Map (omnibase_infra)
- dev
- omnibase_core
- omnibase_spi
- omniclaude
- omniintelligence
- omnimarket
- omnimemory
- probe_placement_variables
- dod_evidence
- security
- metadata
- dev
- source
- dogfood
- effects
- judge
- lakshman
- tenant-projection
- worker
- runtime-entry-points.yaml
- units
- memgraph
- operational_defaults
- operational_defaults
- operational_defaults
- volumes
- volumes
- options
- x-runner-base
- options
- x-runner-base
- volumes
- options
- x-runner-base
- x-runner-base
- x-runner-base
- options
- volumes
- lane_mirror_source_bus
- dev
- local_bus
- ModelProjectionLivenessVerdict
- inputs
- pins
- build-and-push-migrate-image.yml
- steps
- inputs
- contractor-integration-note.yml
- open-bump-pr
- env
- dlq-depth-monitor.yml
- inputs
- evidence-autoclose-sweep
- steps
- apply
- refresh_after_days
- steps
- env
- steps
- success_threshold
- trigger-delivery
- secrets
- with
- detect
- apply
- release-train-lab.yml
- steps
- env
- env
- with
- steps
- with
- validator_host_ports.py
- entries
- _apply_overlays
- model_enabled_protocols_config.py
- ProtocolDeclaredStateSurface
- ModelLockfileCveVerdict
- _api_get
- integration_skip_guard.yaml
- ModelWorkflowJobFact
- runner_fleet_canary.sh
- test_selection_adjacency.yaml
- edge_delegation_worker/cli.py
- monitor_alert_contract.yaml
- compose_files.sh
- HealthGateReport
- ModelDemoResetConfig
- ModelDelegateAttempt
- ModelResetActionResult
- qwen2.5-7b
- qwen2.5-coder-14b
- qwen2-vl
- qwen3.6-27b-mtp-iq4xs
- qwen3.6-35b-a3b
- text-embedding-gte
- defaults
- db.execute
- EnumTopicStandard
- ModelOrphanedDispatch
- ModelKafkaConnectRetryPolicy
- ModelGatewayConfig
- input_fields
- operations
- ModelMcpHandlerConfig
- EnumAlertSeverity
- model_phase_payload_ready_for_merge.py
- EnumCodingAgentFsmState
- settings
- ModelProjectionContractRef
- intent_targets
- ModelBootRosterPartition
- ModelReleaseCutChangelog
- ModelProjectorValidationError
- ModelIntrospectionMetrics
- ModelNodeRegistration
- node_api_fixture_record_effect/contract.yaml
- recording_policy
- node_architecture_validator/handlers/contract.yaml
- handlers
- coordination_rules
- nodes
- NodeAuthGateCompute
- node_baseline_comparison_compute/contract.yaml
- node_broker_disk_watermark_compute/contract.yaml
- handlers
- handlers
- event_bus
- handlers
- health_check
- io_operations
- cloud_leg
- handlers
- ModelGatewayPublishReceipt
- handlers
- node_coding_agent_invoke_effect/contract.yaml
- node_coding_agent_orchestrator/contract.yaml
- handlers
- handlers
- operations
- security
- rules
- node_contract_validate_compute/contract.yaml
- health_check
- handlers
- node_decision_store_query_compute/contract.yaml
- handlers
- handlers
- node_delegation_dispatch_terminal_effect/contract.yaml
- node_deploy_job_reading_effect/contract.yaml
- node_dlq_depth_monitor_effect/contract.yaml
- health_check
- handlers
- health_check
- event_bus
- handlers
- health_check
- handlers
- health_check
- handlers
- health_check
- handlers
- metadata
- node_in_progress_probe_hygiene_effect/contract.yaml
- health_check
- ProtocolKafkaMessage
- node_lab_proof_run_effect/contract.yaml
- handlers
- event_bus
- health_check
- handlers
- node_llm_embedding_effect/contract.yaml
- node_node_broker_resolve_effect/contract.yaml
- inmemory
- rules
- health_check
- handlers
- metadata
- state_machine
- node_runner_usage_effect/contract.yaml
- consumes
- node_runtime_error_triage_effect/contract.yaml
- handlers
- handlers
- config
- node_session_state_effect/contract.yaml
- io_operations
- handlers
- health_check
- handlers
- health_check
- handlers
- event_bus
- metadata
- health_check
- coordination_rules
- model_lifecycle_candidate_upserted_event.py
- build_finding
- node_waste_detection_compute/contract.yaml
- consumes
- ModelLoggingSinkConfig
- ModelModelReviewCapabilityObservation
- steps
- ProtocolClusterMetadata
- HandshakeFailureReason
- load_and_validate_contract_yaml
- select_nonprojection_group_infixes
- EnumBifrostLaneLocale
- EnumEventBusProfile
- create_app
- .__init__
- create_app
- ModelManagedStagingTopicDiff
- ModelParsedDSN
- ProducerZeroOutputError
- EnumWorkflowReceiptTerminalStatus
- HandlerRuntimeLifecycle
- omnibase_infra
- model_review
- pypi_cache
- omnibase_infra
- saturation_alert
- allowed_commands
- OMN-19404.yaml
- dod-omn19404-ac3-slot-topics-accepted
- default_retry_policy
- io_config
- retry_policy
- methodology
- feature_flags
- feature_flags
- prod
- stability-test
- omninode-host-maintenance-sync.sh
- core
- environment
- ci-bus-topics
- postgres
- postgres
- valkey
- volumes
- environment
- projection-delegation-writer
- projection-tenant-registry-writer
- toolcache-seed.sh
- steps
- PULL_REQUEST_TEMPLATE.md
- resolve-lane
- on
- build-runtime-candidate
- steps
- lab-pass-gate
- env
- in-progress-probe-hygiene.yml
- env
- lane-census-refresh.yml
- plugin-pin-cascade.yml
- product-readiness-shadow.yml
- release
- sync-main
- deploy
- release-train-nightly.yml
- steps
- steps
- with
- steps
- route-inline
- route-reusable
- verify-lane-converged
- security-scan.yml
- codeql
- on
- ModelNodeTypeProbe
- resolve_project_tracker
- Record
- gate_runner_supervisor.sh
- diagnose_e2e.sh
- render_desired_state
- runtime_health_wait.sh
- ModelCiBusOverlay
- packages
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- evidence
- event.transform
- handler_routing
- WORKSPACE_ROOT
- handler_routing
- handler_routing
- EnumTopicStatus
- EnumDependsOnCondition
- EnumAdjudicatorState
- EnumDlqFailureClass
- EnumRunVariant
- ModelLaneEntryPointCensus
- kafka_publish_headers.py
- mcp/protocols.py
- ModelDeleteFileResult
- ModelSystemClock
- ModelDlqReplayTarget
- settings
- ModelRuntimeTimeouts
- ModelBrokerGrantBindings
- error_types
- error_types
- health_check
- handlers
- HandlerBaselineComparison
- ModelBaselineComparisonInput
- health_check
- retry_policy
- health_check
- node_broker_grant_derive_compute/contract.yaml
- retry_policy
- metadata
- node_build_loop_projection_compute/contract.yaml
- metadata
- handlers
- input_fields
- metadata
- https_ingest
- NodeBusForwarderEffect
- ModelGatewayCloudBusConfig
- event_bus
- nodes
- NodeChainOrchestrator
- NodeChainReplayCompute
- NodeChainRetrievalEffect
- NodeChainStoreEffect
- NodeChainVerifyReducer
- health_check
- EnumCIRunnerRouteVisibility
- node_coding_agent_fsm_reducer/contract.yaml
- handlers
- node_coding_agent_workspace_compute/contract.yaml
- handlers
- node_consumer_coordinator_rejoin_compute/contract.yaml
- extract_fields
- io_config
- metadata
- metadata
- intents
- config
- metadata
- health_check
- node_declared_state_diff_compute/contract.yaml
- metadata
- ModelLedgerChainRow
- retry_policy
- metadata
- metadata
- metadata
- retry_policy
- metadata
- node_dlq_replay_effect/contract.yaml
- NodeEventForwardEffect
- node_event_provenance_stamp_compute/contract.yaml
- metadata
- health_check
- retry_policy
- metadata
- metadata
- node_gateway_link_health_projection_compute/contract.yaml
- metadata
- io_operations
- metadata
- metadata
- dependencies
- retry_policy
- health_check
- handlers
- metadata
- health_check
- health_check
- retry_policy
- metadata
- handler
- NodeLLMCompletionEffect
- dependencies
- metadata
- node_merge_provenance_observe_effect/contract.yaml
- metadata
- retry_policy
- health_check
- handlers
- health_check
- handlers
- health_check
- retry_policy
- catalog_service
- event_bus
- intents
- NodeRegistrationStorageEffect
- health_check
- health_check
- event_bus
- nodes
- node_rsd_orchestrator/contract.yaml
- event_bus
- nodes
- metadata
- nodes
- metadata
- intents
- methodology
- health_check
- produces
- handlers
- event_bus
- nodes
- definitions
- metadata
- intents
- health_check
- RegistryInfraSessionState
- build
- handlers
- retry_policy
- intents
- intents
- health_check
- model_cross_repo_run_started_event.py
- handler
- NodeVectorStoreEffect
- retry_policy
- handlers
- retry_policy
- write_config_local
- VerificationSpec
- FileSnapshotSink
- ModelLifecycleHookConfig
- ModelLifecycleHookConfig
- ModelLocalRuntimeIngressRequest
- RegistryError
- graph_queries.py
- ModelTopicGraph
- classification
- adapter
- classification
- internal_catalog
- one_database
- role_acl
- schema_qualification
- tenant_rls
- topology_parity
- predecessor_pins
- pinned_backends
- machines
- dns_cache
- response_handling
- publish_topics
- topic_validation
- install-host-maintenance.sh
- omninode-runner-tree-converge.sh
- tracing
- canonical
- canonical
- canonical
- canonical
- canonical
- canonical
- canonical
- canonical
- canonical
- build
- environment
- healthcheck
- environment
- projection-api
- lane.dev.kafka.scram
- env
- inputs
- with
- attest-source-hash
- steps
- env
- resolve-lane
- demo-loop-gate
- docs-only-marker
- repo_key
- steps
- onexbot-occ-app-id
- resolve-lane
- attest-source-hash
- docker-build
- docker-integration-tests
- steps
- steps
- hostile-review
- steps
- env
- on
- steps
- nonrequired-check-failure-rate
- steps
- decide
- steps
- ModelDeclaredStateApplyRequest
- bootstrap-infisical.sh
- check_lane_census_age.py
- violations
- run_post_verdicts
- ensure_ci_env.sh
- runner_broker_dispatch_wedge_rerun.sh
- resolve_host
- OOMKillSweep
- omnidash-lifecycle.sh
- reconcile_privilege_lib.sh
- DriftReport
- ConsumerGroupAudit
- ManifestResponse
- snapshot_cache.py
- allowlist
- operation_bindings
- future_considerations
- metadata
- metadata
- EnumDataProvenance
- EnumSharedListenerState
- ModelEventBusReadiness
- .handle
- ModelHandlerResponse
- ModelHttpHandlerResponse
- ModelQdrantHandlerResponse
- handler_placement_epoch_guard/contract.yaml
- projection_failure_classification.py
- ModelIdempotencyGuardConfig
- ModelIdempotencyRecord
- DiscoveredCapabilitiesCacheDict
- ModelAttributionRecord
- ModelLlmEndpointLaneOverlayBinding
- ModelDelegationFirstInferenceIdentity
- ModelDelegationProviderPolicy
- ModelDelegationTierRetryBound
- TypeCacheInfo
- ModelLiveDispatchSettler
- ModelReadCommittedTopics
- event_bus
- metadata
- metadata
- dependencies
- metadata
- event_bus
- metadata
- metadata
- event_bus
- metadata
- metadata
- retry_policy
- node_chain_canary_effect/contract.yaml
- dependencies
- event_bus
- handler
- metadata
- coordination_rules
- workflow_metadata
- metadata
- descriptor
- descriptor
- retry_policy
- event_bus
- metadata
- node_context_audit_dlq_effect/__init__.py
- retry_policy
- health_check
- metadata
- dependencies
- io_operations
- dependencies
- event_bus
- metadata
- health_check
- metadata
- retry_policy
- health_check
- health_check
- handler_routing
- node_deploy_runners/node.py
- metadata
- dependencies
- metadata
- event_bus
- metadata
- quarantine
- .__init__
- response_handling
- topic_validation
- HandlerEventBusWiring
- node_event_forward_effect/contract.yaml
- handler_routing
- retry_policy
- node_evidence_autoclose_sweep_effect/contract.yaml
- dependencies
- event_bus
- event_bus
- input_fields
- dependencies
- event_bus
- handlers
- dependencies
- retry_policy
- metadata
- node_invariant_evaluate_compute/contract.yaml
- metadata
- metadata
- dependencies
- metadata
- event_bus
- handler_routing
- metadata
- health_check
- config
- event_bus
- metadata
- metadata
- metadata
- redpanda
- consumed_events
- feature_flags
- metadata
- metadata
- RegistryInfraNodeRegistrationReducer
- dependencies
- event_bus
- coordination_rules
- workflow_metadata
- event_bus
- handler_routing
- node_row_count_diagnostic_effect/__init__.py
- coordination_rules
- workflow_metadata
- event_bus
- coordination_rules
- event_bus
- workflow_metadata
- event_bus
- event_bus
- event_bus
- metadata
- metadata
- env_dependencies
- event_bus
- metadata
- event_bus
- metadata
- coordination_rules
- workflow_metadata
- node_secret_seed_effect/contract.yaml
- dependencies
- handler
- metadata
- metadata
- metadata
- overlay_revision_render
- dependencies
- metadata
- severity
- node_sync_revert_watchdog_effect/contract.yaml
- dependencies
- metadata
- node_topic_migration_executor_effect/contract.yaml
- event_bus
- metadata
- node_topic_migration_projection/contract.yaml
- event_bus
- metadata
- handler_routing
- metadata
- event_bus
- metadata
- event_bus
- metadata
- dependencies
- metadata
- event_bus
- metadata
- ModelCrossRepoRunCompletedEvent
- event_bus
- analyze_low_cache
- first_attribution
- event_bus
- ProtocolEmissionCountSource
- ModelBoundedLaneBrokerTopology
- ModelRetryPolicy
- owners
- service_owned
- service_owned
- checks
- entries
- action_archive_cache
- tool_cache
- docker_ci_runner_variable
- omninode_infra_fleet_runner_variable
- private_repo_hosted_exception
- security_scan_runner_variable
- test-failure-baseline.yaml
- requirements
- dod-omn19404-ac1-slot-roles-accepted
- dod-omn19404-ac2-slot-migration-accepted
- circuit_breaker
- conditions
- conditions
- secret_resolver_namespaces
- bindings
- canary-s7-override.yml
- canary
- fault-injection
- omnimemory
- runtime
- runtime-core
- runtime-infrastructure
- runtime-integrations
- runtime-observability-projections
- secrets
- tenant-path
- healthcheck
- labels
- healthcheck
- healthcheck
- healthcheck
- healthcheck
- healthcheck
- labels
- healthcheck
- healthcheck
- healthcheck
- healthcheck
- labels
- healthcheck
- gateway-dns-bastion
- projection-live-events-writer
- projection-savings-writer
- projection-tenant-credentials-writer
- projection-tenant-registry-writer
- runtime-worker
- tenant-projection-writer
- redpanda
- entries
- shape_reconciliation_exemptions
- resolve-lab-lane/action.yml
- setup-python-uv/action.yml
- detectors
- build-and-push
- with
- steps
- exposed-identifier-gate
- lockfile-registry-allowlist
- topic-enum-drift
- topic-naming-lint
- on
- steps
- build-migrate-bundle
- saturation-record
- dlq-depth-monitor
- steps
- suppress_alert_exit
- build-summary
- env
- notify-on-failure
- omni-standards-gate
- env
- steps
- reason-graph
- post-release-dev-bump
- trigger-runtime-rebuild
- env
- route-wiring-proof
- steps
- env
- secret-scan
- is_runtime_affecting
- model_graph_handler_config.py
- ModelLaneEntryPointPolicy
- ModelLaneEntryPointCensusResult
- model_lane_entry_point_census_request.py
- ModelApplicabilityScope
- .public-repo-hygiene.yaml
- emit_verification_result
- carries_failure_conclusion
- _capture_render_inputs
- lane-census-check.sh
- ModelMonitorAlertContract
- render_dev_lane_tenant_path_env.sh
- EnumResetAction
- metadata
- retry_policy
- descriptor
- authentication
- retry_policy
- feature_flags
- EnumBackendType
- EnumBuildLoopIntentType
- EnumBuildLoopPhase
- EnumBuildability
- EnumCaptureOutcome
- EnumConfirmationEventType
- EnumCostTier
- EnumDedupeStrategy
- EnumEnvironment
- EnumEventBusLostBrokerPolicy
- EnumExecutionShapeViolation
- EnumLlmFinishReason
- EnumLlmOperationType
- EnumRegistrationStatus
- EnumRegistryResponseStatus
- EnumResponseStatus
- EnumUpdatePlanState
- datetime_handling
- retry_policy
- metadata
- EnumGraphOperationType
- EnumMcpOperationType
- ModelFileSystemConfig
- EnumQdrantOperationType
- HookActivationResolver
- EnumAgentSandbox
- EnumAgentStatus
- EnumCliBackendStatus
- EnumCodingAgentEventKind
- delegate_flag_checks.py
- ModelLlmToolDefinition
- ModelLlmMessage
- ModelLlmToolChoice
- ModelWorkLedgerReadRefusal
- ModelNetworkBindingSettings
- ModelOtlpTracesEndpoint
- get_security_level
- masking_rules
- dependencies
- NodeArtifactReconciliationOrchestrator
- NodeBaselineComparisonCompute
- consumer_flow_topics
- handler
- NodeBoardProbeEffect
- metadata
- version
- NodeBootCatalogEffect
- NodeBrokerDiskWatermarkCompute
- metadata
- NodeBrokerPrincipalEffect
- dependencies
- dependencies
- metadata
- NodeCheckpointEffect
- NodeCheckpointValidateCompute
- NodeCIRunnerRouteCompute
- descriptor
- descriptor
- metadata
- node_consumer_health_triage_effect/contract.yaml
- circuit_breaker
- NodeContractLoaderEffect
- dependencies
- error_codes
- retry_policy
- version
- idempotency
- version
- produced_events
- dependencies
- dependencies
- error_codes
- retry_policy
- node_capabilities
- NodeDecisionStoreQueryCompute
- metadata
- dependencies
- NodeDelegationDispatchTerminalEffect
- dependencies
- error_codes
- retry_policy
- dependencies
- error_codes
- retry_policy
- NodeDeveloperPrincipalPlanCompute
- node_dlq_depth_evaluate_compute/contract.yaml
- error_types
- retry_policy
- handler
- handlers
- error_types
- circuit_breaker
- NodeEventBusWiringEffect
- retry_policy
- version
- metadata
- metadata
- fault_injection
- NodeFaultInjectFixtureCompute
- NodeGatewayAttachEffect
- dependencies
- dependencies
- handler
- metadata
- NodeHostileReviewFixVerifierEffect
- handler
- handlers
- dependencies
- NodeKafkaReplayCompute
- NodeLabProofPlanCompute
- NodeLabProofRunEffect
- NodeLaneEntryPointCensusEffect
- dependencies
- metadata
- node_llm_completion_effect/contract.yaml
- retry_policy
- retry_policy
- version
- metadata
- dependencies
- error_codes
- retry_policy
- NodeMergeProvenanceCompute
- dependencies
- metadata
- NodeNodeBrokerResolveEffect
- conditions
- NodeNodeGraphReducer
- version
- NodePatternLifecycleEffect
- NodeRegistrationOrchestrator
- idempotency
- retry_policy
- metadata
- node_capabilities
- handler
- dependencies
- tags
- timestamp
- metadata
- NodeRemoteAgentInvokeEffect
- NodeRewardBinderEffect
- ModelPlanOverrideData
- NodeRsdOrchestrator
- node_runner_fleet_health_compute/contract.yaml
- topic_config
- NodeRunnerFleetMaintainOrchestrator
- NodeRuntimeOrchestrator
- node_runtime_source_attestor_effect/contract.yaml
- NodeSavingsEstimationCompute
- NodeScopeWorkflowOrchestrator
- error_types
- retry_policy
- handlers
- NodeSecretSeedEffect
- NodeServiceDiscoveryEffect
- NodeSessionStateEffect
- NodeInfisicalSetupEffect
- llm_endpoint_env_render
- overlay_home_lane_env
- NodeLocalProvisionEffect
- NodeSetupOrchestrator
- NodePreflightEffect
- NodeSetupValidateEffect
- metadata
- title
- dependencies
- retry_policy
- NodeValidationOrchestrator
- node_vector_store_effect/contract.yaml
- metadata
- config
- metadata
- NodeWorkspaceConfigRootResolveEffect
- EnumRunnerHostArch
- write_config_cloud
- setup.yaml
- standalone_quickstart.yaml
- ProtocolActionAuthorizationClaimPort
- HandshakeFailureReason
- EnumContractSource
- util_mcp_auth.py
- ModelScopePredicate
- _load_overlay_entries
- summary_generator.py
- EnumHealthStatus
- CaptureLifecycleFSM
- EnumTopicProvisioningProfile
- instances/local.yaml
- omninode_runtime_service
- omninode_runtime_service
- ModelApplicationDatabaseActivityEvidence
- ModelApplicationDatabasePolicyState
- ModelApplicationDatabaseRoleMembership
- probe_schema_validator.py
- EnumWorkflowReceiptReplayVerdict
- allowlisted_handlers
- v0.31.0 (2026-03-31)
- v0.23.0 (2026-03-20)
- tag_clone_consumed_paths
- known_entrypointless
- observability
- disk-gc/install-worktree-prune.sh
- omnidash
- resources
- labels
- resources
- labels
- resources
- labels
- resources
- resources
- resources
- labels
- resources
- labels
- optional_directory_bind_mounts
- resources
- labels
- resources
- resources
- redpanda-partition-cap
- redpanda
- redpanda-partition-cap
- projection-api
- projection-api
- redpanda
- runtime-effects
- redpanda
- run-cloud-migrations.sh
- broker_readiness_probe.sh
- steps
- on
- lineage-guard
- pinned_repo
- check_delegation_legs
- env
- plant_canary_in_answer
- c12-provider-catalogue
- steps
- dry_run
- secrets
- drill_ticket
- dev-lane-liveness
- env
- apply
- hostile-review-gate
- integration-tests
- on
- lane
- compose_pg_port
- pg_port
- runtime_host
- consume-inline
- with
- secrets
- dry_run
- infra_ref
- labels
- allowlisted_handlers
- baseline_subscriptions.py
- tui/__main__.py
- traced_dispatch
- FileSnapshotSource
- audit-merge-method-on-main.sh
- buildx-orphan-sweep.sh
- check_shell_hygiene.sh
- headless-close-out.sh
- validate-state.sh
- DesiredState
- VerifyInterrupted
- setup-infisical-identity.sh
- .enrich
- adapters/protocol_postgres_adapter.py
- ModelDelegateDefaultBus
- ModelDelegateHumanOutcome
- ModelDelegateTerminalMetrics
- summary_stats
- defaults
- version
- audit_logging
- version
- security
- model_filesystem_delete_payload.py
- model_filesystem_ensure_directory_payload.py
- model_filesystem_ensure_directory_result.py
- model_filesystem_list_directory_payload.py
- model_filesystem_read_payload.py
- model_filesystem_read_result.py
- model_filesystem_write_payload.py
- model_filesystem_write_result.py
- model_manifest_store_payload.py
- model_qdrant_collection_payload.py
- model_qdrant_handler_config.py
- ModelIdempotencyStoreHealthCheckResult
- EnumCutoverEventKind
- EnumCutoverFamilyStatus
- ModelBaselineRunConfig
- ModelBaselineRunResult
- model_phase_payload_create_pr.py
- model_phase_payload_implement.py
- model_phase_payload_pr_release_ready.py
- ModelAgentInvocation
- ModelSubprocessInvocation
- ModelDelegationBuildIdentity
- ModelDebugTraceSnapshot
- model_eval_budget_cap.py
- health/model_health_check_result.py
- ModelEntryPointFamily
- ModelLlmFunctionCall
- ModelMCPToolParameter
- ModelDevBootPoint
- model_infisical_policy.py
- model_validation_error_params.py
- metadata
- node_api_fixture_record_effect/node.py
- dependencies
- config
- io_operations
- node_artifact_change_detector_effect/node.py
- ProtocolPublisher
- NodeBaselineCapture
- input_model
- board_checks
- published_events
- EnumBoardCheckId
- input_model
- io_operations
- transitions
- input_model
- input_model
- node_broker_grant_derive_compute/node.py
- input_model
- io_operations
- NodeBuildLoopProjectionCompute
- NodeBuildLoopWriteEffect
- canary
- dedupe
- node_c11_delegation_legs_effect/node.py
- node_c16_pinned_backend_effect/node.py
- io_operations
- published_events
- node_chain_canary_effect/node.py
- metadata
- node_ci_live_contact_effect/node.py
- input_model
- EnumCIRunnerRouteDecision
- EnumCIRunnerRouteForce
- EnumCIRunnerRouteReason
- ModelCIRunnerFleetObservation
- dependencies
- metadata
- dependencies
- metadata
- node_coding_agent_invoke_effect/node.py
- dependencies
- metadata
- node_coding_agent_orchestrator/node.py
- dependencies
- metadata
- node_coding_agent_workspace_compute/node.py
- input_model
- node_consumer_coordinator_rejoin_compute/node.py
- dependencies
- feature_flags
- io_operations
- NodeConsumerHealthTriageEffect
- metadata
- input_model
- io_operations
- NodeContractPersistenceEffect
- dependencies
- io_operations
- node_contract_subscribe_preflight_effect/node.py
- metadata
- node_contract_validate_compute/node.py
- io_operations
- NodeDbErrorLinearEffect
- input_model
- io_operations
- NodeDecisionStoreEffect
- input_model
- node_declared_state_apply_effect/node.py
- input_model
- node_declared_state_diff_compute/node.py
- node_declared_state_reconcile_orchestrator/node.py
- bind_handler_db
- chain_topology
- node_capabilities
- NodeDelegationChainLedgerEffect
- descriptor
- input_model
- io_operations
- handler_routing
- io_operations
- NodeDeltaBundleEffect
- handler_routing
- io_operations
- NodeDeltaMetricsEffect
- metadata
- node_deploy_job_reading_effect/node.py
- node_dev_boot_guard_effect/node.py
- node_dev_lane_code_ref_pin_effect/node.py
- node_dev_lane_pin_compose_guard_effect/node.py
- node_dev_roll_events_effect/node.py
- input_model
- io_operations
- alert_policy
- io_operations
- ModelDlqTopicThresholdOverride
- node_dlq_depth_evaluate_compute/node.py
- alert_policy
- io_operations
- node_dlq_depth_monitor_effect/node.py
- dependencies
- io_operations
- replay_policy
- NodeDlqReplayEffect
- observability
- metadata
- circuit_breaker
- input_model
- node_event_provenance_stamp_compute/node.py
- io_operations
- EnumEvidenceAutocloseArm
- node_evidence_autoclose_sweep_effect/node.py
- input_model
- io_operations
- NodeGatewayLinkHealthProjectionCompute
- NodeGatewayLinkHealthWriteEffect
- errors
- node_gmail_archive_cleanup_effect/node.py
- events_published
- io_operations
- messages_archived
- node_gmail_intent_poller_effect/node.py
- input_model
- io_operations
- node_hotpatch_ledger_reconcile_effect/node.py
- input_model
- io_operations
- node_in_progress_probe_hygiene_effect/node.py
- io_operations
- metadata
- node_invariant_evaluate_compute/node.py
- output_model
- input_model
- input_model
- io_operations
- published_events
- consumed_events
- node_capabilities
- circuit_breaker
- metadata
- io_operations
- backends
- consumed_events
- input_model
- io_operations
- NodeMergeGateEffect
- EnumActionsExecutionStatus
- node_merge_provenance_observe_effect/node.py
- node_model_health_effect/node.py
- node_model_router_compute/node.py
- input_model
- io_operations
- metadata
- NodeOnboardingOrchestrator
- input_model
- NodeOnexApiImageBuildEffect
- input_model
- NodeOnexApiImageImportCheckEffect
- ModelLifecycleResult
- metadata
- node_placement_runtime_effect/node.py
- io_operations
- metadata
- node_reconcile_workspace_venvs/node.py
- dependencies
- intent_consumption
- mcp
- time_injection
- model_orchestrator_config.py
- dependencies
- NodeRegistrationReducer
- io_operations
- backend_id
- service_name
- node_registry_effect/node.py
- node_release_identity_compute/node.py
- node_release_notes_apply_effect/node.py
- consumed_events
- metadata
- node_routing_orchestrator/node.py
- node_routing_score_reducer/contract.yaml
- metadata
- transitions
- node_routing_score_reducer/node.py
- NodeRRHEmitEffect
- NodeRRHStorageEffect
- NodeRRHValidateCompute
- ModelDependencyEdge
- node_rsd_data_fetch_effect/node.py
- input_model
- metadata
- node_rsd_score_compute/node.py
- node_rsd_state_reducer/node.py
- published_events
- EnumReadinessSignalOutcome
- EnumRunnerFleetHealthState
- node_runner_fleet_health_compute/node.py
- metadata
- node_runner_health_snapshot_effect/contract.yaml
- node_runner_health_snapshot_effect/node.py
- input_model
- NodeRunnerUsageEffect
- dependencies
- io_operations
- published_events
- NodeRuntimeErrorTriageEffect
- dependencies
- NodeRuntimeManifestReducer
- dependencies
- feature_flags
- io_operations
- NodeRuntimeSourceAttestorEffect
- node_runtime_start_first_effect/node.py
- input_model
- node_scope_extract_compute/node.py
- node_scope_file_read_effect/node.py
- node_scope_manifest_write_effect/node.py
- descriptor
- metadata
- node_scope_workflow_reducer/node.py
- io_operations
- io_operations
- dependencies
- NodeSessionLifecycleReducer
- RegistryInfraSessionLifecycle
- input_model
- output_model
- source_configuration
- io_operations
- io_operations
- node_sibling_clone_pin_effect/node.py
- duration_ms
- message
- retry_count
- node_slack_alerter_effect/node.py
- node_standalone_compose_recreate_effect/node.py
- io_operations
- node_sync_revert_watchdog_effect/node.py
- node_ticket_contract_compose_effect/node.py
- dependencies
- io_operations
- NodeTopicMigrationExecutorEffect
- io_operations
- NodeTopicMigrationProjection
- dependencies
- NodeUpdatePlanReducer
- dependencies
- NodeValidationAdjudicator
- RegistryInfraValidationAdjudicator
- input_model
- NodeValidationExecutor
- NodeValidationLedgerProjectionCompute
- node_capabilities
- input_model
- descriptor
- io_operations
- metadata
- node_venv_atomic_install/node.py
- analyze_retry_waste
- input_model
- node_waste_detection_compute/node.py
- input_model
- io_operations
- ModelDnsCacheConfig
- ModelPyPICacheConfig
- ModelOnboardingStepVerification
- contributor_cloud.yaml
- contributor_hybrid.yaml
- contributor_local.yaml
- full_platform.yaml
- gcp
- new_employee.yaml
- omnimarket_quickstart.yaml
- ModelVerificationSpec
- EnumActionAuthorizationClaimOutcome
- EnumActionAuthorizationClaimState
- normalize_fanout_sequence
- auto_wiring/model_handshake_config.py
- ModelHandshakeConfig
- model_emit_daemon_runtime_input.py
- model_emit_daemon_runtime_output.py
- ModelBoundedDelegationRoute
- ModelBoundedDelegationRouteDeclaration
- model_policy_result.py
- model_transition_notification_outbox_config.py
- ModelActivationScope
- agent_learning_extraction/config.py
- ModelAgentStatusEvent
- ModelDetectionFailure
- ModelExecutionLog
- ModelPerformanceMetric
- ModelRoutingDecision
- ModelTransformationEvent
- ModelConfigGraphProjector
- model_graph_mutation.py
- ModelApplicationDatabaseActivityPrincipalObservation
- ModelApplicationInventoryRelationCounts
- arch-002-no-transport-imports
- bandit-security
- banned-compose-env-vars
- detect-secrets
- enum-governance
- hardcoded-local-paths
- naming-conventions
- pydantic-patterns
- single-class-per-file
- v0.24.0 (2026-03-22)
- v0.27.0 (2026-03-25)
- gates
- entries
- check-dependency-pins
- check-doc-content-scan-compute
- check-duplicate-registry-ids
- check-fsm-handler-drift
- check-github-token-env-reads
- check-imperative-contracts
- check-invariant-analysis
- check-local-paths-compute
- check-local-paths
- check-localhost-url-compute
- check-merge-queue-parity
- check-no-backward-compat
- check-no-faked-boundary
- check-pin-hygiene-compute
- check-private-ip-compute
- check-todo-fixme-marker-compute
- omni-agent
- omninode-runtime-host-dev
- onex
- onex-validate-links
- onex-validate-topics
- onex-work-ledger
- resolve-private-vocabulary
- validate-string-versions
- onex-gateway-forwarder
- onex-linear-relay
- onex-status
- emergency_bypass
- evidence_requirements
- install-lane-census.sh
- provision-vllm-0.27-venv.sh
- nofile
- labels
- operational_defaults
- nofile
- nofile
- nofile
- phoenix
- phoenix
- runtime-effects
- environment
- forward-migration
- intelligence-migration
- valkey
- lane_mirror_binding
- tenant_identity
- cutover-proof/prove.sh
- fenced_node_migrations
- grandfathered_force_rls_migrations
- git-mirror-refresh.sh
- playwright-smoke.sh
- aws-region
- backoff-factor
- fail-on-missing
- initial-delay
- max-delay
- deploy-wait-seconds
- lanes-var
- match
- wait-while-busy
- cache-key-prefix
- cache-version
- github-token
- install-args
- lock-file-path
- python-version
- shared-env-enabled
- shared-env-install-args
- skip-install
- sync-attempts
- sync-retry-delay-seconds
- uv-version
- working-directory
- codeql-config.yml
- env
- git-ref
- bus_lane
- deploy_wait_seconds
- gateway_url
- check_pinned_backends
- pinned_backend_override_free_cloud
- pinned_backend_override_local
- pinned_backend_override_paid
- env
- deploy_wait_seconds
- gateway_url
- quarantine_bootstrap_servers
- task_type
- on
- version
- bus_lane
- topic_prefix
- lookback_hours
- max_comments_per_run
- project_id
- refresh
- ci-naming-convention
- type-safety
- type-union-check
- changed_files
- source_sha
- ticket
- lint-shadow
- tests-shadow
- typecheck-shadow
- source_sha
- node_boundary_import_baseline.yaml
- build_parser
- _run_read_only
- ModelLaneBinding
- default_runner
- resolve_modern_bash.sh
- git-gc-auto.sh
- infra-watchdog.sh
- compute_desired_state_sha256
- sync-node-migrations.sh
- check_no_infra_inmemory_import.sh
- verify-plugin-cache.sh
- compute.validate
- kafka.stream
- postgres.query
- workflow.reduce
- contract_version
- retry_policy
- graph.delete_node
- graph.delete_relationship
- graph.execute_query_batch
- graph.traverse
- contract_version
- http.post
- contract_version
- future_considerations
- input_validation
- allow_any
- validate_env
- get_resolution
- derive_dlq_topic_for_event_type
- omnibase_infra/handlers/contract.yaml
- handler_broker_grant_derive/contract.yaml
- handler_consumer_coordinator_rejoin/contract.yaml
- handler_declared_state_diff/contract.yaml
- handler_developer_principal_plan/contract.yaml
- handler_event_provenance_stamp/contract.yaml
- read_live_inspection
- HandlerLaneEntryPointCensus
- contract_version
- node_version
- storage
- version
- handler_registration_from_announcement/contract.yaml
- handler_validation_adjudicate/contract.yaml
- handler_validator_event_ledger_project/contract.yaml
- hook_venv_reconcile/contract.yaml
- venv_atomic_install/contract.yaml
- settings
- contract_version
- descriptor
- input_model
- node_version
- output_model
- input_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- consumed_events
- contract_version
- input_model
- node_version
- output_model
- contract_version
- node_version
- output_model
- ProtocolPublisher
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- capabilities
- contract_version
- input_model
- output_model
- capabilities
- contract_version
- input_model
- node_capabilities
- output_model
- config
- node_version
- output_model
- contract_version
- input_model
- output_model
- input_model
- output_model
- contract_version
- node_version
- contract_version
- event_model
- input_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- runtime_dispatch
- contract_version
- event_model
- input_model
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- node_version
- output_model
- health_check
- output_model
- staleness
- state_model
- contract_version
- health_check
- input_model
- node_version
- output_model
- contract_version
- event_bus
- input_model
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- input_model
- contract_version
- node_version
- output_model
- node_version
- output_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- input_model
- output_model
- input_model
- output_model
- contract_version
- input_model
- node_version
- output_model
- should_replay
- node_version
- contract_version
- dependencies
- input_model
- output_model
- contract_version
- node_version
- output_model
- contract_version
- input_model
- output_model
- node_version
- output_model
- config
- contract_version
- input_model
- node_version
- output_model
- capabilities
- contract_version
- input_model
- output_model
- capabilities
- contract_version
- input_model
- node_capabilities
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- input_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- node_version
- contract_version
- node_version
- output_model
- contract_version
- descriptor
- node_version
- output_model
- capabilities
- contract_version
- input_model
- output_model
- capabilities
- contract_version
- input_model
- node_version
- output_model
- published_events
- contract_version
- dependencies
- input_model
- output_model
- contract_version
- health_check
- input_model
- output_model
- contract_version
- input_model
- node_version
- output_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- node_version
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- io_operations
- node_version
- output_model
- contract_version
- event_bus
- input_model
- node_version
- output_model
- contract_version
- descriptor
- input_model
- node_version
- output_model
- output_model
- contract_version
- health_check
- input_model
- output_model
- contract_version
- node_version
- output_model
- contract_version
- node_version
- output_model
- contract_version
- handler_routing
- input_model
- node_version
- output_model
- input_model
- output_model
- contract_version
- input_model
- output_model
- contract_version
- output_model
- contract_version
- input_model
- output_model
- input_model
- output_model
- runtime_dispatch
- contract_version
- input_model
- node_version
- output_model
- contract_version
- metadata
- node_version
- output_model
- contract_version
- input_model
- node_version
- contract_version
- health_check
- input_model
- output_model
- contract_version
- input_model
- node_version
- output_model
- node_version
- output_model
- input_model
- output_model
- contract_version
- input_model
- output_model
- contract_version
- health_check
- contract_version
- health_check
- idempotency
- input_model
- node_version
- output_model
- contract_version
- dependencies
- node_version
- output_model
- capabilities
- contract_version
- input_model
- output_model
- contract_version
- input_model
- contract_version
- input_model
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- input_model
- node_version
- output_model
- contract_version
- health_check
- idempotency
- input_model
- node_version
- output_model
- published_events
- state_model
- contract_version
- health_check
- idempotency
- node_version
- output_model
- state_model
- contract_version
- io_operations
- node_version
- output_model
- capabilities
- contract_version
- input_model
- output_model
- contract_version
- node_version
- output_model
- contract_version
- dependencies
- health_check
- node_version
- output_model
- contract_version
- dependencies
- input_model
- output_model
- contract_version
- descriptor
- input_model
- node_version
- output_model
- contract_version
- metadata
- node_version
- output_model
- contract_version
- node_version
- output_model
- auto_wiring/contract.yaml
- contract_topic_router.py
- _is_command_topic
- tracing_contract.yaml
- derive_event_type_from_topic
- no-untracked-todos
- stub-implementations
- code_excerpt
- ratchet
- repos
- bootstrap.sh
- onex-prepush-idle-stop.sh
- depends_on
- ports
- depends_on
- ports
- depends_on
- ports
- depends_on
- depends_on
- ports
- depends_on
- ports
- ports
- depends_on
- depends_on
- omninode-contract-resolver
- postgres
- redpanda-sasl-enable
- agent-actions-consumer
- context-audit-consumer
- redpanda-sasl-enable
- redpanda-scram-user
- gateway-dns-bastion
- volumes
- baselines_breakdown
- baselines_comparisons
- baselines_trend
- pr_state
- savings_estimates
- session_outcomes
- cgroup-cpu-env.sh
- model-review-healthcheck.sh
- ecr-repository
- reject-deploy-gate-skip-token.sh
- omnibase_infra
- caia-watchdog.sh
- check_llm_endpoints.sh
- check_migrations_complete.sh
- reject-node-implementations.sh
- validate_runner_image.sh
- cloud-bus-tunnel.sh
- deploy-agent-launch.sh
- render-deploy-agent-plist.sh
- deploy-build-loop-trigger.sh
- disk-gc.sh
- disk-watermark-check.sh
- docker-volume-gc.sh
- fire-build-loop.sh
- generate-local-env.sh
- install-git-hooks.sh
- FindingKindSpec
- parse_version
- repowise_sync_after_pull.sh
- run-intelligence-migrations.sh
- runner-disk-admission-restore.sh
- worktree-gc.sh
- read_committed_topics.yaml
- AdapterMcpMetadataDict
- capabilities
- event_bus
- supported_rules
- event_bus
- input_subscriptions
- capabilities
- capabilities
- http_mount
- NodeContractResolverBridge
- health_check
- capabilities
- writer
- ModelDelegationTerminalPayload
- capabilities
- capabilities
- DlqDependencyLease
- capabilities
- health_check
- capabilities
- capabilities
- capabilities
- capabilities
- NodeRegistryApiEffect
- capabilities
- ModelArtifactEnforcement
- ModelUnavailableBehavior
- ModelToggleRequestBody
- migration_classes.yaml
- prod_promotion_grants.yaml
- tenant_scoped_ingress_allowlist.yaml
- install-buildx-orphan-sweep.sh
- install-disk-gc.sh
- install-runner-disk-guard.sh
- install-worktree-reaper.sh
- install-infra-watchdog.sh
- omninode-workspace-reconcile.sh
- warm_cache.sh
- dns-cache/entrypoint.sh
- docker-compose.model-review-canary.yml
- entrypoint-runtime.sh
- dns-bastion/entrypoint.sh
- 001_create_omniintelligence_schema.sh
- agent_execution_logs
- agent_routing_decisions
- agent_status_events
- capability_scores
- context_audit_events
- evidence_correlation_trace_projection
- evidence_dashboard_projection
- evidence_readiness_aggregate_projection
- live_events
- llm_call_metrics
- llm_cost_aggregates
- llm_routing_decisions
- log_entries
- router_performance_metrics
- swarm_runs
- skip-manifest.yaml
- pypi-cache/entrypoint.sh
- healthcheck.sh
- omni-curl.sh script
- noncanonical_class_allowlist.yaml
- deploy-agent
- reject-required-check-skip-vector.sh
- audit-env-files.sh
- check_no_cloud_bus_wrapper.sh
- check_placeholder_topics.sh
- check-realm-no-clients.sh
- check-sibling-compat.sh
- check-stale-images.sh
- check-strict-gate-ordering.sh
- check_version_source_sync.sh
- build_runner_image.sh
- normalize_product_clone.sh
- capabilities
- cli
- clients
- configs
- contracts
- decorators
- deep_dive
- diagnostics
- docker
- doctor
- errors
- event_bus
- examples
- gate
- gate_binding
- gateway
- handlers
- hooks
- infrastructure
- lab_proof
- learning
- migration
- migrations
- mixins
- models
- nodes
- observability
- onboarding
- plugins
- probes
- projectors
- protocols
- resources
- runtime
- schemas
- scope
- scripts
- secret_stores
- services
- shared
- sinks
- test_utils
- testing
- tools
- topology
- tui
- types
- validation
- validators
- verification
- workflows
- demo-loop-assert.sh
- deploy-agent-launchd.sh
- preflight_port_free.sh
- deploy-timers.sh
- install-bare-clone-sync.sh
- install-watchdog.sh
- k8s-pod-readiness-check.sh
- install-lab-alarm.sh
- launchd/install-worktree-prune.sh
- onex-build-loop-trigger.sh
- preflight-check.sh
- deploy-push-lanes.sh
- queue-runner.sh script
- rebalance-test-splits.sh
- consumer_groups_stability.yaml
- seed-keycloak.sh
- validate-pr-contract-sync.sh
- check_kafka_no_hardcoded_fallback.sh
- check_no_cloud_bus.sh
- check_precommit_additional_dependencies.sh
- completeness-allowlist.yaml
- run_topic_lint.sh
- verify-merge-queue-no-bypass.sh
- verify-omnidash-health.sh
- cli/topics.yaml
- scope_check.sh
- runtime/topics.yaml
- observability/topics.yaml
- services/topics.yaml
- routing_tiers.yaml

## God Nodes (most connected - your core abstractions)
1. `migrations` - 499 edges
2. `ModelInfraErrorContext` - 354 edges
3. `ProtocolConfigurationError` - 321 edges
4. `EnumHandlerTypeCategory` - 259 edges
5. `EnumHandlerType` - 248 edges
6. `RuntimeHostError` - 238 edges
7. `EventBusKafka` - 145 edges
8. `InfraConnectionError` - 144 edges
9. `RuntimeHostProcess` - 135 edges
10. `bootstrap()` - 133 edges

## Surprising Connections (you probably didn't know these)
- `Error Taxonomy` --references--> `InfraRateLimitedError`  [INFERRED]
  CHANGELOG.md → src/omnibase_infra/errors/error_infra.py
- `Handler & Intent Improvements` --references--> `HandlerGraph`  [INFERRED]
  CHANGELOG.md → src/omnibase_infra/handlers/handler_graph.py
- `Fixed` --references--> `HandlerCatalogRequest`  [INFERRED]
  CHANGELOG.md → src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_catalog_request.py
- `Handler to Dispatcher Terminology Migration` --references--> `ProtocolMessageDispatcher`  [INFERRED]
  CHANGELOG.md → src/omnibase_infra/protocols/protocol_message_dispatcher.py
- `Runtime Startup is a First-Class CI Gate` --references--> `wire_from_manifest()`  [INFERRED]
  CLAUDE.md → src/omnibase_infra/runtime/auto_wiring/handler_wiring.py

## Import Cycles
- None detected.

## Communities (5800 total, 1386 thin omitted)

### Community 0 - "typing"
Cohesion: 0.00
Nodes (22): parse_dlq_topic(), read_runtime_lane_example(), dod_verify_argv(), DoneWriteGateResolutionError, reap_timed_out_process(), run_dod_verify_command(), clear_topic_parse_cache(), is_valid_identifier() (+14 more)

### Community 1 - "pydantic"
Cohesion: 0.00
Nodes (36): main(), sign_envelope(), _day_window(), new_reconcile_run_id(), ModelMcpToolCall, ModelManifestPersistenceConfig, ModelManifestQueryPayload, ModelManifestRetrievePayload (+28 more)

### Community 2 - "ModelInfraErrorContext"
Cohesion: 0.01
Nodes (70): Error Handling, Errors, Handlers, Error Context Factory (MANDATORY), Error Handling, Error Sanitization, cmd_verify(), _run_list_nodes() (+62 more)

### Community 3 - ".with_correlation"
Cohesion: 0.01
Nodes (90): _validate_repo(), ServiceRegistrationError, ProtocolConfigurationError, PolicyRegistryError, build_confluent_auth_config(), build_dlq_topic(), _normalize_category(), load_private_key_from_pem() (+82 more)

### Community 4 - "migrations"
Cohesion: 0.00
Nodes (499): migrations, forward/001_registration_projection.sql, forward/002_updated_at_audit_index.sql, forward/003_capability_fields.sql, forward/004_capability_fields_concurrent.sql, forward/005_create_contracts_topics.sql, forward/020_create_agent_actions_table.sql, forward/021_create_agent_routing_decisions_table.sql (+491 more)

### Community 5 - "pathlib"
Cohesion: 0.01
Nodes (86): main(), check_file(), _is_config(), main(), _yaml_values(), check(), flat_migration_connect_target(), flat_migration_files() (+78 more)

### Community 6 - "json"
Cohesion: 0.01
Nodes (118): check_against_inventory(), declared_host_labels(), LegsError, main(), parse_legs(), evaluate(), Finding, _is_guardish() (+110 more)

### Community 7 - "cli_delegate.py"
Cohesion: 0.01
Nodes (116): auth_token(), _atomic_write_text(), _attempt_evidence(), _backend_pin_defect(), _backend_pin_receipt_block(), _bind_lane_transport(), _budget_outcome_receipt_block(), build_backend_overrides() (+108 more)

### Community 8 - "deploy_agent/executor.py"
Cohesion: 0.01
Nodes (126): BuildSource, EnumRuntimeLane, gateway_services_in(), ModelContainerResidue, ModelHealthCheck, Phase, services_for_scope(), without_gateway_services() (+118 more)

### Community 9 - "gateway_forwarder.py"
Cohesion: 0.01
Nodes (67): ModelGatewayForwarderConfig, requires_egress_redaction(), requires_metadata_scrub(), ModelGatewayForwarderRuntimeConfig, ModelGatewayHttpsIngestConfig, ModelGatewayLaneMirrorHealth, ModelGatewayTenantIdentity, _destination_key() (+59 more)

### Community 10 - "ModelInjectedRequest"
Cohesion: 0.01
Nodes (37): ContainerValidationError, ContainerWiringError, ServiceRegistryUnavailableError, ServiceResolutionError, ModelTopicCatalogQuery, ModelInjectedRequest, read(), read_uuid() (+29 more)

### Community 11 - "lab_pass_receipt.py"
Cohesion: 0.02
Nodes (121): read_receipt_binding_only(), artifact_name(), bind_commit_checks(), build_bus_event(), build_post_merge_event(), build_receipt(), chain_lane_checks(), check_consumer_group_lag() (+113 more)

### Community 12 - "urllib_parse"
Cohesion: 0.01
Nodes (40): apply_migration(), connect_with_retry(), database_exists(), database_url_for_database(), ensure_database_exists(), ensure_schema_migrations_table(), extract_sequence_number(), file_checksum() (+32 more)

### Community 13 - "agent.py"
Cohesion: 0.02
Nodes (83): _declared_branch_install_repos(), DeployAgent, _before_reexec(), _new_settle_pool(), _read_dev_head(), _read_dev_tags(), _read_running_ref(), _running_contains() (+75 more)

### Community 14 - "re"
Cohesion: 0.01
Nodes (67): check_drift(), _compose_env_for_lane(), _is_optional_and_dev(), _load_manifest(), main(), check_d1_drizzle_tables(), check_d2_topic_conflicts(), _check_is_relevant() (+59 more)

### Community 15 - "runtime_host_process.py"
Cohesion: 0.01
Nodes (16): decode_inmemory_event(), get_event_bus_class(), get_event_bus_registry(), get_handler_class(), get_handler_registry(), _current_runtime_profile_name(), _discover_package_node_contracts(), _normalize_prefetch_policy() (+8 more)

### Community 16 - "importlib"
Cohesion: 0.01
Nodes (77): build_record(), main(), __getattr__(), _declared_run_step_event(), _normalize_result_payload(), _resolve_delegation_provenance(), _resolve_measured_actual_cost(), _run_step_topic() (+69 more)

### Community 17 - "handler_wiring.py"
Cohesion: 0.02
Nodes (105): QuarantinePublishUnconfirmedError, _annotation_mentions_event_envelope(), BoundaryPublishError, _build_inference_intent_validation_failure_result(), _build_projection_intents(), _build_projection_terminal_payload(), _build_topic_migration_executor_dependencies(), _coerce_datetime_or_none() (+97 more)

### Community 18 - "ModelRegistrationProjection"
Cohesion: 0.01
Nodes (13): EnumSelectionStrategy, ModelDependencySpec, ModelProjectedFlagMeta, ModelRegistrationProjection, ModelRegistrationSnapshot, SnapshotPublisherRegistration, ProtocolCapabilityProjection, ProtocolCapabilityQuery (+5 more)

### Community 19 - "sys"
Cohesion: 0.01
Nodes (65): find_source_contracts(), main(), check_file(), main(), check_parity(), _default_omni_home(), extract_topics_from_array(), load_registry_topics() (+57 more)

### Community 20 - "collections_abc"
Cohesion: 0.01
Nodes (13): observe_delegate_queue_depth(), check_pin(), contract_path(), _contract_settings(), _load(), load_settings(), publish(), docker_started_at() (+5 more)

### Community 21 - "os"
Cohesion: 0.01
Nodes (59): Infrastructure Primitives for Atomic Operations, backfill(), ConfigurationError, ErrorCode, extract_capability_tags(), extract_contract_type(), extract_contract_version(), extract_intent_types() (+51 more)

### Community 22 - "dataclasses"
Cohesion: 0.01
Nodes (32): evaluate_trivyignore(), _format_report(), main(), ModelTrivyIgnoreEntry, ModelTrivyIgnoreVerdict, ModelTrivyIgnoreViolation, _parse_entries(), _validate_entry() (+24 more)

### Community 23 - "argparse"
Cohesion: 0.02
Nodes (90): _discover_contract_paths(), main(), run_batch_validation(), _validate_contract_node_type(), cmd_stamp(), main(), check(), discover_reconcilers() (+82 more)

### Community 24 - "ModelAutoWiringManifest"
Cohesion: 0.02
Nodes (75): EnumContractAttachStatus, ModelContractAttachResult, ModelTopicReadinessConfig, report_roster_partition(), verify_roster_partition(), contract_publish_topics(), ensure_topics_exist(), provision_unattached_publish_topics() (+67 more)

### Community 25 - "ModelBackendResult"
Cohesion: 0.01
Nodes (26): MixinPostgresErrorResponse, PostgresErrorContext, MixinPostgresOpExecutor, ModelBackendResult, HandlerPostgresCleanupTopics, HandlerPostgresContractUpsert, HandlerPostgresDeactivate, HandlerPostgresMarkStale (+18 more)

### Community 26 - "RuntimeHostProcess"
Cohesion: 0.02
Nodes (12): ModelDuplicateResponse, _baseline_subscription_wiring_disabled(), _parse_contract_event_payload(), _raw_contract_owned_by_runtime_profile(), _requires_raw_event_projection_wiring(), RuntimeHostProcess, _make_factory(), convert_value() (+4 more)

### Community 27 - "service_kernel.py"
Cohesion: 0.01
Nodes (6): configure_logging(), main(), raise_nofile_soft_limit(), configure_tracing(), _omnimemory_enabled(), _validate_all_suffixes()

### Community 28 - "ast"
Cohesion: 0.02
Nodes (62): check(), check_file(), check_run_block(), discover(), imported_names(), main(), module_level_names(), _steps() (+54 more)

### Community 29 - "event_bus_kafka.py"
Cohesion: 0.01
Nodes (3): _cleanup_quietly(), connect_with_bounded_retry(), _resolve_ssl_context()

### Community 30 - "RepositoryExecutionError"
Cohesion: 0.02
Nodes (20): [0.3.2] - 2026-02-02, Changed, Database Repository Models Migration, Dependencies, RepositoryContractError, RepositoryExecutionError, RepositoryTimeoutError, ModelValidationLedgerEntry (+12 more)

### Community 31 - "ProtocolEventBusLike"
Cohesion: 0.02
Nodes (26): EnumRuntimeHeartbeatRefusalReason, HandlerRegistrationFromAnnouncement, HandlerRuntimeIntrospectionStartup, RuntimeHeartbeatRequiredError, ModelDiscoveredCapabilities, ModelRegistrationFromAnnouncementRequest, ModelRegistrationFromAnnouncementResult, ModelRuntimeHeartbeatRefusal (+18 more)

### Community 32 - "lab_alarm.py"
Cohesion: 0.02
Nodes (80): EnumLabPassResult, _append_jsonl(), _build_effects_reader(), failing_reader(), _build_lag_reader(), failing_reader(), build_parser(), CanaryVerdictReader (+72 more)

### Community 33 - "EventBusKafka"
Cohesion: 0.02
Nodes (21): _publish_to_kafka(), ProjectionWedgeExhaustedError, _consume_failure_extra(), EventBusKafka, _commit_low_watermark(), _dispatch_one(), _drain(), _settle_rewinds() (+13 more)

### Community 34 - "rsd_live_delegation_overlay.py"
Cohesion: 0.02
Nodes (45): ModelRsdArtifactSourceRelation, ModelRsdLiveDelegationAuthorityEnvelope, ModelRsdLiveDelegationObservation, ModelRsdLiveDelegationOverlay, ModelRsdLiveDelegationPreflightResult, ModelRsdLiveDelegationResultAnchor, ModelRsdModelArtifactProvenance, ModelRsdPostgresAcceptanceOverlay (+37 more)

### Community 35 - "cli_skill.py"
Cohesion: 0.02
Nodes (42): _task_class_names(), TaskTypeOption, _emit_skill_routing_error(), _resolve_packaged_contract(), run_node_by_name(), _coerce_value(), load_skill_registry(), _parse_skill_args() (+34 more)

### Community 36 - "subprocess"
Cohesion: 0.02
Nodes (75): audit_repo_main(), collect_check_run_names(), collect_default_branch_shas(), collect_merged_pr_head_shas(), find_orphan_contexts(), leaf_form(), main(), normalize_forms() (+67 more)

### Community 37 - "runtime/models/__init__.py"
Cohesion: 0.01
Nodes (3): ModelEventBusAggregateHealth, ModelProtocolRegistrationConfig, ModelRuntimeAggregateHealth

### Community 38 - "tools/contract_topic_extractor.py"
Cohesion: 0.02
Nodes (43): _build_parser(), _build_specs(), main(), _run_dry(), contract_topic_member_key(), contract_topic_value(), _contract_topics(), _parse_contract_topics() (+35 more)

### Community 39 - "RegistrationReducerService"
Cohesion: 0.02
Nodes (25): Fixed, ModelNodeBecameActive, ModelNodeLivenessExpired, ModelNodeRegistrationAckReceived, ModelNodeRegistrationAckTimedOut, ModelNodeRegistrationInitiated, ModelPayloadPostgresUpdateRegistration, ModelRegistrationAckUpdate (+17 more)

### Community 40 - "application_database_acl.py"
Cohesion: 0.02
Nodes (43): _git_blob(), _IndentedSafeDumper, main(), _parse_args(), _parse_repository_roots(), _SourceLock, _write_yaml(), _append_inventory_objects() (+35 more)

### Community 41 - "prepr_runtime_pool.py"
Cohesion: 0.03
Nodes (81): acquire_lease(), assess(), build_parser(), build_publisher(), publisher(), _check_map(), declared_changed_path(), declared_lane_projects() (+73 more)

### Community 42 - "receipt_mode.py"
Cohesion: 0.02
Nodes (44): ModelDelegateLocusDecision, ModelReceiptRuntimeSummary, capture_log_path(), _close_capture_logging(), _configure_capture_logging(), create_receipt_artifact_store(), _declared_correlation_ids(), default_emit_socket_path() (+36 more)

### Community 43 - "sanitize_error_message"
Cohesion: 0.02
Nodes (29): _do_export(), _do_seed(), _extract_requirements(), _load_infisical_credentials(), _load_machine_class_overlay(), main(), _print_diff_summary(), _seed_machine_class() (+21 more)

### Community 44 - "ServiceRegistration"
Cohesion: 0.02
Nodes (12): _partition_endpoints_by_auth(), PluginLlm, _registry_declarations(), PluginDlq, _get_default_projector_contracts_dir(), ServiceRegistration, ModelDomainPluginConfig, ModelDomainPluginResult (+4 more)

### Community 45 - "HandlerHttpRest"
Cohesion: 0.02
Nodes (16): EnumInfraResourceType, EnumKafkaAcks, MSKTokenProvider, OAuthBearerTokenProvider, _categorize_size(), HandlerHttpRest, ModelHttpBodyContent, DependencyMaterializer (+8 more)

### Community 46 - "SecretResolver"
Cohesion: 0.02
Nodes (15): SecretResolutionError, ModelCachedSecret, ModelRuntimeProfilePolicy, ModelSecretCacheStats, ModelSecretMapping, ModelSecretNamespaceRule, ModelSecretResolverConfig, ModelSecretResolverMetrics (+7 more)

### Community 47 - "release_train.py"
Cohesion: 0.03
Nodes (73): _check_host_field(), check_lockfile(), count_packages(), _extract_host(), main(), ModelLockfileHostFinding, build_parser(), bump_declared_version() (+65 more)

### Community 48 - "ModelSessionStateResult"
Cohesion: 0.02
Nodes (15): EnumSessionLifecycleState, HandlerRunContextRead, HandlerRunContextWrite, HandlerSessionIndexRead, HandlerSessionIndexWrite, HandlerStaleRunGC, ModelRunContext, ModelRunContextReadRequest (+7 more)

### Community 49 - "models/validation/__init__.py"
Cohesion: 0.02
Nodes (16): ModelCategoryMatchResult, RegistryProtocolBinding, MessageOutputCategory, check_routing_coverage_ci(), discover_message_types(), discover_registered_routes(), _discover_routes_static(), _get_base_classes() (+8 more)

### Community 50 - "CHANGELOG.md"
Cohesion: 0.01
Nodes (141): Added, Added, Added, Added, Added, Bug Fixes, Bug Fixes, Bug Fixes (+133 more)

### Community 51 - "_parse_contract"
Cohesion: 0.02
Nodes (52): announcer_identity(), apply_announcement(), build_announcements(), build_catalog_direct(), _declared_topics(), _entry_point_is_active(), HandlerPackageAnnouncer, installed_file_reader() (+44 more)

### Community 52 - "JobStore"
Cohesion: 0.03
Nodes (32): EnumAcceptBacklogStatus, EnumJobNotStarted, ModelLineageDecision, ModelOneShotRerun, ModelOnexApiDelivery, ModelVerifyRecreate, PhaseStatus, _accept_backlog_block() (+24 more)

### Community 53 - "ContractRegistryReducer"
Cohesion: 0.02
Nodes (11): load_intent_target_uris(), ModelIntentTargetUris, ModelNodeHeartbeatEvent, ContractRegistrationEventRouter, ModelContractRegistryState, ContractRegistryReducer, ModelNodeGraphState, ModelReducerOutput (+3 more)

### Community 54 - "check_dev_lane_staleness.py"
Cohesion: 0.03
Nodes (70): agent_is_idle(), Ancestry, _AncestryResolver, assert_lane_fence(), bind_generation_to_job_end(), bind_reemission_candidates(), _binding_only_receipts(), _contains_on_branch() (+62 more)

### Community 55 - "post_merge/consumer.py"
Cohesion: 0.02
Nodes (38): _build_event(), _extract_ticket_ids(), main(), _publish(), ModelPRMergedEvent, load_network_binding_settings(), load_third_party_endpoints(), _check_compose_version() (+30 more)

### Community 56 - "VerifiedProjectionTenantAuthority"
Cohesion: 0.02
Nodes (30): ProjectionNotMaterializedError, ProjectionQueryRowBudgetError, HandlerProjectionDerivedWrite, HandlerProjectionWriteCursor, ModelProjectionDerivedWriteRequest, ProtocolProjectionDriverCursor, ProtocolProjectionWriteConnection, ProtocolProjectionWriteCursor (+22 more)

### Community 57 - "backend_probe.py"
Cohesion: 0.02
Nodes (45): discover_backends(), EventBusResolutionAmbiguousError, _import_event_bus_inmemory(), _normalize_bus_value(), resolve_bus_type(), select_event_bus(), BootstrapCauseRecorder, ConsumerGroupDescribeDeniedError (+37 more)

### Community 58 - "handler_contract_subscribe_preflight.py"
Cohesion: 0.02
Nodes (36): _committed(), ConsumerGroupFenceError, _fence_retired_group(), hand_over_committed_offsets(), _hand_over_on_kafka(), load_consumer_group_handover(), load_contract_handover(), _member_count() (+28 more)

### Community 59 - "prove_application_database_domain_enforcement.py"
Cohesion: 0.03
Nodes (51): _assert_red(), _authoritative_declarations(), _behavioral_evidence(), _catalog_identities(), _catalog_routine_dependency_states(), _catalog_surface_authority_columns(), _column_states(), _connect() (+43 more)

### Community 60 - "deploy_agent/consumer.py"
Cohesion: 0.03
Nodes (33): canonical_signing_body(), compute_signature(), hmac_key_check(), sign_envelope(), verify_command(), ModelCoalescePlan, ModelQueuedCommand, plan_coalesce() (+25 more)

### Community 61 - "cli_auth.py"
Cohesion: 0.03
Nodes (10): auth_group(), _emit(), ledger_group(), ledger_query(), ledger_read(), _read(), run_work_ledger_read(), read_gateway_block() (+2 more)

### Community 62 - "TopicProvisioner"
Cohesion: 0.03
Nodes (25): _run_live(), _broker_count(), _ensure_topic(), EnumTopicReadinessFailureReason, EnumTopicReadinessStatus, ModelTopicReadinessFailure, ModelTopicSetReadiness, _cli_main() (+17 more)

### Community 63 - "handler_evidence_autoclose_sweep.py"
Cohesion: 0.03
Nodes (37): _attachment_urls(), _cited_product_pr_refs(), _closer_flip_was_reverted(), _completed_entry_since(), _counted_ac_bindings(), _evidence_after_revert(), _excluded_tickets(), _extract_ticket_binding() (+29 more)

### Community 64 - "handler_lab_proof_plan.py"
Cohesion: 0.02
Nodes (8): main(), self_test(), _contract_operations(), load_lab_proof_profile_registry(), validate_steps_against_repo(), _broker(), _hashes(), _tree_hash()

### Community 65 - "MixinNodeIntrospection"
Cohesion: 0.02
Nodes (32): [0.2.0] - 2026-01-17, 1. Cache Invalidation Method Signature Change, 2. Configuration Model API, Added, Automated Migration, Breaking Changes, Changed, CI/CD (+24 more)

### Community 66 - "enum"
Cohesion: 0.02
Nodes (20): EnumProjectTrackerIssueStatusType, EnumSkillTerminalPublishOutcome, ModelSkillTerminalPublishReport, EnumContractTransportKey, EnumProjectTrackerBackend, EnumRuntimeRestartPolicy, EnumSecurityRuleId, EnumUpdateTriggerType (+12 more)

### Community 67 - "_prepare_handler_wiring"
Cohesion: 0.02
Nodes (49): read_contract_yaml(), EnumQuarantineReason, _bound_dispatch_identifier(), _commit_handler_wiring(), _contracts_root_for_runtime_dependencies(), _delegation_fault_pre_dispatch_guard(), _derive_dispatcher_id(), derive_entry_message_category() (+41 more)

### Community 68 - "EnumCheckSeverity"
Cohesion: 0.03
Nodes (41): EnumCheckSeverity, EnumContractCheckType, EnumValidationVerdict, is_promotion_blocked(), BatchVerificationConfig, _compute_overall_verdict(), _discover_contracts(), _get_probe_requirement() (+33 more)

### Community 69 - "InfraUnavailableError"
Cohesion: 0.02
Nodes (17): InfraUnavailableError, header_identity_fields_from_envelope(), EventBusInmemory, ModelPublishReceipt, GatewayHttpResponse, _deliver(), publish_batch(), InmemoryTransactionalEpochs (+9 more)

### Community 70 - "check_lane_sibling_revision.py"
Cohesion: 0.03
Nodes (47): AcceptanceUnresolvedError, EnumConvergenceOutcome, EnumLaneBuildOutcome, ModelAcceptanceProbe, ModelAgentAcceptance, ModelConvergenceBudget, read_agent_acceptance(), read_compose_service() (+39 more)

### Community 71 - "KafkaContractSource"
Cohesion: 0.02
Nodes (13): ModelHandlerDescriptor, _parse_handler_routing(), ModelContractVersion, ModelEventBusWiring, ModelHandlerRouting, EnumMaterializationRejection, EnumMaterializationStatus, ContractYamlParser (+5 more)

### Community 72 - "ModelCheckpoint"
Cohesion: 0.02
Nodes (11): EnumCheckpointPhase, ModelCheckpoint, HandlerCheckpointList, HandlerCheckpointRead, HandlerCheckpointWrite, ModelCheckpointEffectInput, RegistryInfraCheckpointEffect, HandlerCheckpointValidate (+3 more)

### Community 73 - "EnumHandlerType"
Cohesion: 0.02
Nodes (10): EnumHandlerType, HandlerSiblingClonePin, ModelDbDescribeResponse, ModelSiblingClonePinRequest, ModelSiblingClonePinResult, HandlerChainLearnInitiate, HandlerChainRetrievalComplete, HandlerChainStoreComplete (+2 more)

### Community 74 - "HandlerQdrant"
Cohesion: 0.02
Nodes (8): AdapterSimilarityEnrichment, _format_results_markdown(), HandlerQdrant, ModelTtlValueCell, HandlerEmbeddingOpenaiCompatible, _parse_openai_embeddings(), _parse_openai_usage(), ModelLlmEmbeddingRequest

### Community 75 - "ModelUpdatePlan"
Cohesion: 0.03
Nodes (17): _anchor_for_plan(), _build_markdown_table(), HandlerPlanToPRComment, _parse_owner_repo(), HandlerPlanToYaml, ModelPRCommentResult, ModelYamlEmitResult, HandlerImpactAnalysis (+9 more)

### Community 76 - "HandlerGraph"
Cohesion: 0.03
Nodes (8): Intent Handler Routing (Demo), HandlerGraph, HandlerIntent, ModelGraphExecutePayload, ModelGraphQueryPayload, ModelGraphRecord, ModelGraphHandlerResponse, MixinEnvelopeExtraction

### Community 77 - "build_budget.py"
Cohesion: 0.03
Nodes (40): _buildable_services(), BuildBudgetError, count_build_steps(), create_command(), derive_image_build_budget(), ensure_builder(), EnumBuildOutcome, _exists() (+32 more)

### Community 78 - "urllib_request"
Cohesion: 0.03
Nodes (41): build_parser(), checks_from_record(), fetch_once(), _gh_compare_status(), main(), poll(), resolve_relation(), resolve_via_latest() (+33 more)

### Community 79 - "handler_runtime_deploy.sh"
Cohesion: 0.06
Nodes (107): acquire_lock(), acquire_lock_with_holder(), assert_broker_reachable(), assert_deployed_migration_tree_synced(), bringup_full_stack(), BROKER_PARTITION_CAP_SERVICE, BROKER_READINESS_SERVICE, build_images() (+99 more)

### Community 80 - "node_chain_orchestrator/models/__init__.py"
Cohesion: 0.03
Nodes (21): ModelChainEntry, ModelChainLearnCommand, ModelChainLearnResult, ModelChainMatch, ModelChainReplayInput, ModelChainReplayResult, ModelChainRetrievalDecision, ModelChainRetrievalResult (+13 more)

### Community 81 - "ServiceLlmEndpointHealth"
Cohesion: 0.02
Nodes (12): is_coding_plan_endpoint(), EnumLlmEndpointProbeState, ModelLlmEndpointHealthConfig, _validate_url_map(), ModelLlmEndpointHealthEvent, ModelLlmEndpointStatus, _declared_probe_urls(), EndpointCircuitBreaker (+4 more)

### Community 82 - "math"
Cohesion: 0.03
Nodes (24): parse_priority_contracts(), parse_ready_bound_seconds(), _process_start_monotonic(), required_topics_without_partitions(), seconds_since_process_start(), start_ready_bound_watch(), watch_ready_bound(), compute_freshness_score() (+16 more)

### Community 83 - "handler_savings_correlation.py"
Cohesion: 0.03
Nodes (4): ProtocolPublisher, format_result(), _load_env_file(), main()

### Community 84 - "._publish_raw_to_dlq"
Cohesion: 0.03
Nodes (38): MixinKafkaDlq, ProtocolKafkaDlqHost, ModelDlqEvent, get_dlq_topic_for_original(), is_dlq_topic(), envelope_consumer_group(), _await_event_bus_publish(), _boundary_dlq_enabled() (+30 more)

### Community 85 - "run_suite.py"
Cohesion: 0.03
Nodes (30): cli(), compose_command(), env(), env_down(), env_up(), _run_compose(), _wait_for_healthy(), get_broker() (+22 more)

### Community 86 - "EnumHandlerTypeCategory"
Cohesion: 0.02
Nodes (9): EnumHandlerTypeCategory, HandlerC11DelegationLegs, ModelC11DelegationLegsRequest, HandlerBoardProbeResultPublisher, EnumChainVerifyState, HandlerChainVerify, HandlerScopeManifestWrite, ModelScopeManifestWriteRequest (+1 more)

### Community 87 - "bootstrap"
Cohesion: 0.02
Nodes (34): ProtocolIntentEffect, decode_event_message(), drop_blank_secret_names(), env_secret_store_is_authoritative(), gate_manifest_on_required_secrets(), required_secret_names(), resolve_present_secret_names(), ModelRuntimeBuildSha (+26 more)

### Community 88 - "prove_application_database_acl.py"
Cohesion: 0.04
Nodes (53): _capture_live_snapshot(), _controlled_roles(), main(), _parse_args(), _acl_object_keyword(), _AclSnapshot, _actual_acl_set(), _admin() (+45 more)

### Community 89 - "MixinLlmHttpTransport"
Cohesion: 0.03
Nodes (8): MixinLlmHttpTransport, _parse_cidr_allowlist(), _parse_cloud_endpoint_host_allowlist(), MixinRetryExecution, ProtocolCircuitBreakerAware, ModelRetryErrorClassification, ModelOperationContext, ModelRetryState

### Community 90 - "omnimarket_drift_guard.py"
Cohesion: 0.03
Nodes (29): EnumOffRegistryReason, EnumOffRegistryVerdict, ModelOffRegistryCheck, ModelOmnimarketLagStamp, canonical_clone_attachment(), canonical_distribution_name(), canonical_local_omnimarket_commit(), CanonicalCloneAttachment (+21 more)

### Community 91 - "generated/__init__.py"
Cohesion: 0.03
Nodes (20): EnumDeployTopic, EnumGitTopic, EnumGithubTopic, EnumLinearTopic, EnumOccTopic, EnumOmnibaseInfraTopic, EnumOmniclaudeTopic, EnumOmniintelligenceTopic (+12 more)

### Community 92 - "ModelCheckResult"
Cohesion: 0.03
Nodes (12): ModelCheckResult, ModelPatternCandidate, HandlerArtifactCompleteness, HandlerReplaySanity, HandlerCheckExecutor, HandlerSubprocessCheckExecutor, ModelCheckExecutorConfig, HandlerCostTokenDelta (+4 more)

### Community 93 - "ConsumerRejoinSupervisor"
Cohesion: 0.03
Nodes (14): evaluate_consumer_stall(), ModelConsumerRejoinPolicy, ConsumerRejoinSupervisor, observed(), ModelConsumerPollBatch, read_coordinator_known(), HandlerConsumerCoordinatorRejoin, EnumConsumerStallReason (+6 more)

### Community 94 - "StatusApp"
Cohesion: 0.03
Nodes (12): StatusApp, consume_all(), HookEventReceived, PRStatusReceived, SnapshotReceived, ScreenStatus, _format_hook_line(), _outcome_text() (+4 more)

### Community 95 - "EventBusSubcontractWiring"
Cohesion: 0.03
Nodes (22): [0.4.0] - 2026-02-05, Added, Breaking Changes, Changed, Contract Dependency Resolution, Dependencies, Deprecated, Event Ledger Integration Tests (+14 more)

### Community 96 - "prove.py"
Cohesion: 0.04
Nodes (36): _adapter(), _admin_rows(), _assert_database_identities(), _initialize_database(), _KeyProvider, main(), _parse_args(), _prove_no_authority_records_the_claim() (+28 more)

### Community 97 - "ModelHandlerValidationError"
Cohesion: 0.03
Nodes (18): EnumValidationSeverity, ModelHandlerValidationError, ModelHandlerIdentifier, ModelEnvironmentPolicy, ModelHandlerSecurityPolicy, _create_io_error(), _create_parse_error(), _create_size_limit_error() (+10 more)

### Community 98 - "ModelRegistryApiRequest"
Cohesion: 0.02
Nodes (11): HandlerRegistryApiGetContract, HandlerRegistryApiGetDiscovery, HandlerRegistryApiGetHealth, HandlerRegistryApiGetNode, HandlerRegistryApiGetTopic, HandlerRegistryApiGetWidgetMapping, HandlerRegistryApiListContracts, HandlerRegistryApiListNodes (+3 more)

### Community 99 - "HandlerLedgerQuery"
Cohesion: 0.03
Nodes (9): ModelPayloadLedgerAppend, bind_handler_db(), HandlerLedgerAppend, HandlerLedgerQuery, ModelLedgerAppendResult, ModelLedgerEntry, ModelLedgerQuery, ModelLedgerQueryResult (+1 more)

### Community 100 - "x-runtime-env"
Cohesion: 0.02
Nodes (93): x-runtime-env, ARCH_GRAPH_BOLT_URI, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_VERIFY_ENDPOINTS, BIFROST_VERTEX_GEMINI_ENDPOINT_URL, BUS_ID, DELEGATION_ROUTING_TIERS_PATH (+85 more)

### Community 101 - "ModelEventMessage"
Cohesion: 0.03
Nodes (6): ModelEventMessage, HandlerValidatorEventLedgerProject, ModelKafkaUngroupedReader, HandlerBuildLoopProjection, HandlerGatewayLinkHealthProjection, HandlerLedgerProjection

### Community 102 - "HandlerGatewayHeartbeat"
Cohesion: 0.03
Nodes (19): HandlerGatewayAttach, HandlerGatewayDetach, SessionNotFoundError, HandlerGatewayHeartbeat, SessionNotFoundError, GatewayCircuitBreakerGuard, introspect_token(), EnumGatewaySessionEventType (+11 more)

### Community 103 - "StoreEffectIdempotencyInmemory"
Cohesion: 0.03
Nodes (7): ModelEffectIdempotencyConfig, ModelRegistryRequest, ModelRegistryResponse, ProtocolEffectIdempotencyStore, NodeRegistryEffect, CacheEntry, StoreEffectIdempotencyInmemory

### Community 104 - "ModelEventHeaders"
Cohesion: 0.03
Nodes (11): EnumConsumerGroupPurpose, ModelEventHeaders, EventBusRecovering, restore(), attach(), on_message(), extract_replay_count_from_raw_headers(), ModelConsumerRejoinAlarm (+3 more)

### Community 105 - "MessageDispatchEngine"
Cohesion: 0.03
Nodes (9): ModelMaterializedDispatch, DispatchEntryInternal, DispatchMatchCriteriaInternal, _find_duplicate_identifiers(), _get_route_dispatcher_id(), MessageDispatchEngine, ModelLogContextKwargs, PayloadScopingOutcomeInternal (+1 more)

### Community 107 - "HandlerDlqDepthMonitor"
Cohesion: 0.03
Nodes (11): HandlerDlqDepthEvaluate, EnumDlqDepthVerdict, ModelDlqDepthEvaluateRequest, ModelDlqDepthEvaluateResult, ModelDlqThresholdPolicy, ModelDlqTopicObservation, ModelDlqTopicVerdict, HandlerDlqDepthMonitor (+3 more)

### Community 108 - "application_database_domain_enforcement.py"
Cohesion: 0.05
Nodes (42): _analyze_application_database_sql(), _analyze_sql_fragment(), application_database_created_catalog_identities(), application_database_sql_target_locations(), application_database_sql_target_requirements(), ApplicationDatabaseSqlTargetRequirement, _balanced_parenthesized(), _compile_relation_target() (+34 more)

### Community 109 - "node_contract_registry_reducer/reducer.py"
Cohesion: 0.03
Nodes (4): NodeContractRegistryReducer, load_artifact_registry(), load_packaged_artifact_registry(), ModelArtifactRegistry

### Community 111 - "handler_a2a_task.py"
Cohesion: 0.04
Nodes (16): A2ASdkTransport, _build_prompt(), _extract_artifacts(), _extract_error(), _extract_texts(), HandlerA2ATask, map_remote_status(), NoopLifecycleEventSink (+8 more)

### Community 112 - "ModelSnapshotTopicConfig"
Cohesion: 0.03
Nodes (13): _get_error_classes(), ModelSnapshotTopicConfig, ModelCircuitBreakerConfig, RegistrationProjectionPool, ModelTransitionNotificationPublisherMetrics, _get_compute_registry_cache_size(), FailedNotificationRecord, TransitionNotificationPublisher (+5 more)

### Community 113 - "sanitize_error_string"
Cohesion: 0.03
Nodes (9): _make_transition_error(), _make_transition_error(), ModelSessionLifecycleState, ModelRuntimeSchedulerConfig, resolve_runtime_scheduler_enabled(), next_tick_slot(), RuntimeScheduler, start_lane_runtime_scheduler() (+1 more)

### Community 114 - "HandlerSetupOrchestrator"
Cohesion: 0.03
Nodes (10): HandlerSetupOrchestrator, ModelSetupEvent, ModelSetupOrchestratorOutput, ModelSetupOrchestratorRequest, ProtocolPreflightEffect, ProtocolProvisionEffect, RegistryInfraSetupOrchestrator, HandlerPreflightCheck (+2 more)

### Community 115 - "prepush_override_grant.py"
Cohesion: 0.05
Nodes (44): pytest_configure(), activation_covers(), activation_path(), append_receipt(), build_grant(), build_parser(), _cmd_consume(), _cmd_mint() (+36 more)

### Community 116 - "environment"
Cohesion: 0.03
Nodes (83): BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BIFROST_VERIFY_ENDPOINTS, BUS_ID (+75 more)

### Community 117 - "HandlerContractSubscribePreflight"
Cohesion: 0.03
Nodes (20): hand_over_from_version_predecessors(), HandlerContractSubscribePreflight, list_groups_on_kafka(), resolve_retired_group(), resolve_successor_group(), supports_offset_handover(), ModelConsumerGroupFence, ModelConsumerGroupHandover (+12 more)

### Community 118 - "environment"
Cohesion: 0.04
Nodes (82): BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BIFROST_VERIFY_ENDPOINTS, BUS_ID (+74 more)

### Community 119 - "lint_topic_names.py"
Cohesion: 0.04
Nodes (23): _build_parser(), _extract_raw_topics_from_contract(), _extract_topics_from_python_file(), _is_excluded_path(), lint_topic(), LintResult, _load_baseline(), _load_placeholder_denylist() (+15 more)

### Community 120 - "seed-keycloak-clients.py"
Cohesion: 0.06
Nodes (46): apply_environment_overrides(), _assert_client_secrets_match_consumers(), _assert_confidential_clients_have_secrets(), _assert_update_preserved_undeclared_access_type(), _attributes_update_value(), _build_create_payload(), _build_update_payload(), _canonical_profile() (+38 more)

### Community 121 - "application_relation_ownership.py"
Cohesion: 0.04
Nodes (25): load_application_database_ownership_identities(), record(), record_with_physical_bridge(), _append_inventory_evidence_violations(), _append_location_conflicts(), _append_service_declarations(), assert_application_relation_ownership(), _declaration_from_table() (+17 more)

### Community 122 - "AdapterLlmProviderOpenai"
Cohesion: 0.04
Nodes (7): AdapterLlmProviderOpenai, ModelLlmAdapterRequest, ModelLlmAdapterResponse, ModelLlmHealthResponse, ModelLlmModelCapabilities, ModelLlmProviderConfig, AdapterLlmProviderScoped

### Community 123 - "environment"
Cohesion: 0.03
Nodes (80): BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BIFROST_VERIFY_ENDPOINTS, BUS_ID (+72 more)

### Community 124 - "handler_reward_binder.py"
Cohesion: 0.03
Nodes (12): _build_evidence_refs(), _compute_idempotency_key(), _compute_objective_fingerprint(), _compute_reward_delta(), HandlerRewardBinder, ModelEvaluationResult, ModelEvidenceBundle, ModelEvidenceItem (+4 more)

### Community 125 - "kafka_contract_source.py"
Cohesion: 0.04
Nodes (15): NodeInventoryError, bind_introspection_manifest_identity(), _unbound_build_sha(), ModelQuarantineRecord, build_node_inventory(), build_parser(), _discovered_inventory(), main() (+7 more)

### Community 126 - "EnumInfraTransportType"
Cohesion: 0.04
Nodes (11): EnumInfraTransportType, ConfigPrefetcher, ModelPrefetchResult, ContractConfigExtractor, _resolve_transport(), ModelConfigRequirement, ModelConfigRequirements, ModelTransportConfigSpec (+3 more)

### Community 127 - "ModelDevLaneCodeRefPin"
Cohesion: 0.04
Nodes (20): followed_branch(), HandlerDevLaneCodeRefPin, pin_path(), read_active_pin(), release_pin(), write_pin(), assert_services_declared(), ComposeServiceMissingError (+12 more)

### Community 128 - "ModelRuleCheckResult"
Cohesion: 0.05
Nodes (9): ModelRRHEnvironmentData, ModelRRHProfile, ModelRRHRuleSeverity, ModelRuleCheckResult, HandlerRRHValidate, _truncate_pattern(), ModelRRHContractGovernance, ModelRRHValidateRequest (+1 more)

### Community 129 - "apply_application_database_acl.py"
Cohesion: 0.04
Nodes (26): _dev_contractor_main(), _execute_sql(), psql(), _public_connect_revoked(), readback_sql(), _restore_from_snapshot(), _rows(), run() (+18 more)

### Community 130 - "HandlerLlmOpenaiCompatible"
Cohesion: 0.04
Nodes (14): LLM Inference Infrastructure, AdapterCodeReviewAnalysis, AdapterSummarizationEnrichment, ModelLlmInferenceResponse, ModelLlmToolCall, _compute_input_hash(), HandlerLlmOpenaiCompatible, _parse_tool_calls() (+6 more)

### Community 131 - "handler_coding_agent_invoke.py"
Cohesion: 0.04
Nodes (13): _as_dict(), _as_mapping(), _coerce(), _hash(), NodeCodingAgentFsmReducer, _diff_hash(), _git(), _git_capture_diff() (+5 more)

### Community 132 - "AdapterLinearGraphQLProjectTracker"
Cohesion: 0.05
Nodes (13): AdapterLinearGraphQLProjectTracker, _coerce_str(), _comment_from_graphql(), _issue_from_graphql(), LinearGraphQLHealthStatus, _project_from_graphql(), LocalStubProjectTracker, _now_iso() (+5 more)

### Community 133 - "DispatchResultApplier"
Cohesion: 0.04
Nodes (8): ModelPrincipalTopic, ModelProjectionResultLocal, ProtocolProjectionEffect, build_contract_result_applier(), build_static_result_applier(), DispatchResultApplier, IntentExecutor, ProtocolIntentEffect

### Community 134 - "ModelKafkaEventBusConfig"
Cohesion: 0.03
Nodes (15): resolve_work_ledger_query_transport(), KafkaReadbackSource, build_aiokafka_auth_kwargs(), create_offset_consumer(), ModelKafkaEventBusConfig, _log_demand_measure(), IdleTopicWatch, nothing_to_stop() (+7 more)

### Community 135 - "HandlerBifrostGateway"
Cohesion: 0.04
Nodes (11): ModelBifrostShadowConfig, ModelShadowDecisionLog, CircuitBreakerState, HandlerBifrostGateway, ProtocolShadowPolicy, _safe_invoke_callback(), ModelBifrostBackendConfig, ModelBifrostConfig (+3 more)

### Community 136 - "x-dogfood-runtime-env"
Cohesion: 0.03
Nodes (73): x-dogfood-runtime-env, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BUS_ID (+65 more)

### Community 137 - "x-judge-runtime-env"
Cohesion: 0.03
Nodes (73): x-judge-runtime-env, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BUS_ID (+65 more)

### Community 138 - "x-lakshman-runtime-env"
Cohesion: 0.03
Nodes (73): x-lakshman-runtime-env, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_LOCAL_CODER_ENDPOINT_URL, BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL, BIFROST_LOCAL_REASONER_ENDPOINT_URL, BIFROST_SOURCE_CONTRACT_PATH, BUS_ID (+65 more)

### Community 139 - "AdapterKafkaAdminLag"
Cohesion: 0.03
Nodes (13): consumer_group_topic_backlog(), build_aiokafka_auth_kwargs_for(), resolve_lane_client_transport(), AdapterKafkaAdminLag, ModelConsumerGroupLag, ModelTopicPartitionOffset, ProtocolKafkaLagConsumer, _coerce_offset() (+5 more)

### Community 140 - "nonrequired_check_failure_rate.py"
Cohesion: 0.06
Nodes (35): active_workflows(), Alert, AlertDecision, AlertDestinationMissingError, _cell(), count_failures(), decide_alerts(), dedup_message() (+27 more)

### Community 141 - "ModelDelegationDiagnosis"
Cohesion: 0.05
Nodes (12): CheckDelegationGateway, CheckDelegationIdentity, CheckDelegationLocalModel, CheckDelegationQuota, gateway_text(), render_diagnosis(), request_whoami_status(), run_async() (+4 more)

### Community 142 - "HandlerManifestPersistence"
Cohesion: 0.04
Nodes (13): HandlerManifestPersistence, _do_query_io(), _do_retrieve_io(), _do_store_io(), ModelManifestMetadata, ModelManifestQueryResult, ModelManifestRetrieveResult, ModelManifestStoreResult (+5 more)

### Community 143 - "audit-runner-routing.py"
Cohesion: 0.06
Nodes (40): audit_github_variables(), audit_local_workflows(), audit_private_repo_hosted_exception(), audit_private_repo_hosted_placement(), _audit_pull_request_target_caller(), audit_route_wiring(), audit_scoped_variables(), audit_selector_markers() (+32 more)

### Community 144 - "worktree_prune_timer.py"
Cohesion: 0.06
Nodes (32): BranchVerdict, checked_out_branches(), classify(), verdict(), default_branch(), delete(), has_open_pr(), head_contains_tip() (+24 more)

### Community 145 - "ContractLinter"
Cohesion: 0.05
Nodes (12): EnumContractViolationSeverity, ContractLinter, convert_violation_to_handler_error(), _derive_handler_id_from_path(), _get_yaml_line_numbers(), lint_contract_file(), lint_contracts_ci(), lint_contracts_in_directory() (+4 more)

### Community 146 - "ModelNodeIdentity"
Cohesion: 0.04
Nodes (16): ModelNodeIdentity, load_probe_settings(), ModelVerificationProbeSettings, _is_core_registration_topic(), _rpk_watermark_fallback(), _derive_consumer_group_id(), _discover_identity_via_rpk(), _discover_split_group_topics() (+8 more)

### Community 147 - "handler_reconcile_workspace_venvs.py"
Cohesion: 0.04
Nodes (11): _approved(), HandlerHookVenvReconcileHost, HandlerReconcileWorkspaceVenvs, _normalize(), _requirement_name(), ModelHookVenvReconcilePolicy, ModelHookVenvReconcileRequest, ModelHookVenvReconcileResult (+3 more)

### Community 148 - "main"
Cohesion: 0.05
Nodes (22): AclRow, _apply_grants(), audit_acl_rows(), Broker, _broker_from_environment(), BrokerConfig, ContainerBroker, Declaration (+14 more)

### Community 149 - "c29_customer_byo_key_probe.py"
Cohesion: 0.07
Nodes (41): _as_dict(), classify_connects(), Clause, Connect, delegate_runs_root(), DnsMessage, _find_refusal(), find_source_trees() (+33 more)

### Community 150 - "bulk_pr_throttle.py"
Cohesion: 0.05
Nodes (33): BulkPrThrottleError, BulkRunReport, CreateSpec, gh_apply_pr_operation(), _gh_auth_token(), gh_create_pr(), gh_queue_depth(), gh_repo_allows_auto_merge() (+25 more)

### Community 151 - "deploy-runners.sh"
Cohesion: 0.09
Nodes (68): add_services_deploy(), add_wait_target_idle(), container_is_running(), declared_compose_services(), deploy_runners(), deploy_with_retry(), deregister_retired_runner(), ensure_runner_tmp_dirs() (+60 more)

### Community 152 - "HandlerRuntimeErrorTriage"
Cohesion: 0.04
Nodes (10): EnumRuntimeErrorCategory, EnumRuntimeErrorSeverity, ModelRuntimeErrorEvent, HandlerRuntimeErrorTriage, ModelRuntimeErrorTriageResult, EnumEscalationReason, load_escalation_rule(), ModelEscalationRule (+2 more)

### Community 153 - "recreate_supervisor.py"
Cohesion: 0.04
Nodes (25): _compose_tag_constructor(), ComposeBudgetError, ComposeDurationError, _ComposeLoader, _ComposeOverrideMap, derive_runtime_health_budget(), derive_runtime_phase_budget(), _merge_service_maps() (+17 more)

### Community 154 - "detect_test_paths.py"
Cohesion: 0.06
Nodes (31): _changed_test_paths(), _collectable_modules_under(), collectable_test_file_count(), compute_selection(), _full_suite(), _full_suite_roots(), full_suite_test_file_count(), is_collectable_test_file_name() (+23 more)

### Community 155 - "deploy_source_ref.py"
Cohesion: 0.07
Nodes (34): _assert_git_repo(), assert_manifest_matches_refs(), _assert_owned_clean_target(), build_parser(), checkout_immutable_selections(), clean_checkout(), _cmd_assert(), _cmd_checkout() (+26 more)

### Community 156 - "scoped_effects_deploy.py"
Cohesion: 0.08
Nodes (40): _admission(), _assert_active_config(), _assert_dependencies(), _assert_lock(), _assert_mutable_resources(), _assert_topics(), _assert_unchanged(), _broker_login() (+32 more)

### Community 157 - "ModelChainViolation"
Cohesion: 0.05
Nodes (12): ChainPropagationError, ModelChainViolation, ChainAwareDispatcher, propagate_chain_context(), validate_dispatch_chain(), ChainPropagationValidator, enforce_chain_propagation(), get_causation_id() (+4 more)

### Community 158 - "HandlerDb"
Cohesion: 0.04
Nodes (8): HandlerDb, ModelDbQueryPayload, ModelDbQueryResponse, ModelPayloadValidationLedgerAppend, ModelValidationLedgerAppendResult, bind_handler_db(), HandlerValidationLedgerAppend, sanitize_dsn()

### Community 159 - "validator_declarative_node.py"
Cohesion: 0.05
Nodes (16): ModelDeclarativeNodeValidationResult, ModelDeclarativeNodeViolation, _find_instance_variables(), _extract_self_attrs(), _get_base_class_names(), _get_source_line(), _is_class_exempted(), _is_node_class() (+8 more)

### Community 160 - "routing.py"
Cohesion: 0.05
Nodes (18): consumer_group_for(), build_router_from_env(), DeployRouter, GitTableAtRef, lane_flags_for_instance(), load_routing_table(), ModelAgentInstance, ModelLaneFlags (+10 more)

### Community 161 - "ModelTopicCatalogResponse"
Cohesion: 0.05
Nodes (6): HandlerTopicCatalogPostgres, ModelTopicInfoPostgres, ModelTopicCatalogEntry, ModelTopicCatalogResponse, ServiceTopicCatalog, TopicAccumulator

### Community 162 - "ContractFileWatcher"
Cohesion: 0.04
Nodes (8): HandlerManualTrigger, HandlerPRWebhookIngestion, ModelManualReconcileCommand, ModelPRWebhookEvent, ModelUpdateTrigger, ContractFileWatcher, HandlerContractFileEvent, NodeImpactAnalyzerCompute

### Community 163 - "HandlerMCP"
Cohesion: 0.04
Nodes (3): HandlerMCP, ModelMcpToolResult, ModelNameKeyedTable

### Community 164 - "handler_onboarding.py"
Cohesion: 0.04
Nodes (12): _collect_capability_evidence(), _execute_dag_step(), _handle_dag(), _skipped_step_result(), _status_from_step_results(), EnumOnboardingStatus, ModelStepResult, ModelOnboardingStep (+4 more)

### Community 165 - "handler_standalone_compose_recreate.py"
Cohesion: 0.05
Nodes (14): EnumStandaloneComposeRecreateOutcome, _changed(), _declared(), HandlerStandaloneComposeDocker, HandlerStandaloneComposeRecreate, recreate_standalone_compose(), refusal(), _resolve() (+6 more)

### Community 166 - "ModelLocalProvisionEffectOutput"
Cohesion: 0.04
Nodes (9): compose_base_command(), HandlerLocalProvision, _make_error_output(), str_tuple(), HandlerLocalStatus, _make_error_output(), HandlerLocalTeardown, _make_error_output() (+1 more)

### Community 167 - "lane_census_inventory.py"
Cohesion: 0.06
Nodes (30): api_get(), api_stream(), boot_time_iso(), build_envelope(), cgroup_dir_for_pid(), collect_inventory(), collect_memory_observation(), consume() (+22 more)

### Community 168 - "apply_topic_namespace"
Cohesion: 0.05
Nodes (8): KafkaTransport, apply_topic_namespace(), apply_topic_namespace_all(), create_topic_resolver(), namespace_consumer_group_id(), resolve_topic_namespace(), strip_topic_namespace(), TopicNamespaceError

### Community 170 - "runtime_cost_estimator_build.py"
Cohesion: 0.05
Nodes (16): dev_lane_build_settings(), HandlerCostEstimatorBuild, validate_source(), validated_image(), ModelCostEstimatorBuildConfig, ModelCostEstimatorBuildRequest, ModelCostEstimatorBuildResult, build_dev_lane_services() (+8 more)

### Community 171 - "handler_dev_lane_pin.py"
Cohesion: 0.06
Nodes (16): EnumDevLanePinKey, EnumDevLanePinOutcome, _build_parser(), _check_reference(), DevLanePinRefusedError, HandlerDevLanePin, _ledger_row(), main() (+8 more)

### Community 172 - "MixinAsyncCircuitBreaker"
Cohesion: 0.04
Nodes (6): Circuit Breaker & Dispatcher Resilience, MixinAsyncCircuitBreaker, ConfigTTLCleanup, ModelTTLCleanupResult, ServiceTTLCleanup, WriterInfraRoutingDecisionsPostgres

### Community 173 - "ModelRuntimeAttachReadiness"
Cohesion: 0.04
Nodes (13): _validate_consumer_declarations(), ContractContentHashError, ModelRuntimeAttachReadiness, derive_event_type_alias_for_topic(), build_registered_events(), HandlerPostgresRuntimeManifestInsert, ModelPayloadInsertRuntimeManifest, build_runtime_manifest() (+5 more)

### Community 175 - "ModelContractDiscoveryResult"
Cohesion: 0.04
Nodes (7): EnumHandlerSourceMode, ModelContractDiscoveryResult, ModelHandlerSourceConfig, handler_identity(), HandlerSourceResolver, ProtocolContractSource, PluginLoaderContractSource

### Community 176 - "service_gateway_forwarder.py"
Cohesion: 0.07
Nodes (13): content_addressed_event_id(), _content_event_id_from_headers(), decode_envelope_strict(), egress_admits(), egress_scrubbed(), envelope_with_event_id(), GatewayRecordRefusedError, _header_text() (+5 more)

### Community 177 - "infra_validators.py"
Cohesion: 0.06
Nodes (28): PathInput, _convert_yaml_exemptions(), _count_non_optional_unions(), _create_filtered_result(), ExemptionPattern, _filter_exempted_errors(), _find_isinstance_union_lines(), find_unused_exemptions() (+20 more)

### Community 178 - "instance_lanes.py"
Cohesion: 0.06
Nodes (29): EnumLaneMirrorOutcome, ModelLaneMirrorForwarder, ModelLaneMirrorReceipt, default_instance_lane_env_exports(), _default_run(), discover_lane_overlay_clone(), _forwarder_state(), _git_text() (+21 more)

### Community 179 - "ProtocolMessageDispatcher"
Cohesion: 0.04
Nodes (5): ModelExecutionShapeValidation, ProtocolMessageDispatcher, coerce_message_category(), DispatchEntryInternal, RegistryDispatcher

### Community 180 - "service_runtime_health_monitor.py"
Cohesion: 0.04
Nodes (15): invoke_entry_method(), engine_type_accepts_delivery(), _request_model(), _extract_method_signature(), _extract_protocol_info(), generate_lockfile(), _get_repo_root(), load_lockfile() (+7 more)

### Community 181 - "health_payload.py"
Cohesion: 0.06
Nodes (26): _body_says_starting(), _clip_detail(), decode_health_body(), default_max_verdict_age(), derive_verdict_wait_bound(), evaluate_health_body(), extract_non_healthy_dimensions(), HealthPayload (+18 more)

### Community 182 - "verify_migration_checksum_adoption.py"
Cohesion: 0.07
Nodes (33): build_parser(), declared_surface(), diff_object(), discover_divergent_canonical_rows(), discover_divergent_omnimarket_rows(), discover_non_canonical_rows(), file_sha256(), _flat_migrations() (+25 more)

### Community 183 - "CLAUDE.md - Omnibase Infrastructure"
Cohesion: 0.04
Nodes (42): Adoption decision and remaining work, Agent Behavioral Rules, Architecture: Four-Node Pattern, Autonomous mode safety rails, Branch Protection, CLAUDE.md - Omnibase Infrastructure, Common Pitfalls, Contract-Driven Config Discovery (+34 more)

### Community 184 - ".run"
Cohesion: 0.05
Nodes (8): AcceptBacklogWatchdog, ModelAcceptBacklogVerdict, ModelAcceptQueueSample, read_accept_queue(), read_verdict_record(), LagRefresher, main(), LagSampler

### Community 185 - "TopicPartition"
Cohesion: 0.04
Nodes (4): ProtocolSeekableConsumer, _AiokafkaDlqOffsetReader, TopicPartition, ProtocolTopicPartition

### Community 186 - "AgentActionsConsumer"
Cohesion: 0.05
Nodes (5): ConfigAgentActionsConsumer, AgentActionsConsumer, EnumHealthStatus, _run_once(), mask_dsn_password()

### Community 187 - "c13_customer_local_probe.py"
Cohesion: 0.07
Nodes (34): bifrost_overlay_yaml(), classify(), Clause, Connect, count_kinds(), declared_model_host(), delegate_runs_root(), find_source_trees() (+26 more)

### Community 188 - "runtime_migration_freeze_check.py"
Cohesion: 0.05
Nodes (16): NodeMigrationFreezeCheckCompute, parse_freeze_date(), ModelMigrationFreezeCheckInput, _added_paths(), main(), _render(), _write(), _excluded() (+8 more)

### Community 189 - "handler_chain_canary.py"
Cohesion: 0.04
Nodes (10): _delegation_traffic_class_from_row(), _is_terminal_fsm_state(), _readback_projection_via_asyncpg(), _readback_terminal_via_aiokafka(), render_receipt(), _replay_ledger_chain_via_asyncpg(), _scan_quarantine_tail_via_aiokafka(), _scan_topics_for_correlation() (+2 more)

### Community 190 - "ModelSecretSeedRequest"
Cohesion: 0.05
Nodes (8): HandlerSecretSeed, parse_source(), ProtocolSeedSecretStore, _redact(), _safe_error(), EnumSecretSeedVerdict, ModelSecretSeedRequest, ModelSecretSeedResult

### Community 191 - "InjectionEffectivenessConsumer"
Cohesion: 0.05
Nodes (5): ConfigInjectionEffectivenessConsumer, EnumHealthStatus, InjectionEffectivenessConsumer, _run_once(), mask_dsn_password()

### Community 192 - "reconcile_verify_movement.py"
Cohesion: 0.07
Nodes (33): append_msg(), build_parser(), CloneHealth, _cmd_clone_health(), _cmd_clone_refusal(), _cmd_floor(), _cmd_floor_from_venv(), _cmd_lock_targets() (+25 more)

### Community 193 - "handler_api_fixture_record.py"
Cohesion: 0.08
Nodes (28): _array(), build_signed_webhook(), _capture(), _counts(), _credential(), _expand(), HandlerApiFixtureRecord, finish() (+20 more)

### Community 194 - "EnumPolicyType"
Cohesion: 0.05
Nodes (7): EnumPolicyType, MixinPolicyValidation, ModelPolicyKey, ModelPolicyRegistration, ProtocolPolicy, normalize_version(), validate_policy_type_value()

### Community 195 - "ModelExecutionShapeViolationResult"
Cohesion: 0.06
Nodes (8): ModelExecutionShapeRule, ModelExecutionShapeViolationResult, MixinExecutionShapeViolationChecks, MixinNodeArchetypeDetection, ExecutionShapeValidator, ModelDetectedNodeInfo, validate_execution_shapes(), validate_execution_shapes_ci()

### Community 196 - "check_lab_probe_windows.py"
Cohesion: 0.09
Nodes (30): busy_intervals(), check(), _check_entry(), CommittedPlacements, derive_lane(), _expand_field(), job_name_lanes(), _jobs() (+22 more)

### Community 197 - "resolve_embedded_runtime_config"
Cohesion: 0.04
Nodes (17): ModelDlqTrackingConfig, ModelBootSubscriptionsConfig, ModelContractRegistryConfig, ModelEventBusConfig, ModelLoggingConfig, ModelRuntimeConfig, ModelShutdownConfig, _build_dlq_replay_tracking() (+9 more)

### Community 198 - "action_verify.py"
Cohesion: 0.07
Nodes (36): _as_mapping(), _blocking_check_names(), _blocking_check_names_from_statuses(), _compute_config_hash(), _compute_pr_diff_hash(), _decision(), _decision_from_event(), _extract_receipt_from_pr_body() (+28 more)

### Community 199 - "AdapterTicketLinear"
Cohesion: 0.05
Nodes (7): AdapterEventBusScoped, AdapterTicketLinearScoped, AdapterTicketLinear, _build_issue_filter(), _normalise_issue(), InvariantViolation, is_done_state()

### Community 200 - "managed_staging_canary_catalog.py"
Cohesion: 0.05
Nodes (17): build_canary_catalog_from_candidate(), derive_candidate_group_bases(), extract_candidate_topic_suffixes(), generate_canary_catalog(), load_canary_namespace(), _resolve_roots(), verify_zero_collision(), build_topic_diff() (+9 more)

### Community 201 - "ProtocolSnapshotStore"
Cohesion: 0.04
Nodes (3): ProtocolSnapshotStore, ServiceSnapshot, SnapshotNotFoundError

### Community 203 - "validate_naming.py"
Cohesion: 0.06
Nodes (6): InfraNamingConventionValidator, _is_in_archived_directory(), main(), NamingViolation, ParsedFileInfo, run_ast_validation()

### Community 204 - "grammar.py"
Cohesion: 0.05
Nodes (14): EnumGateBindingProbe, _compiled_forms(), declared_form_summary(), gate_binding_line(), load_gate_binding_grammar(), _normalizations(), normalize_gate_binding(), _reading_line_pattern() (+6 more)

### Community 205 - "TransitionReducer"
Cohesion: 0.06
Nodes (11): ModelCredentialStoreWrite, ModelInteractivePolicy, ModelTransitionBranch, ModelTransition, _interpolate(), _replacer(), InterpolationError, StateDict (+3 more)

### Community 206 - "ReaderInjectionEffectivenessPostgres"
Cohesion: 0.05
Nodes (7): ModelInjectionEffectivenessQuery, ModelInjectionEffectivenessQueryResult, ModelInjectionEffectivenessRow, ModelLatencyBreakdownRow, ModelPatternHitRateRow, ProtocolInjectionEffectivenessReader, ReaderInjectionEffectivenessPostgres

### Community 207 - "validate_markdown_links.py"
Cohesion: 0.06
Nodes (19): BrokenLink, extract_headings_as_anchors(), extract_links_from_markdown(), find_markdown_files(), generate_report(), _heading_to_anchor(), is_external_link(), is_http_link() (+11 more)

### Community 208 - "ModelLabProofProfileVariant"
Cohesion: 0.04
Nodes (15): EnumLabProofBudgetBasis, EnumLabProofExemptClass, EnumLabProofKind, EnumLabProofProfileStatus, applicable_checks(), classify_exemption(), glob_to_regex(), paths_all_match() (+7 more)

### Community 209 - "ProtocolPostgresAdapter"
Cohesion: 0.04
Nodes (6): HandlerPartialRetry, ProtocolPartialRetryRequest, HandlerPostgresDeactivate, HandlerPostgresUpsert, ProtocolPostgresAdapter, sanitize_backend_error()

### Community 210 - "bus_backed_exposure_readers.py"
Cohesion: 0.08
Nodes (26): BackendReader, BackendReaderRegistration, BackendReaderSurface, build_parser(), check_exposure_readers(), collect_backend_reader_surface(), collect_bus_backed_exposures(), collect_layout_readers() (+18 more)

### Community 211 - "environment"
Cohesion: 0.04
Nodes (54): ARCH_GRAPH_BOLT_URI, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_VERIFY_ENDPOINTS, BUS_ID, DELEGATION_ROUTING_TIERS_PATH, DEPLOY_AGENT_HMAC_SECRET, GH_TOKEN (+46 more)

### Community 212 - "services"
Cohesion: 0.04
Nodes (54): profiles, profiles, profiles, networks, profiles, profiles, container_name, depends_on (+46 more)

### Community 213 - "NodeInvocationAdapter"
Cohesion: 0.05
Nodes (6): EnumRuntimeBackend, _describe_collection_geometry(), ModelLocalStateStoreEntry, ModelLocalStateStore, NodeInvocationAdapter, _on_terminal()

### Community 214 - "ServiceFreshnessMonitor"
Cohesion: 0.04
Nodes (8): EnumDegradedBehavior, ModelProjectionDegradedEvent, ModelProjectionRecoveredEvent, ModelCursorContract, ModelProjectionContract, get_projection_contract(), ProtocolTopicRegistry, ServiceFreshnessMonitor

### Community 215 - "HandlerGmailApi"
Cohesion: 0.05
Nodes (5): _extract_list_of_dicts(), HandlerGmailApi, HandlerGmailArchiveCleanup, ModelGmailCleanupConfig, ModelGmailCleanupResult

### Community 216 - "HandlerVectorSearch"
Cohesion: 0.05
Nodes (8): contract_qdrant_url(), _load_contract(), HandlerVectorSearch, HandlerVectorUpsert, ModelVectorDocument, ModelVectorSearchHit, ModelVectorStoreRequest, ModelVectorStoreResult

### Community 217 - "ModelBrokerAclBinding"
Cohesion: 0.05
Nodes (12): HandlerBrokerPrincipalApply, _labels(), planned_bindings(), BrokerPrincipalKafkaAcls, _enum_name(), kafka_acls_from_env(), _to_binding(), ModelBrokerAclBinding (+4 more)

### Community 218 - "commands.py"
Cohesion: 0.06
Nodes (21): cli(), demo(), _get_db_dsn(), _get_error_count(), _is_result_valid(), _load_env_for_demo(), _print_result(), registry() (+13 more)

### Community 219 - "HandlerActionArchiveCacheFill"
Cohesion: 0.06
Nodes (9): _failure(), HandlerActionArchiveCacheFill, _is_filled(), _refusal(), ModelActionArchiveCacheConfig, ModelActionArchiveCacheFailure, ModelActionArchiveCacheFillRequest, ModelActionArchiveCacheFillResult (+1 more)

### Community 220 - "handler_consumer_flow.py"
Cohesion: 0.07
Nodes (22): walk(), runtime_role(), _add(), grade_boot(), grade_consumer_flow(), grade_cursor(), grade_kinds(), grade_negative() (+14 more)

### Community 221 - "generate_application_database_table_grants.py"
Cohesion: 0.06
Nodes (24): _derivation_for_instance(), _FlowList, _grant_to_document(), _InstanceDumper, main(), _mark_flow(), _render_instance(), _represent_flow_list() (+16 more)

### Community 222 - "application_database.py"
Cohesion: 0.06
Nodes (20): main(), _load_runtime_database_topology(), _drop_undeclared_binding_carriers(), _load_docker_consumer_catalog(), load_environment_topology(), _load_profile_catalog(), load_topology_profile(), ProjectionDumper (+12 more)

### Community 223 - "ModelAdjudicatorState"
Cohesion: 0.05
Nodes (7): HandlerAdjudicate, ModelAdjudicatorState, ModelValidationAdjudicationCompleted, ModelValidationAdjudicationRequested, ModelValidationResultPublished, ModelVerdict, HandlerPublishResult

### Community 224 - "ModelContextSection"
Cohesion: 0.05
Nodes (7): EnumContextSectionCategory, ModelContextSection, ModelStaticContextReport, ServiceAttributionReporter, ServiceLlmCategoryAugmenter, ServiceStaticContextParser, ServiceTokenCounter

### Community 225 - "ModelSecurityValidationResult"
Cohesion: 0.05
Nodes (5): ModelSecurityError, ModelSecurityValidationResult, ModelSecurityWarning, SecurityMetadataValidator, validate_handler_security()

### Community 226 - "_consumer_flow_collection.py"
Cohesion: 0.07
Nodes (19): _changed_boot_containers(), inject(), junit_outcomes(), _mutated_source(), natural_validation_errors(), observe_injection(), observe_lane(), _observe_settled() (+11 more)

### Community 227 - "janitor_docker_network.py"
Cohesion: 0.06
Nodes (10): EnumNetworkDisposition, classify_network(), JanitorDockerNetwork, _parse_docker_timestamp(), _parse_network_line(), _shell_single_quote(), ModelNetworkDecision, ModelNetworkInfo (+2 more)

### Community 228 - "HandlerLabProofVerdict"
Cohesion: 0.06
Nodes (16): _judge(), EnumLabProofCheck, EnumLabProofOutcome, ModelLabProofCheckResult, ModelLabProofPlan, ModelLabProofResult, ModelLabProofRunReport, ModelLabProofVerdictRequest (+8 more)

### Community 229 - "ModelCutoverFamilyContract"
Cohesion: 0.06
Nodes (11): EnumReceiptDimension, ModelControlPlaneDeltaEvidence, ModelCutoverContinuityEvidence, ModelCutoverFamilyContract, ModelProjectionReplayEvidence, ModelReceiptCheck, ModelReconciliationInput, ModelTransformationEvidence (+3 more)

### Community 230 - "HandlerRequestAdjudication"
Cohesion: 0.04
Nodes (8): ModelPlannedCheck, ModelValidationChecksCompleted, ModelValidationExecutionRequested, ModelValidationPlan, HandlerRunChecks, HandlerBuildPlan, HandlerRequestAdjudication, HandlerRequestExecution

### Community 231 - "HandlerChainCanary"
Cohesion: 0.05
Nodes (9): _all_links_unevaluated(), HandlerChainCanary, _kill_switch_engaged(), EnumChainCanaryVerdict, EnumChainLink, EnumDeployWindowStatus, ModelChainCanaryResult, ModelChainLinkVerdict (+1 more)

### Community 232 - "HandlerMetricsPrometheus"
Cohesion: 0.06
Nodes (6): generate_latest(), _get_scrape_duration_histogram(), HandlerMetricsPrometheus, ModelMetricsHandlerConfig, ModelMetricsHandlerPayload, ModelMetricsHandlerResponse

### Community 233 - "ServiceLlmCostAggregator"
Cohesion: 0.06
Nodes (4): EnumHealthStatus, _run_once(), mask_dsn_password(), ServiceLlmCostAggregator

### Community 234 - "with"
Cohesion: 0.05
Nodes (50): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, env (+42 more)

### Community 235 - "in_progress_demotion.py"
Cohesion: 0.06
Nodes (11): _format_demotion_comment(), InProgressDemotionLeg, count(), _normalised_fence(), OpenPullRequestIndex, _parse_utc(), _ticket_cited_in(), EnumInProgressDemotionDecision (+3 more)

### Community 236 - "ModelBoardProbeResultEvent"
Cohesion: 0.05
Nodes (6): EnumBoardProbeOutcome, EnumBoardSubjectKind, board_probe_result_event_from(), ModelBoardProbeResultEvent, _board_outcome_for(), chain_canary_result_event_from()

### Community 237 - "replay_hook_history.py"
Cohesion: 0.08
Nodes (26): build_env(), changed_files(), classify_fire(), _classify_run(), CommitRecord, _entry_executable(), _executable_resolves(), _git() (+18 more)

### Community 238 - "EventRegistry"
Cohesion: 0.05
Nodes (9): _cli_stamp(), _cli_verify(), _compute_registry_diff(), EventRegistry, ModelEventRegistration, _sanitize_artifact_path(), validate_event_registry_fingerprint(), ModelEventRegistryFingerprintElement (+1 more)

### Community 239 - "ModelBatchLifecycleResult"
Cohesion: 0.05
Nodes (3): ModelBatchLifecycleResult, ModelLifecycleResult, ProtocolLifecycleExecutor

### Community 240 - "checker_first_party_sources.py"
Cohesion: 0.09
Nodes (26): _accept_target(), audit_build_args(), audit_environment(), build_context_for_lane(), _build_context_record(), BuildContext, classify_distribution(), _classify_revision() (+18 more)

### Community 241 - "services"
Cohesion: 0.08
Nodes (50): build, profiles, BUILD_SOURCE, EXPECTED_BUILD_SOURCE, additional_contexts, args, context, dockerfile (+42 more)

### Community 242 - "generate_ticket_plan.py"
Cohesion: 0.05
Nodes (11): Attachment, BlockingRelation, categorize_issues(), CategorizedIssues, fetch_project_issues(), generate_document(), get_api_key(), LinearIssue (+3 more)

### Community 243 - "agent_learning_extraction/consumer.py"
Cohesion: 0.05
Nodes (11): EnumLearningMatchType, EnumLearningTaskType, ModelAgentLearningMatch, ModelAgentLearning, ModelAgentLearningQuery, ModelAgentLearningQueryResult, build_learning_record(), classify_task_type() (+3 more)

### Community 244 - "HandlerDockerForwarderStateReader"
Cohesion: 0.06
Nodes (8): HandlerDockerForwarderStateReader, parse_refused_topics(), _brokers(), grade_forwarder_refused_topic(), HandlerForwarderRefusedTopic, ModelForwarderRefusedTopicRequest, ModelForwarderStateObservation, ProtocolForwarderStateReader

### Community 245 - "ModelRunnerFleetConfig"
Cohesion: 0.05
Nodes (6): ModelGitMirrorConfig, default_runner_fleet_config_path(), load_runner_fleet_config(), ModelRunnerFleetConfig, ModelRunnerFleetHost, ModelToolCacheConfig

### Community 246 - "build_effects_candidate.py"
Cohesion: 0.08
Nodes (23): build_candidate(), _image(), main(), _manifest(), ModelEffectsCandidatePlan, _probe(), _snapshot_market(), validate_sources() (+15 more)

### Community 247 - "declared_consumer_groups.py"
Cohesion: 0.06
Nodes (21): _compose_env_value(), _ComposeLoader, compute_base_group_id(), _construct_undefined_as_value(), DerivationError, derive_compose_declared_groups(), derive_expected_keys(), DerivedGroupKey (+13 more)

### Community 248 - "validate.py"
Cohesion: 0.07
Nodes (24): _canonical_repository_name(), main(), _report_known_issues(), _repository_root(), run_all(), run_any_types(), run_architecture(), run_architecture_layers() (+16 more)

### Community 249 - "check_topic_completeness.py"
Cohesion: 0.08
Nodes (22): _build_parser(), check_completeness(), _collect_from_parsed_yaml(), collect_registry_from_constants_file(), collect_registry_from_contracts(), collect_registry_from_manifests(), collect_registry_from_python_enums(), _extract_topics_from_manifest() (+14 more)

### Community 250 - "LongLivedTerminalCorrelator"
Cohesion: 0.06
Nodes (4): LongLivedTerminalCorrelator, make_terminal_event_consumer(), TerminalConsumerSession, TerminalEventConsumer

### Community 251 - "consumers"
Cohesion: 0.04
Nodes (48): consumers, forward-migration, migration-gate, omnidash, omnimarket-projection-delegation, omnimarket-projection-llm-cost, omnimarket-projection-run-trace, omnimarket-projection-savings (+40 more)

### Community 252 - "with"
Cohesion: 0.04
Nodes (48): default, description, required, env, CACHE_VERSION, DRY_RUN, LINEAR_API_KEY, LLM_ENDPOINT_CIDR_ALLOWLIST (+40 more)

### Community 253 - "product_reason_graph.py"
Cohesion: 0.08
Nodes (21): classify_check_runs(), _derive_facts(), _is_dependent(), _load_check_runs(), main(), _norm(), _normalize_runs(), render_summary() (+13 more)

### Community 254 - "rollback_lane_to_known_good.py"
Cohesion: 0.08
Nodes (32): act_on_selection(), catalog_databases(), _declared_at(), declared_keys_from_listing(), _gh_json(), _git(), inflight_trigger_runs(), _is_strict_ancestor() (+24 more)

### Community 255 - "verify_stability_refresh.py"
Cohesion: 0.07
Nodes (26): GroupDescription, parse_group_describe(), parse_group_list(), annotate_dimension_consumer_groups(), check_cluster_health(), check_partition_headroom(), check_service_digest(), consumer_group_topic() (+18 more)

### Community 256 - "ServiceTopicRegistry"
Cohesion: 0.05
Nodes (7): _make_routing_decided_callback(), MixinEmissionCounter, ProtocolConsumptionCountSource, _default_dlq_topic(), _default_routing_decision_topic(), RegistryTopicRegistry, ServiceTopicRegistry

### Community 257 - "RegistryCompute"
Cohesion: 0.05
Nodes (4): ComputeRegistryError, ModelComputeKey, _MetricsTimer, RegistryCompute

### Community 258 - "ConsumerFlowCounters"
Cohesion: 0.05
Nodes (5): ModelConsumerFlowDelta, ModelNodeFlowWindow, ModelTopicProduceDelta, ConsumerFlowCounters, RetainedFlowWindows

### Community 259 - "node_dev_roll_events_effect/contract.yaml"
Cohesion: 0.04
Nodes (48): capabilities, description, name, contract_name, contract_version, major, minor, patch (+40 more)

### Community 260 - "ServiceArtifactStore"
Cohesion: 0.08
Nodes (3): _default_artifact_root(), ModelArtifactStoreConfig, ServiceArtifactStore

### Community 261 - "RegistryInfraLlmInferenceEffect"
Cohesion: 0.05
Nodes (8): Added, LLM Metrics Observability, Zero-Repo-Env Policy, NodeLlmInferenceEffect, _create_transport_adapter(), RegistryInfraLlmInferenceEffect, ProtocolLlmHandler, ServiceLlmMetricsPublisher

### Community 262 - "OMN-19817.yaml"
Cohesion: 0.04
Nodes (47): dod-omn19817-ac1-accepted-20261021t2335z, dod-omn19817-ac2-accepted-20261021t2335z, dod-omn19817-ac3-accepted-20261021t2335z, id, statement, check_type, check_value, cwd (+39 more)

### Community 263 - "OMN-19958.yaml"
Cohesion: 0.04
Nodes (47): dod-omn19958-ac1-accepted-20261113t0945z, dod-omn19958-ac2-accepted-20261022t0245z, dod-omn19958-ac3-accepted-20261022t0245z, id, statement, check_type, check_value, cwd (+39 more)

### Community 264 - "preflight_lane_deploy_attribution.py"
Cohesion: 0.07
Nodes (21): apply_acknowledgement(), build_parser(), build_record(), _coerce_datetime(), collect_actor(), default_grants_repo(), EnumGrantVerdict, evaluate_grant_state() (+13 more)

### Community 265 - "MixinDictLikeAccessors"
Cohesion: 0.06
Nodes (8): MixinDictLikeAccessors, ModelPluginContext, ModelPluginInputData, ModelPluginOutputData, PluginComputeBase, ProtocolPluginCompute, ModelPolicyContext, ModelPolicyResult

### Community 266 - "ApplicationDatabaseFunctionSignature"
Cohesion: 0.04
Nodes (8): EnumApplicationDatabaseObjectKind, EnumInternalTenantColumnTransformStatus, ModelApplicationDatabaseAclRow, ModelApplicationDatabaseCatalogObjectEvidence, ModelApplicationInventoryRelation, ModelDatabaseObjectEvidence, ModelInternalTenantColumnTransform, ApplicationDatabaseFunctionSignature

### Community 267 - "PostgresTransformationEvidenceCollector"
Cohesion: 0.06
Nodes (7): ModelApplicationPathWriteProof, ModelConnectionIdentity, ModelPostgresEvidenceQuerySet, _collect_expression_schema_refs(), _collect_relations(), _collect_schema_names(), PostgresTransformationEvidenceCollector

### Community 268 - "RepositoryPostgresCutoverJournal"
Cohesion: 0.09
Nodes (5): ModelCutoverJournalRequest, _canonical_json(), _receipt_content_signature(), RepositoryPostgresCutoverJournal, _sha256()

### Community 270 - "ModelDiscoveryResult"
Cohesion: 0.05
Nodes (7): ModelDiscoveryError, ModelDiscoveryResult, ModelDiscoveryWarning, ModelDiscoveryError, ContractHandlerDiscovery, ProtocolHandlerDiscovery, ProtocolHandlerPluginLoader

### Community 271 - "node_c11_delegation_legs_effect/contract.yaml"
Cohesion: 0.04
Nodes (47): capabilities, description, name, contract_name, contract_version, major, minor, patch (+39 more)

### Community 272 - "node_c16_pinned_backend_effect/contract.yaml"
Cohesion: 0.04
Nodes (47): capabilities, description, name, contract_name, contract_version, major, minor, patch (+39 more)

### Community 273 - "node_chain_retrieval_effect/contract.yaml"
Cohesion: 0.04
Nodes (47): enabled, failure_threshold, reset_timeout_ms, contract_version, major, minor, patch, description (+39 more)

### Community 274 - "node_ledger_projection_compute/handlers/contract.yaml"
Cohesion: 0.04
Nodes (47): ledger.project, contract_version, major, minor, patch, description, expression, parameter_name (+39 more)

### Community 275 - "node_standalone_compose_recreate_effect/contract.yaml"
Cohesion: 0.04
Nodes (47): allowed_lanes, capabilities, description, name, contract_name, contract_version, major, minor (+39 more)

### Community 276 - "registration_projector.yaml"
Cohesion: 0.04
Nodes (47): columns, description, trigger_event, where, aggregate_type, behavior, idempotency, mode (+39 more)

### Community 277 - "InfraRoutingDecisionsConsumer"
Cohesion: 0.05
Nodes (6): ConfigInfraRoutingDecisionsConsumer, ConsumerMetrics, InfraRoutingDecisionsConsumer, _run_once(), _as_str(), ModelInfraRoutingDecisionIngest

### Community 278 - "SkillLifecycleConsumer"
Cohesion: 0.06
Nodes (5): ConfigSkillLifecycleConsumer, ConsumerMetrics, _run_once(), mask_dsn_password(), SkillLifecycleConsumer

### Community 279 - "ModelPricingTable"
Cohesion: 0.05
Nodes (6): LLM Cost Tracking, ModelComputeCostEntry, ModelCostEstimate, ModelPricingEntry, ModelPricingTable, ModelRunnerCostPolicy

### Community 280 - "service_eval_runner.py"
Cohesion: 0.06
Nodes (9): EnumEvalTaskStatus, EnumEvalTaskType, ModelEvalTask, _capture_env_snapshot(), _check_success_criteria(), _get_git_sha(), _run_setup_commands(), ServiceEvalRunner (+1 more)

### Community 281 - "with"
Cohesion: 0.05
Nodes (46): env, CACHE_VERSION, ONEX_EVENT_BUS_TYPE, POSTGRES_DATABASE, PYTHON_VERSION, UV_VERSION, test_filter, jobs (+38 more)

### Community 282 - "validate_no_direct_env.py"
Cohesion: 0.06
Nodes (13): AllowlistValidationError, detect_aliased_env_access(), EnvAccessASTVisitor, find_repo_root(), _has_inline_exclusion_marker(), is_bootstrap_exception(), load_allowlist(), main() (+5 more)

### Community 283 - "publisher.py"
Cohesion: 0.07
Nodes (9): ModelDepsConvergenceFinding, load_deploy_agent_kafka_config_from_env(), ModelDeployAgentKafkaConfig, _produce(), publish_dev_lane_drift(), publish_dev_tag_heartbeat(), publish_result(), publish_with_retry() (+1 more)

### Community 284 - "HandlerAuthGate"
Cohesion: 0.05
Nodes (4): EnumAuthDecision, HandlerAuthGate, ModelAuthGateDecision, ModelAuthGateRequest

### Community 285 - "lane_settle_budget.py"
Cohesion: 0.07
Nodes (15): ModelSettleBudget, ModelSettleOutcome, settle_budget_check(), affordable_seconds(), assert_declaration_within_bounds(), build_parser(), converge_wall_clock_seconds(), derive_settle_budget() (+7 more)

### Community 286 - "deploy-gateway.sh"
Cohesion: 0.11
Nodes (44): build_image(), BUILD_IMAGE_TAG, build_sidecar_image(), check_git_dirty(), check_sibling_lock_pins(), COMPOSE_PROJECT, CONTAINER_NAME, GATEWAY_REGISTRY_FILE (+36 more)

### Community 287 - "heavy_lock.py"
Cohesion: 0.08
Nodes (23): abort_wrapped_command(), acquire(), build_parser(), clear_holder(), describe_holder(), _describe_progress(), LockTimeoutError, _log() (+15 more)

### Community 288 - "reconcile-workspace-venvs.sh"
Cohesion: 0.11
Nodes (41): acquire_staging_lock(), arm_lane_hooks(), clone_branch_report(), clone_venv_to_staging(), dispatch_base_python(), dispatch_interpreter_ok(), dispatch_requires_base_python(), exchange_dirs_atomically() (+33 more)

### Community 289 - "venv_readback.py"
Cohesion: 0.08
Nodes (21): build_parser(), build_rows(), classify_commit(), classify_sibling(), classify_version(), _declared_floor(), declared_version(), exit_code_for() (+13 more)

### Community 290 - "HandlerBaselinesBatchCompute"
Cohesion: 0.06
Nodes (7): HandlerBaselinesBatchCompute, ModelBaselinesBatchComputeCommand, ModelBaselinesBatchComputeOutput, _baselines_loop(), ModelBaselinesComparisonRow, ModelBaselinesTrendRow, ModelBatchComputeBaselinesResult

### Community 291 - "node_chain_verify_reducer/contract.yaml"
Cohesion: 0.05
Nodes (46): contract_version, major, minor, patch, description, module, name, handler_routing (+38 more)

### Community 292 - "node_declared_state_apply_effect/contract.yaml"
Cohesion: 0.04
Nodes (46): capabilities, description, name, contract_name, contract_version, major, minor, patch (+38 more)

### Community 293 - "node_declared_state_reconcile_orchestrator/contract.yaml"
Cohesion: 0.04
Nodes (46): capabilities, description, name, contract_name, contract_version, major, minor, patch (+38 more)

### Community 294 - "HandlerSavingsEstimation"
Cohesion: 0.06
Nodes (10): CategoryAccumulator, _classify_category(), HandlerSavingsEstimation, _price_per_token(), EnumModelTier, EnumSavingsCategory, ModelEffectivenessEntry, ModelSavingsCategory (+2 more)

### Community 295 - "variants"
Cohesion: 0.04
Nodes (45): check, reason, when_changed, excluded_repos, reason, repo, distribution, import_package (+37 more)

### Community 296 - "with"
Cohesion: 0.04
Nodes (45): name, permissions, runs-on, steps, timeout-minutes, default, description, required (+37 more)

### Community 297 - "catalog/cli.py"
Cohesion: 0.08
Nodes (20): cmd_down(), cmd_generate(), cmd_read_stack(), cmd_seed(), cmd_status(), cmd_up(), cmd_validate(), cmd_validate_runtime() (+12 more)

### Community 298 - "CollectorRunnerHealth"
Cohesion: 0.06
Nodes (5): CollectorRunnerHealth, EnumRunnerHealthState, HandlerRunnerHealthCollector, ModelRunnerHealthSnapshot, ModelRunnerStatus

### Community 299 - "check_dep_provenance.py"
Cohesion: 0.08
Nodes (20): _api_get(), _declared_version_specs(), drop_dev_tag_override(), find_escape_token_violations(), find_lineage_violations(), find_unmovable_cascade_targets(), find_violations(), _is_dev_merge_tag() (+12 more)

### Community 300 - "ProtocolContainerAware"
Cohesion: 0.05
Nodes (3): ModelPoolCheckoutMetrics, ProtocolContainerAware, HandlerPool

### Community 301 - "check_hostile_review_threads.py"
Cohesion: 0.09
Nodes (25): body_digest(), cited_lines(), classify_threads(), _comments(), Disposition, evaluate_thread(), fetch_review_threads(), finding_text() (+17 more)

### Community 302 - "prepr_teardown_slot.py"
Cohesion: 0.13
Nodes (30): authorize_teardown(), check_owner(), claim(), clear(), default_runner(), heartbeat(), install_reaper(), lane_lock_path() (+22 more)

### Community 303 - "HandlerSyncRevertWatchdog"
Cohesion: 0.06
Nodes (9): DoneWriteReceiptGateBinding, DoneWriteReceiptGuard, DoneWriteRefusedError, HandlerSyncRevertWatchdog, EnumPriorDoneActorKind, EnumSyncRevertWatchdogDecision, ModelSyncRevertWatchdogOutcome, ModelSyncRevertWatchdogRequest (+1 more)

### Community 304 - "CloudLedgerReader"
Cohesion: 0.06
Nodes (7): EnumCloudLedgerVerdict, CloudLedgerReader, ProtocolCloudLedgerTransport, TokenRefusedError, ModelCloudLedgerRead, ModelGatewayCredentialBase, ModelGatewayCredential

### Community 306 - "node_chain_store_effect/contract.yaml"
Cohesion: 0.04
Nodes (45): enabled, failure_threshold, reset_timeout_ms, contract_version, major, minor, patch, description (+37 more)

### Community 307 - "node_ci_live_contact_effect/contract.yaml"
Cohesion: 0.04
Nodes (45): capabilities, description, name, contract_name, contract_version, major, minor, patch (+37 more)

### Community 308 - "node_hotpatch_ledger_reconcile_effect/contract.yaml"
Cohesion: 0.04
Nodes (45): capabilities, description, name, contract_name, contract_version, major, minor, patch (+37 more)

### Community 309 - "HandlerSavingsCorrelation"
Cohesion: 0.06
Nodes (15): _assert_idempotency_read_is_scoped(), _build_effectiveness_entries(), _classify_severity(), _coerce_int(), _compute_validator_catch_savings(), HandlerSavingsCorrelation, InjectionRow, LlmCallRow (+7 more)

### Community 310 - "node_setup_orchestrator/contract.yaml"
Cohesion: 0.05
Nodes (45): capabilities, changelog, change, ticket, version, contract_version, major, minor (+37 more)

### Community 311 - "node_ticket_contract_compose_effect/contract.yaml"
Cohesion: 0.04
Nodes (45): capabilities, description, name, contract_name, contract_version, major, minor, patch (+37 more)

### Community 312 - "application_database_red_control_registry.py"
Cohesion: 0.05
Nodes (7): ApplicationDatabaseRedControlBinding, _case(), _control(), _freeze_registry(), EnumApplicationDatabaseEnforcementGate, ModelApplicationDatabaseEnforcementContract, ModelApplicationDatabaseEnforcementGateState

### Community 313 - "verify"
Cohesion: 0.06
Nodes (44): default_instance, consumer_group, flags, hostnames, verify, available, consumer_group, flags (+36 more)

### Community 314 - "RuntimeDelegationDispatchPort"
Cohesion: 0.06
Nodes (8): _publish(), HandlerDelegationDispatchTerminal, ModelDelegationDispatchTerminalRequest, ModelDispatchBudget, ModelPatternBBrokerConfig, parse_active_runtime_packages(), _named_tests_wire(), RuntimeDelegationDispatchPort

### Community 315 - "services"
Cohesion: 0.04
Nodes (45): container_name, profiles, profiles, profiles, container_name, restart, profiles, container_name (+37 more)

### Community 316 - "with"
Cohesion: 0.04
Nodes (44): description, required, author, description, description, required, CI_OVERLAY_REPO, DEST (+36 more)

### Community 317 - "with"
Cohesion: 0.05
Nodes (44): concurrency, cancel-in-progress, group, name, needs, runs-on, steps, timeout-minutes (+36 more)

### Community 318 - "with"
Cohesion: 0.04
Nodes (44): concurrency, cancel-in-progress, group, CURRENT_PIN, GH_TOKEN, TARGET_PIN, TARGET_REF_INPUT, target_ref (+36 more)

### Community 319 - "with"
Cohesion: 0.05
Nodes (44): concurrency, cancel-in-progress, group, default, description, required, type, env (+36 more)

### Community 320 - "handler_invariant_evaluate.py"
Cohesion: 0.05
Nodes (7): HandlerInvariantEvaluate, HandlerInvariantEvaluateAll, HandlerInvariantEvaluateBatch, _new_evaluator(), ModelInvariantEvaluateAllInput, ModelInvariantEvaluateBatchInput, ModelInvariantEvaluateInput

### Community 321 - "validate_timezone_aware_datetime"
Cohesion: 0.05
Nodes (5): ModelRuntimeHeartbeatEvent, _sanitize_url_for_logging(), validate_endpoint_urls_dict(), validate_timezone_aware_datetime(), validate_timezone_aware_datetime_optional()

### Community 322 - "c16_receipt_identity_probe.py"
Cohesion: 0.11
Nodes (28): _accepted_model(), grade(), grade_identity(), grade_skip_never_pass(), grade_typed_death(), _http(), main(), _maybe_json() (+20 more)

### Community 323 - "NodeGatewayDelivery"
Cohesion: 0.08
Nodes (3): _build_quarantine_payload(), NodeGatewayDelivery, ProtocolGatewayConsumer

### Community 324 - "_build_link_verdicts"
Cohesion: 0.06
Nodes (11): _build_link_verdicts(), _link_five(), _link_four(), _link_three(), _link_two(), dsn_shaped_argv_flags(), EnumChainLinkStatus, EnumLedgerReplayStatus (+3 more)

### Community 325 - "node_rrh_emit_effect/contract.yaml"
Cohesion: 0.05
Nodes (44): capabilities, description, name, version, contract_version, major, minor, patch (+36 more)

### Community 326 - "profiles"
Cohesion: 0.04
Nodes (44): deployment_repository, injection_path, instance, runtime_policy_profile, deployment_repository, injection_path, instance, runtime_policy_profile (+36 more)

### Community 327 - "agent-actions-consumer.yaml"
Cohesion: 0.05
Nodes (43): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_AGENT_ACTIONS_HEALTH_CHECK_HOST (+35 more)

### Community 328 - "consumer-health-projection.yaml"
Cohesion: 0.05
Nodes (43): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_CONSUMER_HEALTH_HEALTH_CHECK_HOST (+35 more)

### Community 329 - "worktree_reaper.py"
Cohesion: 0.08
Nodes (20): _build_arg_parser(), build_projection_url(), cursor_file_path(), _default_prune_runner(), _default_prune_script(), fetch_projection_page(), main(), log() (+12 more)

### Community 330 - "DefaultEnvelopeChunker"
Cohesion: 0.06
Nodes (5): ChunkingGateway, DefaultEnvelopeChunker, EnvelopeFactory, _sha256_hex(), ReassemblyGateway

### Community 332 - "handler_c16_pinned_backend.py"
Cohesion: 0.07
Nodes (16): DelegationRow, grade_pinned_backends(), HandlerC16PinnedBackend, load_pinned_config(), load_row_query(), observe_pinned_live(), pinned_observations_from_replay(), pinned_run_from_receipt() (+8 more)

### Community 333 - "handler_deploy_job_reading.py"
Cohesion: 0.10
Nodes (12): _alarm_head(), _append_env(), _failing_step(), HandlerDeployJobReading, _names(), _pin_request_id(), _request_id(), _stamp() (+4 more)

### Community 334 - "host_slot.py"
Cohesion: 0.09
Nodes (12): _result(), _build_parser(), _format(), host_slot_from_env(), HostSlot, HostSlotHeldError, job_lease(), Lease (+4 more)

### Community 335 - "compare_environments.py"
Cohesion: 0.08
Nodes (22): build_parser(), check_credential_parity(), check_db_schema_parity(), check_ecr_tag_validity(), check_env_var_name_alignment(), check_feature_flag_consistency(), check_kafka_topic_parity(), check_package_version_parity() (+14 more)

### Community 336 - "ModelDurabilityConfirmation"
Cohesion: 0.06
Nodes (6): EnumConfirmationState, BrokerReadbackStrategy, PublishReturnOnlyStrategy, ModelDurabilityConfirmation, ProtocolConfirmationStrategy, ProtocolReadbackSource

### Community 337 - "ProtocolDoneWriteReceiptGate"
Cohesion: 0.06
Nodes (14): resolve_done_write_receipt_gate(), _ac_coverage_gap(), _check_id(), _check_result_rows(), _coverage_corpus_counts(), _declared_labels(), _every_criterion_is_state_shaped(), _excerpt() (+6 more)

### Community 338 - "node_deploy_runners/contract.yaml"
Cohesion: 0.05
Nodes (43): capabilities, description, name, contract_name, contract_version, major, minor, patch (+35 more)

### Community 339 - "node_row_count_diagnostic_effect/contract.yaml"
Cohesion: 0.05
Nodes (43): contract_version, major, minor, patch, description, event_bus, publish_topics, subscribe_topics (+35 more)

### Community 340 - "TopicCategoryASTVisitor"
Cohesion: 0.06
Nodes (5): TopicCategoryASTVisitor, collect_static_parts(), TopicCategoryValidator, validate_topic_categories_in_directory(), validate_topic_categories_in_file()

### Community 341 - "compute_schema_fingerprint"
Cohesion: 0.06
Nodes (15): SchemaFingerprintMismatchError, SchemaFingerprintMissingError, ModelSchemaFingerprintResult, ModelSchemaManifest, _build_constraint_record(), _cli_stamp(), _cli_verify(), _compute_schema_diff() (+7 more)

### Community 342 - "Changelog"
Cohesion: 0.05
Nodes (43): [0.12.0] - 2026-02-27, [0.16.0] - 2026-03-07, [0.16.1] - 2026-03-09, [0.2.3] - 2026-01-25, [0.2.7] - 2026-01-30, [0.2.8] - 2026-01-30, [0.3.0] - 2026-02-01, [0.3.1] - 2026-02-02 (+35 more)

### Community 343 - "OMN-18613.yaml"
Cohesion: 0.05
Nodes (42): dod-omn18613-ac1-accepted-20261018t2045z, dod-omn18613-ac4-accepted-20261018t2045z, id, statement, check_type, check_value, cwd, context (+34 more)

### Community 344 - "OMN-19521.yaml"
Cohesion: 0.05
Nodes (42): dod-omn19521-ac1-accepted-20261021t2245z, dod-omn19521-ac2-accepted-20261021t2245z, id, statement, check_type, check_value, cwd, context (+34 more)

### Community 345 - "OMN-20359.yaml"
Cohesion: 0.05
Nodes (42): dod-omn20359-ac1-accepted-20261113t1345z, dod-omn20359-ac3-accepted-20261113t1345z, id, statement, check_type, check_value, cwd, context (+34 more)

### Community 346 - "OMN-20715.yaml"
Cohesion: 0.05
Nodes (42): dod-omn20715-ac1-accepted-20261114t1545z, dod-omn20715-ac2-accepted-20261114t1545z, id, statement, check_type, check_value, cwd, context (+34 more)

### Community 347 - "OMN-21050.yaml"
Cohesion: 0.05
Nodes (42): dod-omn21050-ac2-accepted-20261116t0145z, dod-omn21050-ac3-accepted-20261116t0145z, id, statement, check_type, check_value, cwd, context (+34 more)

### Community 348 - "atomic_venv_install.py"
Cohesion: 0.10
Nodes (22): _clean_env(), commit_stage(), exchange(), exchange_supported(), _journal_write(), _live_footprint(), _load_plan_tool(), _plan_requirements() (+14 more)

### Community 349 - "context-audit-consumer.yaml"
Cohesion: 0.05
Nodes (42): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_CONTEXT_AUDIT_HEALTH_CHECK_HOST (+34 more)

### Community 350 - "infra-routing-decisions-consumer.yaml"
Cohesion: 0.05
Nodes (42): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_ROUTING_DECISIONS_HEALTH_CHECK_HOST (+34 more)

### Community 351 - "llm-cost-aggregation-consumer.yaml"
Cohesion: 0.05
Nodes (42): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_LLM_COST_HEALTH_CHECK_HOST (+34 more)

### Community 352 - "skill-lifecycle-consumer.yaml"
Cohesion: 0.05
Nodes (42): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_SKILL_LIFECYCLE_HEALTH_CHECK_HOST (+34 more)

### Community 353 - "render_bifrost_delegation_contract.py"
Cohesion: 0.08
Nodes (18): _is_typed_decision_backend(), lane_overlay_store_key(), _load_lane_overlay(), _load_mapping(), _load_routing_ladder_rungs(), main(), _parse_store_overlay(), _probe_openai_model_endpoint() (+10 more)

### Community 354 - "env"
Cohesion: 0.05
Nodes (42): name, runs-on, steps, timeout-minutes, types, concurrency, cancel-in-progress, group (+34 more)

### Community 355 - "with"
Cohesion: 0.05
Nodes (42): concurrency, cancel-in-progress, group, jobs, provider-rung-canary, resolve-lane, name, on (+34 more)

### Community 357 - "check_delegation_tenant_conversion_readiness.py"
Cohesion: 0.10
Nodes (22): blank_sql_string_literals(), build_client(), census_direct(), census_reconstructed(), check_privileges(), classify(), derive_role_read_plan(), evaluate() (+14 more)

### Community 358 - "ConsumerHealthEmitter"
Cohesion: 0.07
Nodes (4): ConsumerHealthEmitter, MixinConsumerHealth, EnumConsumerHealthEventType, EnumConsumerHealthSeverity

### Community 359 - "HandlerLabProofRun"
Cohesion: 0.06
Nodes (8): EnumLabProofAttribution, EnumLabProofStepId, EnumLabProofStepPhase, ModelLabProofObservation, ModelLabProofStep, HandlerLabProofRun, _now(), _text()

### Community 361 - "HandlerPluginLoader"
Cohesion: 0.09
Nodes (6): ModelFailedPluginLoad, ModelLoadedHandler, ModelPluginLoadContext, ModelPluginLoadSummary, HandlerPluginLoader, _sanitize_exception_message()

### Community 362 - "HandlerGatewayLinkHealthUpsert"
Cohesion: 0.06
Nodes (6): _require_non_negative_int(), _require_str(), ModelPayloadGatewayLinkHealthUpsert, bind_handler_db(), HandlerGatewayLinkHealthUpsert, ModelGatewayLinkHealthUpsertResult

### Community 363 - "node_merge_provenance_compute/contract.yaml"
Cohesion: 0.05
Nodes (42): capabilities, description, name, contract_name, contract_version, major, minor, patch (+34 more)

### Community 364 - "node_reward_binder_effect/contract.yaml"
Cohesion: 0.05
Nodes (42): capabilities, contract_version, major, minor, patch, description, descriptor, node_archetype (+34 more)

### Community 365 - "RuntimeContractConfigLoader"
Cohesion: 0.07
Nodes (3): ModelContractLoadResult, _contract_declares_subcontract(), RuntimeContractConfigLoader

### Community 366 - "ServiceRetryWorker"
Cohesion: 0.07
Nodes (5): ConfigRetryWorker, EnumDeliveryStatus, ModelDeliveryAttempt, ModelRetryResult, ServiceRetryWorker

### Community 367 - "with"
Cohesion: 0.05
Nodes (41): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, env (+33 more)

### Community 368 - "with"
Cohesion: 0.05
Nodes (41): concurrency, cancel-in-progress, group, CI, GH_TOKEN, LINEAR_API_KEY, SLACK_BOT_TOKEN, jobs (+33 more)

### Community 369 - "with"
Cohesion: 0.05
Nodes (41): concurrency, cancel-in-progress, group, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, OMNIBASE_PATH, ONEX_TENANT_ID, SETUP_OUTCOMES (+33 more)

### Community 370 - "env"
Cohesion: 0.05
Nodes (42): CI_RUNNER_ENVIRONMENT, CONVERGE_EVIDENCE, CONVERGE_GENERATION, CONVERGE_MODE, CONVERGE_OUTCOME, CONVERGE_VERDICT, CONVERGED_VIA, CORRELATION_ID (+34 more)

### Community 371 - "steps"
Cohesion: 0.05
Nodes (41): env, LINEAR_API_KEY, PYTHON_VERSION, UV_VERSION, lookback_hours, max_issues, team_key, jobs (+33 more)

### Community 372 - "boot_gate.py"
Cohesion: 0.09
Nodes (23): _deployment_rows(), _docker_exec(), _event_lines(), explain_not_ready(), HostPathOwnershipError, _kubectl_json(), _kubectl_json_or_error(), _label_selector() (+15 more)

### Community 373 - "node_registration_orchestrator/models/__init__.py"
Cohesion: 0.06
Nodes (3): get_union_intent_types(), ProtocolEffect, ProtocolRegistrationIntent

### Community 374 - "HandlerSlackWebhook"
Cohesion: 0.06
Nodes (3): HandlerSlackWebhook, ConfigWaitlistSignupNotifier, WaitlistSignupNotifier

### Community 375 - "generator.py"
Cohesion: 0.06
Nodes (14): cmd_image_containers(), containers_on_image(), generate_compose(), _override_defaults(), read_env_file(), _reject_redaction_placeholder(), render_bundle(), _render_optional_directory_bind_mount() (+6 more)

### Community 376 - "ModelLaneClientTransport"
Cohesion: 0.06
Nodes (9): DelegateLaneCredentialError, resolve_lane_binding(), resolve_lane_client_transport_for(), sasl_refusal_remediation(), ModelSkillTerminalPublishTarget, resolve_skill_terminal_target(), bound_lane_client_transport(), ModelLaneClientTransport (+1 more)

### Community 377 - "entrypoint_preflight.py"
Cohesion: 0.08
Nodes (17): _echo(), _echo_err(), _env_number(), _exit_code(), handoff(), KernelErrorCapture, main(), _park() (+9 more)

### Community 378 - "check_runner_host_artifact_freshness.py"
Cohesion: 0.09
Nodes (20): compute_local_hashes(), compute_ref_hashes(), compute_remote_hashes(), converge_paths(), find_stale_paths(), _git(), _local_sha256(), main() (+12 more)

### Community 379 - "unit_drift.py"
Cohesion: 0.08
Nodes (18): _unit_drift_report(), _sync_own_unit_after_pull(), check_units(), EnumUnitStatus, has_drift(), _hostname(), _installed_path(), load_manifest() (+10 more)

### Community 380 - "LabOverlayApplier"
Cohesion: 0.09
Nodes (10): LabOverlayApplier, LabOverlayRefusalError, load_record(), load_repair_record(), ModelLabOverlayCheck, ModelLabOverlayRecord, record_path(), repair_record_path() (+2 more)

### Community 381 - "_fail"
Cohesion: 0.07
Nodes (16): auth_lane_login(), auth_lane_logout(), auth_login(), auth_logout(), auth_status(), _fail(), _lane_store(), _load_credential() (+8 more)

### Community 382 - "AdapterProtocolEventPublisherKafka"
Cohesion: 0.06
Nodes (4): AdapterProtocolEventPublisherKafka, AdapterProtocolEventPublisherInmemory, ModelPublisherMetrics, TypedDictEnvelopeBuildParams

### Community 383 - "HandlerCatalogRequest"
Cohesion: 0.05
Nodes (3): ModelTopicCatalogRequest, DispatcherCatalogRequest, HandlerCatalogRequest

### Community 384 - "ModelChainCanaryRequest"
Cohesion: 0.06
Nodes (7): _extract_error(), _extract_terminal(), _readback_window_seconds(), _terminal_topics(), host_aliases_in(), _coerce_topics(), ModelChainCanaryRequest

### Community 385 - "node_release_identity_compute/contract.yaml"
Cohesion: 0.05
Nodes (41): capabilities, description, name, version, contract_name, contract_version, major, minor (+33 more)

### Community 386 - "node_release_notes_apply_effect/contract.yaml"
Cohesion: 0.05
Nodes (41): contract_version, major, minor, patch, description, descriptor, node_archetype, purity (+33 more)

### Community 387 - "WriterInjectionEffectivenessPostgres"
Cohesion: 0.06
Nodes (8): ModelAgentMatchEvent, ModelContextEnrichmentEvent, ModelContextUtilizationEvent, ModelInjectionRecordedEvent, ModelLatencyBreakdownEvent, ModelManifestInjectionLifecycleEvent, ModelPatternUtilization, WriterInjectionEffectivenessPostgres

### Community 388 - "relation_evidence"
Cohesion: 0.05
Nodes (40): database_objects, audit_id, database_ref, definition_sha256, domain, function_signature, kind, name (+32 more)

### Community 389 - "decision-store-consumer.yaml"
Cohesion: 0.05
Nodes (40): command, container_name, depends_on, condition, service, description, hardcoded_env, DECISION_STORE_HEALTH_CHECK_HOST (+32 more)

### Community 390 - "environment"
Cohesion: 0.05
Nodes (40): cap_drop, container_name, environment, healthcheck, image, init, labels, logging (+32 more)

### Community 391 - "with"
Cohesion: 0.05
Nodes (40): description, required, author, description, CI_OVERLAY_REPO, OVERLAY_NAME, inputs, app-id (+32 more)

### Community 392 - "env"
Cohesion: 0.05
Nodes (40): concurrency, cancel-in-progress, group, GH_TOKEN, KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, PR_ACTION (+32 more)

### Community 393 - "env"
Cohesion: 0.05
Nodes (40): default, description, required, concurrency, cancel-in-progress, group, name, runs-on (+32 more)

### Community 394 - "with"
Cohesion: 0.05
Nodes (40): concurrency, cancel-in-progress, group, env, MAX_AGE_DAYS, PYTHON_VERSION, STALE_ADVISORY, UV_VERSION (+32 more)

### Community 395 - "env"
Cohesion: 0.05
Nodes (40): concurrency, cancel-in-progress, group, KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, PR_AUTHOR, PR_BASE_REF (+32 more)

### Community 396 - "validator_any_type.py"
Cohesion: 0.07
Nodes (8): ModelAnyTypeValidationResult, ModelAnyTypeViolation, _find_onex_exclude_lines(), _print_fix_instructions(), _should_skip_file(), validate_any_types(), validate_any_types_ci(), validate_any_types_in_file()

### Community 397 - "build_sanitized_evidence"
Cohesion: 0.06
Nodes (8): build_sanitized_evidence(), _CanaryFixture, collect_model_review_capability_observation(), ModelModelReviewReferenceProbe, EnumModelReviewCapabilityFailure, ModelModelReviewCapabilityConfig, ModelModelReviewCapabilityPreflight, preflight_model_review_capability()

### Community 398 - "ContractTopicExtractor"
Cohesion: 0.06
Nodes (14): check_allowlist_hygiene(), check_wiring_health(), collect_subscribed_topics(), current_event_name(), expiry_horizon_days(), _is_infrastructure_topic(), main(), _parse_allowlist_expiry() (+6 more)

### Community 399 - "lineage_fence.py"
Cohesion: 0.07
Nodes (11): GitAncestryResolver, EnumLineageVerdict, _already_carried(), ci_source_repository(), decide_lineage(), decision(), unproven(), DockerProvenanceReader (+3 more)

### Community 400 - "prepush_dispatch.sh"
Cohesion: 0.10
Nodes (36): pick_capacity_host(), prepush_bundle_tree(), prepush_designated_hostnames(), prepush_developer_shell_path(), prepush_emit_receipt(), prepush_field(), prepush_heavy_local_policy(), prepush_identity_label() (+28 more)

### Community 401 - "omninode-ci-required-context-probe.py"
Cohesion: 0.11
Nodes (16): emit(), GitHub, has_in_flight_runs(), main(), _next_link(), _now(), observed_contexts(), _parse_ts() (+8 more)

### Community 402 - "ServiceAutoEvalRunner"
Cohesion: 0.07
Nodes (6): EnumAutoEvalTaskType, ModelAutoEvalBudgetCap, ModelAutoEvalResult, ModelAutoEvalTask, ModelBudgetState, ServiceAutoEvalRunner

### Community 403 - "ModelComponentHealth"
Cohesion: 0.07
Nodes (8): EnumRuntimeReadinessState, check_config_provenance_health(), check_published_events_map_health(), fold_attach_readiness_into_status(), ModelComponentHealth, ModelDetailedHealthResponse, ModelLocalRuntimeIngressHealth, build_component_health()

### Community 404 - "HandlerDockerConsumerFlowTarget"
Cohesion: 0.07
Nodes (6): HandlerConsumerFlow, HandlerDockerConsumerFlowTarget, ModelConsumerFlowLane, ModelConsumerFlowObservation, ModelConsumerFlowRequest, ProtocolConsumerFlowTarget

### Community 405 - "validate_canonical_topic"
Cohesion: 0.05
Nodes (12): ModelGatewayCanaryConfig, ModelGatewayLaneMirrorConfig, ModelGatewayMirrorTopics, prefix_topic(), resolve_physical_topic(), resolve_tenant_from_wire_topic(), strip_topic_prefix(), validate_canonical_topic() (+4 more)

### Community 406 - "node_contract_subscribe_preflight_effect/contract.yaml"
Cohesion: 0.05
Nodes (40): capabilities, description, name, contract_name, contract_version, major, minor, patch (+32 more)

### Community 407 - "node_service_discovery_effect/models/__init__.py"
Cohesion: 0.06
Nodes (7): EnumHealthStatus, ModelDiscoveryQuery, ModelDiscoveryResult, ModelHealthCheckConfig, ModelQueryMetadata, ModelServiceInfo, ModelServiceRegistration

### Community 410 - "ModelSubscriptionHandle"
Cohesion: 0.06
Nodes (3): EnumSubscriptionPhase, ModelSubscriptionGate, ModelSubscriptionHandle

### Community 411 - "infisical.yaml"
Cohesion: 0.05
Nodes (39): container_name, depends_on, condition, service, description, hardcoded_env, AUTH_SECRET, DB_CONNECTION_URI (+31 more)

### Community 412 - "injection-effectiveness-consumer.yaml"
Cohesion: 0.05
Nodes (39): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_HEALTH_CHECK_HOST (+31 more)

### Community 413 - "ConsumerTopicVisitor"
Cohesion: 0.07
Nodes (4): ConsumerTopicVisitor, HandlerContractValidate, _namespace_violations(), ModelContractValidateInput

### Community 414 - "prove_tenant_uuid_conversion.py"
Cohesion: 0.09
Nodes (17): _column_type(), _connect(), main(), phase_continuity(), phase_fail_closed(), phase_follow_ons(), phase_total_mapping(), _policy_expressions() (+9 more)

### Community 415 - "boundary_failure_terminal.py"
Cohesion: 0.08
Nodes (8): EnumNonRetryableErrorCategory, _candidate_class_names(), classify_boundary_failure(), _customer_facing_reason(), _exception_chain(), _first_onex_code(), is_answered_caller_refusal(), ModelBoundaryFailureTerminal

### Community 416 - "handler_developer_principal_plan/handler.py"
Cohesion: 0.08
Nodes (15): _declaration(), DeclarationRefusedError, _derived_grants(), _grant(), HandlerDeveloperPrincipalPlan, PlannedDeclaration, PlannedGrant, principal_name() (+7 more)

### Community 417 - "main"
Cohesion: 0.06
Nodes (9): load_bundle_policy(), main(), _parse(), _run(), EnumLabProofExecution, ModelLabProofPlanRequest, ModelLabProofSubject, ModelLabProofBundlePolicy (+1 more)

### Community 418 - "governed_helper_gate.py"
Cohesion: 0.11
Nodes (21): Baseline, BaselineEntry, build_parser(), Finding, Findings, _hits(), _in_scope(), line_digest() (+13 more)

### Community 419 - "lane_container_memory_event.py"
Cohesion: 0.11
Nodes (21): alert_key(), build_event(), _check_fields(), ci_runs_truncation_notice(), _delta(), format_ts(), main(), MemoryObservationError (+13 more)

### Community 420 - "check_topology_grant_delivery.py"
Cohesion: 0.10
Nodes (19): _added_serial_columns(), _column_body(), _corpus_files_in_apply_order(), declared_grants(), declared_sequences(), delivered_grants(), delivered_sequences(), GrantKey (+11 more)

### Community 421 - "ValidationAggregator"
Cohesion: 0.06
Nodes (3): EnumHandlerErrorType, EnumHandlerSourceType, ValidationAggregator

### Community 422 - ".run_once"
Cohesion: 0.06
Nodes (10): build_confluent_auth_config_from_env(), idle_watched_topics(), ModelRuntimeHealthCheckEvent, ModelRuntimeHealthDimension, build_runtime_health_block(), _read_runtime_health_verdict(), _consumer_group_state_name(), _list_consumer_group_snapshots() (+2 more)

### Community 423 - "GatewayTokenMinter"
Cohesion: 0.08
Nodes (5): GatewayTokenMinter, ModelGatewayAccessToken, check_gateway_attach(), prove_gateway_attach(), ProtocolGatewayTransport

### Community 424 - "HandlerWriteDecision"
Cohesion: 0.07
Nodes (9): ConflictWritten, DecisionScopeKey, HandlerWriteDecision, _ordered_pair(), _severity_for_confidence(), Stage1Result, Stage2Result, structural_confidence() (+1 more)

### Community 425 - "node_dev_boot_guard_effect/contract.yaml"
Cohesion: 0.05
Nodes (39): capabilities, description, name, contract_name, contract_version, major, minor, patch (+31 more)

### Community 426 - "node_dev_lane_code_ref_pin_effect/contract.yaml"
Cohesion: 0.05
Nodes (39): capabilities, description, name, contract_name, contract_version, major, minor, patch (+31 more)

### Community 427 - "node_dev_lane_pin_compose_guard_effect/contract.yaml"
Cohesion: 0.05
Nodes (39): capabilities, description, name, contract_name, contract_version, major, minor, patch (+31 more)

### Community 429 - "node_runtime_start_first_effect/contract.yaml"
Cohesion: 0.05
Nodes (39): capabilities, description, name, contract_name, contract_version, major, minor, patch (+31 more)

### Community 430 - "RuntimeLogEventBridge"
Cohesion: 0.06
Nodes (6): _categorize_logger(), _log_level_to_severity(), RuntimeLogEventBridge, _templatize_message(), ProtocolRuntimeLogProducer, get_active_flow_key()

### Community 431 - "ConsumerHealthProjectionConsumer"
Cohesion: 0.06
Nodes (6): ConfigConsumerHealthProjection, ConsumerHealthProjectionConsumer, EnumHealthStatus, _main(), _mask_dsn_password(), WriterConsumerHealthPostgres

### Community 432 - "registry_api/routes.py"
Cohesion: 0.14
Nodes (13): get_contract(), get_correlation_id(), get_discovery(), get_node(), get_service(), get_topic(), get_widget_mapping(), health_check() (+5 more)

### Community 433 - "OMN-17292.yaml"
Cohesion: 0.05
Nodes (38): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+30 more)

### Community 434 - "OMN-19383.yaml"
Cohesion: 0.05
Nodes (38): dod-omn19383-ac1-events-package-is-lane-state, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+30 more)

### Community 435 - "OMN-19496.yaml"
Cohesion: 0.05
Nodes (38): dod-omn-19496-ac1-laptop-env-no-lab-secret, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+30 more)

### Community 436 - "ci-relay.yaml"
Cohesion: 0.05
Nodes (38): command, container_name, depends_on, condition, service, description, hardcoded_env, CI_RELAY_HOST (+30 more)

### Community 437 - "linear-relay.yaml"
Cohesion: 0.05
Nodes (38): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BOOTSTRAP_SERVERS (+30 more)

### Community 438 - "projection-api.yaml"
Cohesion: 0.05
Nodes (38): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+30 more)

### Community 439 - "waitlist-signup-notifier.yaml"
Cohesion: 0.05
Nodes (38): command, container_name, depends_on, condition, service, description, hardcoded_env, WAITLIST_NOTIFIER_HEALTH_CHECK_HOST (+30 more)

### Community 440 - "environment"
Cohesion: 0.26
Nodes (39): INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_ENVIRONMENT_SLUG, INFISICAL_PROJECT_ID, KAFKA_CONSUMER_GROUP, KAFKA_ENVIRONMENT, KAFKA_TOPIC_NAMESPACE (+31 more)

### Community 441 - "omnipc2-ci-runner-5"
Cohesion: 0.05
Nodes (38): create_host_path, cgroup_parent, cpus, image, mem_limit, memswap_limit, pids_limit, restart (+30 more)

### Community 442 - "shlex"
Cohesion: 0.07
Nodes (10): _bifrost_text(), _bool_text(), format_dotenv(), format_shell(), load_contract(), main(), render_env(), _secret_resolver_config_json() (+2 more)

### Community 443 - "with"
Cohesion: 0.05
Nodes (38): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, CI_BUS_OVERLAY_LIVE_PATH (+30 more)

### Community 444 - "dependency-cascade-dispatch.yml"
Cohesion: 0.05
Nodes (38): name, secrets, uses, with, concurrency, cancel-in-progress, group, description (+30 more)

### Community 445 - "steps"
Cohesion: 0.05
Nodes (38): concurrency, cancel-in-progress, group, GH_TOKEN, PIP_INDEX_URL, UV_DEFAULT_INDEX, UV_INDEX_URL, packages (+30 more)

### Community 446 - "dlq_replay.py"
Cohesion: 0.08
Nodes (7): cmd_list(), cmd_replay(), cmd_stats(), create_parser(), main(), ModelReplayConfig, safe_truncate()

### Community 447 - "handler_hotpatch_ledger_reconcile.py"
Cohesion: 0.10
Nodes (14): _ancestor_candidate(), _backup(), _candidates(), _git(), HandlerHotpatchLedgerReconcile, _ledger_lock(), _live_prepatch_files(), ReconcileRefusedError (+6 more)

### Community 448 - "check_application_database_sql.py"
Cohesion: 0.09
Nodes (16): _assert_base_revision_resolvable(), changed_sql_paths(), _is_legacy_default_schema_sql_path(), _is_non_deployable_sql_path(), load_sql_baseline(), main(), _parser(), pull_request_merge_ref_base() (+8 more)

### Community 449 - "register"
Cohesion: 0.10
Nodes (10): EnumDevLaneBranchInstallReleaseReason, DevLaneBranchInstallStore, _emit(), main(), register(), release(), status_payload(), _utc() (+2 more)

### Community 450 - "EnumNodeArchetype"
Cohesion: 0.07
Nodes (9): EnumNodeArchetype, ModelOutputValidationParams, ModelValidateAndRaiseParams, get_execution_shape_rules(), enforce_execution_shape(), decorator(), wrapper(), ExecutionShapeViolationError (+1 more)

### Community 451 - "RemoteTaskStateRepository"
Cohesion: 0.10
Nodes (4): EnumPostgresErrorCode, RemoteTaskStateRepository, _op(), _op()

### Community 452 - "ModelDiscoveredContract"
Cohesion: 0.08
Nodes (3): AsgiMounts, ModelHttpMount, ModelDiscoveredContract

### Community 453 - "HandlerGmailIntentPoll"
Cohesion: 0.06
Nodes (9): _decode_body_data(), _extract_body_text(), _extract_list_of_dicts(), _extract_string_list(), ModelGmailMessage, extract_urls(), HandlerGmailIntentPoll, ModelGmailIntentPollerConfig (+1 more)

### Community 454 - "quote_identifier"
Cohesion: 0.08
Nodes (4): quote_identifier(), validate_identifier(), IntentEffectPostgresUpdate, MixinProjectorSqlOperations

### Community 455 - "ModelRuntimeContractConfig"
Cohesion: 0.06
Nodes (4): ModelRoutingEntry, ModelRoutingSubcontract, _create_handler_routing_subcontract(), ModelRuntimeContractConfig

### Community 456 - "node_auth_gate_compute/contract.yaml"
Cohesion: 0.05
Nodes (38): capabilities, description, name, version, contract_name, contract_version, major, minor (+30 more)

### Community 457 - "ModelGatewayEnvelope"
Cohesion: 0.06
Nodes (6): HandlerConsumeInbound, _coerce_dispatch_input(), HandlerForwardOutbound, ModelGatewayEnvelope, ProtocolGatewayPublisher, stamp_verified_tenant_slug()

### Community 458 - "node_migration_append_only_check_compute/contract.yaml"
Cohesion: 0.05
Nodes (38): contract_name, contract_version, major, minor, patch, description, descriptor, idempotent (+30 more)

### Community 459 - "node_migration_freeze_check_compute/contract.yaml"
Cohesion: 0.05
Nodes (38): contract_name, contract_version, major, minor, patch, description, descriptor, idempotent (+30 more)

### Community 460 - "node_migration_sequence_check_compute/contract.yaml"
Cohesion: 0.05
Nodes (38): contract_name, contract_version, major, minor, patch, description, descriptor, idempotent (+30 more)

### Community 461 - "node_setup_validate_effect/contract.yaml"
Cohesion: 0.05
Nodes (38): capabilities, changelog, change, ticket, version, contract_version, major, minor (+30 more)

### Community 462 - "node_sibling_sha_pin_ratchet_compute/contract.yaml"
Cohesion: 0.05
Nodes (38): contract_name, contract_version, major, minor, patch, description, descriptor, idempotent (+30 more)

### Community 463 - "WiringHealthChecker"
Cohesion: 0.08
Nodes (5): ModelTopicWiringHealth, ModelWiringHealthAlert, ModelWiringHealthMetrics, ModelWiringHealthSnapshotEvent, WiringHealthChecker

### Community 464 - "routing_decision_v1.yaml"
Cohesion: 0.05
Nodes (38): ci_gate, failure_criteria, test_class, test_file, collapsed_fields, name, note, ge (+30 more)

### Community 465 - "OMN-18508.yaml"
Cohesion: 0.05
Nodes (37): dod-omn18508-ac6-accepted-20261018t2045z, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 466 - "OMN-19339.yaml"
Cohesion: 0.05
Nodes (37): dod-omn19339-ac4-lane-not-governed, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 467 - "OMN-19565.yaml"
Cohesion: 0.05
Nodes (37): dod-omn19565-ac4-accepted-20261112t1945z, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 468 - "OMN-19623.yaml"
Cohesion: 0.05
Nodes (37): dod-omn19623-ac1-accepted-20261021t2335z, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 469 - "OMN-19812.yaml"
Cohesion: 0.05
Nodes (37): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+29 more)

### Community 470 - "OMN-19957.yaml"
Cohesion: 0.05
Nodes (37): dod-omn19957-ac1-accepted-20261022t0245z, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 471 - "OMN-20764.yaml"
Cohesion: 0.05
Nodes (37): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+29 more)

### Community 472 - "OMN-20959.yaml"
Cohesion: 0.05
Nodes (37): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+29 more)

### Community 473 - "OMN-21052.yaml"
Cohesion: 0.05
Nodes (37): dod-omn21052-ac2-accepted-20261116t0145z, id, statement, check_type, check_value, cwd, context, ac_binding_provenance (+29 more)

### Community 474 - "OMN-21206.yaml"
Cohesion: 0.05
Nodes (37): id, statement, check_type, check_value, cwd, context, vendor_provenance, dod_evidence (+29 more)

### Community 475 - "inject_env"
Cohesion: 0.05
Nodes (38): DELEGATION_ROUTING_TIERS_PATH, DEPLOY_AGENT_HMAC_SECRET, GH_TOKEN, GITHUB_TOKEN, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_PROJECT_ID, KAFKA_SASL_MECHANISM (+30 more)

### Community 476 - "steps"
Cohesion: 0.05
Nodes (37): concurrency, cancel-in-progress, group, BASE_REF, DISPOSITION_RECORD_PATH, GITHUB_TOKEN, HEAD_SHA, PR_NUMBER (+29 more)

### Community 477 - "CatalogManifest"
Cohesion: 0.07
Nodes (10): EnumInfraLayer, Bundle, CatalogManifest, DependsOnEntry, PortMapping, ResourceLimits, ULimit, ModelOptionalDirectoryBindMount (+2 more)

### Community 478 - "handler_c11_delegation_legs.py"
Cohesion: 0.11
Nodes (17): _attempts_of(), grade_canary(), grade_envelope_corpus(), grade_legs(), grade_policy_refusal(), grade_rung_503_control(), LegResult, LegsConfig (+9 more)

### Community 479 - "run-forward-migrations.sh"
Cohesion: 0.15
Nodes (37): acquire_migration_lock(), assert_migration_lock_still_held(), database_exists(), ensure_directive_database(), file_sha256(), grant_corpus_applier_database_create(), import_cloud_history(), import_ledger_stage() (+29 more)

### Community 480 - "HandlerWorkLedgerQuery"
Cohesion: 0.06
Nodes (4): HandlerWorkLedgerQuery, load_work_ledger_query_client_config(), ModelValueHolder, ModelWorkLedgerQueryClientConfig

### Community 481 - "HandlerConsumerHealthTriage"
Cohesion: 0.08
Nodes (5): EnumConsumerIncidentState, ModelConsumerHealthEvent, ModelConsumerRestartCommand, HandlerConsumerHealthTriage, ModelTriageResult

### Community 482 - "node_context_audit_dlq_effect/contract.yaml"
Cohesion: 0.05
Nodes (37): consumed_events, description, event_type, message_category, topic, contract_version, major, minor (+29 more)

### Community 483 - "HandlerDlqReplay"
Cohesion: 0.09
Nodes (8): _drain_state_for(), _halted_coordinate(), HandlerDlqReplay, _lease_for(), _run_lock_for(), ModelDlqReplayResult, ModelDlqReplayRunResult, DlqDrainRecord

### Community 484 - "node_kafka_replay_compute/contract.yaml"
Cohesion: 0.05
Nodes (37): contract_version, major, minor, patch, definitions, progress_model, description, event_bus (+29 more)

### Community 485 - "node_rrh_storage_effect/contract.yaml"
Cohesion: 0.05
Nodes (37): capabilities, description, name, version, contract_version, major, minor, patch (+29 more)

### Community 486 - "OMN-17882.yaml"
Cohesion: 0.05
Nodes (36): id, statement, check_type, check_value, cwd, context, vendor_provenance, dod_evidence (+28 more)

### Community 487 - "OMN-19314.yaml"
Cohesion: 0.05
Nodes (36): dod-omn19314-ac1-failed-apply-records-stderr-tail, dod-omn19314-ac2-empty-stderr-records-stdout-tail, check_type, check_value, cwd, context, ac_binding_provenance, dod_evidence (+28 more)

### Community 488 - "candidate-matrix.yaml"
Cohesion: 0.05
Nodes (36): absent_connect_principals, absent_owner_roles, absent_principals, application, absent_schemas, application, allowed_connect_principals, allowed_memberships (+28 more)

### Community 489 - "migration-gate.yaml"
Cohesion: 0.05
Nodes (36): command, container_name, depends_on, condition, service, description, hardcoded_env, NODE_POSTGRES_DB (+28 more)

### Community 490 - "omnimarket-projection-delegation.yaml"
Cohesion: 0.05
Nodes (36): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BOOTSTRAP_SERVERS (+28 more)

### Community 491 - "omnimarket-projection-llm-cost.yaml"
Cohesion: 0.05
Nodes (36): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BOOTSTRAP_SERVERS (+28 more)

### Community 492 - "omnimarket-projection-run-trace.yaml"
Cohesion: 0.05
Nodes (36): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BOOTSTRAP_SERVERS (+28 more)

### Community 493 - "omnimemory-qdrant.yaml"
Cohesion: 0.05
Nodes (36): container_name, depends_on, description, extra_networks, hardcoded_env, QDRANT__LOG_LEVEL, QDRANT__SERVICE__HOST, QDRANT__STORAGE__ON_DISK_PAYLOAD (+28 more)

### Community 494 - "principal-issuer"
Cohesion: 0.05
Nodes (36): create_host_path, context, dockerfile, external, name, RPK_ADMIN_HOSTS, RPK_BROKERS, RPK_PASS (+28 more)

### Community 495 - "omnipc2-customer-plane-runner-1"
Cohesion: 0.05
Nodes (36): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN, UV_CONCURRENT_BUILDS, UV_CONCURRENT_DOWNLOADS (+28 more)

### Community 496 - "arm64-verify-proof"
Cohesion: 0.06
Nodes (36): if, name, needs, runs-on, steps, strategy, timeout-minutes, concurrency (+28 more)

### Community 497 - "shared-env-runner-parity.yml"
Cohesion: 0.06
Nodes (36): concurrency, cancel-in-progress, group, env, OMNI_CI_ENV_ROOT, PYTHON_VERSION, UV_VERSION, name (+28 more)

### Community 498 - "HandlerDevLaneBranchInstall"
Cohesion: 0.08
Nodes (7): EnumDevLaneBranchInstallFileState, EnumDevLaneBranchInstallState, DevLaneContainerReader, HandlerDevLaneBranchInstall, ModelDevLaneBranchInstallFileReadback, ModelDevLaneBranchInstallReadback, ModelDevLaneBranchInstallReceipt

### Community 499 - "ModelConsumerSyncStatus"
Cohesion: 0.06
Nodes (9): ModelConsumerSyncStatus, ModelDispatchDeadlineStatus, ProtocolConsumerSyncSource, ProtocolDispatchDeadlineSource, describe_consumer_sync(), _describe_one_group(), evaluate_consumer_sync(), evaluate_dispatch_deadline() (+1 more)

### Community 500 - "manifest_fetch.py"
Cohesion: 0.08
Nodes (8): count_manifest_contracts(), EnumManifestFetchFailure, fetch_manifest_contract_count(), fetch_manifest_once(), fetch_manifest_with_budget(), ManifestFetchOutcome, ManifestFetchReport, RetryBudget

### Community 501 - "expand_contract_env_refs"
Cohesion: 0.08
Nodes (7): _declared_overlay_roots(), HandlerNodeBrokerResolve, MissingOrNoKeyOverlayError, _workspace_root_from_environment(), ModelNodeBrokerResolveRequest, ModelNodeEventBusOverlay, expand_contract_env_refs()

### Community 503 - "model_application_relation_evidence_inventory.py"
Cohesion: 0.06
Nodes (4): EnumOwnershipEvidenceStatus, ModelApplicationInventoryRuntimeEvidence, ModelNodeOwnershipDocument, ModelRuntimeEvidenceStatus

### Community 504 - "runtime_llm_endpoint_env.py"
Cohesion: 0.11
Nodes (17): declaration(), _flagged_slots(), LlmEndpointEnvError, _load_contract(), main(), _read_overlay(), _rebound_document(), refresh_exports() (+9 more)

### Community 505 - "handler_git_fix_verifier.py"
Cohesion: 0.10
Nodes (9): CitedThreadLines, _git_argv(), GitFixVerifier, Hunk, _map_line(), map_range(), parse_hunks(), _patch_ids_of() (+1 more)

### Community 506 - "shutil"
Cohesion: 0.11
Nodes (18): bind(), _copy_private(), main(), _nonempty_string(), _read_json_bytes(), _terminal_identity(), _verify_first_stage(), build_final_k1_k6_terminal_evidence_sink() (+10 more)

### Community 507 - "omninode-fleet-failure-probe.py"
Cohesion: 0.12
Nodes (22): _age_seconds(), check_lab_pass_receipts(), check_scheduled_workflows(), _child_env(), emit(), _emit_cached(), _emit_unreadable(), _gh_json() (+14 more)

### Community 508 - "HandlerLabProofPlan"
Cohesion: 0.07
Nodes (8): ModelLabProofRetry, HandlerLabProofPlan, _is_test_file(), LabProofPlanError, _packages(), ModelLabDesiredStateRenderRequest, ModelLabDesiredStateRenderResult, ModelRunnerFleetPool

### Community 510 - "ModelGatewaySession"
Cohesion: 0.08
Nodes (10): ModelGatewaySession, assert_expiry_not_extended(), build_renewal_directive(), ExpiryExtensionError, is_renewal_due(), exceeds_unverified_ceiling(), is_expired(), SessionExpiredError (+2 more)

### Community 511 - "handler_score_models.py"
Cohesion: 0.09
Nodes (12): ModelEndpointHealth, _capability_match(), _compute_score(), _effective_cost_per_1k(), HandlerScoreModels, _passes_hard_constraints(), ScoringContext, ModelLiveMetrics (+4 more)

### Community 512 - "node_onboarding_orchestrator/contract.yaml"
Cohesion: 0.06
Nodes (36): capabilities, contract_version, major, minor, patch, description, descriptor, node_archetype (+28 more)

### Community 513 - "node_rrh_validate_compute/contract.yaml"
Cohesion: 0.05
Nodes (36): capabilities, description, name, version, contract_version, major, minor, patch (+28 more)

### Community 514 - "node_rsd_state_reducer/contract.yaml"
Cohesion: 0.05
Nodes (36): contract_version, major, minor, patch, description, input_model, description, module (+28 more)

### Community 515 - "node_scope_extract_compute/contract.yaml"
Cohesion: 0.05
Nodes (36): contract_version, major, minor, patch, description, descriptor, idempotent, node_archetype (+28 more)

### Community 516 - "node_scope_workflow_reducer/contract.yaml"
Cohesion: 0.05
Nodes (36): contract_version, major, minor, patch, description, input_model, description, module (+28 more)

### Community 517 - "HandlerServiceValidate"
Cohesion: 0.06
Nodes (5): ProtocolValidateEffect, HandlerServiceValidate, ModelSetupNodeHealthResult, ModelSetupValidateEffectOutput, RegistryInfraSetupValidateEffect

### Community 518 - "connect_cloud.yaml"
Cohesion: 0.05
Nodes (36): credential_store_output, write_credentials, description, env_output, write_credentials, max_estimated_minutes, next, set_state (+28 more)

### Community 519 - "OMN-21288.yaml"
Cohesion: 0.06
Nodes (35): id, statement, check_type, check_value, cwd, context, http_mount_remaining, remaining_after_this_change (+27 more)

### Community 520 - "OMN-21386.yaml"
Cohesion: 0.06
Nodes (35): id, statement, check_type, check_value, cwd, context, release_handoff, remaining (+27 more)

### Community 521 - "environment"
Cohesion: 0.13
Nodes (36): BIFROST_VERIFY_ENDPOINTS, CODING_AGENT_CRED_HOME, KAFKA_ENVIRONMENT, KAFKA_INSTANCE_ID, KAFKA_MAX_POLL_INTERVAL_MS, OMNI_HOME, OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS, OMNIINTELLIGENCE_PUBLISH_INTROSPECTION (+28 more)

### Community 522 - "with"
Cohesion: 0.06
Nodes (35): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, BASE_SHA (+27 more)

### Community 523 - "with"
Cohesion: 0.06
Nodes (35): concurrency, cancel-in-progress, group, GH_TOKEN, pr_number, jobs, refresh, name (+27 more)

### Community 524 - "with"
Cohesion: 0.06
Nodes (36): with, build-args, buildkitd-config, cache-enabled, cache-from, cache-key-prefix, cache-to, cache-version (+28 more)

### Community 525 - "with"
Cohesion: 0.06
Nodes (35): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, LINEAR_API_KEY (+27 more)

### Community 526 - "with"
Cohesion: 0.06
Nodes (35): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, LINEAR_API_KEY (+27 more)

### Community 527 - "with"
Cohesion: 0.06
Nodes (35): name, runs-on, steps, timeout-minutes, PROBE_REPO, PROBE_VARS_JSON, repo, vars-json (+27 more)

### Community 528 - "with"
Cohesion: 0.06
Nodes (35): concurrency, cancel-in-progress, group, name, permissions, runs-on, steps, timeout-minutes (+27 more)

### Community 529 - "steps"
Cohesion: 0.06
Nodes (35): concurrency, cancel-in-progress, group, UV_PUBLISH_TOKEN, publish, python-version, jobs, release (+27 more)

### Community 530 - "steps"
Cohesion: 0.06
Nodes (35): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, required (+27 more)

### Community 531 - "ContractAttachReadinessGate"
Cohesion: 0.07
Nodes (5): EnumContractAttachExclusionReason, ModelContractAttachExclusion, EnumContractAttachGatePhase, ContractAttachReadinessGate, ModelContractAttachGateStatus

### Community 532 - "ConsumerMetrics"
Cohesion: 0.06
Nodes (4): ModelBindAllHealthServerBinding, ModelLoopbackHealthServerBinding, ConfigContextAuditConsumer, ConsumerMetrics

### Community 533 - "scanner_orchestration_monolith_ratchet.py"
Cohesion: 0.09
Nodes (12): _baseline_key(), load_monolith_baseline(), main(), monolith_ratchet_violations(), monolith_strict_violations(), MonolithBaselineEntry, MonolithScanResult, _relative_handler_path() (+4 more)

### Community 534 - "e2e_cloud_workflow_harness.py"
Cohesion: 0.10
Nodes (15): build_parser(), HarnessConfig, HarnessConfigError, HarnessPlan, main(), _render(), run(), stage_cross_tenant_denial() (+7 more)

### Community 535 - "ModelMCPToolDefinition"
Cohesion: 0.07
Nodes (3): AdapterONEXToolExecution, ModelMCPToolDefinition, ServiceMCPToolDiscovery

### Community 536 - "DevRollEventsPublisher"
Cohesion: 0.07
Nodes (5): EnumDevStartCause, ModelDevLogCapture, ModelDevRollIncident, ModelDevRuntimeStarted, DevRollEventsPublisher

### Community 537 - "node_setup_infisical_effect/contract.yaml"
Cohesion: 0.06
Nodes (35): capabilities, changelog, change, ticket, version, contract_version, major, minor (+27 more)

### Community 538 - "ServiceContractPublisher"
Cohesion: 0.07
Nodes (6): ContractPublisherError, ContractPublishingInfraError, NoContractsFoundError, ModelContractError, ModelInfraError, ServiceContractPublisher

### Community 539 - "OMN-21923.yaml"
Cohesion: 0.06
Nodes (34): id, statement, check_type, check_value, cwd, context, dev_integration, redpanda_proof (+26 more)

### Community 540 - "lane-mirror-forwarder.yaml"
Cohesion: 0.06
Nodes (34): command, container_name, depends_on, condition, service, description, hardcoded_env, healthcheck (+26 more)

### Community 541 - "environment"
Cohesion: 0.06
Nodes (34): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, OMNIBASE_OPERATOR_ENV_FILE, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN (+26 more)

### Community 542 - "omnipc2-verify-runner-1"
Cohesion: 0.06
Nodes (34): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, LANE_PROBE_HOST, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN, interval (+26 more)

### Community 543 - "deployed_readback"
Cohesion: 0.06
Nodes (34): deployed_feature_release, deployed_readback, cluster_version, container, health, host, image, node_earliest_version (+26 more)

### Community 544 - "HandlerTopicMigrationExecutor"
Cohesion: 0.07
Nodes (5): HandlerTopicMigrationExecutor, ModelTopicMigrationCommand, ModelTopicMigrationLifecycleEvent, HandlerTopicMigrationProjection, ModelTopicMigrationProjectionRow

### Community 545 - "steps"
Cohesion: 0.06
Nodes (34): name, runs-on, steps, timeout-minutes, env, EXPECTED_COMMIT_SHA, INPUT_EXPECTED_COMMIT_SHA, INPUT_RUNTIME_SOURCE_HASH (+26 more)

### Community 546 - "with"
Cohesion: 0.06
Nodes (34): if, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, APP_TOKEN (+26 more)

### Community 547 - "with"
Cohesion: 0.06
Nodes (34): name, runs-on, steps, timeout-minutes, GH_TOKEN, OMNIMARKET_SRC, jobs, deployed-migration-tree-sync-logic (+26 more)

### Community 548 - "AgentRegistry"
Cohesion: 0.09
Nodes (4): EnumAgentStatus, ModelAgentBinding, ModelAgentEntity, AgentRegistry

### Community 549 - ".compute_savings"
Cohesion: 0.07
Nodes (4): HandlerRunnerUsageSavings, _quantize(), ModelRunnerSavingsEstimated, ModelRunnerUsageEvent

### Community 550 - "validator_localhandler.py"
Cohesion: 0.08
Nodes (7): ModelLocalHandlerValidationResult, ModelLocalHandlerViolation, _should_skip_file(), validate_localhandler(), validate_localhandler_ci(), validate_localhandler_in_file(), _validate_localhandler_with_count()

### Community 551 - "HandlerLLMCompletion"
Cohesion: 0.07
Nodes (5): HandlerLLMCompletion, ModelLLMCompletionMessage, ModelLLMCompletionRequest, ModelLLMCompletionResult, RegistryInfraLlmCompletionEffect

### Community 552 - "audit_required_context_parity_cli.py"
Cohesion: 0.09
Nodes (14): cmd_report(), collect_live_state(), fetch_protection(), fetch_queue_enabled(), fetch_workflow_jobs(), main(), real_gh(), _render_human() (+6 more)

### Community 553 - "c11_negative_paths_probe.py"
Cohesion: 0.11
Nodes (17): CaseResult, CaseSpec, _code_of(), grade(), grade_case(), main(), _maybe_json(), _names_missing_field() (+9 more)

### Community 554 - "check_pin_reachability.py"
Cohesion: 0.11
Nodes (15): _default_targets(), _expand_targets(), extract_pins(), extract_precommit_pins(), extract_pyproject_pins(), extract_uv_lock_pins(), extract_workflow_pins(), _is_literal() (+7 more)

### Community 555 - "check_migration_class.py"
Cohesion: 0.11
Nodes (20): _alter_findings(), _alter_table_findings(), _call_findings(), check_tree(), _created_tables(), destructive_findings(), enumerate_forward_migrations(), load_executions() (+12 more)

### Community 556 - "closer_receipt_residual.py"
Cohesion: 0.09
Nodes (17): _build_arg_parser(), collect_held_unbound_tickets(), ContractBindingCounts, count_contract_bindings(), distinct_tickets_currently_held(), _dod_evidence_items(), extract_receipt_json_lines(), fetch_run_log() (+9 more)

### Community 557 - "HandlerWorkspaceRuntimeConfigMaterializer"
Cohesion: 0.06
Nodes (4): HandlerWorkspaceRuntimeConfigMaterializer, ModelMaterializedWorkspaceRuntimeConfig, ModelWorkspaceRuntimeConfigMaterialization, ModelWorkspaceRuntimeConfigSidecar

### Community 561 - "node_model_router_compute/contract.yaml"
Cohesion: 0.06
Nodes (34): contract_version, major, minor, patch, description, module, name, handler_routing (+26 more)

### Community 562 - "node_scope_file_read_effect/contract.yaml"
Cohesion: 0.06
Nodes (34): contract_version, major, minor, patch, description, descriptor, idempotent, node_archetype (+26 more)

### Community 563 - "HandlerInfisicalFullSetup"
Cohesion: 0.07
Nodes (4): HandlerInfisicalFullSetup, ModelInfisicalSetupEffectOutput, RegistryInfraSetupInfisicalEffect, ProtocolInfisicalEffect

### Community 564 - "node_setup_preflight_effect/contract.yaml"
Cohesion: 0.06
Nodes (34): capabilities, contract_version, major, minor, patch, description, descriptor, node_archetype (+26 more)

### Community 565 - "TopicResolver"
Cohesion: 0.07
Nodes (7): ModelTieredResolutionContractResult, ModelTieredResolutionConfigLocal, ModelTrustDomainConfigLocal, ModelBusDescriptor, BusDescriptorNotFoundError, TopicResolutionError, TopicResolver

### Community 566 - "mixin_any_type_exemption.py"
Cohesion: 0.07
Nodes (5): _extract_comment_portion(), _has_type_ignore_with_valid_note(), _is_any_exemption_note(), _is_in_multiline_import(), MixinAnyTypeExemption

### Community 567 - "typing_extensions"
Cohesion: 0.06
Nodes (7): Intent Model Architecture, ModelDevBootContainerConfig, ModelDevBootHealthProbe, ModelDevBootHealthStatus, ModelDevBootHealthcheck, ModelDevBootState, ProtocolIntentPayload

### Community 568 - "OMN-20275.yaml"
Cohesion: 0.06
Nodes (33): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+25 more)

### Community 569 - "runtime_config.yaml"
Cohesion: 0.06
Nodes (33): boot_subscriptions, policy, contract_registry, enabled, tick_interval_seconds, contract_version, description, event_bus (+25 more)

### Community 571 - "forward-migration.yaml"
Cohesion: 0.06
Nodes (33): command, container_name, depends_on, condition, service, description, hardcoded_env, MIGRATIONS_DIR (+25 more)

### Community 572 - "keycloak.yaml"
Cohesion: 0.06
Nodes (33): command, container_name, depends_on, condition, service, description, hardcoded_env, KC_DB (+25 more)

### Community 573 - "omnimemory-memgraph.yaml"
Cohesion: 0.06
Nodes (33): command, container_name, depends_on, description, extra_networks, hardcoded_env, healthcheck, interval_s (+25 more)

### Community 574 - "factory-floor-web"
Cohesion: 0.06
Nodes (33): create_host_path, cap_drop, container_name, healthcheck, image, init, logging, mem_limit (+25 more)

### Community 575 - "environment"
Cohesion: 0.06
Nodes (33): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN, UV_CONCURRENT_BUILDS (+25 more)

### Community 576 - "omninode-studio-runner-1"
Cohesion: 0.06
Nodes (33): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, LANE_PROBE_HOST, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN, interval (+25 more)

### Community 577 - "prepr_pool_group.py"
Cohesion: 0.12
Nodes (16): build_group(), Candidate, _candidates_from_map(), exclusion(), GroupBuild, halves(), LandedVerdict, _load_candidates() (+8 more)

### Community 578 - "compose-config-lint"
Cohesion: 0.06
Nodes (33): name, permissions, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+25 more)

### Community 579 - "steps"
Cohesion: 0.06
Nodes (33): concurrency, cancel-in-progress, group, BASE_SHA, EVENT_NAME, HEAD_SHA, PATH_REGEX, PUSH_TIMEOUT_S (+25 more)

### Community 580 - "with"
Cohesion: 0.06
Nodes (33): concurrency, cancel-in-progress, group, env, PYTHON_VERSION, UV_VERSION, jobs, lab-overlay-tests (+25 more)

### Community 582 - "ModelRelationEvidence"
Cohesion: 0.06
Nodes (4): EnumApplicationDatabaseIdentityRoot, EnumApplicationDatabaseIdentityRootOperation, ModelApplicationDatabaseIdentityRootControlState, ModelRelationEvidence

### Community 583 - "provider_rung_canary_observe.py"
Cohesion: 0.12
Nodes (20): backend_secret_ref(), deployed_post(), post(), deployed_resolve(), resolve(), error_fields(), _exc(), _headers() (+12 more)

### Community 584 - ".__init__"
Cohesion: 0.06
Nodes (4): ModelConsumerRetryConfig, ModelDlqConfig, ModelIdempotencyConfig, ModelOffsetPolicyConfig

### Community 585 - "check_prod_promotion_lineage.py"
Cohesion: 0.10
Nodes (14): assert_image_revision_matches(), assert_prod_build_promoted(), describe_pending_entries(), EnumProdLineageFailure, _git(), head_is_promoted(), inspect_image(), main() (+6 more)

### Community 586 - "preflight_hotpatch_ledger.py"
Cohesion: 0.11
Nodes (18): build_parser(), candidate_commits(), check_bypass(), commit_known(), ContainerAbsentError, fail(), is_ancestor(), load_ledger() (+10 more)

### Community 587 - "register-repo.py"
Cohesion: 0.11
Nodes (13): _bootstrap_keys(), cmd_onboard_repo(), cmd_seed_shared(), _create_folders_via_admin(), _identity_defaults(), _is_abort_error(), _load_infisical_adapter(), _load_registry() (+5 more)

### Community 588 - "ServiceEnvelopeValidator"
Cohesion: 0.06
Nodes (3): EnumValidationErrorCode, ServiceEnvelopeValidator, ValidationResult

### Community 589 - "ModelFileValidationResult"
Cohesion: 0.07
Nodes (3): ModelArchitectureValidationRequest, ModelFileValidationResult, NodeArchitectureValidator

### Community 590 - "node_emit_daemon_runtime/contract.yaml"
Cohesion: 0.06
Nodes (33): consumed_events, description, event_type, message_category, topic, contract_version, major, minor (+25 more)

### Community 591 - "_handle_interactive"
Cohesion: 0.08
Nodes (10): _handle_interactive(), handle_onboarding(), HandlerOnboarding, _load_interactive_policy(), OnboardingHandlerError, _write_credential_store(), _write_declared_credentials(), ModelOnboardingInput (+2 more)

### Community 592 - "BatchResponsePublisher"
Cohesion: 0.07
Nodes (3): BatchResponsePublisher, ModelBatchPublisherConfig, ModelBatchPublisherMetrics

### Community 593 - "ContextAuditConsumer"
Cohesion: 0.08
Nodes (4): ContextAuditConsumer, _run_once(), mask_dsn_password(), WriterContextAuditPostgres

### Community 594 - "validation"
Cohesion: 0.06
Nodes (33): contract_kind, allow_in_init, allowed_class_attributes, allowed_methods, allowed_statements, base_class_patterns, forbidden_decorators, forbidden_patterns (+25 more)

### Community 595 - "OMN-21753.yaml"
Cohesion: 0.06
Nodes (32): id, statement, check_type, check_value, cwd, context, remaining_after_this_change, dod_evidence (+24 more)

### Community 596 - "io"
Cohesion: 0.09
Nodes (11): build_receipt(), _coverage_metrics_for(), coverage_pct_for(), handler_module_sha256(), input_hash(), _JsonDumpable, _measured_arcs(), ModelAdequacyReceipt (+3 more)

### Community 597 - "generate_entry_points.py"
Cohesion: 0.10
Nodes (14): build_entry_point_section(), check_mode(), collect_node_dirs(), generate_mode(), main(), _read_pyproject(), _replace_entry_point_section(), _resolve_repo() (+6 more)

### Community 598 - "postgres.yaml"
Cohesion: 0.06
Nodes (32): container_name, depends_on, description, hardcoded_env, POSTGRES_DB, healthcheck, interval_s, retries (+24 more)

### Community 599 - "volumes"
Cohesion: 0.06
Nodes (33): name, name, name, name, name, name, name, name (+25 more)

### Community 600 - "volumes"
Cohesion: 0.06
Nodes (33): name, name, name, name, name, name, name, name (+25 more)

### Community 601 - "services"
Cohesion: 0.06
Nodes (33): container_name, profiles, container_name, profiles, container_name, container_name, ports, container_name (+25 more)

### Community 602 - "handler_runner_fleet_health_evaluate.py"
Cohesion: 0.12
Nodes (15): _annotate_indeterminate(), _classify_runner(), _container_stability_signal(), _diag_heartbeat_signal(), _disk_capacity_signal(), _docker_health_signal(), _evaluate_readiness_signals(), _github_registration_signal() (+7 more)

### Community 603 - "dispatch-parity-gate.yml"
Cohesion: 0.06
Nodes (32): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, env (+24 more)

### Community 604 - "with"
Cohesion: 0.06
Nodes (32): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, jobs (+24 more)

### Community 605 - "with"
Cohesion: 0.07
Nodes (32): concurrency, cancel-in-progress, group, jobs, runner-image-build-smoke, runner-image-multiarch-smoke, name, on (+24 more)

### Community 606 - "s6-runtime-seam-gate.yml"
Cohesion: 0.06
Nodes (32): concurrency, cancel-in-progress, group, env, CACHE_VERSION, LLM_ENDPOINT_CIDR_ALLOWLIST, ONEX_EVENT_BUS_TYPE, PYTHON_VERSION (+24 more)

### Community 607 - "with"
Cohesion: 0.06
Nodes (32): concurrency, cancel-in-progress, group, OMNIMARKET_SRC, jobs, skill-node-mapping-sync, name, on (+24 more)

### Community 608 - "url-authority-gate.yml"
Cohesion: 0.06
Nodes (32): concurrency, cancel-in-progress, group, env, UV_CONCURRENT_BUILDS, UV_CONCURRENT_DOWNLOADS, UV_CONCURRENT_INSTALLS, UV_HTTP_TIMEOUT (+24 more)

### Community 609 - "JsonPayload"
Cohesion: 0.09
Nodes (6): DictBundleAdapter, ProtocolEvidenceBundle, _sha256_file(), write_evidence_bundle(), _write_artifact(), _write_json()

### Community 611 - "HandlerBootFailureLedger"
Cohesion: 0.09
Nodes (8): BootFailureStreak, HandlerBootFailureLedger, _ledger_dir(), _streak_path(), _write_atomic(), ModelBootAttemptOutcome, ModelBootIdentity, ModelRuntimeBootFailedEvent

### Community 612 - "handlers.py"
Cohesion: 0.13
Nodes (14): _breakdown_from_rows(), _decimal(), fetch_cost_by_model(), fetch_cost_by_repo(), fetch_cost_summary(), fetch_cost_trend(), fetch_savings_summary(), fetch_token_usage() (+6 more)

### Community 613 - "application_database_source_tenant_authority.py"
Cohesion: 0.08
Nodes (13): ApplicationDatabaseAuthorityResolutionError, _composite_argument_is_consumed(), _dollar_quote_at(), _has_escape_string_prefix(), _identifier_is_referenced(), _is_identifier_character(), _named_composite_argument_is_consumed(), _normalize_identifier() (+5 more)

### Community 614 - "prepr_verify_lane.sh"
Cohesion: 0.09
Nodes (32): BUILD_SOURCE, cleanup(), compose(), EXPECTED_BUILD_SOURCE, fail(), first_party_sources(), GIT_SHA, KAFKA_ENVIRONMENT (+24 more)

### Community 615 - "GatewayTransportHttpx"
Cohesion: 0.08
Nodes (5): CheckDelegationKey, default_onex_home(), request_models(), ProtocolWhoamiTransport, GatewayTransportHttpx

### Community 616 - "HandlerOnexApiImageImportCheck"
Cohesion: 0.09
Nodes (8): build_import_check_handler(), HandlerOnexApiImageImportCheck, heal_pin(), import_gate(), _local_images(), _read_pin(), ModelOnexApiImageImportRequest, ModelOnexApiImageImportVerdict

### Community 617 - "node_baselines_batch_compute/contract.yaml"
Cohesion: 0.06
Nodes (32): contract_version, major, minor, patch, description, event_bus, publish_topics, subscribe_topics (+24 more)

### Community 618 - "HandlerBrokerDiskWatermark"
Cohesion: 0.08
Nodes (7): _classify(), HandlerBrokerDiskWatermark, _resolve_volume_mountpoint(), EnumDiskSeverity, ModelBrokerDiskCheck, ModelBrokerDiskWatermarkInput, ModelBrokerDiskWatermarkOutput

### Community 619 - "RuntimeLocalIngressServer"
Cohesion: 0.08
Nodes (7): normalize_correlation_id(), validate_envelope(), ModelLocalRuntimeIngressError, ModelLocalRuntimeIngressResponse, _dispatch_through_broker(), _preferred_request_name(), RuntimeLocalIngressServer

### Community 620 - "set_statement_timeout"
Cohesion: 0.10
Nodes (4): parse_execute_count(), ServiceBatchComputeBaselines, set_statement_timeout(), transaction_context()

### Community 622 - "suites"
Cohesion: 0.06
Nodes (31): suites, tests/integration/event_bus/test_adapter_protocol_event_publisher_kafka.py, tests/integration/migrations/test_node_migration_shape_drift_omn15376.py, tests/integration/migrations/test_omn15683_mixed_representation_conversion.py, tests/integration/migrations/test_omn17316_role_set_membership_guard.py, tests/integration/runtime/test_kafka_contract_source_integration.py, tests/integration/services/mcp/test_service_mcp_tool_sync.py, env_vars (+23 more)

### Community 623 - "OMN-15629.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 624 - "OMN-18087.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 625 - "OMN-19215.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 626 - "OMN-19988.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 627 - "OMN-20117.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 628 - "OMN-20915.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 629 - "OMN-21031.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 630 - "OMN-21041.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 631 - "OMN-21053.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 632 - "OMN-21056.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 633 - "OMN-21071.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 634 - "OMN-21271.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 635 - "OMN-21371.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 636 - "OMN-21403.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 637 - "OMN-21481.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 638 - "OMN-21536.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 639 - "OMN-21587.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 640 - "OMN-21623.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 641 - "OMN-21690.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 642 - "OMN-21696.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 643 - "OMN-21697.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 644 - "OMN-21698.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 645 - "OMN-21699.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 646 - "OMN-21700.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 647 - "OMN-21703.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 648 - "OMN-21704.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 649 - "OMN-21705.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 650 - "OMN-21706.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 651 - "OMN-21719.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 652 - "OMN-21792.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 653 - "OMN-21802.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 654 - "OMN-21920.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 655 - "OMN-21934.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 656 - "OMN-22211.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 657 - "OMN-22530.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 658 - "OMN-22532.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 659 - "OMN-22534.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 660 - "OMN-22535.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 661 - "OMN-22543.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 662 - "OMN-22555.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 663 - "OMN-22561.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 664 - "OMN-22622.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 665 - "OMN-22634.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 666 - "OMN-22635.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 667 - "OMN-22644.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 668 - "OMN-22650.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 669 - "OMN-22668.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 670 - "OMN-22691.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 671 - "OMN-22711.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 672 - "OMN-22724.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 673 - "OMN-22756.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 674 - "OMN-22808.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 675 - "OMN-22815.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 676 - "OMN-22870.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 677 - "OMN-23009.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 678 - "OMN-23027.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 679 - "OMN-23039.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 680 - "OMN-23043.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 681 - "OMN-23055.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 682 - "OMN-23062.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 683 - "OMN-23066.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 684 - "OMN-23072.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 685 - "OMN-23106.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 686 - "OMN-23143.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 687 - "OMN-23304.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 688 - "OMN-23332.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 689 - "OMN-23333.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 690 - "OMN-23342.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 691 - "OMN-23431.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 692 - "OMN-23459.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 693 - "OMN-23513.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 694 - "OMN-23521.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 695 - "OMN-23524.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 696 - "OMN-23525.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 697 - "OMN-23540.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 698 - "OMN-23562.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 699 - "OMN-23579.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 700 - "OMN-23624.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 701 - "OMN-23659.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 702 - "OMN-23715.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 703 - "OMN-23858.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 704 - "OMN-23884.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 705 - "OMN-23896.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 706 - "OMN-23947.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 707 - "OMN-23951.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 708 - "OMN-23981.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 709 - "OMN-23988.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 710 - "OMN-24009.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 711 - "OMN-24013.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 712 - "OMN-24014.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 713 - "OMN-24017.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 714 - "OMN-24032.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 715 - "OMN-24048.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 716 - "OMN-24067.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 717 - "OMN-24077.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 718 - "OMN-24081.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 719 - "OMN-24119.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 720 - "OMN-24127.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 721 - "OMN-24134.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 722 - "OMN-24153.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 723 - "OMN-24186.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 724 - "OMN-24202.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 725 - "OMN-24218.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 726 - "OMN-24223.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 727 - "OMN-24260.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 728 - "OMN-24280.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 729 - "OMN-24299.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 730 - "OMN-24306.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 731 - "OMN-24308.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 732 - "OMN-24323.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 733 - "OMN-24446.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 734 - "OMN-24463.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 735 - "OMN-24465.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 736 - "OMN-24466.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 737 - "OMN-24471.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 738 - "OMN-24480.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 739 - "OMN-24496.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 740 - "OMN-24524.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 741 - "OMN-24526.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 742 - "OMN-24562.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 743 - "OMN-24617.yaml"
Cohesion: 0.06
Nodes (31): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+23 more)

### Community 744 - "environment"
Cohesion: 0.07
Nodes (31): context, dockerfile, labels, target, postgres, ADMIN_DSN, OWNERSHIP_MANIFEST, POOL_DSN_APP_DASHBOARD (+23 more)

### Community 745 - "local-tenant-registry-seed.yaml"
Cohesion: 0.06
Nodes (31): command, container_name, depends_on, condition, service, description, entrypoint, hardcoded_env (+23 more)

### Community 746 - "omnidash.yaml"
Cohesion: 0.06
Nodes (31): container_name, depends_on, condition, service, description, hardcoded_env, healthcheck, interval_s (+23 more)

### Community 747 - "omnimemory-valkey.yaml"
Cohesion: 0.06
Nodes (31): command, container_name, depends_on, description, extra_networks, hardcoded_env, healthcheck, interval_s (+23 more)

### Community 748 - "docker-compose.dev-lane.yml"
Cohesion: 0.06
Nodes (31): infra-runtime, INFRA_RUNTIME_IMAGE, x-dev-lane-analytics-env, OMNIDASH_ANALYTICS_DB_URL, x-dev-lane-broker-auth-env, KAFKA_SASL_MECHANISM, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME (+23 more)

### Community 749 - "services"
Cohesion: 0.06
Nodes (31): create_host_path, name, container_name, image, labels, networks, ports, restart (+23 more)

### Community 750 - "services"
Cohesion: 0.06
Nodes (32): container_name, profiles, profiles, profiles, container_name, labels, restart, profiles (+24 more)

### Community 751 - "dev"
Cohesion: 0.06
Nodes (31): auto_offset_reset, bootstrap_servers, enable_auto_commit, environment, sasl_credential_ref, sasl_mechanism, security_protocol, timeout_seconds (+23 more)

### Community 752 - "with"
Cohesion: 0.06
Nodes (31): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, jobs (+23 more)

### Community 753 - "steps"
Cohesion: 0.06
Nodes (31): runs-on, steps, timeout-minutes, PACKAGE_NAME, RELEASE_TAG, SIMULATED_TAG, TAG_VERSION, package-name (+23 more)

### Community 754 - "runtime_change_classifier.py"
Cohesion: 0.09
Nodes (15): attribute_runtime_paths(), classify_runtime_paths(), find_lane_state_paths(), format_runtime_path_attribution(), git_manifest_reader(), _read(), _show(), _glob_segments() (+7 more)

### Community 755 - "validator_registry.py"
Cohesion: 0.09
Nodes (10): clear_builtin_validators(), execute_validator(), _invoke_validator(), _normalize_validator_result(), OmniGateValidatorCallable, register_builtin_validator(), resolve_validator(), _validator_result() (+2 more)

### Community 756 - "handler_architecture_validation.py"
Cohesion: 0.10
Nodes (10): _baseline_services(), _command_allowed(), _command_argv(), _command_digest(), _command_files(), _command_key(), _command_rows(), _row_digest() (+2 more)

### Community 757 - "expr"
Cohesion: 0.09
Nodes (12): expr, MixinAnyTypeClassification, AnySignatureFinding, _contains_any(), _is_any(), _is_dict_any(), _iter_python_files(), main() (+4 more)

### Community 758 - "runtime_sibling_sha_pin_ratchet.py"
Cohesion: 0.08
Nodes (10): count_pins(), in_scope(), is_pin_line(), NodeSiblingShaPinRatchetCompute, ModelSiblingShaPinRatchetInput, collect_request(), _git(), main() (+2 more)

### Community 759 - "ServiceResolutionEventPublisher"
Cohesion: 0.07
Nodes (4): ModelResolutionEventLocal, ModelResolutionProofLocal, ModelTierAttemptLocal, ServiceResolutionEventPublisher

### Community 760 - "RunStepSequence"
Cohesion: 0.10
Nodes (7): _bus_attempts(), _record_bus_run(), RunStepSequence, _text(), send(), ModelDelegationDispatchTerminalResult, ModelDelegationRunStepEvent

### Community 761 - "check_architecture.sh"
Cohesion: 0.11
Nodes (30): build_grep_excludes(), check_import(), checkout_identity_is_canonical(), core_checkout_git(), count_python_files(), escape_regex(), find_omnibase_core_path(), GREP_EXCLUDE_ARGS (+22 more)

### Community 762 - "check_no_credential_in_log.py"
Cohesion: 0.12
Nodes (18): _bare_word_whole_name(), _check_call(), _check_file(), _collect_paths(), _collect_py_files(), _credential_hit(), _dict_key_findings(), _expr_credential_name() (+10 more)

### Community 763 - "post_hostile_review_threads.py"
Cohesion: 0.11
Nodes (15): _api(), build_review(), collect_findings(), finding_fingerprint(), format_finding_body(), main(), _paginate(), parse_right_side_lines() (+7 more)

### Community 764 - "load_bearing_gates"
Cohesion: 0.09
Nodes (31): load_bearing_gates, merge_policy, aggregator, aggregator_job_id, aggregator_workflow, context, coverage, gate_job_id (+23 more)

### Community 765 - "queue-runner.py"
Cohesion: 0.19
Nodes (27): atomic_write(), capacity_available(), claim_or_recover(), commit_verified_completion(), config_has_url_rewrite(), file_is_owner_regular(), git_text(), initialize() (+19 more)

### Community 766 - "trigger_rebuild_on_merge.py"
Cohesion: 0.10
Nodes (13): assert_consumer_model_accepts_payload(), assert_declared_overlay_loads(), build_kafka_producer_config(), build_source_for_base_branch(), _class_declares_extra_forbid(), emit_github_output(), _field_call_is_required(), lane_for_base_branch() (+5 more)

### Community 767 - "verify_deployed_versions.py"
Cohesion: 0.10
Nodes (12): format_report_text(), format_revision_text(), get_image_revision(), get_installed_version(), main(), parse_versions_string(), RevisionCheckResult, _revisions_match() (+4 more)

### Community 769 - "node_chain_replay_compute/contract.yaml"
Cohesion: 0.07
Nodes (31): contract_version, major, minor, patch, description, module, name, handler_routing (+23 more)

### Community 770 - "node_checkpoint_effect/contract.yaml"
Cohesion: 0.06
Nodes (31): capabilities, contract_version, major, minor, patch, description, module, name (+23 more)

### Community 771 - ".from_contracts_dir"
Cohesion: 0.08
Nodes (10): HandlerContractScan, _env_override_float(), _env_override_int(), _get_dict(), _get_float(), _get_int(), _get_list(), _get_str() (+2 more)

### Community 772 - "node_rsd_data_fetch_effect/contract.yaml"
Cohesion: 0.07
Nodes (31): contract_version, major, minor, patch, description, descriptor, node_archetype, module (+23 more)

### Community 773 - "node_scope_manifest_write_effect/contract.yaml"
Cohesion: 0.07
Nodes (31): contract_version, major, minor, patch, description, descriptor, node_archetype, module (+23 more)

### Community 775 - "ModelWarning"
Cohesion: 0.08
Nodes (7): ModelContractView, ModelPaginationInfo, ModelResponseListContracts, ModelResponseListNodes, ModelResponseListTopics, ModelTopicSummary, ModelWarning

### Community 776 - "DemoLoopGate"
Cohesion: 0.09
Nodes (4): DemoLoopGate, EnumAssertionStatus, ModelAssertionResult, ModelDemoLoopResult

### Community 777 - "HandlerServiceDiscoveryMock"
Cohesion: 0.08
Nodes (4): Handler Renames, HandlerServiceDiscoveryMock, ModelServiceDiscoveryHealthCheckDetails, ModelServiceDiscoveryHealthCheckResult

### Community 778 - "OMN-11187.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 779 - "OMN-13902.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 780 - "OMN-13912.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 781 - "OMN-13966.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 782 - "OMN-14108.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 783 - "OMN-14186.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 784 - "OMN-15361.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 785 - "OMN-15645.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 786 - "OMN-16106.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 787 - "OMN-16553.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 788 - "OMN-16632.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 789 - "OMN-16812.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 790 - "OMN-17105.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 791 - "OMN-17232.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 792 - "OMN-17305.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 793 - "OMN-17353.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 794 - "OMN-17991.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 795 - "OMN-18143.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 796 - "OMN-18144.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 797 - "OMN-18475.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 798 - "OMN-18489.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 799 - "OMN-18534.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 800 - "OMN-18550.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 801 - "OMN-18594.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 802 - "OMN-18595.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 803 - "OMN-18596.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 804 - "OMN-18630.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 805 - "OMN-18632.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, checks, description (+22 more)

### Community 806 - "OMN-18648.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 807 - "OMN-18713.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 808 - "OMN-18843.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 809 - "OMN-18866.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 810 - "OMN-18893.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 811 - "OMN-18894.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 812 - "OMN-18896.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 813 - "OMN-19043.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 814 - "OMN-19064.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 815 - "OMN-19077.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 816 - "OMN-19091.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 817 - "OMN-19092.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 818 - "OMN-19124.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 819 - "OMN-19127.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 820 - "OMN-19131.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 821 - "OMN-19155.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 822 - "OMN-19175.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 823 - "OMN-19211.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 824 - "OMN-19214.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 825 - "OMN-19414.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 826 - "OMN-19416.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 827 - "OMN-19417.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 828 - "OMN-19424.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 829 - "OMN-19427.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 830 - "OMN-19437.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 831 - "OMN-19438.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 832 - "OMN-19445.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 833 - "OMN-19448.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 834 - "OMN-19452.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 835 - "OMN-19455.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 836 - "OMN-19507.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 837 - "OMN-19513.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 838 - "OMN-19558.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 839 - "OMN-19561.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 840 - "OMN-19566.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 841 - "OMN-19655.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 842 - "OMN-19747.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 843 - "OMN-19795.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 844 - "OMN-19807.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 845 - "OMN-19854.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 846 - "OMN-19911.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 847 - "OMN-19944.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 848 - "OMN-19972.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 849 - "OMN-19975.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 850 - "OMN-19983.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 851 - "OMN-19986.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 852 - "OMN-19987.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 853 - "OMN-19991.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 854 - "OMN-19992.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 855 - "OMN-19996.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 856 - "OMN-19998.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 857 - "OMN-20008.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 858 - "OMN-20032.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 859 - "OMN-20068.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 860 - "OMN-20074.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 861 - "OMN-20079.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 862 - "OMN-20112.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 863 - "OMN-20113.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 864 - "OMN-20174.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 865 - "OMN-20232.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 866 - "OMN-20266.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 867 - "OMN-20298.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 868 - "OMN-20376.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 869 - "OMN-20381.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 870 - "OMN-20385.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 871 - "OMN-20387.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 872 - "OMN-20410.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 873 - "OMN-20413.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 874 - "OMN-20422.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 875 - "OMN-20489.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 876 - "OMN-20528.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 877 - "OMN-20597.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 878 - "OMN-20604.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 879 - "OMN-20637.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 880 - "OMN-20696.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 881 - "OMN-20717.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 882 - "OMN-20737.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 883 - "OMN-20740.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 884 - "OMN-20743.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 885 - "OMN-20767.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 886 - "OMN-20772.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 887 - "OMN-20782.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 888 - "OMN-20794.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 889 - "OMN-20798.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 890 - "OMN-20802.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 891 - "OMN-20805.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 892 - "OMN-20820.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 893 - "OMN-20821.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 894 - "OMN-20822.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 895 - "OMN-20827.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 896 - "OMN-20834.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 897 - "OMN-20837.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 898 - "OMN-20844.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 899 - "OMN-20864.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 900 - "OMN-20867.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 901 - "OMN-20870.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 902 - "OMN-20871.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 903 - "OMN-20876.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 904 - "OMN-20910.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 905 - "OMN-20919.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 906 - "OMN-20922.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 907 - "OMN-20923.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 908 - "OMN-20931.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 909 - "OMN-20932.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 910 - "OMN-20933.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 911 - "OMN-20934.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 912 - "OMN-20935.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 913 - "OMN-20940.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 914 - "OMN-20942.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 915 - "OMN-20945.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 916 - "OMN-20947.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 917 - "OMN-20949.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 918 - "OMN-20963.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 919 - "OMN-20991.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 920 - "OMN-20999.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 921 - "OMN-21000.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 922 - "OMN-21015.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 923 - "OMN-21029.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+22 more)

### Community 924 - "OMN-21034.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 925 - "OMN-21045.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 926 - "OMN-21048.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 927 - "OMN-21084.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 928 - "OMN-21087.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 929 - "OMN-21088.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 930 - "OMN-21090.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 931 - "OMN-21099.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 932 - "OMN-21110.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 933 - "OMN-21146.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 934 - "OMN-21147.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+22 more)

### Community 935 - "OMN-21164.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 936 - "OMN-21217.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 937 - "OMN-21226.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 938 - "OMN-21256.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 939 - "OMN-21266.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 940 - "OMN-21268.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 941 - "OMN-21272.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 942 - "OMN-21279.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 943 - "OMN-21298.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 944 - "OMN-21334.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 945 - "OMN-21336.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 946 - "OMN-21339.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 947 - "OMN-21341.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 948 - "OMN-21350.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 949 - "OMN-21351.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 950 - "OMN-21396.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 951 - "OMN-21400.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 952 - "OMN-21401.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 953 - "OMN-21402.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 954 - "OMN-21418.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 955 - "OMN-21421.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 956 - "OMN-21423.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 957 - "OMN-21430.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 958 - "OMN-21431.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 959 - "OMN-21433.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 960 - "OMN-21436.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 961 - "OMN-21487.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 962 - "OMN-21497.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 963 - "OMN-21523.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 964 - "OMN-21529.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 965 - "OMN-21538.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 966 - "OMN-21551.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 967 - "OMN-21590.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+22 more)

### Community 968 - "OMN-21604.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 969 - "OMN-21611.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 970 - "OMN-21622.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 971 - "OMN-21645.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 972 - "OMN-21689.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 973 - "OMN-21708.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 974 - "OMN-21722.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 975 - "OMN-21731.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 976 - "OMN-21799.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 977 - "OMN-21864.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 978 - "OMN-21882.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 979 - "OMN-21891.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 980 - "OMN-21906.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 981 - "OMN-21909.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 982 - "OMN-21924.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 983 - "OMN-21928.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 984 - "OMN-21933.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 985 - "OMN-21935.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 986 - "OMN-21939.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 987 - "OMN-21950.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 988 - "OMN-21968.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 989 - "OMN-21970.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 990 - "OMN-21996.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 991 - "OMN-22006.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 992 - "OMN-22052.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 993 - "OMN-22103.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 994 - "OMN-22111.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 995 - "OMN-22123.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 996 - "OMN-22130.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 997 - "OMN-22151.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 998 - "OMN-22153.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 999 - "OMN-22162.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1000 - "OMN-22166.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1001 - "OMN-22172.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1002 - "OMN-22177.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1003 - "OMN-22216.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1004 - "OMN-22258.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1005 - "OMN-22275.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1006 - "OMN-22291.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1007 - "OMN-22294.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1008 - "OMN-22302.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1009 - "OMN-22320.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1010 - "OMN-22356.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1011 - "OMN-22358.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1012 - "OMN-22359.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1013 - "OMN-22362.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1014 - "OMN-22364.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1015 - "OMN-22376.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1016 - "OMN-22377.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1017 - "OMN-22386.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1018 - "OMN-22393.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1019 - "OMN-22430.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1020 - "OMN-22547.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1021 - "OMN-22552.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1022 - "OMN-22574.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1023 - "OMN-22580.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1024 - "OMN-22600.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1025 - "OMN-22604.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1026 - "OMN-22607.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1027 - "OMN-22617.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1028 - "OMN-22638.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1029 - "OMN-22648.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1030 - "OMN-22664.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1031 - "OMN-22665.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1032 - "OMN-22675.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1033 - "OMN-22683.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1034 - "OMN-22685.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1035 - "OMN-22704.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1036 - "OMN-22712.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1037 - "OMN-22714.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1038 - "OMN-22715.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1039 - "OMN-22716.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1040 - "OMN-22812.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1041 - "OMN-22844.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1042 - "OMN-22845.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1043 - "OMN-22863.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1044 - "OMN-22867.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1045 - "OMN-22877.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1046 - "OMN-22896.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1047 - "OMN-22930.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1048 - "OMN-22946.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1049 - "OMN-23002.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1050 - "OMN-23024.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1051 - "OMN-23029.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1052 - "OMN-23050.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1053 - "OMN-23058.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1054 - "OMN-23063.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1055 - "OMN-23075.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1056 - "OMN-23156.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1057 - "OMN-23163.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1058 - "OMN-23164.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1059 - "OMN-23169.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1060 - "OMN-23176.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1061 - "OMN-23177.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1062 - "OMN-23235.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1063 - "OMN-23241.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1064 - "OMN-23324.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1065 - "OMN-23341.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1066 - "OMN-23356.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1067 - "OMN-23359.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1068 - "OMN-23362.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1069 - "OMN-23387.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1070 - "OMN-23395.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1071 - "OMN-23402.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1072 - "OMN-23410.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1073 - "OMN-23418.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1074 - "OMN-23466.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1075 - "OMN-23468.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1076 - "OMN-23471.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1077 - "OMN-23508.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1078 - "OMN-23518.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1079 - "OMN-23535.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1080 - "OMN-23539.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1081 - "OMN-23544.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1082 - "OMN-23613.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1083 - "OMN-23732.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1084 - "OMN-23750.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1085 - "OMN-23838.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1086 - "OMN-23841.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1087 - "OMN-23843.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1088 - "OMN-23888.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1089 - "OMN-23903.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1090 - "OMN-23907.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1091 - "OMN-23932.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1092 - "OMN-23966.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1093 - "OMN-23967.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1094 - "OMN-23973.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1095 - "OMN-24004.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1096 - "OMN-24037.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1097 - "OMN-24063.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1098 - "OMN-24118.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1099 - "OMN-24179.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1100 - "OMN-24206.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1101 - "OMN-24214.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1102 - "OMN-24215.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1103 - "OMN-24216.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1104 - "OMN-24341.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1105 - "OMN-24397.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1106 - "OMN-24426.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1107 - "OMN-24427.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1108 - "OMN-24428.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1109 - "OMN-24429.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1110 - "OMN-24447.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1111 - "OMN-24474.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1112 - "OMN-24476.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1113 - "OMN-24491.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1114 - "OMN-24508.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1115 - "OMN-24509.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1116 - "OMN-24544.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1117 - "OMN-24674.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1118 - "OMN-24695.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1119 - "OMN-24716.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1120 - "OMN-24780.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1121 - "OMN-9050.yaml"
Cohesion: 0.06
Nodes (30): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+22 more)

### Community 1122 - "cloud-migration.yaml"
Cohesion: 0.06
Nodes (30): command, container_name, depends_on, condition, service, description, hardcoded_env, DB_HOST (+22 more)

### Community 1123 - "retry-worker.yaml"
Cohesion: 0.06
Nodes (30): command, container_name, depends_on, condition, service, description, hardcoded_env, OMNIBASE_INFRA_RETRY_WORKER_POSTGRES_DSN (+22 more)

### Community 1124 - "services"
Cohesion: 0.06
Nodes (31): container_name, ports, container_name, container_name, ports, profiles, container_name, profiles (+23 more)

### Community 1125 - "redpanda"
Cohesion: 0.07
Nodes (31): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, hard, soft, build (+23 more)

### Community 1126 - "options"
Cohesion: 0.08
Nodes (31): container_name, entrypoint, image, logging, networks, profiles, restart, volumes (+23 more)

### Community 1127 - "omninode-pypi-cache"
Cohesion: 0.06
Nodes (30): context, DEVPI_HOST, DEVPI_PORT, DEVPI_SERVERDIR, interval, retries, start_period, test (+22 more)

### Community 1128 - "labels"
Cohesion: 0.09
Nodes (30): context, dockerfile, labels, target, fresh-postgres, legacy-postgres, POSTGRES_HOST_AUTH_METHOD, build (+22 more)

### Community 1129 - "with"
Cohesion: 0.06
Nodes (31): with, aws-region, build-args, commit-sha, context, driver, ecr-registry, ecr-repository (+23 more)

### Community 1130 - "steps"
Cohesion: 0.06
Nodes (30): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, AWS_REGION (+22 more)

### Community 1131 - "steps"
Cohesion: 0.06
Nodes (30): concurrency, cancel-in-progress, group, EVENT_NAME, PYPI_PIN_RESOLVE_TIMEOUT_SECONDS, jobs, pin-resolvability, name (+22 more)

### Community 1132 - "ModelLlmEndpointEntry"
Cohesion: 0.09
Nodes (3): EnumLlmEndpointStatus, ModelLlmEndpointContract, ModelLlmEndpointEntry

### Community 1133 - "apply_governed_compose_config.py"
Cohesion: 0.14
Nodes (15): ApplyUsageError, build_parser(), canonical_digest(), checkout_refusals(), cmd_apply(), cmd_digest(), compose_argv(), fetch_anchor_from_main() (+7 more)

### Community 1134 - "prepush_smart_tests.sh"
Cohesion: 0.16
Nodes (29): assert_no_integration_path_runs_locally(), consume_override_grant(), die(), dispatch_to_lab_host(), filter_prepush_runnable_paths(), guard_full_suite_host(), host_is_fit(), host_load_ratio() (+21 more)

### Community 1135 - "ledger_watermark.py"
Cohesion: 0.17
Nodes (18): advance(), _archive_entries(), archive_files(), build_parser(), find_in_archive(), _load_section_parser(), main(), migrate() (+10 more)

### Community 1136 - "ONEXToMCPAdapter"
Cohesion: 0.09
Nodes (3): MCPToolDefinition, MCPToolParameter, ONEXToMCPAdapter

### Community 1137 - "HandlerBuildLoopAppend"
Cohesion: 0.09
Nodes (4): ModelPayloadBuildLoopAppend, bind_handler_db(), HandlerBuildLoopAppend, ModelBuildLoopAppendResult

### Community 1138 - "node_rsd_score_compute/contract.yaml"
Cohesion: 0.06
Nodes (30): contract_version, major, minor, patch, description, module, name, handler_routing (+22 more)

### Community 1139 - "OMN-14936.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1140 - "OMN-17101.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1141 - "OMN-20073.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, cwd, dod_evidence, checks, description (+21 more)

### Community 1142 - "OMN-20085.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1143 - "OMN-20134.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1144 - "OMN-20589.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, cwd, dod_evidence, checks, description (+21 more)

### Community 1145 - "OMN-20769.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1146 - "OMN-20967.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1147 - "OMN-21004.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1148 - "OMN-21009.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1149 - "OMN-21028.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1150 - "OMN-21093.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1151 - "OMN-21228.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1152 - "OMN-21335.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1153 - "OMN-21337.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+21 more)

### Community 1154 - "OMN-21666.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1155 - "OMN-21932.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1156 - "OMN-21947.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1157 - "OMN-22144.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1158 - "OMN-22152.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1159 - "OMN-22183.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1160 - "OMN-22536.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1161 - "OMN-22545.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1162 - "OMN-22713.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1163 - "OMN-22821.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1164 - "OMN-22858.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1165 - "OMN-22899.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1166 - "OMN-23003.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1167 - "OMN-23152.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1168 - "OMN-23705.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1169 - "OMN-23996.yaml"
Cohesion: 0.07
Nodes (29): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+21 more)

### Community 1170 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1171 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1172 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1173 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1174 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1175 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1176 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1177 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1178 - "object_type"
Cohesion: 0.12
Nodes (30): bypass_rls, grants, login, principals, object_type, objects, privileges, schema (+22 more)

### Community 1179 - "intelligence-migration.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, MIGRATIONS_DIR (+21 more)

### Community 1180 - "omnimarket-projection-baselines.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1181 - "omnimarket-projection-cost-by-repo.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1182 - "omnimarket-projection-live-events.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1183 - "omnimarket-projection-registration.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1184 - "omnimarket-projection-savings.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1185 - "omnimarket-projection-session-outcome.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, condition, service, description, hardcoded_env, KAFKA_BROKERS (+21 more)

### Community 1186 - "hardcoded_env"
Cohesion: 0.07
Nodes (30): hardcoded_env, BUS_ID, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KAFKA_INSTANCE_ID, KAFKA_MAX_POLL_INTERVAL_MS, KAFKA_SASL_MECHANISM, KAFKA_SASL_PASSWORD (+22 more)

### Community 1187 - "redpanda.yaml"
Cohesion: 0.07
Nodes (29): command, container_name, depends_on, description, hardcoded_env, healthcheck, interval_s, retries (+21 more)

### Community 1188 - "redpanda"
Cohesion: 0.07
Nodes (29): name, interval, retries, start_period, test, timeout, name, networks (+21 more)

### Community 1189 - "runtime-effects"
Cohesion: 0.08
Nodes (30): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, resources, ONEX_TRAJECTORY_EVALUATION_BACKEND, ONEX_WIRING_STRICT_MODE (+22 more)

### Community 1190 - "x-prepr-slot-db-env"
Cohesion: 0.07
Nodes (29): networks, omnibase-infra-network, external, name, x-lane-overlay-mount, read_only, source, target (+21 more)

### Community 1191 - "with"
Cohesion: 0.07
Nodes (29): name, permissions, runs-on, steps, timeout-minutes, GH_TOKEN, jobs, audit (+21 more)

### Community 1192 - "with"
Cohesion: 0.07
Nodes (29): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, jobs (+21 more)

### Community 1193 - "playwright-launch"
Cohesion: 0.07
Nodes (29): concurrency, cancel-in-progress, group, runner_label, jobs, playwright-launch, name, on (+21 more)

### Community 1194 - "env"
Cohesion: 0.07
Nodes (30): APP_ID, APP_PRIVATE_KEY, CI_RUNNER_ENVIRONMENT, CONVERGE_EVIDENCE, CONVERGE_GENERATION, CONVERGE_OUTCOME, CONVERGE_VERDICT, CORRELATION_ID (+22 more)

### Community 1195 - "delegate_request_payload.py"
Cohesion: 0.08
Nodes (6): build_delegate_provenance(), build_delegate_request_payload(), load_probe_ladder(), _merge_metadata(), traffic_options(), ModelDelegateSessionModel

### Community 1196 - ".discover_handlers"
Cohesion: 0.07
Nodes (6): ModelBootstrapHandlerDescriptor, BootstrapEffectDefinition, _ensure_model_rebuilt(), _get_active_bootstrap_definitions(), HandlerBootstrapSource, _is_mcp_enabled()

### Community 1197 - "ModelInteractiveStep"
Cohesion: 0.11
Nodes (4): AdapterCliInput, AdapterFakeInput, ModelInteractiveStep, ProtocolInputAdapter

### Community 1198 - "ModelDockerDatabaseConsumerCatalog"
Cohesion: 0.07
Nodes (3): ModelDeferredDockerDatabaseConsumer, ModelDockerDatabaseConsumerCatalog, ModelDockerDatabaseConsumer

### Community 1199 - "deploy_lane_verify_route.py"
Cohesion: 0.13
Nodes (14): _append(), job_env(), job_outputs(), load_table(), ModelVerifyRoute, ModelVerifyTargets, requester_repository(), resolve() (+6 more)

### Community 1200 - "provider_rung_canary_probe.py"
Cohesion: 0.15
Nodes (15): Check, checked_payload(), classify(), _docker(), expected_disposition(), grade(), _host_matches(), main() (+7 more)

### Community 1201 - "runner_saturation_record.py"
Cohesion: 0.10
Nodes (12): Alert, build_record(), _consecutive(), evaluate_alerts(), is_sustained(), extract_route_reason(), load_alert_policy(), main() (+4 more)

### Community 1202 - "provision-keycloak.py"
Cohesion: 0.13
Nodes (14): _build_parser(), ensure_keycloak_db(), get_admin_token(), log(), main(), patch_master_realm_ssl_none(), provision_onex_admin_client(), provision_onex_service_client() (+6 more)

### Community 1203 - "prepr_verify_rendered_slot.py"
Cohesion: 0.11
Nodes (13): assert_target_is_a_pool_slot(), build_parser(), _emit(), main(), RefusalError, resolve_slot(), slot_environment(), SlotPolicy (+5 more)

### Community 1204 - "HandlerRunnerHookMountDrift"
Cohesion: 0.09
Nodes (6): DockerCli, HandlerRunnerHookMountDrift, ModelRunnerHookMountDriftRequest, ModelRunnerHookMountDriftResult, ModelRunnerHookMountFinding, _handler_with_docker()

### Community 1205 - "ModelUpsertResult"
Cohesion: 0.08
Nodes (3): ModelUpdateRegistrationRequest, ModelUpsertResult, ProtocolRegistrationPersistence

### Community 1206 - "ModelArchitectureViolation"
Cohesion: 0.08
Nodes (4): ModelArchitectureViolation, _evaluate_monolith_signals(), MonolithNodeAnalysis, RuleOrchestrationMonolith

### Community 1207 - "node_checkpoint_validate_compute/contract.yaml"
Cohesion: 0.07
Nodes (29): capabilities, contract_version, major, minor, patch, description, module, name (+21 more)

### Community 1208 - "HandlerLlmInferenceCommand"
Cohesion: 0.09
Nodes (7): _flatten_metric_extensions(), HandlerLlmInferenceCommand, LlmInferenceCommandTransport, _optional_str(), ModelLlmCallCompletedEvent, ModelLlmCallCompletedInfraEvent, ModelLlmInferenceCommand

### Community 1209 - "node_model_health_effect/contract.yaml"
Cohesion: 0.07
Nodes (29): contract_version, major, minor, patch, description, handler_type, module, name (+21 more)

### Community 1210 - "node_registry_effect/contract.yaml"
Cohesion: 0.07
Nodes (29): capabilities, description, name, contract_name, contract_version, major, minor, patch (+21 more)

### Community 1211 - "HandlerRunnerFleetSnapshot"
Cohesion: 0.09
Nodes (5): HandlerRunnerFleetSnapshot, _optional_count(), parse_run_jobs(), ModelRunnerJobFact, ModelZombieRunCandidate

### Community 1212 - "provenance"
Cohesion: 0.07
Nodes (28): delta, measured_at, measured_by, node_ids, reason, baseline_collected, job, max_skips (+20 more)

### Community 1213 - "omnibase_infra"
Cohesion: 0.07
Nodes (29): omnibase_infra, reason, status, target, omni-infra, onex-action-authorization-claim, onex-gateway-canary-probe, onex-git-hook-relay (+21 more)

### Community 1214 - "OMN-20439.yaml"
Cohesion: 0.07
Nodes (28): id, statement, check_type, check_value, dod_evidence, checks, description, id (+20 more)

### Community 1215 - "performance_benchmarks"
Cohesion: 0.10
Nodes (29): contract_scanning, contract_validation, event_bus_wiring, node_instantiation, description, metrics, scaling, description (+21 more)

### Community 1216 - "omninode-system-slack-report.sh"
Cohesion: 0.14
Nodes (27): _census_owner_home(), check_census_drift(), check_ci_required_contexts(), check_dev_broker(), check_fleet_failures(), check_gh_read_identity(), check_http(), check_postgres_backup_freshness() (+19 more)

### Community 1217 - "labels"
Cohesion: 0.08
Nodes (28): context, dockerfile, labels, target, postgres, ADMIN_DSN, POSTGRES_DB, POSTGRES_PASSWORD (+20 more)

### Community 1218 - "environment"
Cohesion: 0.07
Nodes (28): CODING_AGENT_CRED_HOME, INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_ENVIRONMENT_SLUG, INFISICAL_PROJECT_ID, OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS, OMNIMARKET_PROJECTION_READ_BINDING_OVERLAY (+20 more)

### Community 1219 - "services"
Cohesion: 0.07
Nodes (28): command, container_name, image, networks, profiles, read_only, restart, tmpfs (+20 more)

### Community 1220 - "preflight_required_compose_env.py"
Cohesion: 0.10
Nodes (12): main(), PasswordFormat, render_env_example_block(), render_refusal(), violations(), _broker_memory_bytes(), _check_broker_memory(), _check_dev_broker_stack() (+4 more)

### Community 1221 - "name"
Cohesion: 0.10
Nodes (29): name, runs-on, steps, timeout-minutes, name, runs-on, steps, timeout-minutes (+21 more)

### Community 1222 - "HandlerEventProvenanceStamp"
Cohesion: 0.08
Nodes (9): _clean(), HandlerEventProvenanceStamp, provenance_from_overlay(), read_provenance_overlay(), ModelEventProvenanceOverlay, ModelEventProvenanceStampRequest, ModelEventProvenanceStampResult, dump_envelope_json() (+1 more)

### Community 1227 - "c12_provider_catalogue_probe.py"
Cohesion: 0.17
Nodes (16): Check, checked_payload(), _docker(), _get(), grade(), add(), _has_finding(), main() (+8 more)

### Community 1228 - "c28_consumer_flow_probe.py"
Cohesion: 0.14
Nodes (18): grade(), grade_boot(), grade_cursor(), grade_kinds(), grade_negative(), _is_int(), kind_rows(), main() (+10 more)

### Community 1229 - "merge_queue_enqueue.py"
Cohesion: 0.11
Nodes (13): arm_args(), classify_pr(), EnumEnqueueAction, is_armed(), is_behind(), is_benign_enqueue_error(), is_in_queue(), is_no_merge_queue_error() (+5 more)

### Community 1230 - "validate_runner_image.py"
Cohesion: 0.13
Nodes (12): check_image_identity(), check_receipt_gate_readiness(), check_zero_uv_sync(), CheckResult, _env_ready_marker_present(), _load_identity_module(), _load_lock(), main() (+4 more)

### Community 1231 - "queue-contract-validator.py"
Cohesion: 0.14
Nodes (15): _canonical_remote_urls(), _construct_mapping(), _exact_string(), _expected_push_argv(), _fail(), _is_canonical_remote_url(), _keys(), _load_contract() (+7 more)

### Community 1232 - "_Teardown"
Cohesion: 0.20
Nodes (4): _chunks(), parse_rpk_column(), select_slot_names(), _Teardown

### Community 1233 - "ServicePolicyEngine"
Cohesion: 0.09
Nodes (3): EnumPolicyDecision, PolicyDecision, ServicePolicyEngine

### Community 1234 - "ProtocolBrokerPrincipalAdmin"
Cohesion: 0.08
Nodes (6): admin_api_from_env(), environment_value(), HandlerBrokerPrincipalList, ModelBrokerPrincipalListRequest, ModelBrokerPrincipalListResult, ProtocolBrokerPrincipalAdmin

### Community 1235 - "ModelStorageResult"
Cohesion: 0.08
Nodes (3): ProtocolRegistrationPersistence, ModelStorageQuery, ModelStorageResult

### Community 1236 - "ModelTopicParser"
Cohesion: 0.08
Nodes (3): ModelParsedTopic, ModelTopicParser, _parse_topic_cached()

### Community 1237 - "scanner_imperative_orchestrator_ratchet.py"
Cohesion: 0.12
Nodes (12): _baseline_key(), BaselineEntry, load_baseline(), main(), ratchet_violations(), _relative_handler_path(), render_baseline_yaml(), _resolve_repo_root() (+4 more)

### Community 1238 - "ProtocolChainVectorClient"
Cohesion: 0.08
Nodes (4): ProtocolChainEmbeddingClient, ProtocolChainVectorClient, HandlerChainRetrieval, HandlerChainStore

### Community 1239 - "performance_benchmarks"
Cohesion: 0.10
Nodes (29): contract_scanning, contract_validation, event_bus_wiring, node_instantiation, description, metrics, scaling, description (+21 more)

### Community 1240 - "HookObservability"
Cohesion: 0.11
Nodes (3): configure_global_hook(), get_global_hook(), HookObservability

### Community 1243 - "ModelApplicationDatabaseAclPolicy"
Cohesion: 0.07
Nodes (4): EnumApplicationDatabaseAclPolicySourceKind, ModelApplicationDatabaseAclPolicy, ModelApplicationDatabaseConnectionPolicy, ModelApplicationDatabaseRoleState

### Community 1244 - "OMN-21485.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1245 - "OMN-21683.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1246 - "OMN-21688.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1247 - "OMN-21916.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1248 - "OMN-22054.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1249 - "OMN-22158.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+19 more)

### Community 1250 - "OMN-22391.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1251 - "OMN-24191.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1252 - "OMN-24290.yaml"
Cohesion: 0.07
Nodes (27): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+19 more)

### Community 1253 - "omnibase-infra-memgraph.yaml"
Cohesion: 0.07
Nodes (27): command, container_name, depends_on, description, hardcoded_env, healthcheck, interval_s, retries (+19 more)

### Community 1254 - "environment"
Cohesion: 0.08
Nodes (28): BIFROST_LANE_OVERLAY_PATH, INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_ENVIRONMENT_SLUG, INFISICAL_PROJECT_ID, OMNIMARKET_PROJECTION_READ_BINDING_OVERLAY, ONEX_RUNTIME_LANE (+20 more)

### Community 1255 - "environment"
Cohesion: 0.49
Nodes (28): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, ONEX_LAB_RUN_HOSTS, ONEX_LANE_HOST, RUNNER_GROUP (+20 more)

### Community 1256 - "runner-monitor.sh"
Cohesion: 0.15
Nodes (27): ALERT_ONLY_RUNNER_FAMILY, auto_bounce(), bounce_timeout_seconds(), collect_remediation_targets(), compose_preflight(), config_field(), config_host_prefixes(), container_is_crashlooping() (+19 more)

### Community 1257 - "socket"
Cohesion: 0.15
Nodes (17): _first_site_packages(), _holder_is_live(), _load_verify_movement(), _parse_record(), promote_floor(), _record(), _repo_root(), run_promotion_command() (+9 more)

### Community 1258 - "steps"
Cohesion: 0.07
Nodes (27): description, ACTION_PATH, PACKAGE, REPO_ROOT, inputs, package, repo-root, name (+19 more)

### Community 1259 - "with"
Cohesion: 0.07
Nodes (27): name, permissions, runs-on, steps, timeout-minutes, env, CACHE_VERSION, PYTHON_VERSION (+19 more)

### Community 1260 - "steps"
Cohesion: 0.07
Nodes (27): concurrency, cancel-in-progress, group, name, permissions, runs-on, steps, timeout-minutes (+19 more)

### Community 1261 - "steps"
Cohesion: 0.07
Nodes (27): default, description, required, REPOS_CSV, RUNNER_GITHUB_TOKEN, dry_run, jobs, wedge-rerun (+19 more)

### Community 1262 - "steps"
Cohesion: 0.07
Nodes (27): concurrency, cancel-in-progress, group, BRANCH, LINEAR_API_KEY, PR_NUMBER, REPO, TICKET_ID (+19 more)

### Community 1263 - "validate_application_migration_manifest.py"
Cohesion: 0.13
Nodes (17): BlockedMigration, CloudAlias, _content_sha256(), LegacyNodeMigrationDeclaration, main(), ManifestError, MigrationDeclaration, _node_identity() (+9 more)

### Community 1264 - "ModelPayloadHttp"
Cohesion: 0.08
Nodes (5): EnumHttpOperationType, ModelHttpGetPayload, ModelHttpPostPayload, ModelPayloadHttp, RegistryPayloadHttp

### Community 1265 - "handler_dev_lane_branch_install.py"
Cohesion: 0.08
Nodes (4): _build_parser(), report(), report_one(), _short()

### Community 1266 - "check_dockerfile_pins.py"
Cohesion: 0.12
Nodes (13): _check_no_exact_pin_for_plugins(), _check_no_exact_pin_without_no_deps(), _check_npm_exact_pin(), _check_range_covers_latest(), _iter_logical_lines(), _latest_pypi_version(), main(), NpmPinEntry (+5 more)

### Community 1267 - "lockfile_cve_autobump.py"
Cohesion: 0.16
Nodes (14): _body(), compute_bump_plan(), _ensure_labels(), _execute(), is_base_inherited_run(), main(), ModelAutobumpConfig, ModelBumpItem (+6 more)

### Community 1268 - "skip_count_ratchet.py"
Cohesion: 0.12
Nodes (10): BaselineEntry, _build_parser(), evaluate(), InputError, main(), node_id(), Observation, observe() (+2 more)

### Community 1269 - "prove_migration_revision_equivalence.py"
Cohesion: 0.13
Nodes (17): build_parser(), executable_text(), find_recorded_revision(), _git(), _git_bytes(), load_canonical_adoptions(), main(), Proof (+9 more)

### Community 1270 - "onex-setup.py"
Cohesion: 0.11
Nodes (11): _build_command_envelope_json(), _build_parser(), _build_setup_command(), main(), _omnibase_dir(), _print_topology_summary(), _prompt_preset(), _publish_setup_command() (+3 more)

### Community 1271 - "repoint_dev_lane_onex_api.py"
Cohesion: 0.13
Nodes (12): build_parser(), choose(), inspect_resident(), list_candidates(), main(), ModelCandidate, read_pin(), RepointRefusalError (+4 more)

### Community 1272 - "resolve_workspace_pins.py"
Cohesion: 0.11
Nodes (12): assert_pins_satisfied(), build_comparisons(), compare_pin(), LockPin, parse_lock_pins(), PinComparison, read_repo_head_sha(), read_repo_version() (+4 more)

### Community 1273 - "validate_test_root_collection.py"
Cohesion: 0.13
Nodes (13): check_collocated_selector_coverage(), check_full_suite_invocation(), collected_roots(), find_test_dirs(), find_violations(), _full_suite_run_block(), _is_collected(), _load_workflow() (+5 more)

### Community 1274 - "TransportHolderLlmHttp"
Cohesion: 0.08
Nodes (3): AdapterCodeAnalysisEnrichment, AdapterDocumentationGeneration, TransportHolderLlmHttp

### Community 1275 - ".graphql_transport"
Cohesion: 0.09
Nodes (6): _build_ticket_description(), _build_ticket_title(), _fingerprint_to_lock_key(), HandlerLinearDbErrorReporter, ModelDbErrorEvent, ModelDbErrorTicketResult

### Community 1276 - "render_settings"
Cohesion: 0.10
Nodes (4): render_settings(), EnumMachineRole, ModelMachineEntry, ModelMachineRegistry

### Community 1277 - "DelegateLaneSelectionError"
Cohesion: 0.12
Nodes (11): declared_lane_ids(), DelegateLaneSelectionError, _incomplete_selection_message(), resolve_lane_declaration_path(), resolve_lane_selection(), resolve_lane_selection_at(), resolve_lane_target(), ModelDelegateLaneSelection (+3 more)

### Community 1278 - "DevBootFailedError"
Cohesion: 0.13
Nodes (14): causes_from_logs(), _checked(), _dev_containers(), DevBootFailedError, DevOnexApiStepFailedError, _folded_profiles(), _json_object(), preflight_dev_images() (+6 more)

### Community 1279 - "node_baseline_capture/contract.yaml"
Cohesion: 0.07
Nodes (27): contract_version, major, minor, patch, description, event_bus, publish_topics, module (+19 more)

### Community 1280 - "offer_companion_sources.py"
Cohesion: 0.13
Nodes (14): _candidate(), _default_ref(), find_product_contract(), _gateway_args(), _gateway_result(), list_contract_tickets(), merge_candidates(), read_clone_candidates() (+6 more)

### Community 1281 - "HandlerReleaseIdentity"
Cohesion: 0.09
Nodes (3): HandlerReleaseIdentity, ModelReleaseIdentityDecision, ModelReleaseIdentityRequest

### Community 1285 - "EffectMockRegistry"
Cohesion: 0.08
Nodes (4): EffectMockRegistry, clear_thread_local_registry(), get_thread_local_registry(), scoped_effect_mock_registry()

### Community 1286 - "ValidatorSecurity"
Cohesion: 0.12
Nodes (3): CompiledPatterns, MethodContext, ValidatorSecurity

### Community 1287 - "OMN-20320.yaml"
Cohesion: 0.07
Nodes (26): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+18 more)

### Community 1288 - "OMN-20388.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1289 - "OMN-21115.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1290 - "OMN-21324.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1291 - "OMN-21733.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1292 - "OMN-21757.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1293 - "OMN-21863.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1294 - "OMN-22027.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1295 - "OMN-22119.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1296 - "OMN-22132.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1297 - "OMN-22233.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1298 - "OMN-22311.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1299 - "OMN-22360.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1300 - "OMN-22392.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1301 - "OMN-22692.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1302 - "OMN-22958.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1303 - "OMN-23011.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1304 - "OMN-23067.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, cwd, dod_evidence, binds_ac, checks (+18 more)

### Community 1305 - "OMN-23368.yaml"
Cohesion: 0.07
Nodes (26): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+18 more)

### Community 1306 - "omnimemory-kreuzberg.yaml"
Cohesion: 0.07
Nodes (26): container_name, depends_on, description, extra_networks, hardcoded_env, healthcheck, interval_s, retries (+18 more)

### Community 1307 - "phoenix.yaml"
Cohesion: 0.07
Nodes (26): container_name, depends_on, description, hardcoded_env, PHOENIX_WORKING_DIR, healthcheck, interval_s, retries (+18 more)

### Community 1308 - "environment"
Cohesion: 0.21
Nodes (27): environment, environment, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKERS, KAFKA_CONSUMER_GROUP, KAFKA_SASL_MECHANISM, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME (+19 more)

### Community 1309 - "runtime-effects"
Cohesion: 0.09
Nodes (27): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, resources, cpus, memory (+19 more)

### Community 1310 - "services"
Cohesion: 0.07
Nodes (26): optional_secret_refs, admin_port_publish, advertised_kafka_addr, kafka_port_publish, name, container_name, image, networks (+18 more)

### Community 1311 - "docker-compose.runners-hp051-ci-mq-runner.yml"
Cohesion: 0.08
Nodes (26): create_host_path, services, x-lab-host-table, bind, read_only, source, target, type (+18 more)

### Community 1312 - "docker-compose.runners-hp053-ci-runner.yml"
Cohesion: 0.08
Nodes (26): create_host_path, services, x-lab-host-table, bind, read_only, source, target, type (+18 more)

### Community 1313 - "labels"
Cohesion: 0.09
Nodes (26): context, dockerfile, labels, target, postgres, ADMIN_DSN, POSTGRES_DB, POSTGRES_PASSWORD (+18 more)

### Community 1314 - "fcntl"
Cohesion: 0.13
Nodes (11): acquire_flock(), check_env_file(), check_infisical_addr(), check_throttle(), check_uv_available(), filter_bootstrap_keys(), main(), _parse_env_file() (+3 more)

### Community 1315 - "resolve_lab_lane.py"
Cohesion: 0.13
Nodes (13): _busy_agents(), _check_url(), choose_lane(), deploy_agent_busy_reason(), _env_value(), http_health(), main(), _open() (+5 more)

### Community 1316 - "with"
Cohesion: 0.07
Nodes (27): with, aws-region, build-args, context, driver, exit-code, fetch-depth, file (+19 more)

### Community 1317 - "steps"
Cohesion: 0.07
Nodes (26): concurrency, cancel-in-progress, group, runs-on, steps, timeout-minutes, jobs, contract-topic-graph (+18 more)

### Community 1318 - "steps"
Cohesion: 0.07
Nodes (26): concurrency, cancel-in-progress, group, name, runs-on, steps, BASE_REF, jobs (+18 more)

### Community 1319 - "direct-model-call.yml"
Cohesion: 0.07
Nodes (26): concurrency, cancel-in-progress, group, name, runs-on, steps, env, BASE_REF (+18 more)

### Community 1320 - "env"
Cohesion: 0.07
Nodes (27): ACCEPT_SRC, BASE_REF, CI_OVERLAY_CONFIGURED, CI_OVERLAY_REPO, DISPOSITION_RECORD_PATH, GITHUB_TOKEN, HEAD_SHA, HOSTILE_REVIEW_IS_FORK (+19 more)

### Community 1321 - "steps"
Cohesion: 0.07
Nodes (26): default, description, required, type, details_url, receipt_gzip_b64, jobs, post (+18 more)

### Community 1322 - "runner-disk-preflight"
Cohesion: 0.07
Nodes (26): concurrency, cancel-in-progress, group, OMNI_RUNNER_ENVIRONMENT, RUNNER_DISK_WARN_GB, warn_gb, jobs, runner-disk-preflight (+18 more)

### Community 1323 - "DevLaneBranchInstallBlobs"
Cohesion: 0.11
Nodes (9): assert_dev_project(), DevLaneBranchInstallBlobs, DevLaneBranchInstallRefusedError, git_merged_probe(), probe(), _merge_commit(), _pairs(), _probe() (+1 more)

### Community 1324 - "cli_identity.py"
Cohesion: 0.09
Nodes (8): assert_target_command(), declaration_from_build_provenance(), identity_group(), _load_stamp(), stamp_command(), collect_runtime_identity(), render_identity_line(), _short()

### Community 1325 - "ProtocolCutoverJournalRepository"
Cohesion: 0.08
Nodes (3): ModelCutoverFamilyState, ModelCutoverJournalEvent, ProtocolCutoverJournalRepository

### Community 1326 - "HandlerOnexApiImageBuild"
Cohesion: 0.11
Nodes (5): build_image_build_handler(), build_onex_api_image(), HandlerOnexApiImageBuild, ModelOnexApiImageBuildRequest, ModelOnexApiImageBuildResult

### Community 1327 - "lane_transport.py"
Cohesion: 0.09
Nodes (8): ledger_readback_env(), _load_lane_dsn_declaration(), load_lane_ledger_readback(), load_lane_projection_readback(), looks_like_a_dsn(), projection_readback_env(), ModelLaneLedgerReadback, ModelLaneProjectionReadback

### Community 1328 - "routing_api/routes.py"
Cohesion: 0.11
Nodes (3): health(), list_models(), _load_registry()

### Community 1329 - "_run_registration_only"
Cohesion: 0.09
Nodes (10): _aggregate_verdict(), build_parser(), _default_contracts_dir(), _find_all_contracts(), _find_registration_contracts(), main(), _make_runtime_config(), _make_runtime_db_query_fn() (+2 more)

### Community 1330 - "disk_gc_plan.py"
Cohesion: 0.12
Nodes (13): build_plan(), _is_build_stamped_tag(), _is_disposable_ci_tag(), _is_generation_bounded_repo(), _is_in_use(), _load_ndjson(), main(), make_github_pr_state_lookup() (+5 more)

### Community 1331 - "delegation_channel.py"
Cohesion: 0.13
Nodes (6): AiokafkaDelegationChannel, _decode_envelope(), _KafkaMessageLike, ProtocolAsyncKafkaConsumer, ProtocolAsyncKafkaProducer, ModelDelegationEnvelope

### Community 1332 - "generate_feature_inventory.py"
Cohesion: 0.14
Nodes (13): _all_node_names_in_repo(), _extract_topics(), find_design_spec(), _find_git_repos(), _fixture_paths_for_repo(), _load_yaml(), main(), NodeEntry (+5 more)

### Community 1333 - "lane_census_plan.py"
Cohesion: 0.13
Nodes (13): _census_model(), default_manifest_path(), default_snapshot_path(), lane_census_dir(), LaneCensusConfigError, load_manifest(), load_registry(), main() (+5 more)

### Community 1334 - "refresh_stability_lane.sh"
Cohesion: 0.11
Nodes (24): compose_up_bounded(), container_status(), _cwt_err(), _cwt_log(), reconcile_container_running_state(), compose_wait_timeout.sh script, ALL_TRACKED_REPOS, COMPOSE_PROJECT (+16 more)

### Community 1335 - "lane_lock.py"
Cohesion: 0.17
Nodes (16): build_parser(), cmd_acquire(), cmd_describe(), cmd_path(), cmd_release(), cmd_status(), describe_holder(), holder_path() (+8 more)

### Community 1336 - "prepr_pool_prove.sh"
Cohesion: 0.15
Nodes (23): composed_build(), derive_live_grep(), derive_live_not_applicable(), drop_source_trees(), fetch_group(), fetch_pr(), fetch_ref(), identity_readback() (+15 more)

### Community 1337 - "refresh_dev_lane.sh"
Cohesion: 0.13
Nodes (25): ALL_TRACKED_REPOS, CLOUD_MIGRATION_FAILED_DEPLOY_EXIT, COMPOSE_PROJECT, compose_ps_q(), compose_ps_q_any(), CORE_SERVICES, err(), git_clone() (+17 more)

### Community 1338 - "stage_pinned_proof_root.py"
Cohesion: 0.17
Nodes (15): _build_parser(), _clone_local(), _git(), _git_env(), _load_manifest(), main(), _parse_assignments(), _read_bash_array() (+7 more)

### Community 1339 - "note"
Cohesion: 0.09
Nodes (27): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, confidence (+19 more)

### Community 1341 - "HandlerCodingAgentInvoke"
Cohesion: 0.12
Nodes (5): EnumCodingAgent, ModelCodingAgentInvokeCommand, ModelCodingAgentResult, build_agent_invocation(), HandlerCodingAgentInvoke

### Community 1342 - "timeouts"
Cohesion: 0.07
Nodes (26): timeouts, boundary_dlq_max_attempts, circuit_breaker_reset_seconds, consumer_restart_backoff_factor, consumer_restart_initial_backoff_seconds, consumer_restart_max_backoff_seconds, consumer_restart_stable_threshold_seconds, database_query_seconds (+18 more)

### Community 1343 - "ModelCheckpointEffectOutput"
Cohesion: 0.08
Nodes (5): _checkpoint_dir(), _checkpoint_filename(), _serialize_checkpoint(), attempt_number(), ModelCheckpointEffectOutput

### Community 1344 - "cascade_supersession.py"
Cohesion: 0.11
Nodes (11): _merged_pr_that_moved_the_pin(), normalise_distribution(), parse_cascade_provenance(), pinned_version_from_lockfile(), resolved(), pinned_version_from_pyproject(), _pull_request_moved_the_pin(), _read_pinned_version() (+3 more)

### Community 1347 - "DecisionProjector"
Cohesion: 0.10
Nodes (5): build_embedding_text(), DecisionEmbedderWriter, EmbeddingClient, ModelDecisionRecord, DecisionProjector

### Community 1348 - "OMN-18202.yaml"
Cohesion: 0.08
Nodes (25): check_type, check_value, dod_evidence, checks, description, evidence_artifact, execution_scope, id (+17 more)

### Community 1349 - "OMN-20557.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1350 - "OMN-21229.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1351 - "OMN-21629.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1352 - "OMN-21642.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1353 - "OMN-21702.yaml"
Cohesion: 0.08
Nodes (25): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, execution_scope (+17 more)

### Community 1354 - "OMN-21714.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1355 - "OMN-21717.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1356 - "OMN-21736.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1357 - "OMN-21758.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1358 - "OMN-21761.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1359 - "OMN-21765.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1360 - "OMN-21784.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1361 - "OMN-21789.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1362 - "OMN-21807.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1363 - "OMN-21853.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1364 - "OMN-21894.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1365 - "OMN-21913.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1366 - "OMN-21981.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1367 - "OMN-22002.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1368 - "OMN-22097.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1369 - "OMN-22128.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1370 - "OMN-23246.yaml"
Cohesion: 0.08
Nodes (25): id, statement, check_type, check_value, dod_evidence, binds_ac, checks, description (+17 more)

### Community 1371 - "valkey.yaml"
Cohesion: 0.08
Nodes (25): command, container_name, depends_on, description, hardcoded_env, healthcheck, interval_s, retries (+17 more)

### Community 1372 - "migration-gate"
Cohesion: 0.08
Nodes (26): forward-migration, intelligence-migration, migration-gate, postgres, redpanda-partition-cap, valkey, condition, depends_on (+18 more)

### Community 1373 - "postgres"
Cohesion: 0.08
Nodes (25): optional_secret_refs, required_secret_refs, admin_port_publish, advertised_kafka_addr, kafka_port_publish, name, container_name, image (+17 more)

### Community 1374 - "docker-compose.runners-hp053-ci-mq-runner.yml"
Cohesion: 0.09
Nodes (25): create_host_path, services, x-lab-host-table, bind, read_only, source, target, type (+17 more)

### Community 1375 - "rebuilt-postgres16-proof"
Cohesion: 0.08
Nodes (25): concurrency, cancel-in-progress, group, COMPOSE_PROJECT_NAME, jobs, rebuilt-postgres16-proof, name, on (+17 more)

### Community 1376 - "integration-test-check.yml"
Cohesion: 0.09
Nodes (25): name, runs-on, steps, timeout-minutes, name, runs-on, steps, timeout-minutes (+17 more)

### Community 1377 - "steps"
Cohesion: 0.08
Nodes (25): concurrency, cancel-in-progress, group, jobs, verify, name, on, pull_request (+17 more)

### Community 1378 - "steps"
Cohesion: 0.08
Nodes (25): CROSS_REPO_PAT, RUNNER_FLEET_STATUS_TOKEN, SPREAD_LABELS, SPREAD_MIN_HOSTS, jobs, runner-pool-spread, name, on (+17 more)

### Community 1379 - "shellcheck-gate.yml"
Cohesion: 0.08
Nodes (25): concurrency, cancel-in-progress, group, env, SHELLCHECK_VERSION, jobs, shell-hygiene, name (+17 more)

### Community 1380 - "HandlerCostApiDependencies"
Cohesion: 0.10
Nodes (3): HandlerCostApiDependencies, ModelCostSnapshotTopics, install_cost_routes()

### Community 1381 - "handler_update_scores.py"
Cohesion: 0.10
Nodes (6): _find_score(), HandlerUpdateScores, _update_rolling_avg(), ModelCapabilityScore, ModelReducerState, ModelRoutingOutcome

### Community 1382 - "HandlerRunnerFleetHealthEvaluate"
Cohesion: 0.09
Nodes (7): HandlerRunnerFleetHealthEvaluate, _recommend_for_assessment(), _zombie_restart_corroboration(), EnumRecommendedActionType, EnumRunnerReadinessState, ModelRecommendedAction, ModelRunnerHealthAssessment

### Community 1383 - "runner_health/__init__.py"
Cohesion: 0.11
Nodes (5): _emit_to_kafka(), main(), _run_network_pass(), _send_slack_alert(), ModelRunnerHealthAlert

### Community 1384 - "InteractiveExecutor"
Cohesion: 0.10
Nodes (4): InteractiveExecutor, InteractiveExecutorError, ModelInteractiveResult, ModelStepResult

### Community 1385 - "omnibase_infra/types/__init__.py"
Cohesion: 0.08
Nodes (3): TypedDictCapabilities, TypedDictIntrospectionCache, TypedDictPerformanceMetricsCache

### Community 1386 - "AdapterModelRouter"
Cohesion: 0.08
Nodes (3): AdapterModelRouter, _provider_to_tier(), ModelRoutingDecidedEvent

### Community 1387 - "ModelLabPassReceipt"
Cohesion: 0.11
Nodes (7): build_pr_head_bus_event(), _missing_mandatory(), ModelLabPassReceipt, pr_head_receipt_key(), pr_head_receipt_key_text(), reemit_receipt(), replace()

### Community 1388 - "check_sibling_lock_pins.py"
Cohesion: 0.13
Nodes (11): check_pins(), _classify_drift(), compare_pins(), _is_clone_strictly_ahead(), main(), ModelActualPin, ModelLockPin, ModelPinComparison (+3 more)

### Community 1390 - "attempt"
Cohesion: 0.10
Nodes (6): attempt(), BrokerPrincipalAdminApi, BrokerPrincipalKafkaSession, kafka_session_from_env(), scrub(), ProtocolBrokerPrincipalTopicLister

### Community 1391 - "ModelPlacementServingReport"
Cohesion: 0.10
Nodes (6): EnumPlacementHealthVerdict, evaluate_serving_health(), read_serving_report(), ModelPlacementServingCount, ModelPlacementServingReport, ProtocolPlacementServingReader

### Community 1393 - "RegistryInfraPlacementRuntime"
Cohesion: 0.09
Nodes (4): ProtocolPlacedNode, ModelPlacementPolicy, ModelPlacementRuntimeSettings, RegistryInfraPlacementRuntime

### Community 1394 - "OperationBindingResolver"
Cohesion: 0.09
Nodes (4): ModelBindingResolutionResult, _is_json_compatible(), _is_json_compatible_recursive(), OperationBindingResolver

### Community 1396 - "ModelPostMergeLabPassReceiptEvent"
Cohesion: 0.08
Nodes (5): EnumPostMergeLabPassErrorClass, ModelPostMergeLabPassCheckOutcome, ModelPostMergeLabPassGeneration, ModelPostMergeLabPassIdentity, ModelPostMergeLabPassReceiptEvent

### Community 1398 - "HandlerBaselineCapture"
Cohesion: 0.10
Nodes (5): HandlerBaselineCapture, ModelBaselineCaptureCommand, ModelBaselineCaptureOutput, ModelBaselinesBreakdownRow, ModelBaselinesSnapshotEvent

### Community 1399 - "HandlerInProgressProbeHygiene"
Cohesion: 0.10
Nodes (7): _comment_body(), HandlerInProgressProbeHygiene, _well_formed_probe_lines(), EnumProbeHygieneDecision, ModelProbeHygieneOutcome, ModelProbeHygieneRequest, ModelProbeHygieneResult

### Community 1400 - "ModelReducerExecutionResult"
Cohesion: 0.10
Nodes (3): ModelReducerExecutionResult, ModelReducerState, ProtocolReducer

### Community 1401 - "properties"
Cohesion: 0.08
Nodes (26): description, ref, type, default, description, nullable, type, properties (+18 more)

### Community 1402 - "node_runtime_orchestrator/contract.yaml"
Cohesion: 0.08
Nodes (25): contract_version, major, minor, patch, description, fingerprint, input_model, module (+17 more)

### Community 1403 - "ModelLifecycleHooks"
Cohesion: 0.12
Nodes (3): ModelLifecycleHooks, _classify_failure(), LifecycleHookExecutor

### Community 1404 - "ServiceNodeIntrospection"
Cohesion: 0.11
Nodes (8): _build_introspection_service(), _attach_event_bus(), _map_node_type(), _parse_contract_dict(), _parse_event_bus_subcontract(), ServiceNodeIntrospection, _try_load_contract(), _try_read_raw_yaml()

### Community 1405 - "validation"
Cohesion: 0.08
Nodes (25): contract_kind, patterns, sensitive_params, description, enabled, parameters, rule_id, severity (+17 more)

### Community 1406 - "OMN-15092.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, checks, description, evidence_artifact, id (+16 more)

### Community 1407 - "OMN-15484.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+16 more)

### Community 1408 - "OMN-15642.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+16 more)

### Community 1409 - "OMN-16343.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+16 more)

### Community 1410 - "OMN-18353.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, checks, description, evidence_artifact, id (+16 more)

### Community 1411 - "OMN-18638.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1412 - "OMN-18656.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1413 - "OMN-19312.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1414 - "OMN-19802.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1415 - "OMN-19925.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1416 - "OMN-20339.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, checks, description, evidence_artifact, id (+16 more)

### Community 1417 - "OMN-22478.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+16 more)

### Community 1418 - "OMN-8468.yaml"
Cohesion: 0.08
Nodes (24): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+16 more)

### Community 1419 - "redpanda-local-sasl.yaml"
Cohesion: 0.08
Nodes (24): command, container_name, depends_on, condition, service, description, entrypoint, hardcoded_env (+16 more)

### Community 1420 - "hardcoded_env"
Cohesion: 0.08
Nodes (25): hardcoded_env, BUS_ID, CODING_AGENT_CRED_HOME, DELEGATION_ROUTING_TIERS_PATH, ENABLE_RUNTIME_LOG_BRIDGE, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KAFKA_MAX_POLL_INTERVAL_MS (+17 more)

### Community 1421 - "omninode-runtime"
Cohesion: 0.12
Nodes (25): create_host_path, container_name, depends_on, labels, ports, profiles, restart, volumes (+17 more)

### Community 1422 - "environment"
Cohesion: 0.22
Nodes (25): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, LANE_PROBE_HOST, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, npm_config_cache, ONEX_LAB_RUN_HOSTS (+17 more)

### Community 1423 - "environment"
Cohesion: 0.38
Nodes (25): ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE, ACTIONS_RUNNER_HOOK_JOB_STARTED, AGENT_TOOLSDIRECTORY, GITHUB_ORG_URL, KUBECONFIG, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, ONEX_LAB_RUN_HOSTS (+17 more)

### Community 1424 - "runtime_lane_env_overlay.py"
Cohesion: 0.13
Nodes (9): declaration(), _document_problems(), lane_env_with_overlay_home(), LoggedLaneEnv, merge_lane_env_sources(), overlay_home_clone(), OverlayHomeLaneEnvError, parse_overlay_home_lane_env() (+1 more)

### Community 1425 - "steps"
Cohesion: 0.08
Nodes (24): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, jobs (+16 more)

### Community 1426 - "steps"
Cohesion: 0.08
Nodes (24): concurrency, cancel-in-progress, group, if, permissions, runs-on, steps, timeout-minutes (+16 more)

### Community 1427 - "steps"
Cohesion: 0.08
Nodes (24): CROSS_REPO_PAT, RUNNER_CANARY_MAX_OFFLINE, RUNNER_FLEET_STATUS_TOKEN, SLACK_BOT_TOKEN, SLACK_CHANNEL_ID, name, runs-on, steps (+16 more)

### Community 1428 - "adapter_project_tracker_linear.py"
Cohesion: 0.12
Nodes (8): AdapterProjectTrackerLinear, _extract_nodes(), _nested_id(), _optional_str(), _required_str(), ModelProjectTrackerIssueStatus, ModelProjectTrackerLabel, ModelProjectTrackerTeam

### Community 1429 - "qdrant/__init__.py"
Cohesion: 0.09
Nodes (4): _get_qdrant_operation_type(), ModelQdrantHandlerPayload, ModelQdrantSearchPayload, ModelQdrantSearchResult

### Community 1430 - "check_dev_candidate_delivery_liveness.py"
Cohesion: 0.14
Nodes (12): assert_supported_patterns(), collect(), evaluate(), Finding, _gh(), main(), matches_filter(), normalize_commit() (+4 more)

### Community 1431 - "JobState"
Cohesion: 0.14
Nodes (12): cancellation_is_provisional(), evaluate_external_contexts(), is_decided(), JobState, latest_check_run_by_name(), _parse_timestamp(), provisional_cancellations(), provisional_external_verdicts() (+4 more)

### Community 1432 - "VerificationResult"
Cohesion: 0.12
Nodes (9): display_summary(), main(), run_verification(), _seed_mock_handlers(), VerificationResult, verify_contract_handler_structure(), verify_contract_routing(), verify_dispatch_routing() (+1 more)

### Community 1433 - "gateway_control_plane.py"
Cohesion: 0.15
Nodes (11): attach(), _auth_headers(), _context(), detach(), heartbeat(), _parse_datetime(), _parse_renewal(), resolve_access_token() (+3 more)

### Community 1434 - "recompute_measured_pricing.py"
Cohesion: 0.15
Nodes (14): _as_mapping(), async_main(), compute_pricing_by_model(), ComputedPricing, _fallback_source(), fetch_db_samples(), load_dotenv(), load_fixture_samples() (+6 more)

### Community 1435 - "release_drift_monitor_lib.py"
Cohesion: 0.17
Nodes (11): _age_days(), _age_hours(), blind_verdicts_for_deploy_target(), DeployFacts, DriftFinding, DriftThresholds, evaluate_all(), evaluate_deploy_target() (+3 more)

### Community 1436 - "run_baselines_batch_compute.py"
Cohesion: 0.13
Nodes (10): _BaselinesBatchCommand, build_command(), build_envelope(), _command_model_module_path(), _load_command_model_class(), _load_command_topic(), main(), _node_contract_dir() (+2 more)

### Community 1437 - "validate_clean_root.py"
Cohesion: 0.13
Nodes (9): _declared_guard_configurations(), generate_report(), _get_gitignored_set(), main(), _matches_pattern(), RootViolation, _suggest_action(), validate_root_directory() (+1 more)

### Community 1438 - "StoreLaneCredential"
Cohesion: 0.11
Nodes (3): ModelLaneCredential, _remediation(), StoreLaneCredential

### Community 1439 - "HandlerBrokerPrincipalVerify"
Cohesion: 0.10
Nodes (5): BrokerPrincipalError, HandlerBrokerPrincipalVerify, ModelBrokerPrincipalVerifyRequest, ModelBrokerPrincipalVerifyResult, ProtocolBrokerPrincipalSession

### Community 1440 - "HandlerDeclaredStateDiff"
Cohesion: 0.12
Nodes (5): HandlerDeclaredStateDiff, EnumDeclaredStateOperationKind, ModelDeclaredStateDiff, ModelDeclaredStateDiffRequest, ModelDeclaredStateOperation

### Community 1441 - "build_shared_listener"
Cohesion: 0.10
Nodes (5): build_shared_listener(), is_consuming(), is_passive(), is_ready(), ModelSharedListenerPolicy

### Community 1444 - "ModelRuntimeFoldPolicy"
Cohesion: 0.09
Nodes (3): ModelRuntimeBootPolicy, ModelRuntimeFoldPolicy, ModelRuntimeProcessPolicy

### Community 1445 - "node_architecture_validator/contract.yaml"
Cohesion: 0.08
Nodes (24): capabilities, description, name, contract_name, contract_version, major, minor, patch (+16 more)

### Community 1447 - "DLQConsumer"
Cohesion: 0.09
Nodes (6): DLQConsumer, DlqGroupBacklogProbe, ModelDlqReplayEngineConfig, ModelGatewayQuarantinedDlqRecord, _build_runtime_handler_dependencies(), _dlq_replay_subscribe_topics()

### Community 1448 - "HandlerMergeProvenanceObserve"
Cohesion: 0.11
Nodes (5): HandlerMergeProvenanceObserve, ModelMergeGroupRunFact, ModelMergeProvenanceObservation, ModelMergeProvenanceObserveRequest, ProtocolMergeGroupRunReader

### Community 1449 - "correlation_id"
Cohesion: 0.08
Nodes (25): default, default_factory, description, nullable, type, default, description, ge (+17 more)

### Community 1450 - "FactoryObservabilitySink"
Cohesion: 0.11
Nodes (3): FactoryObservabilitySink, ModelMetricsSinkConfig, SinkMetricsPrometheus

### Community 1451 - "registry_infra_ledger_write.py"
Cohesion: 0.09
Nodes (10): [0.9.0] - 2026-02-20, Added, Changed, Dependencies, LLM-Driven Code Generation Handlers, OmniMemory Topics, Tests, Topic Catalog (+2 more)

### Community 1452 - "build_core_runtime"
Cohesion: 0.09
Nodes (7): build_core_runtime(), build_delegation_dlq_resolver(), _resolver(), derive_canonical_dlq_topic(), build_routing_map(), _resolve_owning_entry(), assert_single_owner_split()

### Community 1453 - "OMN-10278.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, contract_completeness, dod_evidence, checks, description, id, source (+15 more)

### Community 1454 - "OMN-13419.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, id, source, status (+15 more)

### Community 1455 - "OMN-13445.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, id, source, status (+15 more)

### Community 1456 - "OMN-13554.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, id, source, status (+15 more)

### Community 1457 - "OMN-13713.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, id, source, status (+15 more)

### Community 1458 - "OMN-15310.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1459 - "OMN-15358.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+15 more)

### Community 1460 - "OMN-15741.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+15 more)

### Community 1461 - "OMN-16171.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+15 more)

### Community 1462 - "OMN-16367.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1463 - "OMN-16769.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1464 - "OMN-16788.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+15 more)

### Community 1465 - "OMN-16821.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1466 - "OMN-16926.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1467 - "OMN-16947.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1468 - "OMN-16961.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1469 - "OMN-17079.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1470 - "OMN-17106.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1471 - "OMN-17135.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1472 - "OMN-17137.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1473 - "OMN-17152.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1474 - "OMN-17166.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1475 - "OMN-17221.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1476 - "OMN-17277.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+15 more)

### Community 1477 - "OMN-17296.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1478 - "OMN-17317.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1479 - "OMN-17379.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1480 - "OMN-17395.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1481 - "OMN-17398.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1482 - "OMN-17432.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1483 - "OMN-17443.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1484 - "OMN-17447.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1485 - "OMN-17485.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1486 - "OMN-17648.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1487 - "OMN-17664.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1488 - "OMN-17678.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1489 - "OMN-17792.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1490 - "OMN-17855.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1491 - "OMN-17862.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1492 - "OMN-17886.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1493 - "OMN-17888.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1494 - "OMN-17893.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1495 - "OMN-17949.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1496 - "OMN-17981.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1497 - "OMN-18124.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1498 - "OMN-18134.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1499 - "OMN-18266.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1500 - "OMN-18272.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1501 - "OMN-18296.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1502 - "OMN-18411.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1503 - "OMN-18693.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, dod_evidence, checks, description, id, source, status (+15 more)

### Community 1504 - "OMN-19315.yaml"
Cohesion: 0.08
Nodes (23): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+15 more)

### Community 1505 - "cloud-migration-files.yaml"
Cohesion: 0.08
Nodes (23): command, container_name, depends_on, description, hardcoded_env, healthcheck, image, labels (+15 more)

### Community 1506 - "hardcoded_env"
Cohesion: 0.08
Nodes (24): hardcoded_env, BUS_ID, CODING_AGENT_CRED_HOME, DELEGATION_ROUTING_TIERS_PATH, ENABLE_RUNTIME_LOG_BRIDGE, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KAFKA_MAX_POLL_INTERVAL_MS (+16 more)

### Community 1507 - "environment"
Cohesion: 0.09
Nodes (24): INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_ENVIRONMENT_SLUG, INFISICAL_PROJECT_ID, KAFKA_MAX_POLL_INTERVAL_MS, OMNIMARKET_PROJECTION_READ_BINDING_OVERLAY, ONEX_BOOT_PRIORITY_CONTRACTS (+16 more)

### Community 1508 - "gateway-attach-test-lane"
Cohesion: 0.08
Nodes (23): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, GATEWAY_ATTACH_TENANT_CLIENT_CREDENTIALS_REF, ONEX_TOPIC_ENFORCEMENT_MODE, build (+15 more)

### Community 1509 - "x-runner-base"
Cohesion: 0.09
Nodes (23): create_host_path, services, bind, read_only, source, target, type, x-lab-host-table (+15 more)

### Community 1510 - "environment"
Cohesion: 0.34
Nodes (24): ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE, ACTIONS_RUNNER_HOOK_JOB_STARTED, AGENT_TOOLSDIRECTORY, GITHUB_ORG_URL, KUBECONFIG, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, ONEX_LAB_RUN_HOSTS (+16 more)

### Community 1511 - "healthcheck"
Cohesion: 0.08
Nodes (23): context, dockerfile, labels, postgres, POSTGRES_DB, POSTGRES_PASSWORD, interval, retries (+15 more)

### Community 1512 - "env"
Cohesion: 0.08
Nodes (24): BRANCH, BUDGET_MS, BUS_LANE, CHAIN_CANARY_GATEWAY_API_KEY, CHAIN_CANARY_PROJECTION_DSN, CHAIN_RECEIPT_DIR, DEPLOY_WAIT_SECONDS, EXPECTED_LEDGER_HOPS (+16 more)

### Community 1513 - "steps"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, BASE_REF, name, runs-on, steps, jobs (+15 more)

### Community 1514 - "env"
Cohesion: 0.08
Nodes (24): BASE_BRANCH, CI_RUNNER_ENVIRONMENT, CONVERGE_EVIDENCE, CONVERGE_GENERATION, CONVERGE_OUTCOME, CORRELATION_ID, DEPLOY_AGENT_URL, DEV_LANE_CONTAINER (+16 more)

### Community 1515 - "steps"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, jobs, receipt-honesty, name, on, merge_group (+15 more)

### Community 1517 - "run_k1_k6_delegation_evidence.py"
Cohesion: 0.16
Nodes (10): DbConnection, main(), _object_from_file(), _positive_int(), run(), _strict_plan(), _string(), _validate_finalization_plan() (+2 more)

### Community 1520 - ".get_widget_mapping"
Cohesion: 0.09
Nodes (4): ModelCapabilityWidgetMapping, ModelRegistryHealthResponse, ModelWidgetDefaults, ModelWidgetMapping

### Community 1521 - "check_lockfile_cve.py"
Cohesion: 0.14
Nodes (12): _bucket_from_score(), _classify_severity(), _cmd_evaluate(), _cmd_relevant(), _cvss_v3_base_score(), EnumSeverityBucket, evaluate_osv_results(), _has_upstream_fix() (+4 more)

### Community 1522 - "run_validators_in_process.py"
Cohesion: 0.15
Nodes (11): _escape_annotation(), _escape_property(), _exit_code_from_system_exit(), main(), parse_specs(), run_all(), run_one(), SpecError (+3 more)

### Community 1523 - "pr_body.py"
Cohesion: 0.14
Nodes (11): assert_identity(), build_parser(), compose_append(), fetch_pull(), GhError, IdentityMismatchError, _log(), main() (+3 more)

### Community 1524 - "EnumReplayStatus"
Cohesion: 0.11
Nodes (5): EnumReplayStatus, ModelDlqReplayRecord, _coerce_uuid(), _extract_envelope_correlation_id(), _extract_envelope_id()

### Community 1525 - "_run_errors"
Cohesion: 0.12
Nodes (8): HandlerRuntimeStartFirst, _ready_status(), _remove(), _run_errors(), _running(), _start(), ModelRuntimeStartFirstRequest, ModelRuntimeStartFirstResult

### Community 1528 - "node_node_graph_reducer/contract.yaml"
Cohesion: 0.08
Nodes (23): contract_version, major, minor, patch, description, fingerprint, handlers, optional (+15 more)

### Community 1529 - "handlers"
Cohesion: 0.08
Nodes (24): module, name, module, name, handlers, description, direct_handler, event_model (+16 more)

### Community 1530 - "validations"
Cohesion: 0.08
Nodes (24): backoff_strategy, base_delay_ms, jitter_factor, max_delay_ms, max_retries, timing_constraints, documentation, reducer_notes (+16 more)

### Community 1532 - "ModelNetworkPoolStatus"
Cohesion: 0.10
Nodes (3): _capacity_from_pool_json(), CollectorNetworkPool, ModelNetworkPoolStatus

### Community 1533 - "LifecycleHookExecutor"
Cohesion: 0.13
Nodes (3): ModelAutoWiringContext, _classify_failure(), LifecycleHookExecutor

### Community 1536 - "findings_for_module"
Cohesion: 0.10
Nodes (10): _event_type_values(), findings_for_module(), record(), _is_envelope_ctor(), LiteralEventTypeFinding, main(), _parse_args(), scan() (+2 more)

### Community 1537 - "pr_ci_zombie_detector.py"
Cohesion: 0.17
Nodes (14): CancellationCandidate, determine_zombie_cancellations(), _error_text(), fetch_active_pull_request_runs(), fetch_job_count(), force_cancel_run(), main(), _parse_timestamp() (+6 more)

### Community 1538 - "OMN-10089.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1539 - "OMN-10123.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1540 - "OMN-10207.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1541 - "OMN-10281.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1542 - "OMN-10332.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1543 - "OMN-10334.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1544 - "OMN-10338.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1545 - "OMN-10340.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1546 - "OMN-10341.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1547 - "OMN-10342.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1548 - "OMN-10343.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1549 - "OMN-10345.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1550 - "OMN-10348.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1551 - "OMN-10382.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1552 - "OMN-10387.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1553 - "OMN-10388.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1554 - "OMN-10392.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1555 - "OMN-10445.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1556 - "OMN-10447.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1557 - "OMN-10448.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1558 - "OMN-10449.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1559 - "OMN-10450.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1560 - "OMN-10453.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1561 - "OMN-10489.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1562 - "OMN-10500.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1563 - "OMN-10518.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1564 - "OMN-10519.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1565 - "OMN-10530.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1566 - "OMN-10532.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1567 - "OMN-10534.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1568 - "OMN-10563.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1569 - "OMN-10586.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1570 - "OMN-10587.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1571 - "OMN-10635.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1572 - "OMN-10782.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1573 - "OMN-10784.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1574 - "OMN-10786.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1575 - "OMN-10790.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1576 - "OMN-10796.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1577 - "OMN-10864.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1578 - "OMN-10868.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1579 - "OMN-10943.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1580 - "OMN-11050.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1581 - "OMN-11067.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1582 - "OMN-11068.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1583 - "OMN-11069.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1584 - "OMN-11073.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1585 - "OMN-11095.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1586 - "OMN-11104.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1587 - "OMN-11108.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1588 - "OMN-11130.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1589 - "OMN-11197.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1590 - "OMN-11241.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1591 - "OMN-11246.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1592 - "OMN-11248.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1593 - "OMN-11315.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1594 - "OMN-11318.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1595 - "OMN-11319.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1596 - "OMN-11320.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1597 - "OMN-11350.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1598 - "OMN-11673.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1599 - "OMN-12184.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1600 - "OMN-12503.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1601 - "OMN-12513.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1602 - "OMN-12522.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1603 - "OMN-12536.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1604 - "OMN-12559.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1605 - "OMN-12597.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1606 - "OMN-12627.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1607 - "OMN-12631.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1608 - "OMN-12633.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1609 - "OMN-12682.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1610 - "OMN-12700.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1611 - "OMN-12723.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1612 - "OMN-12755.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1613 - "OMN-12827.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1614 - "OMN-12832.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1615 - "OMN-12841.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1616 - "OMN-12848.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1617 - "OMN-12860.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1618 - "OMN-12912.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1619 - "OMN-12940.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1620 - "OMN-13008.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1621 - "OMN-13011.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1622 - "OMN-13109.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1623 - "OMN-13141.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1624 - "OMN-13149.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1625 - "OMN-13190.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1626 - "OMN-13191.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1627 - "OMN-13201.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1628 - "OMN-13203.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1629 - "OMN-13237.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1630 - "OMN-13238.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1631 - "OMN-13247.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1632 - "OMN-13403.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1633 - "OMN-13405.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1634 - "OMN-13417.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1635 - "OMN-13532.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1636 - "OMN-13533.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1637 - "OMN-13546.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1638 - "OMN-13581.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1639 - "OMN-13654.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1640 - "OMN-13656.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1641 - "OMN-13664.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1642 - "OMN-13666.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1643 - "OMN-13712.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1644 - "OMN-13719.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1645 - "OMN-13725.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1646 - "OMN-13726.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1647 - "OMN-14115.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1648 - "OMN-14953.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1649 - "OMN-14993.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1650 - "OMN-15009.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1651 - "OMN-15330.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1652 - "OMN-15384.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1653 - "OMN-15807.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1654 - "OMN-15837.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1655 - "OMN-15838.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1656 - "OMN-16056.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1657 - "OMN-16107.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1658 - "OMN-16418.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1659 - "OMN-16420.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1660 - "OMN-16661.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1661 - "OMN-19532.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1662 - "OMN-19741.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1663 - "OMN-20568.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1664 - "OMN-20600.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1665 - "OMN-20602.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1666 - "OMN-7841.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1667 - "OMN-8065.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1668 - "OMN-8701.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1669 - "OMN-9159.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1670 - "OMN-9211.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1671 - "OMN-9471.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1672 - "OMN-9532.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1673 - "OMN-9552.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1674 - "OMN-9553.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1675 - "OMN-9554.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1676 - "OMN-9648.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1677 - "OMN-9877.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1678 - "OMN-9905.yaml"
Cohesion: 0.09
Nodes (22): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+14 more)

### Community 1679 - "redpanda-partition-cap.yaml"
Cohesion: 0.09
Nodes (22): command, container_name, depends_on, condition, service, description, entrypoint, hardcoded_env (+14 more)

### Community 1680 - "migration-gate"
Cohesion: 0.09
Nodes (23): forward-migration, postgres, condition, depends_on, container_name, depends_on, entrypoint, image (+15 more)

### Community 1681 - "x-runner-base"
Cohesion: 0.12
Nodes (23): driver, options, logging, logging, logging, logging, logging, logging (+15 more)

### Community 1682 - "projection_apply_flow.py"
Cohesion: 0.13
Nodes (8): ModelProjectionApplyFlowVerdict, describe_projection_apply_divergence(), describe_projection_delta_dropped(), evaluate_projection_apply_flow(), _name_list(), projection_apply_divergence_status(), projection_delta_dropped_status(), _unobserved()

### Community 1683 - "steps"
Cohesion: 0.09
Nodes (22): author, description, OMNI_RUNNER_LOCK_FILE, inputs, lock-file, working-directory, default, description (+14 more)

### Community 1684 - "canonical-inference-gate.yml"
Cohesion: 0.09
Nodes (22): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, jobs (+14 more)

### Community 1685 - "env"
Cohesion: 0.09
Nodes (23): AUTOCLOSE_DISARMED_BY, BACKFILL_LOOKBACK_HOURS, CONTRACT_REPOS, CONTRACT_REPOS_PRIVATE, COREPACK_ENABLE_DOWNLOAD_PROMPT, DIAGNOSE_TICKETS, DOD_VERIFY_CHECK_TIMEOUT_S, DOD_VERIFY_HERMETIC_NODE_ROOT (+15 more)

### Community 1686 - "steps"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, BASE_REF, name, runs-on, steps, jobs (+14 more)

### Community 1687 - "legacy-rds-fixture-proof.yml"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, jobs (+14 more)

### Community 1688 - "pr-title"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, OMNIMARKET_VERSION, PR_AUTHOR, PR_TITLE, jobs, pr-title (+14 more)

### Community 1689 - "precommit-parity-gate.yml"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, jobs, precommit-parity-gate, name, on, merge_group (+14 more)

### Community 1690 - "with"
Cohesion: 0.09
Nodes (23): with, app-id, cache-enabled, cache-key-prefix, dest, fetch-depth, if-no-files-found, install-args (+15 more)

### Community 1691 - "steps"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, BASE_SHA, LINEAR_API_KEY, jobs, stale-todo-gate, name (+14 more)

### Community 1692 - "validate-validator-requirements.yml"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, env, UV_HTTP_TIMEOUT, jobs, validate-validator-requirements, name (+14 more)

### Community 1693 - "validator-runtime-profiles.yml"
Cohesion: 0.09
Nodes (22): concurrency, cancel-in-progress, group, env, PYTHON_VERSION, UV_HTTP_TIMEOUT, UV_VERSION, jobs (+14 more)

### Community 1694 - "Any"
Cohesion: 0.16
Nodes (12): _allowed_by(), _broker_sync_findings(), _container_sync_findings(), _desired_documents(), _distribution(), _finding(), lab_sync_findings(), _revision_matches() (+4 more)

### Community 1697 - "gate/executor.py"
Cohesion: 0.11
Nodes (7): _looks_secret_bearing(), redact_if_secret_bearing(), SecretBearingOutputError, _stdout_preview_and_hash(), CredentialRedactionFilter, install_credential_redaction_filter(), _loggers_to_cover()

### Community 1699 - "ModelGateBindingFixtures"
Cohesion: 0.09
Nodes (4): ModelGateBindingAcceptedFixture, ModelGateBindingFixtures, ModelGateBindingLineFixture, ModelGateBindingRejectedFixture

### Community 1701 - "deploy_agent_window.py"
Cohesion: 0.15
Nodes (9): _as_dict(), deploys_in_window(), lane_ready_via_httpx(), _parse_ts(), _parse_uuid(), queued_commands_from_payload(), read_deploy_agent_via_httpx(), snapshot_from_payloads() (+1 more)

### Community 1702 - "declared_supersession.py"
Cohesion: 0.10
Nodes (6): _looks_like_a_named_successor(), parse_declared_supersessions(), _qualified_refs(), resolve_declared_supersession(), ModelDeclaredSupersession, ModelSupersessionVerdict

### Community 1703 - "bounded_delegation_routes.py"
Cohesion: 0.13
Nodes (10): claimed_bounded_lane(), _claimed_lane(), _parse_topology(), _read_overlay(), _read_packaged_overlay(), _record_sha256_hex(), _refuse_unbounded_declarations(), resolve_bounded_delegation_route() (+2 more)

### Community 1705 - "delegation_health_check.py"
Cohesion: 0.13
Nodes (8): _comma_separated(), evaluate_delegation_health(), load_config(), main(), ModelFixForwardReading, ModelHealthConfig, ModelVerdictSource, parse_fix_forward_labels()

### Community 1706 - "post_release_dev_bump.py"
Cohesion: 0.14
Nodes (10): apply_decision(), _build_parser(), BumpConfigError, BumpDecision, decide(), main(), next_patch(), parse_final_version() (+2 more)

### Community 1707 - "resolve_node_migration_source_ref.py"
Cohesion: 0.15
Nodes (14): parse_trailer(), _body_from_event(), _body_from_merge_group(), _body_from_push(), _declared_is_ancestor_of_head(), main(), _paired_pr_url(), _parse_ref() (+6 more)

### Community 1708 - "build_plan"
Cohesion: 0.12
Nodes (11): build_plan(), _container_health_findings(), _exit_code(), _is_optional_and_entirely_down(), _labels_to_dict(), _parse_instant(), reconcile_lane(), _restart_loop_detail() (+3 more)

### Community 1709 - "onex-cli.sh"
Cohesion: 0.19
Nodes (20): infra-down(), infra-status(), infra-up(), infra-up-auth(), infra-up-memory(), infra-up-runtime(), _onex_continue_locally_or_return(), onex_down() (+12 more)

### Community 1710 - "sync-topic-registry.py"
Cohesion: 0.15
Nodes (11): check_mode(), _default_omni_home(), dry_run_mode(), extract_existing_block(), generate_constants_block(), load_registry(), main(), read_topics_ts() (+3 more)

### Community 1711 - "system_health_check.sh"
Cohesion: 0.21
Nodes (22): check_bus_endpoint(), check_cloud_bus_refs(), CHECK_DETAILS, check_dev_lane_liveness(), check_env_audit(), check_infisical_folders(), check_infra_containers(), check_keycloak() (+14 more)

### Community 1712 - "check_tenant_guc_domain_parity.py"
Cohesion: 0.15
Nodes (12): _bridge(), _executable_sql(), _file_order(), logical_domains(), main(), net_corpus_state(), _relation_name(), _unquote() (+4 more)

### Community 1713 - "validate_db_quality_gate.py"
Cohesion: 0.15
Nodes (8): DbQualityGateResult, DbQualityViolation, generate_report(), _is_exempt(), _is_in_string_or_comment(), main(), scan_file(), validate_db_quality_gate()

### Community 1714 - "venv_install_plan.py"
Cohesion: 0.13
Nodes (10): classify(), _describe(), main(), normalize_name(), parse_plan(), _parse_version(), plan_and_apply(), PlanChange (+2 more)

### Community 1715 - "build_events_at"
Cohesion: 0.11
Nodes (10): kafka(), agent_send(), send(), announce_preflight_failure(), build_events(), build_events_at(), declaration(), declaration_at() (+2 more)

### Community 1717 - "ModelFileSystemRequest"
Cohesion: 0.09
Nodes (3): EnumFileSystemOperation, ModelFileSystemRequest, ModelFileSystemResult

### Community 1718 - "handler_ledger_write_guard.py"
Cohesion: 0.14
Nodes (11): check_dsn(), check_file(), check_topic(), dsn_is_canonical(), file_is_canonical(), _inside(), LedgerTestWriteRefusedError, _refuse_if_test() (+3 more)

### Community 1719 - "ModelLlmEndpointOverlay"
Cohesion: 0.12
Nodes (5): ModelLlmEndpointOverlay, _parse_overlay(), resolve_overlay_endpoints(), _searched_locations(), _validate_endpoint_url()

### Community 1720 - "RowCountProbe"
Cohesion: 0.12
Nodes (4): ModelRowCountProbeResult, ModelTableRowCount, PgStatRow, RowCountProbe

### Community 1721 - "ModelGeneratedComputeNodeSpec"
Cohesion: 0.10
Nodes (6): ModelGeneratedComputeNodeSpec, ModelMCPWrapperEmission, ServiceMCPWrapperContractEmitter, build_generated_tool_invoke_suffix(), build_generated_tool_result_suffix(), _to_kebab_event_name()

### Community 1724 - "HandlerMergeProvenance"
Cohesion: 0.12
Nodes (5): HandlerMergeProvenance, EnumMergeProvenanceReason, EnumMergeProvenanceVerdict, ModelMergeProvenanceRequest, ModelMergeProvenanceResult

### Community 1725 - "node_registration_orchestrator/contract.yaml"
Cohesion: 0.09
Nodes (22): contract_version, major, minor, patch, description, descriptor, liveness_interval_seconds, node_archetype (+14 more)

### Community 1726 - "properties"
Cohesion: 0.09
Nodes (23): default_factory, description, key_type, type, value_type, default, description, key_type (+15 more)

### Community 1727 - "response_fields"
Cohesion: 0.09
Nodes (23): description, path, response_fields, buckets, description, labels, name, states (+15 more)

### Community 1728 - "HandlerScopeCheckInitiate"
Cohesion: 0.10
Nodes (5): HandlerScopeFileRead, ModelScopeFileReadRequest, ModelScopeFileReadResult, HandlerScopeCheckInitiate, ModelScopeCheckCommand

### Community 1732 - "ModelCostApiBase"
Cohesion: 0.10
Nodes (8): ModelCostApiBase, ModelCostSummary, ModelCostTrend, ModelCostTrendPoint, ModelSavingsSummaryItem, ModelSavingsSummary, ModelSavingsUnavailable, ModelTokenUsage

### Community 1733 - "ServiceRegistryDiscovery"
Cohesion: 0.10
Nodes (4): ModelFeatureFlagChangedEvent, ModelFeatureFlagToggleResult, ModelFeatureFlagView, ServiceRegistryDiscovery

### Community 1734 - "util_dlq_credential_redaction.py"
Cohesion: 0.13
Nodes (11): is_credential_field_name(), _normalise(), _redact(), redact_credential_fields(), redact_credential_fields_in_text(), redact_credential_value_patterns(), _redact_text_pairs(), _sub() (+3 more)

### Community 1735 - "OMN-10081.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1736 - "OMN-10137.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1737 - "OMN-10168.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1738 - "OMN-10169.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1739 - "OMN-10204.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1740 - "OMN-10237.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1741 - "OMN-10238.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1742 - "OMN-14060.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1743 - "OMN-1419.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1744 - "OMN-9215.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1745 - "OMN-9410.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1746 - "OMN-9420.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1747 - "OMN-9550.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1748 - "OMN-9556.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1749 - "OMN-9557.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1750 - "OMN-9600.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1751 - "OMN-9727.yaml"
Cohesion: 0.09
Nodes (21): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+13 more)

### Community 1752 - "c12_provider_catalogue_observe.py"
Cohesion: 0.16
Nodes (10): _customer_only(), _exc(), _findings(), _intake(), main(), observe(), _sha256(), fold_manifest() (+2 more)

### Community 1753 - "ci"
Cohesion: 0.10
Nodes (21): builders, ci, lane, bootstrap_timeout_seconds, buildkitd_config, driver, driver_opts, image (+13 more)

### Community 1754 - "hardcoded_env"
Cohesion: 0.09
Nodes (22): hardcoded_env, BUS_ID, DELEGATION_ROUTING_TIERS_PATH, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KAFKA_MAX_POLL_INTERVAL_MS, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL (+14 more)

### Community 1755 - "com.omninode.lane"
Cohesion: 0.09
Nodes (22): container_name, labels, com.omninode.lane, container_name, labels, container_name, labels, container_name (+14 more)

### Community 1756 - "environment"
Cohesion: 0.11
Nodes (22): BIFROST_LANE_OVERLAY_PATH, INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_ENVIRONMENT_SLUG, INFISICAL_PROJECT_ID, OMNIMARKET_PROJECTION_READ_BINDING_OVERLAY, ONEX_CONSUMER_FLOW_ARCHIVE_DIR (+14 more)

### Community 1757 - "com.omninode.lane"
Cohesion: 0.09
Nodes (22): container_name, labels, com.omninode.lane, container_name, labels, container_name, labels, container_name (+14 more)

### Community 1758 - "omninode-dns-cache"
Cohesion: 0.09
Nodes (21): context, interval, retries, start_period, test, timeout, driver, options (+13 more)

### Community 1759 - "environment"
Cohesion: 0.36
Nodes (22): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, ONEX_LAB_RUN_HOSTS, ONEX_LANE_HOST, RUNNER_GROUP (+14 more)

### Community 1760 - "environment"
Cohesion: 0.36
Nodes (22): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, KUBECONFIG, LLM_ENDPOINT_CIDR_ALLOWLIST, LOCAL_LLM_SHARED_SECRET, ONEX_LAB_RUN_HOSTS, ONEX_LANE_HOST, RUNNER_GROUP (+14 more)

### Community 1761 - "hp053-ci-runner-2"
Cohesion: 0.11
Nodes (22): logging, cgroup_parent, container_name, cpus, entrypoint, group_add, image, logging (+14 more)

### Community 1762 - "omninode-prod-deploy-runner-1"
Cohesion: 0.12
Nodes (22): driver, options, logging, logging, logging, container_name, cpus, entrypoint (+14 more)

### Community 1763 - "runtime-worker"
Cohesion: 0.13
Nodes (22): create_host_path, replicas, volumes, volumes, container_name, depends_on, deploy, labels (+14 more)

### Community 1764 - "runner-job-started.sh"
Cohesion: 0.22
Nodes (21): _c1_audit(), _c2_budget_remaining(), _c2_budget_spent(), _c2_budget_timeout(), _c2_ensure_head_sha_fetched(), _c2_env_add_config(), _c2_rewrite_add_pair(), _C2_REWRITE_ENV_LINES (+13 more)

### Community 1765 - "contract_resolver/routes.py"
Cohesion: 0.11
Nodes (4): contract_resolve(), _emit_completed_event(), health(), ModelHealthResponse

### Community 1766 - "steps"
Cohesion: 0.09
Nodes (21): runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, jobs, check-env-reads-gate (+13 more)

### Community 1767 - "with"
Cohesion: 0.09
Nodes (22): with, app-id, fetch-depth, filter, github-token, if-no-files-found, name, owner (+14 more)

### Community 1768 - "non-dev-base-guard.yml"
Cohesion: 0.09
Nodes (21): concurrency, cancel-in-progress, group, BASE_REF, HEAD_REF, PR_BODY, jobs, non-dev-base-guard (+13 more)

### Community 1769 - "with"
Cohesion: 0.09
Nodes (22): with, app-id, fetch-depth, if-no-files-found, name, owner, path, permission-actions (+14 more)

### Community 1770 - "deliver_reports"
Cohesion: 0.10
Nodes (6): deliver_reports(), _dev_lane_services(), retire_lane_services(), run_preflight(), ModelDevBootReportConfig, ProtocolDevPreflightRunner

### Community 1771 - "capture_delegation_evidence.py"
Cohesion: 0.22
Nodes (12): _actual_receipt(), _copy_once(), main(), pack(), _read_json(), _readback(), _required_string(), _route_provenance() (+4 more)

### Community 1772 - "check-pinned-wheels.py"
Cohesion: 0.16
Nodes (11): _build_report(), _check_pypi_range_pins(), _check_uv_sources(), DriftReport, _gh_latest_release(), main(), _minor_behind(), _parse_uv_sources() (+3 more)

### Community 1773 - "generate_claude_lane_block.py"
Cohesion: 0.16
Nodes (10): comparable(), extract_block(), generate_block(), _load_manifest(), _load_snapshot(), main(), _run_check(), _service_count() (+2 more)

### Community 1774 - ".run"
Cohesion: 0.14
Nodes (4): _discover_postgres_containers(), LogMonitor, main(), _restart_containers_from_env()

### Community 1775 - "runner_fleet_event.py"
Cohesion: 0.15
Nodes (9): build_event(), classify(), _counter(), _label_names(), main(), _oom_kill_counters(), _parse_name_prefixes(), resolve_host() (+1 more)

### Community 1776 - "ModelGatewayRenewalDirective"
Cohesion: 0.12
Nodes (3): GatewayRenewalPlanner, EnumGatewayRenewalMode, ModelGatewayRenewalDirective

### Community 1777 - "HandlerSourceAttestation"
Cohesion: 0.11
Nodes (3): ModelRuntimeBootedEvent, HandlerSourceAttestation, ModelSourceAttestationResult

### Community 1778 - "timeouts"
Cohesion: 0.09
Nodes (21): timeouts, capability_probe_seconds, capability_probe_ttl_seconds, circuit_breaker_reset_seconds, handler_shutdown_default_seconds, handler_shutdown_max_seconds, handler_shutdown_min_seconds, health_check_default_seconds (+13 more)

### Community 1779 - "ServiceFlakeDetector"
Cohesion: 0.10
Nodes (4): ModelFlakeDetectionResult, ModelFlakeRecord, ServiceFlakeDetector, should_quarantine_verdict()

### Community 1780 - "node_contract_loader_effect/contract.yaml"
Cohesion: 0.09
Nodes (21): contract_version, major, minor, patch, description, fingerprint, input_model, module (+13 more)

### Community 1781 - "ModelDlqMessage"
Cohesion: 0.13
Nodes (6): DLQProducer, unwrap_nested_dlq_record(), ModelDlqMessage, _parse_consumer_group(), _parse_redacted_fields(), DlqRecordUnparseableError

### Community 1782 - "ProtocolKafkaReplayConsumer"
Cohesion: 0.12
Nodes (3): ProtocolKafkaReplayConsumer, ProtocolOffsetAndTimestamp, ProtocolTopicPartition

### Community 1783 - "ModelRegistryIntent"
Cohesion: 0.10
Nodes (4): ModelPostgresUpsertIntent, validate_union_registry_sync(), ModelRegistryIntent, RegistryIntent

### Community 1784 - "error_handling"
Cohesion: 0.09
Nodes (22): backend, code, description, operation, retriable, error_handling, error_codes, error_sanitization (+14 more)

### Community 1785 - "node_service_discovery_effect/contract.yaml"
Cohesion: 0.09
Nodes (21): capabilities, description, name, dependencies, class_name, description, module, name (+13 more)

### Community 1786 - "node_slack_alerter_effect/contract.yaml"
Cohesion: 0.09
Nodes (21): capabilities, description, name, contract_name, description, io_operations, description, input_fields (+13 more)

### Community 1787 - "steps"
Cohesion: 0.09
Nodes (21): description, steps, depends_on, description, estimated_duration_seconds, name, produces_capabilities, required_capabilities (+13 more)

### Community 1789 - "intelligence_hook_event_v1.yaml"
Cohesion: 0.09
Nodes (21): consumer, file, model, repo, description, optional_fields, description, name (+13 more)

### Community 1791 - "stack_shape.yaml"
Cohesion: 0.10
Nodes (20): command_baseline, argv_sha256, command_hint, field, files, service, command_only_compose_files, compose_files (+12 more)

### Community 1792 - "omnibase_core"
Cohesion: 0.10
Nodes (21): reason, status, target, reason, status, target, reason, status (+13 more)

### Community 1793 - "strip_runtime_entry_points.py"
Cohesion: 0.18
Nodes (10): _dist_name(), _iter_site_packages(), main(), normalize_distribution_name(), _parse_args(), _read_entry_points(), strip_runtime_entry_points(), StripEntryPointReport (+2 more)

### Community 1794 - "OMN-16322.yaml"
Cohesion: 0.10
Nodes (20): check_type, check_value, dod_evidence, checks, description, evidence_artifact, id, source (+12 more)

### Community 1795 - "autoheal.yaml"
Cohesion: 0.10
Nodes (20): container_name, depends_on, description, hardcoded_env, AUTOHEAL_CONTAINER_LABEL, AUTOHEAL_INTERVAL, AUTOHEAL_START_PERIOD, healthcheck (+12 more)

### Community 1796 - "hardcoded_env"
Cohesion: 0.10
Nodes (21): hardcoded_env, BUS_ID, ENABLE_RUNTIME_LOG_BRIDGE, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KAFKA_DEAD_LETTER_TOPIC, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL (+13 more)

### Community 1797 - "hardcoded_env"
Cohesion: 0.10
Nodes (21): hardcoded_env, BROKER_ACL_PROVIDER, BROKER_QUOTA_PROVIDER, GATEWAY_ATTACH_TOKEN_EXCHANGE, GATEWAY_SIGNUP_TOPIC_PROVISIONING, GATEWAY_TOKEN_MAX_TTL_SECONDS, GATEWAY_USER_CLIENT_ID, KAFKA_BOOTSTRAP_SERVERS (+13 more)

### Community 1798 - "projection-tenant-registry-writer"
Cohesion: 0.11
Nodes (21): migration-gate, postgres, redpanda, redpanda-partition-cap, condition, condition, depends_on, depends_on (+13 more)

### Community 1799 - "redpanda-sasl-enable"
Cohesion: 0.10
Nodes (21): redpanda-scram-user, DEV_KAFKA_SASL_PASSWORD, DEV_KAFKA_SASL_USERNAME, command, environment, mem_limit, mem_reservation, command (+13 more)

### Community 1800 - "environment"
Cohesion: 0.10
Nodes (21): AUTH_SECRET, DB_CONNECTION_URI, ENCRYPTION_KEY, REDIS_URL, SITE_URL, SMTP_FROM_ADDRESS, SMTP_FROM_NAME, SMTP_HOST (+13 more)

### Community 1801 - "keycloak"
Cohesion: 0.10
Nodes (21): KC_BOOTSTRAP_ADMIN_PASSWORD, KC_BOOTSTRAP_ADMIN_USERNAME, KC_DB, KC_DB_PASSWORD, KC_DB_URL, KC_DB_USERNAME, KC_HEALTH_ENABLED, KC_HOSTNAME_STRICT (+13 more)

### Community 1802 - "volumes"
Cohesion: 0.10
Nodes (21): name, name, name, name, name, name, name, name (+13 more)

### Community 1803 - "volumes"
Cohesion: 0.10
Nodes (21): name, name, name, name, name, name, name, name (+13 more)

### Community 1804 - "docker-compose.stability-test.yml"
Cohesion: 0.10
Nodes (20): name, admin_port_publish, advertised_kafka_addr, advertised_pandaproxy_addr, kafka_lan_port_publish, kafka_port_publish, x-omninode-contract-overlay, stability-test-redpanda (+12 more)

### Community 1805 - "HandlerMergeGroupRunReadGithub"
Cohesion: 0.13
Nodes (4): HandlerMergeGroupRunReadGithub, MergeGroupReadError, _workflow_path(), ModelWorkflowRunFact

### Community 1806 - "lane_census_event.py"
Cohesion: 0.19
Nodes (9): _alert_key(), build_event(), build_observed_event(), _findings(), main(), _ticket_body(), _ticket_title(), validate_event() (+1 more)

### Community 1807 - "router.py"
Cohesion: 0.24
Nodes (6): get_cost_by_model(), get_cost_by_repo(), get_cost_summary(), get_cost_trend(), get_savings_summary(), get_token_usage()

### Community 1808 - "resolve-arm64-proof-legs/action.yml"
Cohesion: 0.10
Nodes (20): author, description, ARM64_VERIFY_PROOF_LEGS_JSON, inputs, legs-json, description, description, required (+12 more)

### Community 1809 - "with"
Cohesion: 0.10
Nodes (20): concurrency, cancel-in-progress, group, jobs, repo-evidence, name, on, pull_request_target (+12 more)

### Community 1810 - "with"
Cohesion: 0.10
Nodes (21): with, app-id, cache-dependency-glob, deploy-wait-seconds, dest, enable-cache, fetch-depth, if-no-files-found (+13 more)

### Community 1811 - "route"
Cohesion: 0.10
Nodes (21): route, runner-routing-audit, topic-naming-baseline-oneway, actions, contents, name, secrets, uses (+13 more)

### Community 1812 - "delegation-health-check.yml"
Cohesion: 0.10
Nodes (20): concurrency, cancel-in-progress, group, secrets, uses, with, jobs, delegation-health-check (+12 more)

### Community 1813 - "main-target-guard.yml"
Cohesion: 0.10
Nodes (20): concurrency, cancel-in-progress, group, BASE_REF, HEAD_REF, jobs, main-target-guard, name (+12 more)

### Community 1814 - "steps"
Cohesion: 0.10
Nodes (20): name, permissions, runs-on, steps, timeout-minutes, CROSS_REPO_PAT, GH_TOKEN, jobs (+12 more)

### Community 1815 - "no-bare-compose-teardown"
Cohesion: 0.10
Nodes (20): concurrency, cancel-in-progress, group, jobs, no-bare-compose-teardown, name, name, runs-on (+12 more)

### Community 1816 - "inputs"
Cohesion: 0.10
Nodes (21): description, required, type, description, required, type, base_branch, changed_files (+13 more)

### Community 1817 - "with"
Cohesion: 0.10
Nodes (21): with, app-id, cache-enabled, cache-key-prefix, dest, fetch-depth, if-no-files-found, install-args (+13 more)

### Community 1818 - ".create_app"
Cohesion: 0.12
Nodes (3): _load_mcp_server_class(), TransportMCPStreamableHttp, _make_tool_handler()

### Community 1821 - "build_and_start_core_runtime"
Cohesion: 0.11
Nodes (7): build_and_start_core_runtime(), build_kernel_handler_resolver(), _resolve(), _construct_handler(), _param_wants_container(), _provision_dlq_topics(), DefBTarget

### Community 1822 - "check_dispatcher_route_coverage.py"
Cohesion: 0.14
Nodes (8): check_route_coverage(), ContractInfo, main(), _parse_contract(), _parse_topic_list(), RouteCoverageFailure, RouteCoverageReport, _scan_contracts_dir()

### Community 1823 - "check_integration_skips.py"
Cohesion: 0.19
Nodes (10): classify_skip(), evaluate(), GuardConfig, main(), parse_junit(), ReportStats, run_gate(), selftest() (+2 more)

### Community 1824 - "verify_pypi_pin_resolvability.py"
Cohesion: 0.13
Nodes (8): conflicting_requirements(), _decode_stream(), find_single_wheel(), main(), PinResolveTimeoutError, _report_unresolvable(), _resolve_uv(), verify_pin_resolvability()

### Community 1825 - "worker_cycle.py"
Cohesion: 0.16
Nodes (7): _build_wire_payload(), _context(), run_chat_completion(), ModelLocalInferenceRequest, ModelLocalInferenceResult, _envelope_to_inference_request(), run_single_cycle()

### Community 1826 - "prepush_remote_verify.py"
Cohesion: 0.17
Nodes (10): _build_parser(), check(), _fetch_jobs(), _fetch_runs_for_sha(), _gh_json(), main(), RemoteVerifyError, resolve_repo_slug() (+2 more)

### Community 1827 - "lint_compose_dangling_deps.py"
Cohesion: 0.15
Nodes (8): _collect_violations(), _find_disabled_services(), _is_empty_or_interpolated_empty(), lint_file(), main(), _make_loader(), _override_constructor(), _value_references_hostname()

### Community 1828 - "monitor_logs.py"
Cohesion: 0.14
Nodes (10): _apply_monitor_environment_config(), _bootstrap_monitor_environment(), _extract_table_name(), _load_omnibase_env(), _MonitorEnvironmentConfig, _parse_postgres_error_block(), post_slack(), _post_slack_plain_text() (+2 more)

### Community 1829 - "detect_foreign_prepush.py"
Cohesion: 0.17
Nodes (11): _build_parser(), collect_processes(), find_queue_runner_pid(), format_report(), is_descendant(), main(), parse_ps_output(), ProcessInfo (+3 more)

### Community 1830 - "ModelCiBusLane"
Cohesion: 0.11
Nodes (3): build_redeploy_start_payload(), ModelCiBusLane, ModelRedeployStartCommandWire

### Community 1832 - "HandlerDeclaredStateApply"
Cohesion: 0.12
Nodes (3): HandlerDeclaredStateApply, EnumDeclaredStateApplyStatus, ModelDeclaredStateApplyResult

### Community 1833 - "HandlerDeclaredStateReconcile"
Cohesion: 0.11
Nodes (4): HandlerDeclaredStateReconcile, ModelDeclaredStateItem, ModelDeclaredStateSnapshot, ProtocolDeclaredStateObserver

### Community 1834 - "HandlerDeployRunners"
Cohesion: 0.14
Nodes (4): HandlerDeployRunners, ModelDeployRunnersRequest, ModelDeployRunnersResult, ModelRunnerTmpPolicy

### Community 1835 - "HandlerVenvAtomicInstall"
Cohesion: 0.11
Nodes (4): HandlerVenvAtomicInstall, _load(), ModelVenvAtomicInstallRequest, ModelVenvAtomicInstallResult

### Community 1837 - "properties"
Cohesion: 0.10
Nodes (21): description, type, description, items, type, type, properties, description (+13 more)

### Community 1838 - "properties"
Cohesion: 0.10
Nodes (21): default, description, type, default, description, type, default_factory, description (+13 more)

### Community 1839 - "local_bundle"
Cohesion: 0.10
Nodes (21): local_bundle, bundle, bus_bootstrap_servers, clone_base_url, compose_project, consumer_modules, container_migration_gate, container_runtime_effects (+13 more)

### Community 1840 - "properties"
Cohesion: 0.10
Nodes (21): description, type, ModelServiceRegistration, description, type, description, type, properties (+13 more)

### Community 1841 - "overseer_tick_projector.yaml"
Cohesion: 0.10
Nodes (20): aggregate_type, behavior, mode, default, name, source, type, consumed_events (+12 more)

### Community 1843 - "ModelConfigRef"
Cohesion: 0.10
Nodes (3): EnumConfigRefScheme, ModelConfigRef, ModelConfigRefParseResult

### Community 1846 - "entries"
Cohesion: 0.10
Nodes (19): description, entries, accepted_rationale, complexity_score, fan_in, fan_out, finding_codes, line_count (+11 more)

### Community 1847 - "ConfigSessionStorage"
Cohesion: 0.12
Nodes (5): [0.10.0] - 2026-02-23, Changed, Dependencies, Fixed, ConfigSessionStorage

### Community 1848 - "update_env_file"
Cohesion: 0.10
Nodes (12): [0.13.0] - 2026-02-28, [0.14.0] - 2026-03-03, Added, Added, Changed, Changed, Dependencies, Fixed (+4 more)

### Community 1849 - "shared"
Cohesion: 0.10
Nodes (19): bootstrap_only, identity_defaults, db, infisical, kafka, llm, service_override_required, services (+11 more)

### Community 1850 - "OMN-12245.yaml"
Cohesion: 0.10
Nodes (19): check_type, check_value, dod_evidence, checks, description, id, source, status (+11 more)

### Community 1851 - "OMN-13350.yaml"
Cohesion: 0.10
Nodes (19): check_type, check_value, dod_evidence, checks, description, id, source, status (+11 more)

### Community 1852 - "OMN-13531.yaml"
Cohesion: 0.10
Nodes (19): check_type, check_value, dod_evidence, checks, description, id, source, status (+11 more)

### Community 1853 - "OMN-16293.yaml"
Cohesion: 0.10
Nodes (19): check_type, check_value, cwd, dod_evidence, checks, description, id, source (+11 more)

### Community 1854 - "OMN-18326.yaml"
Cohesion: 0.10
Nodes (19): check_type, check_value, dod_evidence, binds_ac, checks, description, id, source (+11 more)

### Community 1855 - "coordination_rules"
Cohesion: 0.10
Nodes (20): allow_partial_completion, dependency_resolution_enabled, detect_cycles, emit_dependency_graph, emit_step_metrics, emit_workflow_events, fail_on_cycle, failure_recovery_strategy (+12 more)

### Community 1856 - "workflow_coordination"
Cohesion: 0.10
Nodes (20): publish_topics, subscribe_topics, graceful_shutdown_timeout_ms, health_check_interval_ms, shutdown_sequence, startup_sequence, description, payload_schema (+12 more)

### Community 1857 - "hardcoded_env"
Cohesion: 0.10
Nodes (20): hardcoded_env, BUS_ID, ENABLE_RUNTIME_LOG_BRIDGE, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL, OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS (+12 more)

### Community 1858 - "forward-migration"
Cohesion: 0.10
Nodes (20): postgres, container_name, depends_on, entrypoint, image, networks, profiles, restart (+12 more)

### Community 1859 - "environment"
Cohesion: 0.11
Nodes (20): DELEGATION_ROUTING_TIERS_PATH, INFISICAL_ADDR, KAFKA_BOOTSTRAP_SERVERS, OMNIBASE_INFRA_DB_URL, OMNIINTELLIGENCE_DB_URL, ONEX_CONTRACTS_DIR, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT (+12 more)

### Community 1860 - "gate-runner"
Cohesion: 0.10
Nodes (19): context, dockerfile, PYTHONPATH, UV_PROJECT_ENVIRONMENT, build, container_name, cpus, environment (+11 more)

### Community 1861 - "environment"
Cohesion: 0.16
Nodes (20): CHAIN_CANARY_READER_PASSWORD, MIGRATIONS_DIR, NODE_POSTGRES_DB, OMNINODE_RUNTIME_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT (+12 more)

### Community 1862 - "forward-migration"
Cohesion: 0.10
Nodes (20): postgres, container_name, depends_on, entrypoint, image, networks, profiles, restart (+12 more)

### Community 1863 - "runtime-effects"
Cohesion: 0.10
Nodes (20): omninode-runtime, redpanda, condition, condition, command, container_name, depends_on, entrypoint (+12 more)

### Community 1864 - "environment"
Cohesion: 0.26
Nodes (20): BIFROST_VERIFY_ENDPOINTS, KAFKA_ENVIRONMENT, KAFKA_INSTANCE_ID, OMNIINTELLIGENCE_PUBLISH_INTROSPECTION, OMNIMEMORY_ENABLED, OMNIMEMORY_MEMGRAPH_HOST, ONEX_BOX_ID, ONEX_DATABASE_TOPOLOGY_PROFILE (+12 more)

### Community 1865 - "docker-compose.runners-hp051-ci-runner.yml"
Cohesion: 0.11
Nodes (19): create_host_path, services, x-lab-host-table, bind, read_only, source, target, type (+11 more)

### Community 1866 - "no-cache"
Cohesion: 0.12
Nodes (20): no-cache, pinned_revision, sibling_ref, default, description, required, type, on (+12 more)

### Community 1867 - "with"
Cohesion: 0.10
Nodes (20): with, app-id, cache-dependency-glob, deploy-wait-seconds, dest, enable-cache, fetch-depth, if-no-files-found (+12 more)

### Community 1868 - "with"
Cohesion: 0.10
Nodes (20): with, app-id, aws-region, fetch-depth, if-no-files-found, name, owner, path (+12 more)

### Community 1869 - "steps"
Cohesion: 0.18
Nodes (20): steps, steps, steps, name, runs-on, steps, timeout-minutes, name (+12 more)

### Community 1870 - "with"
Cohesion: 0.10
Nodes (20): with, app-id, cache-dependency-glob, enable-cache, fetch-depth, node-version, owner, path (+12 more)

### Community 1871 - "with"
Cohesion: 0.10
Nodes (20): with, app-id, fetch-depth, filter, github-token, if-no-files-found, name, owner (+12 more)

### Community 1872 - "steps"
Cohesion: 0.10
Nodes (19): name, runs-on, steps, timeout-minutes, jobs, contract-evidence, name, on (+11 more)

### Community 1873 - "probe-placement-drift.yml"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, uses, with, jobs, drift, name (+11 more)

### Community 1874 - "seed-provenance-check"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, jobs, seed-provenance-check, name, on, merge_group (+11 more)

### Community 1875 - "wheel-content-parity.yml"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, jobs, wheel-content-parity, name, on, merge_group (+11 more)

### Community 1878 - "handler_rsd_score_calculate.py"
Cohesion: 0.13
Nodes (5): ModelTicketData, _calculate_dependency_distance(), _calculate_failure_surface(), _calculate_time_decay(), _count_downstream()

### Community 1880 - "assert_image_config_paths.py"
Cohesion: 0.13
Nodes (8): _build_probe_script(), _load_registry_module(), main(), _parse_verdicts(), _run_probe(), _validate_paths(), ModelRequiredImageConfigPath, required_image_config_paths()

### Community 1881 - "contract_topic_graph.py"
Cohesion: 0.16
Nodes (9): build_graph(), _collect_topics(), _dig(), discover_contract_roots(), _has_dispatch_entry(), _has_top_level_handler(), _load_catalog_ingress_nodes(), main() (+1 more)

### Community 1882 - "validate-kafka-schema-handshake.py"
Cohesion: 0.16
Nodes (7): _changed_module_prefixes(), _load_boundary_pairs(), main(), _pair_touches_changed(), PairResult, _print_human(), _validate_pair()

### Community 1883 - "emit_baselines_raw_snapshot.py"
Cohesion: 0.18
Nodes (9): _build_empty_snapshot(), _compute_breakdown_from_comparisons(), _emit_to_kafka(), main(), _make_delta_metric(), _read_comparison_data(), _read_raw_measurements(), _read_trend_data() (+1 more)

### Community 1884 - "desired_state.py"
Cohesion: 0.21
Nodes (11): _cmd_self_check(), _cmd_validate(), DesiredStateRefusalError, load_desired_state(), _load_schema(), main(), _no_duplicate_keys(), parse_desired_state() (+3 more)

### Community 1885 - "Any"
Cohesion: 0.18
Nodes (8): _check_canonical_order(), _compile(), _describe_condition(), _is_type(), _json_equal(), _json_type(), _render(), _Schema

### Community 1886 - "record_down_migration_execution.py"
Cohesion: 0.18
Nodes (11): _append_record(), _IndentedDumper, _load_checker(), main(), ModelRoundTrip, _psql(), round_trip(), _run() (+3 more)

### Community 1887 - "check_cli_registry_vocabulary.py"
Cohesion: 0.15
Nodes (9): _callee_name(), check_file(), _exempt(), _findings(), main(), _python_files(), _string_elements(), _subscript_name() (+1 more)

### Community 1888 - "check_topic_literals.py"
Cohesion: 0.17
Nodes (8): _build_parent_map(), collect_files(), find_topic_literals(), _is_docstring(), load_baseline(), main(), _should_skip(), violation_key()

### Community 1889 - "integrations"
Cohesion: 0.10
Nodes (19): catalog_version, major, minor, patch, description, endpoint, host, port (+11 more)

### Community 1891 - "HandlerDeclaredStateLeaseLocal"
Cohesion: 0.12
Nodes (3): HandlerDeclaredStateLeaseLocal, LeaseKey, ModelLeaseTable

### Community 1892 - "HandlerCodingAgentOrchestrator"
Cohesion: 0.14
Nodes (4): ModelCodingAgentEvent, ModelCodingAgentFsmAdvance, _coerce_invoke(), HandlerCodingAgentOrchestrator

### Community 1893 - "HandlerRepoStateCollect"
Cohesion: 0.12
Nodes (3): ModelRRHRepoState, HandlerRepoStateCollect, ModelRepoStateCollectRequest

### Community 1894 - "properties"
Cohesion: 0.10
Nodes (20): description, type, description, type, description, type, properties, description (+12 more)

### Community 1895 - "node_bus_forwarder_effect/contract.yaml"
Cohesion: 0.10
Nodes (19): capabilities, description, name, contract_name, contract_version, major, minor, patch (+11 more)

### Community 1896 - "settings"
Cohesion: 0.10
Nodes (20): settings, alarm_job_owner_line, alarm_owner, alarm_pin_owner_line, append_command_env, append_timeout_seconds, error_cell_chars, excluded_lanes (+12 more)

### Community 1897 - "._process_message"
Cohesion: 0.13
Nodes (4): generate_replay_correlation_id(), DlqConsumerDrainState, _source_coordinate(), ModelDlqTargetReplayBound

### Community 1898 - "node_event_bus_wiring_effect/contract.yaml"
Cohesion: 0.10
Nodes (19): contract_version, major, minor, patch, description, fingerprint, input_model, module (+11 more)

### Community 1899 - "node_gmail_archive_cleanup_effect/contract.yaml"
Cohesion: 0.10
Nodes (19): capabilities, description, name, config, archive_labels, retention_days, contract_name, description (+11 more)

### Community 1900 - "properties"
Cohesion: 0.10
Nodes (20): default_factory, description, item_type, type, default, description, type, default (+12 more)

### Community 1902 - "coordination_rules"
Cohesion: 0.10
Nodes (20): allow_partial_completion, dependency_resolution_enabled, detect_cycles, emit_dependency_graph, emit_step_metrics, emit_workflow_events, fail_on_cycle, failure_recovery_strategy (+12 more)

### Community 1903 - "verification_receipt_projector.yaml"
Cohesion: 0.10
Nodes (19): aggregate_type, behavior, mode, name, source, type, consumed_events, columns (+11 more)

### Community 1907 - "ServiceBatchComputeEffectivenessMetrics"
Cohesion: 0.15
Nodes (3): ModelBatchComputeResult, parse_execute_count(), ServiceBatchComputeEffectivenessMetrics

### Community 1909 - "PublisherContractCorpus"
Cohesion: 0.13
Nodes (3): _publisher_topics(), PublisherContractCorpus, PublisherTopicError

### Community 1910 - "Added"
Cohesion: 0.11
Nodes (12): [0.2.6] - 2026-01-30, Added, Changed, Contract Registry System, Declarative Configuration & Routing, Dependencies, Emit Daemon, Event Ledger Persistence (+4 more)

### Community 1912 - "console_scripts"
Cohesion: 0.11
Nodes (19): console_scripts, omnibase_spi, omnimarket, omninode-claude, omninode-intelligence, omninode-memory, reason, status (+11 more)

### Community 1913 - "endpoints"
Cohesion: 0.11
Nodes (18): endpoints, context_window_budgeted, endpoint_from_overlay, endpoint_url, hardware, host, lane_overlay_binding, launchd_unit_or_none (+10 more)

### Community 1914 - "OMN-10079.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1915 - "OMN-10615.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1916 - "OMN-10616.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1917 - "OMN-10637.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1918 - "OMN-10638.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1919 - "OMN-10657.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1920 - "OMN-10712.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1921 - "OMN-10728.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1922 - "OMN-10741.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1923 - "OMN-10755.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1924 - "OMN-10780.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, evidence_requirements (+10 more)

### Community 1925 - "OMN-10792.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1926 - "OMN-10794.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1927 - "OMN-10797.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1928 - "OMN-10814.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1929 - "OMN-10867.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1930 - "OMN-10972.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1931 - "OMN-11061.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1932 - "OMN-11131.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1933 - "OMN-11161.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1934 - "OMN-11162.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1935 - "OMN-11163.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1936 - "OMN-11185.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1937 - "OMN-11196.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1938 - "OMN-11200.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1939 - "OMN-11201.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1940 - "OMN-11242.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1941 - "OMN-11244.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1942 - "OMN-11247.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1943 - "OMN-11299.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1944 - "OMN-12815.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1945 - "OMN-13037.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1946 - "OMN-13362.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1947 - "OMN-13404.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1948 - "OMN-13406.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1949 - "OMN-13507.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1950 - "OMN-13530.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1951 - "OMN-14974.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1952 - "OMN-16994.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, evidence_requirements (+10 more)

### Community 1953 - "OMN-8131.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1954 - "OMN-9120.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1955 - "OMN-9139.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1956 - "OMN-9266.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1957 - "OMN-9529.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1958 - "OMN-9713.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1959 - "OMN-9769.yaml"
Cohesion: 0.11
Nodes (18): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+10 more)

### Community 1960 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1961 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1962 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1963 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1964 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1965 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1966 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1967 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1968 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1969 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1970 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1971 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1972 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1973 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1974 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1975 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1976 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 1977 - "omnibase_infra"
Cohesion: 0.11
Nodes (19): service_owned, omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 1978 - "hardcoded_env"
Cohesion: 0.11
Nodes (19): hardcoded_env, BUS_ID, ENABLE_RUNTIME_LOG_BRIDGE, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL, OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS (+11 more)

### Community 1979 - "hardcoded_env"
Cohesion: 0.11
Nodes (19): hardcoded_env, BUS_ID, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL, OMNIINTELLIGENCE_DB_URL, ONEX_ACTIVE_RUNTIME_PACKAGES (+11 more)

### Community 1980 - "projection-delegation-writer"
Cohesion: 0.11
Nodes (19): resources, cpus, memory, command, container_name, deploy, labels, networks (+11 more)

### Community 1981 - "forward-migration"
Cohesion: 0.11
Nodes (19): resources, container_name, deploy, entrypoint, image, labels, networks, profiles (+11 more)

### Community 1982 - "projection-savings-writer"
Cohesion: 0.11
Nodes (19): resources, cpus, memory, deploy, deploy, command, container_name, deploy (+11 more)

### Community 1984 - "alerter-injected-red-canary.yml"
Cohesion: 0.11
Nodes (18): name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group, INJECT_RED_UNTIL (+10 more)

### Community 1985 - "hostile-review-thread-gate"
Cohesion: 0.11
Nodes (18): concurrency, cancel-in-progress, group, if, name, needs, runs-on, timeout-minutes (+10 more)

### Community 1986 - "with"
Cohesion: 0.11
Nodes (19): with, app-id, cache-enabled, cache-key-prefix, dest, if-no-files-found, install-args, name (+11 more)

### Community 1987 - "with"
Cohesion: 0.11
Nodes (19): with, app-id, client-payload, event-type, fetch-depth, files, generate_release_notes, owner (+11 more)

### Community 1988 - "model_lab_proof_match.py"
Cohesion: 0.11
Nodes (4): EnumLabProofMatchPredicate, is_workflow_only_glob(), is_workflow_path(), ModelLabProofMatch

### Community 1989 - "ModelLabProofProfileRegistry"
Cohesion: 0.11
Nodes (3): ModelLabProofExcludedRepo, ModelLabProofProfile, ModelLabProofProfileRegistry

### Community 1991 - "dogfood_delegation_fault_routes.py"
Cohesion: 0.18
Nodes (7): _guard(), is_dogfood_fault_pin(), load_dogfood_delegation_fault_routes(), _load_lane_document(), ModelDogfoodDelegationFaultRoute, resolve_dogfood_delegation_fault_route(), validate_dogfood_delegation_fault_request()

### Community 1993 - "check_workflow_uv_available.py"
Cohesion: 0.20
Nodes (9): installs_uv(), invokes_uv(), main(), runs_on_hosted(), scan_document(), scan_paths(), _uncommented(), Violation (+1 more)

### Community 1994 - "compute_golden.py"
Cohesion: 0.19
Nodes (8): apply_mask(), _canonical(), compare_output(), diff_paths(), _JsonDumpable, outputs_equivalent(), record_golden(), _remove_path()

### Community 1995 - "runner_image_identity.py"
Cohesion: 0.22
Nodes (9): compute_identity(), compute_shared_env_digest(), emit_startup_evidence(), format_startup_evidence(), generate_lock(), _load_lock(), main(), _read_manifest_digest() (+1 more)

### Community 1996 - "adjacency"
Cohesion: 0.11
Nodes (19): reverse_deps, adjacency, adapters, backends, dlq, enums, gateways, idempotency (+11 more)

### Community 1997 - "lane_census_refresh_decision.py"
Cohesion: 0.18
Nodes (7): decide_refresh(), Decision, _load(), main(), MalformedCandidateError, _parse_emitted_at(), validate_candidate()

### Community 1998 - "lane_scratch.py"
Cohesion: 0.17
Nodes (9): build_parser(), create_exclusive(), default_root(), label_of(), _log(), main(), mint_name(), mint_path() (+1 more)

### Community 1999 - "RuntimeErrorEmitter"
Cohesion: 0.12
Nodes (6): _classify_runtime_error(), _compute_runtime_fingerprint(), _extract_exception_type(), _extract_log_timestamp(), _extract_logger_family(), RuntimeErrorEmitter

### Community 2000 - "._maybe_warning_alert"
Cohesion: 0.14
Nodes (6): ContainerTailer, flush_batch(), _cooldown_write(), MonitorAlertEmitter, _warning_backoff_seconds(), _warning_issue_label()

### Community 2001 - "sibling_clone_manifest.sh"
Cohesion: 0.16
Nodes (15): err(), log(), ensure_runner_clones.sh script, err(), log(), reconcile_deploy_clones.sh script, resolve_deploy_source_clone_root(), resolve_runtime_deploy() (+7 more)

### Community 2002 - "BaseModel"
Cohesion: 0.11
Nodes (8): ModelCiBusLaneBrokerTopology, ModelCiBusLaneDelegationFaultRoute, ModelCiBusLaneDelegationRoute, ModelCiBusLaneDsnReference, ModelCiBusLaneSecretMapping, ModelCiBusLaneSecretNamespace, ModelCiBusLaneSecretResolver, ModelCiBusLaneSecretSource

### Community 2003 - "validator_healthcheck_semantic_probe.py"
Cohesion: 0.16
Nodes (9): HealthCheck, _command_text(), _command_tokens(), _expected_health_url(), _last_option_value(), probes_runtime_health(), SemanticProbeValidationResult, SemanticProbeViolation (+1 more)

### Community 2004 - "BrokerSession"
Cohesion: 0.13
Nodes (3): EnumDetachAnswer, BrokerSession, BrokerUnavailableError

### Community 2006 - "TopicStartupValidator"
Cohesion: 0.13
Nodes (3): EnumTopicValidationStatus, ModelTopicValidationResult, TopicStartupValidator

### Community 2007 - "ModelWorkspaceValidateCommand"
Cohesion: 0.13
Nodes (6): ModelWorkspaceValidateCommand, ModelWorkspaceValidateResult, _as_mapping(), _coerce_validated(), HandlerWorkspaceValidate, validate_workspace()

### Community 2009 - "ModelLaneCensusRegistry"
Cohesion: 0.12
Nodes (3): ModelLaneCensusRegistry, _short_name(), _strings()

### Community 2011 - "gateway_forwarder"
Cohesion: 0.11
Nodes (19): gateway_forwarder, auto_commit, retry_initial_seconds, retry_max_seconds, semantics, source_offset_acknowledgement, retained_payload_fields, scrubbed_topics (+11 more)

### Community 2012 - "effect_operations"
Cohesion: 0.11
Nodes (19): backoff_strategy, base_delay_ms, enabled, jitter_factor, max_delay_ms, max_retries, retryable_errors, effect_operations (+11 more)

### Community 2013 - "HandlerPostgresTopicUpdate"
Cohesion: 0.12
Nodes (3): HandlerPostgresTopicUpdate, normalize_topic_for_storage(), ModelPayloadUpdateTopic

### Community 2014 - "node_gmail_intent_poller_effect/contract.yaml"
Cohesion: 0.11
Nodes (18): capabilities, description, name, config, max_per_label, contract_name, description, input_model (+10 more)

### Community 2015 - "HandlerKafkaReplay"
Cohesion: 0.13
Nodes (4): HandlerKafkaReplay, ModelReplayBounds, ModelKafkaReplayOutput, ModelKafkaReplayProgress

### Community 2016 - "ModelBackendResult"
Cohesion: 0.11
Nodes (19): definitions, ModelBackendResult, ModelRegistryRequest, ModelRegistryResponse, description, extra, frozen, required (+11 more)

### Community 2017 - "properties"
Cohesion: 0.11
Nodes (19): default, description, nullable, type, default, description, nullable, type (+11 more)

### Community 2019 - "CredentialsWriter"
Cohesion: 0.15
Nodes (6): _assert_mode_0600(), CredentialsWriter, CredentialsWriterError, _deep_merge(), _validate_keys(), write_credentials_file()

### Community 2022 - "util_semver.py"
Cohesion: 0.11
Nodes (5): ModelComputeRegistration, normalize_version(), normalize_version_cached(), validate_semver(), validate_version_lenient()

### Community 2023 - "OverlayFileLoader"
Cohesion: 0.15
Nodes (6): OverlayMergeConflictError, OverlayNotFoundError, OverlayPermissionError, OverlaySchemaInvalidError, UnsupportedOverlayVersionError, OverlayFileLoader

### Community 2027 - "principals"
Cohesion: 0.11
Nodes (19): bypass_rls, login, principals, bypass_rls, login, bypass_rls, login, app_dashboard (+11 more)

### Community 2028 - "omninode_runtime_service"
Cohesion: 0.11
Nodes (19): database_ref, dsn_env, principal, bindings, app_dashboard, omninode_runtime_service, onex_api, tenant_projection (+11 more)

### Community 2029 - "omniintelligence"
Cohesion: 0.11
Nodes (19): service_owned, omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals (+11 more)

### Community 2030 - "principals"
Cohesion: 0.11
Nodes (19): bypass_rls, login, principals, bypass_rls, login, bypass_rls, login, app_dashboard (+11 more)

### Community 2031 - "application"
Cohesion: 0.11
Nodes (19): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, checksum_column, domain_column, relation (+11 more)

### Community 2032 - "principals"
Cohesion: 0.11
Nodes (19): bypass_rls, login, principals, bypass_rls, login, bypass_rls, login, app_dashboard (+11 more)

### Community 2033 - "application"
Cohesion: 0.11
Nodes (19): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, checksum_column, domain_column, relation (+11 more)

### Community 2034 - "pattern_exemptions"
Cohesion: 0.11
Nodes (18): pattern_exemptions, class_pattern, documentation, field_pattern, file_pattern, method_pattern, reason, ticket (+10 more)

### Community 2036 - "runner_routing_policy.yaml"
Cohesion: 0.11
Nodes (17): cloud_canary_runner_variable, expected_json, name, org_scope, repositories, revert_when, hosted_runner_allowlist, path (+9 more)

### Community 2038 - "OMN-10048.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2039 - "OMN-10122.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2040 - "OMN-9620.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2041 - "OMN-9629.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2042 - "OMN-9637.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2043 - "OMN-9646.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2044 - "OMN-9728.yaml"
Cohesion: 0.11
Nodes (17): check_type, check_value, dod_evidence, checks, description, id, source, emergency_bypass (+9 more)

### Community 2045 - "redpanda-scram-user"
Cohesion: 0.14
Nodes (18): driver, options, max-file, max-size, logging, logging, logging, logging (+10 more)

### Community 2046 - "omninode-runtime"
Cohesion: 0.11
Nodes (18): intelligence-migration, migration-gate, redpanda-partition-cap, valkey, condition, condition, container_name, depends_on (+10 more)

### Community 2047 - "redpanda-partition-cap"
Cohesion: 0.11
Nodes (18): omninode-runtime, redpanda, REDPANDA_PROFILE_MEMORY, file, service, condition, condition, container_name (+10 more)

### Community 2048 - "omninode-runtime"
Cohesion: 0.11
Nodes (18): intelligence-migration, migration-gate, redpanda-partition-cap, valkey, condition, condition, container_name, depends_on (+10 more)

### Community 2049 - "omninode-runtime"
Cohesion: 0.11
Nodes (18): intelligence-migration, migration-gate, redpanda-partition-cap, valkey, condition, condition, container_name, depends_on (+10 more)

### Community 2050 - "projection-registration-writer"
Cohesion: 0.16
Nodes (18): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, build, build, command (+10 more)

### Community 2051 - "runtime-effects"
Cohesion: 0.16
Nodes (18): migration-gate, postgres, redpanda, condition, condition, depends_on, depends_on, depends_on (+10 more)

### Community 2052 - "projection-tenant-credentials-writer"
Cohesion: 0.14
Nodes (18): driver, options, max-file, max-size, logging, logging, logging, command (+10 more)

### Community 2053 - "chain-canary"
Cohesion: 0.12
Nodes (17): environment, name, permissions, runs-on, timeout-minutes, concurrency, cancel-in-progress, group (+9 more)

### Community 2054 - "sibling-compat"
Cohesion: 0.11
Nodes (17): concurrency, cancel-in-progress, group, jobs, sibling-compat, name, on, pull_request (+9 more)

### Community 2055 - "env"
Cohesion: 0.11
Nodes (17): concurrency, cancel-in-progress, group, env, CACHE_VERSION, GIT_CONFIG_COUNT, GIT_CONFIG_KEY_0, GIT_CONFIG_VALUE_0 (+9 more)

### Community 2056 - "drill_red_until"
Cohesion: 0.11
Nodes (18): default, description, required, type, drill_red_until, sibling_ref, on, push (+10 more)

### Community 2057 - "env"
Cohesion: 0.11
Nodes (18): CI_OVERLAY_REPO, GH_TOKEN, GITHUB_TOKEN, HEAD_SHA, HOSTILE_REVIEW_IS_FORK, JUDGE_PENDING_PATH, JUDGE_RESULT_PATH, JUDGE_STATUS_PATH (+10 more)

### Community 2058 - "with"
Cohesion: 0.11
Nodes (18): with, app-id, cache-enabled, cache-key-prefix, dest, overlay-repo, owner, path (+10 more)

### Community 2059 - "runner-label-exists-caller.yml"
Cohesion: 0.11
Nodes (17): concurrency, cancel-in-progress, group, jobs, runner-label-exists, types, name, on (+9 more)

### Community 2060 - "service_keycloak_token_validator.py"
Cohesion: 0.19
Nodes (7): _assert_bounded_lifetime(), _assert_exact_audience(), ClaimSet, _integer_timestamp_claim(), _resolve_signing_key(), TokenValidationError, verify_and_decode_claims()

### Community 2063 - "cli_profile.py"
Cohesion: 0.16
Nodes (6): bind_lane_cmd(), _fail(), profile_group(), show_cmd(), _store(), unbind_lane_cmd()

### Community 2064 - "assemble_delegation_cohort_key.py"
Cohesion: 0.25
Nodes (9): assemble(), CohortKeyRefusedError, main(), _object(), _read_object(), _string(), _terminal(), _terminal_fields() (+1 more)

### Community 2065 - "handler_dev_boot_guard.py"
Cohesion: 0.12
Nodes (4): DevBootRecoveryError, DevCompanionLaneFailedError, fold_compose_env(), roll_expected()

### Community 2066 - "EnumComposeMountType"
Cohesion: 0.12
Nodes (4): EnumComposeMountType, ModelRunnerTmpCompose, ModelRunnerTmpContainer, ModelRunnerTmpVolume

### Community 2067 - "HandlerFaultInjectFixture"
Cohesion: 0.13
Nodes (3): HandlerFaultInjectFixture, ModelFaultInjectFixtureCommand, ModelFaultInjectFixtureResult

### Community 2069 - "handler_scope_extract.py"
Cohesion: 0.13
Nodes (3): HandlerScopeExtract, ModelScopeExtractInput, ModelScopeExtracted

### Community 2070 - "db_table_validator.py"
Cohesion: 0.12
Nodes (4): _table_exists(), validate_db_tables(), ModelDbTableValidationWarning, ProtocolDbTableCatalogConnection

### Community 2071 - "hooks"
Cohesion: 0.11
Nodes (17): additional_dependencies, always_run, args, entry, exclude, files, id, language (+9 more)

### Community 2072 - "check_reconciler_privilege.py"
Cohesion: 0.19
Nodes (9): check(), _code(), discover_guarded_scripts(), discover_reconcilers(), discover_scheduled_host_scripts(), host_artifact_map(), logical_lines(), main() (+1 more)

### Community 2073 - "check_probe_placement_drift.py"
Cohesion: 0.16
Nodes (8): _display(), drift(), load_declared(), load_live(), main(), placement_equal(), _placement_labels(), PolicyError

### Community 2074 - "ci_summary_gate.py"
Cohesion: 0.16
Nodes (9): drop_superseded_non_verdicts(), _is_non_verdict_row(), _load_check_runs(), _load_jobs(), _load_workflow_runs(), main(), scoped_to_base(), strict_gates_for_event() (+1 more)

### Community 2075 - "service_suites_skip_guard.yaml"
Cohesion: 0.11
Nodes (17): allowed_optional_skip_patterns, curated_test_paths, missing_skip_patterns, provisioned_by, missing_skip_patterns, provisioned_by, missing_skip_patterns, provisioned_by (+9 more)

### Community 2076 - "lane_lock"
Cohesion: 0.13
Nodes (9): _clear_hold(), _held_projects(), lane_lock(), _lane_lock_module_path(), lane_lock_path(), LaneLockContendedError, _record_hold(), _reentrant_hold() (+1 more)

### Community 2077 - "infra-signature-rerun.sh"
Cohesion: 0.16
Nodes (10): already_rerun(), get_recovery_ts(), log(), record_rerun(), record_runner_recovery(), infra-signature-rerun.sh script, ts_before(), merge_sweep_common.sh script (+2 more)

### Community 2078 - "provision_db_slot.sh"
Cohesion: 0.27
Nodes (17): assert_in_fence(), assert_owned_by_slot(), connection_limit_for(), database_exists(), database_holds_no_lane_data(), derive(), fail(), new_password() (+9 more)

### Community 2079 - "release_drift_monitor.py"
Cohesion: 0.17
Nodes (9): _branch_head(), _deploy_affecting_changes(), DeployTargetConfig, emit_friction(), gather_deploy_facts(), main(), _render_human(), _report_to_dict() (+1 more)

### Community 2080 - "check_migration_rls_policy_atomicity.py"
Cohesion: 0.18
Nodes (9): check(), _enclosing_block(), _enforcement_ends_in_block(), load_superseded(), main(), _relation(), scrub(), _Scrubbed (+1 more)

### Community 2082 - "settings"
Cohesion: 0.11
Nodes (17): settings, broker_address, cli_subcommand_timeout_seconds, compose_command_timeout_seconds, compose_start_timeout_seconds, docker_query_timeout_seconds, health_poll_interval_seconds, health_wait_timeout_seconds (+9 more)

### Community 2083 - "processing.py"
Cohesion: 0.14
Nodes (7): collect_all_ticket_ids(), effectiveness_score_v2(), extract_pr_numbers(), extract_ticket_ids(), is_primary_onex_repo(), parse_runtime_state_timestamp(), validate_deep_dive_ingest()

### Community 2084 - "ModelEvalFinding"
Cohesion: 0.12
Nodes (4): EnumEvalFindingCategory, EnumEvalFindingSeverity, ModelEvalFinding, ModelEvalResult

### Community 2085 - "HandlerBrokerGrantDerive"
Cohesion: 0.13
Nodes (3): HandlerBrokerGrantDerive, ModelBrokerGrantDerivation, ModelBrokerGrant

### Community 2086 - "HandlerGitFixVerifier"
Cohesion: 0.12
Nodes (3): HandlerGitFixVerifier, ModelFixVerifyRequest, ModelFixVerifyResult

### Community 2088 - "fields"
Cohesion: 0.11
Nodes (18): bytes_written, correlation_id, created, created_at, file_path, file_size, found, manifest (+10 more)

### Community 2091 - "ModelCodingAgentFsmState"
Cohesion: 0.17
Nodes (6): EnumCodingAgentIntentKind, ModelCodingAgentFsmState, ModelCodingAgentIntent, delta(), _intent(), _on_invoke_failed()

### Community 2092 - "ModelBifrostLaneBackendBinding"
Cohesion: 0.13
Nodes (4): check_one_serving_backend_per_capability(), validate_overlay_bindings(), ModelBifrostLaneBackendBinding, ModelBifrostLaneOverlay

### Community 2099 - "ModelRegistryNodeView"
Cohesion: 0.12
Nodes (7): ModelRegistryNodeDetailView, ModelRegistryNodeView, _build_capability_details(), _build_node_display_name(), _build_registry_node_detail_view(), _build_registry_node_view(), _build_registry_node_view_payload()

### Community 2100 - "find_defects"
Cohesion: 0.13
Nodes (6): find_defects(), _find_disconnected_subgraphs(), _nodes_by_name(), _resolve_node_for_topic(), ModelContractNode, ModelGraphFinding

### Community 2101 - "tenant_scoped_ingress_schema.py"
Cohesion: 0.24
Nodes (10): AllowlistEntry, collect_published_topics(), _iter_contract_files(), load_allowlist(), main(), _parse_args(), _resolve_producer_scan_root(), TenantScopedIngressFinding (+2 more)

### Community 2102 - "entries"
Cohesion: 0.12
Nodes (16): description, entries, accepted_rationale, finding_codes, line_count, max_handler_path, node, owner_ticket (+8 more)

### Community 2103 - "overrides"
Cohesion: 0.12
Nodes (16): description, infisical_path, machine_class, overrides, KAFKA_BOOTSTRAP_SERVERS, LOG_LEVEL, ONEX_EFFECTS_URL, ONEX_RUNTIME_URL (+8 more)

### Community 2104 - "overrides"
Cohesion: 0.12
Nodes (16): description, infisical_path, machine_class, overrides, KAFKA_BOOTSTRAP_SERVERS, LOG_LEVEL, ONEX_EFFECTS_URL, ONEX_RUNTIME_URL (+8 more)

### Community 2105 - "overrides"
Cohesion: 0.12
Nodes (16): description, infisical_path, machine_class, overrides, KAFKA_BOOTSTRAP_SERVERS, LOG_LEVEL, ONEX_EFFECTS_URL, ONEX_RUNTIME_URL (+8 more)

### Community 2106 - "contract_registry_reducer.yaml"
Cohesion: 0.12
Nodes (16): contract_version, major, minor, patch, fingerprint, handlers, optional, required (+8 more)

### Community 2107 - "node_graph_reducer.yaml"
Cohesion: 0.12
Nodes (16): contract_version, major, minor, patch, fingerprint, handlers, optional, required (+8 more)

### Community 2108 - "runtime_orchestrator.yaml"
Cohesion: 0.12
Nodes (16): contract_version, major, minor, patch, fingerprint, node_type, profile_tags, timing_constraints (+8 more)

### Community 2109 - "integration_testing"
Cohesion: 0.12
Nodes (17): event_ordering, fsm_consistency, resource_cleanup, timing, mock_container, mock_contracts_dir, mock_event_bus, integration_testing (+9 more)

### Community 2110 - "local"
Cohesion: 0.12
Nodes (17): redpanda-local-sasl, consumer-health-projection, omninode-runtime, runtime-effects, tenant-projection-writer, local, description, extra_depends_on (+9 more)

### Community 2111 - "projection-registration-writer"
Cohesion: 0.12
Nodes (17): hard, soft, ulimits, command, container_name, labels, networks, profiles (+9 more)

### Community 2112 - "args"
Cohesion: 0.18
Nodes (17): infra-runtime, BUILD_SOURCE, EXPECTED_BUILD_SOURCE, INFRA_RUNTIME_IMAGE, additional_contexts, args, context, dockerfile (+9 more)

### Community 2113 - "x-lane-overlay-mount"
Cohesion: 0.12
Nodes (16): create_host_path, name, networks, omnibase-infra-network, omnimemory-network, driver, name, driver (+8 more)

### Community 2114 - "volumes"
Cohesion: 0.12
Nodes (17): name, name, name, name, name, name, name, name (+9 more)

### Community 2115 - "environment"
Cohesion: 0.38
Nodes (17): KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKERS, KAFKA_CONSUMER_GROUP, OMNIDASH_ANALYTICS_DB_URL, OMNINODE_INTERNAL_DB_URL, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_LOG_LEVEL (+9 more)

### Community 2116 - "gates"
Cohesion: 0.12
Nodes (16): classification, gates, caller_job, caller_workflow, cross_repo_ref, job_path, mode, name (+8 more)

### Community 2117 - "call-reject-skip.yml"
Cohesion: 0.12
Nodes (16): secrets, uses, concurrency, cancel-in-progress, group, jobs, call-reject-skip-token, name (+8 more)

### Community 2118 - "with"
Cohesion: 0.12
Nodes (17): with, cache-dependency-glob, deploy-wait-seconds, enable-cache, fetch-depth, if-no-files-found, lanes-json, lanes-var (+9 more)

### Community 2119 - "env"
Cohesion: 0.12
Nodes (17): EVENT_NAME, FREEZE_BASE_REF, GH_TOKEN, OSV_SCANNER_SHA256, OSV_SCANNER_URL, no-plugin-daemon-classes-gate, pin-reachability, name (+9 more)

### Community 2120 - "deploy-gate"
Cohesion: 0.12
Nodes (16): concurrency, cancel-in-progress, group, if, secrets, uses, with, jobs (+8 more)

### Community 2121 - "inputs"
Cohesion: 0.12
Nodes (17): default, description, required, default, description, required, default, description (+9 more)

### Community 2122 - "env"
Cohesion: 0.12
Nodes (17): env, CACHE_VERSION, INTEGRATION_POSTGRES_DB, INTEGRATION_POSTGRES_HOST, INTEGRATION_POSTGRES_PASSWORD, INTEGRATION_POSTGRES_USER, LAB_LANES_JSON, LLM_ENDPOINT_CIDR_ALLOWLIST (+9 more)

### Community 2123 - "with"
Cohesion: 0.12
Nodes (17): with, app-id, if-no-files-found, name, owner, path, permission-actions, permission-administration (+9 more)

### Community 2124 - "required-check-skip-guard-caller.yml"
Cohesion: 0.12
Nodes (16): concurrency, cancel-in-progress, group, jobs, required-check-skip-guard, types, name, on (+8 more)

### Community 2125 - "env"
Cohesion: 0.12
Nodes (17): DEV_LANE_CLASS_STORE_ADDR, DEV_LANE_COMPOSE_FILE, GITHUB_TOKEN, INFRA_COMPOSE_FILE, INTEGRATION_POSTGRES_DB, INTEGRATION_POSTGRES_HOST, INTEGRATION_POSTGRES_PORT, INTEGRATION_POSTGRES_USER (+9 more)

### Community 2127 - "delegate_caller.py"
Cohesion: 0.18
Nodes (8): _as_session(), _derived(), _fallback_lane(), _is_lane(), _registry_record(), resolve_delegate_caller(), _slug(), _with_skipped()

### Community 2129 - "onboarding/loader.py"
Cohesion: 0.17
Nodes (6): GraphValidationError, load_canonical_graph(), load_graph_from_path(), _validate_dag(), _dfs(), ModelOnboardingGraph

### Community 2131 - "check_runner_fleet_image_drift.py"
Cohesion: 0.18
Nodes (8): _discover_container_names(), discover_observed_versions(), find_stale_containers(), _load_expected_version(), main(), ModelDriftFinding, _read_baked_version(), render_report()

### Community 2132 - "check_workflow_input_default_types.py"
Cohesion: 0.19
Nodes (7): _check_default(), main(), scan_document(), scan_paths(), _triggers(), Violation, workflow_files()

### Community 2133 - "dependabot_lock_refresh.py"
Cohesion: 0.22
Nodes (8): get_pr(), _gh_json(), is_same_repo_dependabot_branch(), list_dependabot_prs(), main(), push_refresh(), regenerate_for_ref(), _run()

### Community 2134 - "check_migration_applied_on_lane.py"
Cohesion: 0.21
Nodes (9): build_parser(), declared_checksum(), file_checksum(), main(), _parse_psql_exec(), probe(), ProbeError, ProbeResult (+1 more)

### Community 2136 - "lane_lock.sh"
Cohesion: 0.19
Nodes (15): lane_lock_acquire(), lane_lock_is_held(), lane_lock_log(), lane_lock_python_bin(), lane_lock_release(), lane_lock.sh script, cleanup(), fail() (+7 more)

### Community 2137 - "lane_rollback_decision.py"
Cohesion: 0.20
Nodes (7): decide_lane_rollback(), _flag(), health_dimension_failures(), LaneRollbackDecision, main(), _names(), provenance_dimension_failures()

### Community 2138 - "integration_config.yaml"
Cohesion: 0.12
Nodes (16): contract_version, major, minor, patch, dependencies, description, name, required (+8 more)

### Community 2139 - "_git_sha_at"
Cohesion: 0.15
Nodes (8): _current_git_sha(), _git_sha_at(), _identity_pairs(), _image_identity_build_args(), _placeholder_keys(), runtime_image_build_args(), _runtime_version(), _runtime_version_at()

### Community 2140 - "EnumBifrostLaneMediaCapability"
Cohesion: 0.13
Nodes (3): EnumBifrostLaneMediaCapability, check_media_capability_endpoint_shape(), validate_binding_endpoint()

### Community 2141 - "EnumContractType"
Cohesion: 0.12
Nodes (3): EnumContractType, ModelCapabilityFields, validate_contract_type_value()

### Community 2142 - "HandlerCILiveContact"
Cohesion: 0.14
Nodes (5): HandlerCILiveContact, git(), read(), ModelCILiveContactRequest, ModelCILiveContactResult

### Community 2143 - "ModelDeclaredStateReconcileRequest"
Cohesion: 0.12
Nodes (4): EnumDeclaredStateReconcileMode, ModelDeclaredStateApplyPolicy, ModelDeclaredStateReconcileRequest, ModelDeclaredStateReconcileResult

### Community 2144 - "HandlerDevBootGuard"
Cohesion: 0.17
Nodes (7): _announce_start(), HandlerDevBootGuard, _load_roll_events(), _roll_events_module(), runtime_started_at(), ModelDevBootRequest, ModelDevBootResult

### Community 2145 - "ModelWorkspaceConfigRootRequest"
Cohesion: 0.13
Nodes (3): HandlerWorkspaceConfigRootResolve, ModelWorkspaceConfigRootRequest, ModelWorkspaceConfigRootResolution

### Community 2148 - "RoutingGate"
Cohesion: 0.13
Nodes (3): ProtocolRoutingClassifier, RoutingGate, TypedDictRoutingAudit

### Community 2152 - "properties"
Cohesion: 0.12
Nodes (17): items, type, $ref, properties, description, type, passed, summary (+9 more)

### Community 2155 - "service_gateway_egress_health.py"
Cohesion: 0.18
Nodes (6): ModelGatewayEgressHealth, evaluate_egress_health(), load_egress_health(), publish_egress_health(), record_delivery(), record_denial()

### Community 2156 - "HandlerCIRunnerRoute"
Cohesion: 0.24
Nodes (5): HandlerCIRunnerRoute, is_fleet(), parse_labels(), ModelCIRunnerRouteDecision, ModelCIRunnerRouteRequest

### Community 2158 - "effect"
Cohesion: 0.12
Nodes (17): effect, description, execution_mode, operation_name, operation_timeout_ms, operation_version, retry_policy, major (+9 more)

### Community 2159 - "max_per_label"
Cohesion: 0.12
Nodes (17): description, type, default, description, ge, le, type, properties (+9 more)

### Community 2160 - "node_llm_inference_effect/contract.yaml"
Cohesion: 0.12
Nodes (16): capabilities, description, name, contract_name, description, event_bus, publish_topics, subscribe_topics (+8 more)

### Community 2161 - "EnumTaskType"
Cohesion: 0.13
Nodes (4): EnumTaskType, ModelRoutingConstraints, ModelRoutingRequest, ModelRouteRequestBody

### Community 2162 - "circuit_breaker"
Cohesion: 0.12
Nodes (17): check_type, failure_threshold, name, reset_timeout_ms, timeout_ms, backends, enabled, failure_threshold (+9 more)

### Community 2163 - "HandlerRsdScoreCalculate"
Cohesion: 0.15
Nodes (4): HandlerRsdScoreCalculate, ModelRsdFactorScore, ModelRsdScoreResult, ModelRsdTicketScore

### Community 2164 - "integration_testing"
Cohesion: 0.12
Nodes (17): event_ordering, fsm_consistency, resource_cleanup, timing, mock_container, mock_contracts_dir, mock_event_bus, integration_testing (+9 more)

### Community 2165 - "event_bus"
Cohesion: 0.18
Nodes (17): event_bus, plugin_managed, publish_topics, subscribe_topics, version, handlers, optional, required (+9 more)

### Community 2166 - "workflow_coordination"
Cohesion: 0.12
Nodes (17): graceful_shutdown_timeout_ms, health_check_interval_ms, shutdown_sequence, startup_sequence, description, payload_schema, topic, major (+9 more)

### Community 2169 - "condition_evaluator.py"
Cohesion: 0.20
Nodes (9): ConditionEvaluationError, evaluate_condition(), _evaluate_single(), _is_quoted(), _parse_inline_list(), _resolve_key(), _resolve_lhs(), _split_and_clauses() (+1 more)

### Community 2170 - "BindingExpressionParseError"
Cohesion: 0.12
Nodes (3): BindingExpressionParseError, BindingExpressionParser, EnumBindingParseErrorCode

### Community 2171 - "ModelConfigProvenance"
Cohesion: 0.13
Nodes (4): build_config_provenance(), compute_sha256(), ModelConfigProvenance, _emit_config_provenance()

### Community 2173 - "venv_purity.py"
Cohesion: 0.19
Nodes (7): assert_venv_purity(), _canonicalize(), _declared_package_names(), find_undeclared_onex_providers(), _locate_uv_lock(), UndeclaredProvider, VenvPurityError

### Community 2178 - "UUID"
Cohesion: 0.13
Nodes (3): CorrelationContext, get_correlation_id(), set_correlation_id()

### Community 2179 - "Added"
Cohesion: 0.12
Nodes (16): [0.8.0] - 2026-02-19, Added, Baselines and Effectiveness Metrics, Build Tooling, Changed, CI/CD, Demo and Test Tooling, Dependencies (+8 more)

### Community 2180 - "compose-dev"
Cohesion: 0.15
Nodes (15): compose_file, observed_boot_seconds, readiness_services, settle_budget_seconds, compose_file, observed_boot_seconds, readiness_services, settle_budget_seconds (+7 more)

### Community 2181 - "repository_overrides"
Cohesion: 0.12
Nodes (16): expected_json, revert_when, expected_json, revert_when, expected_json, revert_when, expected_json, revert_when (+8 more)

### Community 2182 - "key_metrics"
Cohesion: 0.12
Nodes (16): critical, warning, action, condition, description, name, buckets, description (+8 more)

### Community 2183 - "metadata"
Cohesion: 0.12
Nodes (16): cache_compiled_contracts, contracts_dir, dependency_resolution, enable_hot_reload, event_bus_type, exclude_patterns, max_contracts, max_nodes (+8 more)

### Community 2184 - "volumes"
Cohesion: 0.23
Nodes (16): create_host_path, container_name, ports, volumes, volumes, container_name, volumes, omninode-runtime (+8 more)

### Community 2185 - "options"
Cohesion: 0.22
Nodes (16): logging, logging, logging, logging, driver, options, logging, logging (+8 more)

### Community 2186 - "options"
Cohesion: 0.20
Nodes (16): logging, logging, logging, driver, options, logging, logging, max-file (+8 more)

### Community 2187 - "environment"
Cohesion: 0.22
Nodes (16): CHAIN_CANARY_READER_PASSWORD, MIGRATIONS_DIR, NODE_POSTGRES_DB, OMNINODE_RUNTIME_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT (+8 more)

### Community 2188 - "environment"
Cohesion: 0.22
Nodes (16): MIGRATIONS_DIR, NODE_POSTGRES_DB, OMNINODE_RUNTIME_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER (+8 more)

### Community 2189 - "runtime-effects"
Cohesion: 0.12
Nodes (16): migration-gate, replicas, condition, container_name, depends_on, image, labels, ports (+8 more)

### Community 2190 - "services"
Cohesion: 0.12
Nodes (16): container_name, container_name, container_name, container_name, ports, container_name, image, services (+8 more)

### Community 2191 - "env"
Cohesion: 0.12
Nodes (16): AWS_CLI_CONNECT_TIMEOUT, AWS_CLI_READ_TIMEOUT, CANDIDATE_IMAGE, CONSUMER_LOCK, DEPLOY_REF, DEPLOY_SIBLING_FALLBACK_REF, DEV_TAG_WAIT_ATTEMPTS, DEV_TAG_WAIT_SECONDS (+8 more)

### Community 2192 - "jobs"
Cohesion: 0.12
Nodes (16): jobs, lint, migration-freeze, onex-validation, zone-filter, name, needs, runs-on (+8 more)

### Community 2193 - "with"
Cohesion: 0.12
Nodes (16): with, app-id, cache-dependency-glob, enable-cache, fetch-depth, fetch-tags, owner, path (+8 more)

### Community 2194 - "with"
Cohesion: 0.12
Nodes (16): with, app-id, fetch-depth, filter, if-no-files-found, name, owner, path (+8 more)

### Community 2195 - "imperative-contract-guard.yml"
Cohesion: 0.12
Nodes (15): concurrency, cancel-in-progress, group, uses, with, jobs, imperative-contract-guard, name (+7 more)

### Community 2200 - "merge_provenance.py"
Cohesion: 0.15
Nodes (3): evaluate(), evaluate_and_write(), main()

### Community 2203 - "check_version_pins.py"
Cohesion: 0.18
Nodes (7): check_compliance(), extract_dependencies(), load_matrix(), main(), parse_requirement(), parse_version(), parse_yaml_simple()

### Community 2204 - "check_auto_merge_hold.py"
Cohesion: 0.23
Nodes (8): _event_sort_key(), latest_auto_merge_decision(), _load_json_list(), _load_labels(), _load_timeline(), main(), _normalize_label(), should_hold()

### Community 2205 - "main"
Cohesion: 0.18
Nodes (10): accuracy(), main(), print_report(), _read_records(), _request_fields(), run_pending(), run_read_request(), run_write_request() (+2 more)

### Community 2206 - "_Resolver"
Cohesion: 0.18
Nodes (5): _compare_url(), Resolution, _Resolver, status_is_reachable(), _resolve_ref_from_dev()

### Community 2207 - "merge_queue_janitor.py"
Cohesion: 0.24
Nodes (11): cancel_run(), CancellationCandidate, determine_cancellations(), fetch_active_runs(), fetch_live_queue_refs(), _is_live_queue_ref(), main(), _run_gh_json() (+3 more)

### Community 2208 - "runtime_boot_suites_skip_guard.yaml"
Cohesion: 0.12
Nodes (15): allowed_optional_skip_patterns, curated_test_paths, missing_skip_patterns, provisioned_by, missing_skip_patterns, provisioned_by, require_each_curated_module, require_executed_min (+7 more)

### Community 2209 - "credential_loader.py"
Cohesion: 0.18
Nodes (6): _check_permissions(), _context(), CredentialFileFormatError, CredentialFilePermissionError, load_worker_credential(), ModelWorkerCredential

### Community 2210 - "verify_dev_refresh.py"
Cohesion: 0.23
Nodes (11): HealthDimension, check_cluster_health(), check_service_digest(), get_container_state(), get_image_id(), get_revision_label(), main(), _revisions_match() (+3 more)

### Community 2212 - "artifact_reconcile_cmd"
Cohesion: 0.14
Nodes (5): artifact_reconcile_cmd(), _get_bootstrap_servers(), _publish_command(), _resolve_actor(), EnumArtifactTopic

### Community 2214 - "scan.py"
Cohesion: 0.23
Nodes (8): compute_drift(), find_git_repos_direct_children(), get_active_worktrees(), get_branch(), get_commit_entries(), get_dirty(), get_merge_entries(), run_git()

### Community 2217 - "EnumKafkaPermanentFailure"
Cohesion: 0.17
Nodes (4): EnumKafkaPermanentFailure, classify_permanent_kafka_failure(), _declaration(), kafka_failure_types()

### Community 2220 - "HandlerRuntimeTargetCollect"
Cohesion: 0.14
Nodes (3): ModelRRHRuntimeTarget, HandlerRuntimeTargetCollect, ModelRuntimeTargetCollectRequest

### Community 2223 - "node_board_probe_effect/contract.yaml"
Cohesion: 0.12
Nodes (15): capabilities, description, name, consumer_flow_endpoints, projection_api_base_url, consumer_flow_execution, node_timeout_seconds, process_timeout_seconds (+7 more)

### Community 2226 - "HandlerEventForward"
Cohesion: 0.14
Nodes (3): HandlerEventForward, ModelEventForwardRequest, ModelEventForwardResult

### Community 2227 - "retry_policy"
Cohesion: 0.12
Nodes (16): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, retry_policy, description (+8 more)

### Community 2228 - "input_fields"
Cohesion: 0.12
Nodes (16): correlation_id, end_time, envelope_id, event_key, event_timestamp, event_type, event_value, kafka_offset (+8 more)

### Community 2229 - "handlers"
Cohesion: 0.12
Nodes (16): module, name, handler_type, module, name, handler_routing, execution_mode, handlers (+8 more)

### Community 2230 - "retry_policy"
Cohesion: 0.12
Nodes (16): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, retry_policy, description (+8 more)

### Community 2231 - "retry_policy"
Cohesion: 0.12
Nodes (16): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, retry_policy, description (+8 more)

### Community 2232 - "properties"
Cohesion: 0.12
Nodes (16): default, description, nullable, type, default_factory, description, type, default_factory (+8 more)

### Community 2233 - "artifacts"
Cohesion: 0.12
Nodes (15): artifacts, artifact_id, artifact_type, owner_hint, path, repo, source_triggers, tags (+7 more)

### Community 2234 - "projection_dispatch_ledger.py"
Cohesion: 0.15
Nodes (7): dispatch_skipped_entries(), dispatch_skipped_projections(), _entry_key(), projections_with_no_live_dispatcher(), record_dispatch_skipped_projection(), record_live_projection_dispatch(), reset_dispatch_skipped_projections()

### Community 2237 - "first_party_sources.yaml"
Cohesion: 0.12
Nodes (15): build_contexts, name, requires_declared_pr, target_accepts, default_context, open_pr_context, open_pr_lanes, packages (+7 more)

### Community 2239 - "model_bifrost_lane_media_capability_error.py"
Cohesion: 0.18
Nodes (6): BifrostLaneMediaCapabilityAmbiguousError, BifrostLaneMediaCapabilityError, BifrostLaneMediaCapabilityNotServingError, BifrostLaneMediaCapabilityShapeError, BifrostLaneMediaCapabilityUnknownError, resolve_media_capability()

### Community 2240 - "contract_loader_effect.yaml"
Cohesion: 0.13
Nodes (14): contract_version, major, minor, patch, fingerprint, metadata, author, description (+6 more)

### Community 2241 - "recommended_actions"
Cohesion: 0.13
Nodes (15): condition, event, metric, action, level, trigger, description, detection (+7 more)

### Community 2242 - "effect"
Cohesion: 0.13
Nodes (15): enabled, failure_threshold, half_open_requests, success_threshold, timeout_ms, effect, circuit_breaker, description (+7 more)

### Community 2243 - "nodes"
Cohesion: 0.13
Nodes (15): description, edge_type, from_node, to_node, edges, nodes, contract_ref, depends_on (+7 more)

### Community 2244 - "scenarios"
Cohesion: 0.13
Nodes (15): contract_registry, node_graph, scenarios, description, expected_error_code, expected_events, expected_fsm_states, name (+7 more)

### Community 2245 - "validations"
Cohesion: 0.13
Nodes (15): backoff_strategy, base_delay_ms, jitter_factor, max_delay_ms, max_retries, validations, calculation, contract_ref (+7 more)

### Community 2246 - "bundles.yaml"
Cohesion: 0.13
Nodes (14): auth, description, includes, inject_required_env, services, observability, description, includes (+6 more)

### Community 2247 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2248 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2249 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2250 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2251 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2252 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2253 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2254 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2255 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2256 - "hardcoded_env"
Cohesion: 0.13
Nodes (15): hardcoded_env, BUS_ID, CONTRACT_RESOLVER_PORT, KAFKA_BOOTSTRAP_SERVERS, KAFKA_BROKER_ALLOWLIST, KEYCLOAK_ADMIN_URL, OMNIBASE_INFRA_DB_URL, OMNIINTELLIGENCE_DB_URL (+7 more)

### Community 2257 - "projection-claude-hook-events-writer.yaml"
Cohesion: 0.13
Nodes (14): container_name, depends_on, condition, service, description, image, layer, name (+6 more)

### Community 2258 - "environment"
Cohesion: 0.13
Nodes (14): REDPANDA_CONTROL_PLANE_TOPIC, REDPANDA_GROUP_OFFSET_RETENTION_SEC, REDPANDA_LOG_SEGMENT_MS, REDPANDA_PARTITIONS_PER_SHARD, REDPANDA_PROFILE_MEMORY, REDPANDA_RETENTION_BYTES, REDPANDA_SEGMENT_FALLOCATION_STEP, command (+6 more)

### Community 2259 - "docker-compose.dev-105.yml"
Cohesion: 0.13
Nodes (14): advertised_kafka_addr, advertised_pandaproxy_addr, name, networks, omnibase-infra-network, driver, name, x-lane-overlay-mount (+6 more)

### Community 2260 - "docker-compose.dev-202.yml"
Cohesion: 0.13
Nodes (14): advertised_kafka_addr, advertised_pandaproxy_addr, name, networks, omnibase-infra-network, driver, name, x-lane-overlay-mount (+6 more)

### Community 2261 - "redpanda-partition-cap"
Cohesion: 0.13
Nodes (15): redpanda, REDPANDA_PROFILE_MEMORY, file, service, condition, container_name, depends_on, entrypoint (+7 more)

### Community 2262 - "volumes"
Cohesion: 0.13
Nodes (15): name, name, name, name, name, name, name, volumes (+7 more)

### Community 2263 - "volumes"
Cohesion: 0.13
Nodes (15): name, name, name, name, name, name, name, volumes (+7 more)

### Community 2264 - "environment"
Cohesion: 0.47
Nodes (15): ACTIONS_RUNNER_HOOK_JOB_STARTED, GITHUB_ORG_URL, RUNNER_GROUP, RUNNER_LABELS, RUNNER_NAME, RUNNER_TOKEN, UV_CONCURRENT_BUILDS, UV_CONCURRENT_DOWNLOADS (+7 more)

### Community 2265 - "interval"
Cohesion: 0.48
Nodes (15): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+7 more)

### Community 2266 - "bind"
Cohesion: 0.48
Nodes (15): volumes, volumes, volumes, volumes, volumes, volumes, volumes, volumes (+7 more)

### Community 2267 - "volumes"
Cohesion: 0.13
Nodes (15): name, name, name, name, name, name, name, volumes (+7 more)

### Community 2268 - "runners/entrypoint.sh"
Cohesion: 0.29
Nodes (13): _as_runner(), _cache_key(), _clear_cached_creds(), _deregister(), _fix_docker_socket_gid(), _is_registration_error(), _listener_heartbeat_stale(), _reap_orphaned_listeners() (+5 more)

### Community 2269 - "with"
Cohesion: 0.13
Nodes (15): with, cache-enabled, cache-version, fetch-depth, fetch-tags, files, path, persist-credentials (+7 more)

### Community 2270 - "with"
Cohesion: 0.13
Nodes (15): with, app-id, fetch-depth, owner, path, permission-contents, permission-pull-requests, private-key (+7 more)

### Community 2271 - "with"
Cohesion: 0.13
Nodes (15): with, app-id, cache-enabled, cache-key-prefix, dest, overlay-repo, owner, permission-checks (+7 more)

### Community 2272 - "with"
Cohesion: 0.13
Nodes (15): with, app-id, cache-enabled, cache-version, fetch-depth, owner, permission-contents, permission-pull-requests (+7 more)

### Community 2273 - "inputs"
Cohesion: 0.13
Nodes (15): description, required, type, default, description, required, type, base_branch (+7 more)

### Community 2274 - "outputs"
Cohesion: 0.16
Nodes (15): description, value, description, value, workflow_call, decision, labels, reason (+7 more)

### Community 2275 - "env"
Cohesion: 0.13
Nodes (15): CROSS_REPO_PAT, DRY_RUN, EVENT_NAME, FORCE, GH_TOKEN, GITHUB_TOKEN, HEAD_REPO, LAB_RECORD (+7 more)

### Community 2276 - "verify-sibling-converged"
Cohesion: 0.13
Nodes (14): name, needs, runs-on, timeout-minutes, jobs, deliver-sibling-candidate, verify-sibling-converged, name (+6 more)

### Community 2277 - "scrape_configs"
Cohesion: 0.14
Nodes (14): alerting, alertmanagers, static_configs, global, evaluation_interval, scrape_interval, scrape_timeout, rule_files (+6 more)

### Community 2278 - "validator_healthcheck_start_period.py"
Cohesion: 0.17
Nodes (5): _healthcheck_polls_migration_completion(), is_migration_completion_gate(), StartPeriodValidationResult, StartPeriodViolation, validate_migration_gate_start_period()

### Community 2280 - "EnumProjectionOrderingDirection"
Cohesion: 0.15
Nodes (3): EnumProjectionOrderingDirection, get_projection_ordering_contract(), ModelProjectionOrderingContract

### Community 2281 - "ModelRunnerFleetSnapshot"
Cohesion: 0.16
Nodes (3): ModelRunnerFleetHealthEvaluateCommand, HandlerRunnerFleetSnapshotComplete, ModelRunnerFleetSnapshot

### Community 2282 - "ModelRunnerFleetHealthVerdict"
Cohesion: 0.16
Nodes (3): ModelRunnerFleetHealthVerdict, HandlerRunnerFleetHealthVerdictComplete, ModelRunnerFleetMaintainCompletedEvent

### Community 2285 - "check_dashboard_topic_parity.py"
Cohesion: 0.24
Nodes (7): check_dashboard_parity(), _default_omni_home(), _extract_topics_block(), _extract_topics_from_tree(), _extract_yaml_topic_names(), main(), ModelDashboardTopicParityPaths

### Community 2286 - "check_lane_census_drift.py"
Cohesion: 0.23
Nodes (7): declared_containers(), evaluate(), Finding, load_manifest(), load_snapshot(), main(), orphan_map_entries()

### Community 2287 - "check_schema_fingerprint.py"
Cohesion: 0.23
Nodes (6): cmd_stamp(), cmd_verify(), compute_migration_fingerprint(), main(), read_artifact(), write_artifact()

### Community 2289 - "SweepExclusion"
Cohesion: 0.15
Nodes (7): active_sweep_exclusions(), load_check_modes(), load_sweep_exclusions(), _parse_exclusion_date(), SweepExclusion, _validate_exclusion_fields(), validate_sweep_exclusions()

### Community 2290 - "loaded_code.py"
Cohesion: 0.17
Nodes (6): loaded_code_sha(), LoadedCodeShaNotRecordedError, LoadedCodeShaUnavailableError, record_loaded_code_sha(), reset_loaded_code_sha(), resolve_clone_sha()

### Community 2291 - "emit_repowise_freshness_receipt.py"
Cohesion: 0.31
Nodes (10): _age_days(), _docs_mode(), emit(), _git_branch(), _git_head(), _load_workspace_repos(), main(), _print_summary() (+2 more)

### Community 2292 - "PostgresErrorEmitter"
Cohesion: 0.15
Nodes (4): _compute_fingerprint(), _normalize_text(), PostgresErrorEmitter, PostgresErrorTailer

### Community 2294 - "_gh_api"
Cohesion: 0.16
Nodes (7): _failed_job_names(), _gh_api(), _git_overrides(), WorkflowRun, _pyproject_at(), _workflow_last_run(), _workflow_last_success()

### Community 2296 - "smoke_delegation.sh"
Cohesion: 0.21
Nodes (14): api_python_env(), check_dev_tenants_control(), compose_container(), db_query(), dev_tenants_control(), pod_for(), smoke_delegation.sh script, SMOKE_ATTRIBUTION (+6 more)

### Community 2299 - "refresh_interval_seconds"
Cohesion: 0.13
Nodes (15): health.check, kafka.consumer, kafka.producer, postgres.storage, refresh_interval_seconds, show_connection_count, show_lag, defaults (+7 more)

### Community 2300 - "filesystem/handler_contract.yaml"
Cohesion: 0.13
Nodes (14): capability_outputs, contract_version, major, minor, patch, description, handler_id, input_model (+6 more)

### Community 2301 - "descriptor"
Cohesion: 0.13
Nodes (15): enabled, failure_threshold, timeout_ms, descriptor, circuit_breaker, concurrency_policy, graph_bolt_uri, idempotent (+7 more)

### Community 2302 - "mcp/handler_contract.yaml"
Cohesion: 0.13
Nodes (14): capability_outputs, contract_version, major, minor, patch, description, handler_id, input_model (+6 more)

### Community 2305 - "CircuitBreakerEventPublisher"
Cohesion: 0.17
Nodes (3): EnumCircuitState, ModelCircuitBreakerStateEvent, CircuitBreakerEventPublisher

### Community 2307 - ".protect"
Cohesion: 0.18
Nodes (8): _all_healthy(), health_probe_reading(), health_probe_summary(), _healthy(), _json_string(), redact_credential_spans(), _unchanged_and_healthy(), ModelDevBootContainer

### Community 2309 - "HandlerReleaseNotesApply"
Cohesion: 0.15
Nodes (3): HandlerReleaseNotesApply, ModelReleaseNotesApplyRequest, ModelReleaseNotesApplyResult

### Community 2314 - "ModelLlmUsage"
Cohesion: 0.15
Nodes (4): to_call_metrics(), to_usage_normalized(), to_usage_raw(), ModelLlmUsage

### Community 2315 - "timeouts"
Cohesion: 0.13
Nodes (14): timeouts, audit_offsets_seconds, audit_poll_seconds, audit_topic_discovery_seconds, close_seconds, git_diff_seconds, linear_graphql_circuit_reset_seconds, linear_graphql_request_seconds (+6 more)

### Community 2318 - "handlers"
Cohesion: 0.15
Nodes (15): module, name, handler_routing, handlers, routing_strategy, description, event_model, input_model (+7 more)

### Community 2319 - "node_coding_agent_orchestrator/contract_topics.py"
Cohesion: 0.21
Nodes (6): contract_allowed_workspace_roots(), contract_publish_topics(), contract_subscribe_topics(), _expand_env_refs(), _load_contract(), _resolve_allowed_roots()

### Community 2320 - "node_contract_persistence_effect/contract.yaml"
Cohesion: 0.13
Nodes (14): capabilities, description, name, contract_name, contract_version, major, minor, patch (+6 more)

### Community 2321 - "gateway_attach"
Cohesion: 0.13
Nodes (15): gateway_attach, circuit_breaker_reset_timeout_seconds, circuit_breaker_threshold, heartbeat_interval_seconds, keycloak_admin_client_ref, keycloak_introspection_ref, keycloak_issuer_ref, keycloak_jwks_ref (+7 more)

### Community 2322 - "step_timeouts"
Cohesion: 0.13
Nodes (15): step_timeouts, build_base_seconds, build_override_seconds, default_seconds, down_seconds, focused_tests_seconds, generate_seconds, git_fetch_seconds (+7 more)

### Community 2323 - "handlers"
Cohesion: 0.13
Nodes (15): module, name, module, name, handler_routing, default_handler, handlers, routing_strategy (+7 more)

### Community 2324 - "HandlerProbeHealth"
Cohesion: 0.17
Nodes (3): HandlerProbeHealth, ModelHealthProbeTarget, ModelHealthSnapshot

### Community 2325 - "nodes"
Cohesion: 0.13
Nodes (15): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph (+7 more)

### Community 2328 - "nodes"
Cohesion: 0.13
Nodes (15): description, edge_type, from_node, to_node, edges, nodes, contract_ref, depends_on (+7 more)

### Community 2329 - "scenarios"
Cohesion: 0.13
Nodes (15): contract_registry, node_graph, scenarios, description, expected_error_code, expected_events, expected_fsm_states, name (+7 more)

### Community 2330 - "metadata"
Cohesion: 0.13
Nodes (15): cache_compiled_contracts, contracts_dir, dependency_resolution, enable_hot_reload, event_bus_type, exclude_patterns, max_contracts, max_nodes (+7 more)

### Community 2332 - "runtime_overlay_revision.py"
Cohesion: 0.23
Nodes (5): apply(), declaration(), _git(), OverlayRevisionError, _repo_root()

### Community 2333 - "node_validation_adjudicator/contract.yaml"
Cohesion: 0.13
Nodes (14): contract_name, description, descriptor, node_archetype, input_model, description, module, name (+6 more)

### Community 2334 - "node_validation_ledger_write_effect/contract.yaml"
Cohesion: 0.13
Nodes (14): capabilities, description, name, version, contract_name, description, descriptor, node_archetype (+6 more)

### Community 2335 - "node_validation_orchestrator/contract.yaml"
Cohesion: 0.13
Nodes (14): consumed_events, description, event_type, topic, contract_name, description, descriptor, node_archetype (+6 more)

### Community 2344 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2345 - "instances/onex-dev.yaml"
Cohesion: 0.13
Nodes (14): active_preset, databases, mode, mode, mode, mode, presets, application-cloud (+6 more)

### Community 2346 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2347 - "instances/onex-prod.yaml"
Cohesion: 0.13
Nodes (14): active_preset, databases, mode, mode, mode, mode, presets, application-cloud (+6 more)

### Community 2348 - "schemas"
Cohesion: 0.13
Nodes (15): domain, owner, schemas, schemas, schemas, domain, owner, domain (+7 more)

### Community 2349 - "OMN-17427.yaml"
Cohesion: 0.14
Nodes (13): id, statement, interface_change, interfaces_touched, is_seam_ticket, requirements, acceptance, id (+5 more)

### Community 2350 - "response_fields"
Cohesion: 0.14
Nodes (14): description, path, response_fields, health_endpoints, error_count, healthy_nodes, last_event_processed, last_health_check (+6 more)

### Community 2351 - "effects"
Cohesion: 0.18
Nodes (14): max_concurrent_contract_attach, memory_limit, priority_contracts, ready_bound_seconds, bifrost_verify_endpoints, boot, capabilities, omnimemory_enabled (+6 more)

### Community 2352 - "omninode-runtime.yaml"
Cohesion: 0.14
Nodes (13): container_name, description, image, layer, name, ports, external, internal (+5 more)

### Community 2353 - "environment"
Cohesion: 0.29
Nodes (14): MIGRATIONS_DIR, NODE_POSTGRES_DB, OMNINODE_RUNTIME_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER (+6 more)

### Community 2354 - "gateway-forwarder"
Cohesion: 0.14
Nodes (14): command, container_name, devices, dns, group_add, image, labels, networks (+6 more)

### Community 2355 - "omnibase-infra-memgraph"
Cohesion: 0.14
Nodes (14): MEMGRAPH_BOLT_PORT, MEMGRAPH_HTTP_PORT, command, container_name, environment, image, labels, mem_limit (+6 more)

### Community 2356 - "redpanda-partition-cap"
Cohesion: 0.14
Nodes (14): omninode-runtime, redpanda, condition, condition, command, container_name, depends_on, entrypoint (+6 more)

### Community 2357 - "options"
Cohesion: 0.24
Nodes (14): logging, logging, driver, options, logging, logging, max-file, max-size (+6 more)

### Community 2358 - "omninode-runtime"
Cohesion: 0.19
Nodes (14): create_host_path, container_name, image, labels, ports, volumes, volumes, omninode-runtime (+6 more)

### Community 2359 - "cloud_bus"
Cohesion: 0.15
Nodes (14): cloud_bus, acl_provisioner_ref, auto_offset_reset, broker_provider_id, cloud_auth_ref, cloud_broker_ref, enable_auto_commit, environment (+6 more)

### Community 2360 - "forwarder"
Cohesion: 0.14
Nodes (14): forwarder, canary_topic_set, dedupe_retention_hours, dedupe_store_path, drain_deadline_seconds, forward_retry_initial_seconds, forward_retry_max_seconds, heartbeat_interval_seconds (+6 more)

### Community 2361 - "steps"
Cohesion: 0.14
Nodes (14): name, permissions, runs-on, steps, timeout-minutes, build-workspace-candidate, contents, id-token (+6 more)

### Community 2362 - "with"
Cohesion: 0.14
Nodes (14): with, cache-dependency-glob, deploy-wait-seconds, enable-cache, fetch-depth, if-no-files-found, lanes-json, name (+6 more)

### Community 2363 - "chain-canary-c16-receipt-identity.yml"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, env, UV_VERSION, jobs, name, on (+5 more)

### Community 2364 - "steps"
Cohesion: 0.15
Nodes (14): if, name, needs, runs-on, steps, timeout-minutes, effect-assertion-gate, lockfile-cve-scan (+6 more)

### Community 2365 - "inputs"
Cohesion: 0.14
Nodes (14): description, required, type, evidence_source, package, ticket, workflow_call, description (+6 more)

### Community 2366 - "run_full_tests"
Cohesion: 0.14
Nodes (14): run_full_tests, on, pull_request, push, workflow_dispatch, branches, paths, branches (+6 more)

### Community 2367 - "evidence-autoclose-sweep.yml"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, env, NODE_VERSION_DEFAULT, PYTHON_VERSION, UV_VERSION, jobs (+5 more)

### Community 2368 - "env"
Cohesion: 0.14
Nodes (14): APP_SLUG, DETAILS_URL, DIGEST, GH_TOKEN, HEAD_SHA, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, KIND (+6 more)

### Community 2369 - "with"
Cohesion: 0.14
Nodes (14): with, app-id, cache-enabled, cache-key-prefix, github-token, install-args, owner, permission-contents (+6 more)

### Community 2370 - "env"
Cohesion: 0.14
Nodes (14): DISPATCH_PACKAGE, DISPATCH_SOURCE_REPO, DISPATCH_TICKET, DISPATCH_VERSION, EVENT_NAME, EXPECTED_VER, GH_TOKEN, MANUAL_PACKAGE (+6 more)

### Community 2371 - "public-repo-hygiene.yml"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, jobs, public-repo-hygiene, name, on, pull_request (+5 more)

### Community 2372 - "env"
Cohesion: 0.14
Nodes (14): APP_ID, APP_PRIVATE_KEY, APP_TOKEN, BRANCH, GH_TOKEN, RELEASE_TAG, RELEASED_VERSION, REPOSITORY (+6 more)

### Community 2373 - "outputs"
Cohesion: 0.14
Nodes (14): trigger-rebuild, correlation_id, git_ref, published, receipt_lane, routed_instance, runtime_lane, sibling_sha (+6 more)

### Community 2377 - "analyze_high_output"
Cohesion: 0.19
Nodes (3): analyze_agent_loop(), analyze_high_output(), sorted_calls()

### Community 2379 - "generate_onboarding_docs.py"
Cohesion: 0.18
Nodes (3): check_docs(), generate_docs(), main()

### Community 2380 - "service_llm_usage_normalizer.py"
Cohesion: 0.21
Nodes (4): _estimate_tokens_from_text(), normalize_llm_usage(), normalize_streaming_usage(), _safe_int()

### Community 2381 - "check_contract_topic_parity.py"
Cohesion: 0.24
Nodes (6): _find_contracts_root(), _load_contract_suffixes(), _load_provisioned_suffixes(), main(), _repo_root(), run_parity_check()

### Community 2382 - "check_deployed_migration_tree_sync.py"
Cohesion: 0.29
Nodes (7): _clone_blob_bytes(), _clone_files_at_ref(), _compare(), _deployed_files(), _git(), main(), _resolve_ref()

### Community 2383 - "check_wheel_content_parity.py"
Cohesion: 0.21
Nodes (5): assert_build_root_is_not_vcs_ignored(), build_wheel(), main(), wheel_package_files(), _declared_resource()

### Community 2384 - "run_compliance_sweep.py"
Cohesion: 0.27
Nodes (9): _check_hardcoded_topics(), _check_logic_in_node(), _check_transport_imports(), _find_handler_files(), _infer_node_name(), main(), parse_args(), _read_lines() (+1 more)

### Community 2386 - "run_worker_loop"
Cohesion: 0.18
Nodes (5): _AttachedSession, _fresh_attach(), run_worker_loop(), SessionTerminatedError, UnsupportedRenewalModeError

### Community 2387 - "main"
Cohesion: 0.16
Nodes (3): find_python_files(), main(), ModelDumpModeDetector

### Community 2388 - "transport"
Cohesion: 0.14
Nodes (14): metadata, author, ticket, transport, cancel_seconds, shutdown_seconds, startup_seconds, default_host (+6 more)

### Community 2392 - "ModelLedgerEventBase"
Cohesion: 0.14
Nodes (4): ModelDbQueryFailed, ModelDbQueryRequested, ModelDbQuerySucceeded, ModelLedgerEventBase

### Community 2396 - "node_boot_catalog_effect/contract.yaml"
Cohesion: 0.14
Nodes (13): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+5 more)

### Community 2397 - "thresholds"
Cohesion: 0.14
Nodes (14): default, description, env_var, performance, thresholds, default, description, env_var (+6 more)

### Community 2399 - "ModelDeclaredChainHop"
Cohesion: 0.22
Nodes (5): assemble_replay_and_verify(), _declared_hop_for(), _envelopes_on(), _replay_one_hop(), ModelDeclaredChainHop

### Community 2400 - "handlers"
Cohesion: 0.14
Nodes (14): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+6 more)

### Community 2401 - "node_fault_inject_fixture_compute/contract.yaml"
Cohesion: 0.14
Nodes (13): capabilities, description, name, contract_name, contract_version, major, minor, patch (+5 more)

### Community 2404 - "handlers"
Cohesion: 0.16
Nodes (14): module, name, handler_routing, handlers, routing_strategy, description, handler, input_model (+6 more)

### Community 2405 - "handlers"
Cohesion: 0.14
Nodes (14): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+6 more)

### Community 2406 - "output_fields"
Cohesion: 0.14
Nodes (14): io_operations, description, idempotent, operation, output_fields, duplicate, entries, has_more (+6 more)

### Community 2407 - "retry_policy"
Cohesion: 0.14
Nodes (14): enabled, per_backend, error_handling, circuit_breaker, error_sanitization, retry_policy, enabled, sanitize_patterns (+6 more)

### Community 2408 - "handlers"
Cohesion: 0.14
Nodes (14): handler_type, module, name, handler_routing, aggregation_strategy, execution_mode, handlers, partial_failure_handling (+6 more)

### Community 2409 - "ModelRoutingCommand"
Cohesion: 0.16
Nodes (3): ModelHealthRequest, HandlerRoutingInitiate, ModelRoutingCommand

### Community 2410 - "thresholds"
Cohesion: 0.14
Nodes (14): default, description, env_var, default, description, env_var, performance, thresholds (+6 more)

### Community 2411 - "node_registration_storage_effect/contract.yaml"
Cohesion: 0.14
Nodes (13): capabilities, description, name, version, contract_name, description, input_model, description (+5 more)

### Community 2413 - "config"
Cohesion: 0.14
Nodes (14): config, cors_allow_credentials, cors_allow_headers, cors_allow_methods, cost_snapshot_topics, default_page_limit, default_widget_mapping_path, max_node_type_filter_fetch (+6 more)

### Community 2415 - "handlers"
Cohesion: 0.14
Nodes (14): module, name, handler_routing, aggregation_strategy, execution_mode, handlers, partial_failure_handling, routing_strategy (+6 more)

### Community 2416 - "ModelRsdState"
Cohesion: 0.14
Nodes (4): ModelRsdResult, ModelRsdRankChange, ModelRsdScoreSnapshot, ModelRsdState

### Community 2417 - "node_savings_estimation_compute/contract.yaml"
Cohesion: 0.14
Nodes (13): capabilities, description, name, contract_name, contract_version, major, minor, patch (+5 more)

### Community 2418 - "node_setup_local_provision_effect/contract.yaml"
Cohesion: 0.14
Nodes (13): capabilities, contract_version, major, minor, patch, description, descriptor, node_archetype (+5 more)

### Community 2419 - "ConfigWriter"
Cohesion: 0.16
Nodes (4): ConfigWriter, ConfigWriterError, _validate_env_pair(), write_env_file()

### Community 2421 - "_merge_lane_overlay"
Cohesion: 0.18
Nodes (8): _added_backend_entry(), _append_dogfood_fault_backends(), _disable_local_backends(), _index_base_backends(), _is_local_backend(), _merge_lane_overlay(), _render_dark(), _routed_local_backend_ids()

### Community 2425 - "llm_registry_validator.py"
Cohesion: 0.21
Nodes (6): Finding, main(), _parse_args(), _str_field(), _validate_model_entry(), validate_registry_yaml()

### Community 2426 - "verification/contract.yaml"
Cohesion: 0.14
Nodes (13): config, probe_settings, contract_version, major, minor, patch, event_bus, publish_topics (+5 more)

### Community 2427 - "replay_and_diff"
Cohesion: 0.19
Nodes (7): canonical_json(), sha256_of(), terminal_event_payload(), _make_capturing_deserializer(), _deserialize(), replay_and_diff(), ReplayCapture

### Community 2428 - "allowed"
Cohesion: 0.15
Nodes (13): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+5 more)

### Community 2429 - "forbidden"
Cohesion: 0.15
Nodes (13): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+5 more)

### Community 2430 - "bandit-security"
Cohesion: 0.15
Nodes (13): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, emits_exempt_reason, excludes (+5 more)

### Community 2431 - "detect-secrets"
Cohesion: 0.15
Nodes (13): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, emits_exempt_reason, excludes (+5 more)

### Community 2432 - "mypy-type-check"
Cohesion: 0.15
Nodes (13): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, emits_exempt_reason, excludes (+5 more)

### Community 2433 - "ruff-format-check"
Cohesion: 0.15
Nodes (13): ruff-format-check, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, emits_exempt_reason (+5 more)

### Community 2434 - "nonrequired_check_alert"
Cohesion: 0.15
Nodes (13): alert_dedup_window_hours, failure_threshold, fleet_repos, heads_observed, scheduled_actions_repos, scheduled_failure_threshold_pct, scheduled_window_days, scheduled_workflow_thresholds (+5 more)

### Community 2435 - "operations"
Cohesion: 0.15
Nodes (13): operations, description, idempotent, operation_name, operation_timeout_ms, retry_policy, backoff_strategy, base_delay_ms (+5 more)

### Community 2436 - "automatic_recovery"
Cohesion: 0.15
Nodes (13): condition, max_attempts, mechanism, trigger, bypasses_retry_limit, description, mechanism, requires_permission (+5 more)

### Community 2437 - "operations"
Cohesion: 0.15
Nodes (13): description, is_destructive, operation_name, operation_type, requires_permission, rollback_transitions, target_state, trigger (+5 more)

### Community 2438 - "event_bus_wiring_effect.yaml"
Cohesion: 0.15
Nodes (12): contract_version, major, minor, patch, fingerprint, metadata, author, description (+4 more)

### Community 2439 - "operations"
Cohesion: 0.15
Nodes (13): description, is_destructive, operation_name, operation_type, requires_permission, rollback_transitions, target_state, trigger (+5 more)

### Community 2440 - "runtime_policy.contract.yaml"
Cohesion: 0.15
Nodes (12): active_runtime_packages, arch_graph_bolt_uri, auxiliary_services_omnimemory_enabled, bifrost_vertex_gemini_endpoint_url, dev_redpanda_mem_limit, dev_redpanda_memory, google_cloud_location, google_cloud_project (+4 more)

### Community 2441 - "keep-list.yaml"
Cohesion: 0.15
Nodes (12): builder_cache_max_size, dangling_min_age_hours, generation_bounded_repos, generation_keep, generation_min_age_hours, generation_tag_patterns, keep_image_repos, keep_image_tags (+4 more)

### Community 2442 - "bindings"
Cohesion: 0.15
Nodes (12): bindings, broker, consumer_group, direction, physical_topic, described_groups, broker, group (+4 more)

### Community 2443 - "local-market"
Cohesion: 0.15
Nodes (13): ONEX_LOCAL_RUNTIME_PACKAGES, BIFROST_CONTRACT_PATH, BIFROST_LANE_OVERLAY_PATH, BIFROST_VERIFY_ENDPOINTS, local-market, description, env_default_overrides, includes (+5 more)

### Community 2444 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2445 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2446 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2447 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2448 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2449 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2450 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2451 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2452 - "omniintelligence"
Cohesion: 0.15
Nodes (13): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+5 more)

### Community 2453 - "models"
Cohesion: 0.15
Nodes (12): models, api_key_env, base_url_env, capabilities, context_window, hardware, health_path, model_key (+4 more)

### Community 2454 - "contract-resolver.yaml"
Cohesion: 0.15
Nodes (12): catalog_env, BIFROST_CONTRACT_PATH, command, container_name, description, image, layer, name (+4 more)

### Community 2455 - "runtime-effects.yaml"
Cohesion: 0.15
Nodes (12): container_name, depends_on, condition, service, description, image, layer, name (+4 more)

### Community 2456 - "ci-bus-sasl-enable"
Cohesion: 0.15
Nodes (13): command, container_name, depends_on, entrypoint, image, labels, networks, restart (+5 more)

### Community 2457 - "redpanda"
Cohesion: 0.15
Nodes (13): hard, soft, command, container_name, image, labels, networks, ports (+5 more)

### Community 2458 - "volumes"
Cohesion: 0.15
Nodes (13): create_host_path, volumes, bind, read_only, source, target, type, x-lane-overlay-mount (+5 more)

### Community 2459 - "redpanda"
Cohesion: 0.15
Nodes (13): hard, soft, command, container_name, image, networks, ports, profiles (+5 more)

### Community 2460 - "infisical"
Cohesion: 0.15
Nodes (13): AUTH_SECRET, DB_CONNECTION_URI, ENCRYPTION_KEY, REDIS_URL, SITE_URL, TELEMETRY_ENABLED, container_name, environment (+5 more)

### Community 2461 - "volumes"
Cohesion: 0.15
Nodes (13): create_host_path, volumes, bind, read_only, source, target, type, x-lane-overlay-mount (+5 more)

### Community 2462 - "redpanda"
Cohesion: 0.15
Nodes (13): hard, soft, command, container_name, image, networks, ports, profiles (+5 more)

### Community 2463 - "volumes"
Cohesion: 0.15
Nodes (13): create_host_path, volumes, bind, read_only, source, target, type, x-lane-overlay-mount (+5 more)

### Community 2464 - "redpanda"
Cohesion: 0.15
Nodes (13): hard, soft, command, container_name, image, networks, ports, profiles (+5 more)

### Community 2465 - "interval"
Cohesion: 0.51
Nodes (13): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+5 more)

### Community 2466 - "interval"
Cohesion: 0.51
Nodes (13): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+5 more)

### Community 2467 - "dogfood_delegation_fault_provider.py"
Cohesion: 0.22
Nodes (7): error_payload(), handler_for(), do_GET(), do_POST(), _write_json(), main(), parse_status()

### Community 2469 - "env"
Cohesion: 0.15
Nodes (13): CACHE_ENABLED, GIT_FETCH_TOKEN, INSTALL_ARGS, OMNI_CI_ENV_INSTALL_ARGS, OMNI_CI_ENV_ROOT, OMNI_CI_REPO, PYTHON_VERSION, SHARED_ENV_ENABLED (+5 more)

### Community 2470 - "advisory-job-gate.yml"
Cohesion: 0.15
Nodes (12): secrets, uses, concurrency, cancel-in-progress, group, jobs, advisory-job-gate, name (+4 more)

### Community 2471 - "env"
Cohesion: 0.15
Nodes (12): concurrency, cancel-in-progress, group, env, AWS_REGION, ECR_REGISTRY, ECR_REPOSITORY, PYTHON_VERSION (+4 more)

### Community 2472 - "inputs"
Cohesion: 0.15
Nodes (13): default, description, required, budget_ms, probe_url, terminal_bootstrap_servers, default, description (+5 more)

### Community 2473 - "with"
Cohesion: 0.15
Nodes (13): with, deploy-wait-seconds, fetch-depth, if-no-files-found, lanes-json, match, name, path (+5 more)

### Community 2474 - "inputs"
Cohesion: 0.15
Nodes (13): default, description, required, default, description, required, default, description (+5 more)

### Community 2475 - "env"
Cohesion: 0.15
Nodes (13): BUDGET_SECONDS, BUS_LANE, CHAIN_CANARY_GATEWAY_API_KEY, GATEWAY_URL_OVERRIDE, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, OVERRIDE_FREE_CLOUD, OVERRIDE_LOCAL (+5 more)

### Community 2476 - "base_url"
Cohesion: 0.15
Nodes (13): default, description, required, default, description, required, base_url, deploy_wait_seconds (+5 more)

### Community 2477 - "steps"
Cohesion: 0.15
Nodes (13): post-integration-note, if, name, runs-on, steps, timeout-minutes, id, if (+5 more)

### Community 2478 - "env"
Cohesion: 0.15
Nodes (13): EVENT_NAME, GH_TOKEN, MQ_BASE_SHA, MQ_HEAD_REF, MQ_HEAD_SHA, PR_BASE_SHA, PR_CHANGED_FILES, PR_HEAD_SHA (+5 more)

### Community 2479 - "env"
Cohesion: 0.15
Nodes (13): APP_ID, APP_PRIVATE_KEY, DRILL_RED_UNTIL, DRILL_TICKET, GH_TOKEN, IMAGE_DIGEST, IMAGE_REF, MIGRATE_IMAGE_REF (+5 more)

### Community 2480 - "jobs"
Cohesion: 0.15
Nodes (12): jobs, lab-load-probe, name, runs-on, timeout-minutes, name, on, schedule (+4 more)

### Community 2481 - "kb-doc-gate.yml"
Cohesion: 0.15
Nodes (12): concurrency, cancel-in-progress, group, jobs, kb-doc-gate, uses, with, name (+4 more)

### Community 2482 - "sync_main_to_tag"
Cohesion: 0.15
Nodes (13): sync_main_to_tag, tag, on, push, workflow_dispatch, tags, default, description (+5 more)

### Community 2483 - "jobs"
Cohesion: 0.15
Nodes (12): concurrency, cancel-in-progress, group, name, needs, runs-on, timeout-minutes, jobs (+4 more)

### Community 2484 - "outputs"
Cohesion: 0.15
Nodes (13): trigger-rebuild, correlation_id, published, receipt_lane, routed_instance, runtime_lane, source_sha, verify_runs_on (+5 more)

### Community 2485 - "precommit_diff_scope.py"
Cohesion: 0.23
Nodes (7): core_pin_moved(), decide(), GitError, _locked_package(), main(), run_git(), Scope

### Community 2487 - "rules"
Cohesion: 0.15
Nodes (12): description, runbook, summary, groups, name, rules, severity, team (+4 more)

### Community 2488 - "contract_capability_extractor.py"
Cohesion: 0.21
Nodes (5): _dependency_name(), _dependency_protocol_name(), _dependency_type_is_protocol(), _dependency_uses_protocol_marker(), _extract_dependency_protocols()

### Community 2492 - "projection_liveness.py"
Cohesion: 0.19
Nodes (4): describe_dlq_saturation(), describe_projection_attachment(), describe_projection_write_path(), _name_list()

### Community 2493 - "handler_lane_entry_point_census.py"
Cohesion: 0.27
Nodes (7): build_census(), _entry(), entry_point_of(), _family_of(), _is_launched(), _mapping(), _words()

### Community 2495 - "_evaluate_alerts"
Cohesion: 0.17
Nodes (3): _evaluate_alerts(), EnumRunnerFleetAlertKind, ModelRunnerFleetAlert

### Community 2501 - "check_context_field_presence.py"
Cohesion: 0.23
Nodes (5): check_contract(), _find_claim_block(), _has_pinning_hash(), _iter_targets(), main()

### Community 2502 - "evaluate_external_sweep"
Cohesion: 0.21
Nodes (7): check_run_workflow_run_id(), draft_ready_check_runs(), evaluate_external_sweep(), latest_check_run_rows(), replacement_run_in_flight(), resolve_check_run_event(), _run_int()

### Community 2503 - "run_on_gate_runner.sh"
Cohesion: 0.27
Nodes (11): die(), poll_receipt(), RC_REFUSED_LOAD, RC_REFUSED_PROBE, RC_REFUSED_SLOT, RC_SETUP, read_receipt(), receipt_field() (+3 more)

### Community 2504 - "default_out_path"
Cohesion: 0.18
Nodes (6): default_out_path(), _is_git_checkout(), _today_local(), base_week_monday(), deep_dive_filename(), month_name()

### Community 2505 - "main"
Cohesion: 0.19
Nodes (6): main(), _now_utc_iso_minutes(), RepoDay, family_key(), repo_commit_sums(), unique_commit_entries()

### Community 2508 - "args"
Cohesion: 0.15
Nodes (12): arg_type, default, name, payload_field, positional, required, skills, args (+4 more)

### Community 2510 - "models"
Cohesion: 0.15
Nodes (12): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, models (+4 more)

### Community 2511 - "compute_cost"
Cohesion: 0.15
Nodes (13): compute_cost, m2_ultra, rtx_4090, rtx_5090, amortization_per_hour, electricity_per_hour, note, amortization_per_hour (+5 more)

### Community 2512 - "widget_mapping.yaml"
Cohesion: 0.15
Nodes (12): show_capabilities, show_name, show_state, show_type, show_version, fallback, defaults, widget_type (+4 more)

### Community 2513 - "capability_mappings"
Cohesion: 0.19
Nodes (13): capability_mappings, compute.transform, effect.grpc, effect.http, defaults, widget_type, show_error_rate, show_latency (+5 more)

### Community 2514 - "graph/handler_contract.yaml"
Cohesion: 0.15
Nodes (12): activation, optional, required_env, description, handler_id, input_model, module, name (+4 more)

### Community 2515 - "bindings"
Cohesion: 0.15
Nodes (13): graph.create_node, graph.create_relationship, graph.execute_query, expression, parameter_name, required, expression, parameter_name (+5 more)

### Community 2516 - "security"
Cohesion: 0.15
Nodes (13): include_correlation_id, never_log, required_events, security_event_level, security, deferred_to, enabled, audit_logging (+5 more)

### Community 2519 - "TopicViolationAlerter"
Cohesion: 0.17
Nodes (3): _best_effort_caller(), DebounceEntry, TopicViolationAlerter

### Community 2524 - "ModelCheckCatalogEntry"
Cohesion: 0.18
Nodes (3): ModelCheckCatalogEntry, _build_registry(), get_check_executor()

### Community 2526 - "ModelArchitectureValidationRequest"
Cohesion: 0.15
Nodes (13): definitions, ModelArchitectureValidationRequest, ModelArchitectureValidationResult, ModelArchitectureViolation, description, required, type, description (+5 more)

### Community 2527 - "validation_rules"
Cohesion: 0.15
Nodes (13): checks, excluded_contexts, patterns, target_classes, target_files, type, validation_rules, description (+5 more)

### Community 2528 - "node_artifact_reconciliation_orchestrator/contract.yaml"
Cohesion: 0.15
Nodes (12): contract_name, description, health_check, enabled, endpoint, interval_seconds, name, node_name (+4 more)

### Community 2529 - "state_transitions"
Cohesion: 0.15
Nodes (13): major, minor, patch, state_transitions, description, error_states, initial_state, state_machine_name (+5 more)

### Community 2530 - "node_ci_runner_route_compute/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, description, name, node_name, node_type (+4 more)

### Community 2531 - "config"
Cohesion: 0.15
Nodes (13): config, capacity_downgrade_reasons, hosted_labels, lab_labels, lab_record_max_age_seconds, max_busy_fraction, max_lab_load_ratio, min_idle_runners (+5 more)

### Community 2532 - "handlers"
Cohesion: 0.15
Nodes (13): description, module, name, module, name, handler_routing, handlers, routing_strategy (+5 more)

### Community 2533 - "version"
Cohesion: 0.23
Nodes (13): version, event_bus, publish_topics, version, handlers, optional, required, version (+5 more)

### Community 2534 - "handlers"
Cohesion: 0.15
Nodes (13): module, name, handler_routing, aggregation_strategy, execution_mode, handlers, partial_failure_handling, routing_strategy (+5 more)

### Community 2535 - "node_contract_registry_reducer/contract.yaml"
Cohesion: 0.15
Nodes (12): contract_version, major, minor, patch, description, input_model, description, module (+4 more)

### Community 2536 - "node_db_error_linear_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, description, input_model, description, module (+4 more)

### Community 2537 - "node_decision_store_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, description, name, node_name, node_type (+4 more)

### Community 2538 - "handlers"
Cohesion: 0.15
Nodes (13): module, name, handler_routing, aggregation_strategy, execution_mode, handlers, partial_failure_handling, routing_strategy (+5 more)

### Community 2539 - "ModelObservedHop"
Cohesion: 0.19
Nodes (5): derive_chain_state(), _derive_graded_chain_state(), _partition_observed(), EnumLedgerChainState, ModelObservedHop

### Community 2540 - "node_delegation_chain_ledger_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): chain_states, description, descriptor, node_archetype, event_bus, subscribe_topics, in_process_terminal_only, terminal_sources (+4 more)

### Community 2541 - "node_delta_bundle_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, contract_version, major, minor, patch (+4 more)

### Community 2542 - "handlers"
Cohesion: 0.17
Nodes (13): module, name, handlers, backend, description, handler, input_model, output_fields (+5 more)

### Community 2543 - "node_delta_metrics_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, contract_version, major, minor, patch (+4 more)

### Community 2544 - "handlers"
Cohesion: 0.17
Nodes (13): module, name, handlers, backend, description, handler, input_model, output_fields (+5 more)

### Community 2546 - "ModelGmailCleanupConfig"
Cohesion: 0.15
Nodes (13): definitions, ModelGmailCleanupConfig, ModelGmailCleanupResult, description, extra, frozen, required, type (+5 more)

### Community 2547 - "ModelGmailIntentPollerConfig"
Cohesion: 0.15
Nodes (13): definitions, ModelGmailIntentPollerConfig, ModelGmailIntentPollerResult, description, extra, frozen, required, type (+5 more)

### Community 2549 - "node_lab_proof_plan_compute/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, config, contract_name, description, descriptor, node_archetype (+4 more)

### Community 2550 - "node_merge_gate_effect/contract.yaml"
Cohesion: 0.15
Nodes (12): capabilities, description, name, contract_name, contract_version, major, minor, patch (+4 more)

### Community 2551 - "operations"
Cohesion: 0.15
Nodes (13): description, is_destructive, operation_name, operation_type, requires_permission, rollback_transitions, target_state, trigger (+5 more)

### Community 2552 - "handlers"
Cohesion: 0.17
Nodes (13): module, name, handler_routing, handlers, routing_strategy, description, handler, input_model (+5 more)

### Community 2553 - "step_config"
Cohesion: 0.15
Nodes (13): step_config, aggregation_strategy, effect_node, event_pattern, event_type, intent_filter, operation, output_type (+5 more)

### Community 2554 - "io_operations"
Cohesion: 0.15
Nodes (13): correlation_id, node_id, query, record, updates, io_operations, description, idempotent (+5 more)

### Community 2555 - "ModelRsdScoreInput"
Cohesion: 0.15
Nodes (4): ModelAgentRequestData, _calculate_agent_utility(), ModelRsdFactorWeights, ModelRsdScoreInput

### Community 2556 - "ModelDiscoveryResult"
Cohesion: 0.15
Nodes (13): ModelDiscoveryResult, $ref, description, properties, required, type, query_metadata, services (+5 more)

### Community 2557 - "handlers"
Cohesion: 0.17
Nodes (13): module, name, handler_routing, handlers, routing_strategy, description, handler, input_model (+5 more)

### Community 2558 - "ModelSlackAlert"
Cohesion: 0.15
Nodes (13): definitions, ModelSlackAlert, ModelSlackAlertResult, description, extra, frozen, required, type (+5 more)

### Community 2559 - "handlers"
Cohesion: 0.15
Nodes (13): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+5 more)

### Community 2560 - "handlers"
Cohesion: 0.17
Nodes (13): module, name, handler_routing, handlers, routing_strategy, description, handler, input_model (+5 more)

### Community 2561 - "handlers"
Cohesion: 0.15
Nodes (13): module, name, module, name, handler_routing, handlers, routing_strategy, description (+5 more)

### Community 2562 - "responses"
Cohesion: 0.18
Nodes (13): next, next, set_state, next, set_state, next, set_state, aws (+5 more)

### Community 2564 - "ModelRuntimeNodeGraph"
Cohesion: 0.15
Nodes (3): ModelNodeConfig, ModelNodeEdge, ModelRuntimeNodeGraph

### Community 2566 - "bindings"
Cohesion: 0.15
Nodes (13): database_ref, dsn_env, principal, bindings, app_dashboard, onex_api, tenant_projection, database_ref (+5 more)

### Community 2567 - "bindings"
Cohesion: 0.15
Nodes (13): database_ref, dsn_env, principal, bindings, app_dashboard, onex_api, tenant_projection, database_ref (+5 more)

### Community 2568 - "required_validators"
Cohesion: 0.17
Nodes (11): known_repos, metadata, generated_by, last_updated, next_review, related_tickets, required_validators, schema_version (+3 more)

### Community 2569 - "banned-compose-env-vars"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2570 - "branch-protection-rollout-guard"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2571 - "enum-governance"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2572 - "handler-di"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2573 - "hardcoded-local-paths"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2574 - "hardcoded-model-config"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2575 - "hardcoded-topic-compute"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2576 - "naming-conventions"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2577 - "no-hardcoded-topics"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2578 - "no-io-outside-effects"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2579 - "no-raw-sqlite3"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2580 - "no-untracked-todos"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2581 - "pydantic-patterns"
Cohesion: 0.17
Nodes (12): applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes, pre_commit (+4 more)

### Community 2582 - "self-gating-workflows"
Cohesion: 0.17
Nodes (12): self-gating-workflows, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes (+4 more)

### Community 2583 - "single-class-per-file"
Cohesion: 0.17
Nodes (12): single-class-per-file, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes (+4 more)

### Community 2584 - "spdx-headers"
Cohesion: 0.17
Nodes (12): spdx-headers, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes (+4 more)

### Community 2585 - "stub-implementations"
Cohesion: 0.17
Nodes (12): stub-implementations, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes (+4 more)

### Community 2586 - "tenant-scoped-ingress"
Cohesion: 0.17
Nodes (12): tenant-scoped-ingress, applies_to_repos, central_source, ci_workflow, ci_workflow_keywords, description, emits, excludes (+4 more)

### Community 2587 - "probes"
Cohesion: 0.17
Nodes (11): probes, cron, id, lane, lane_source, max_duration_minutes, name, reads (+3 more)

### Community 2588 - "executions"
Cohesion: 0.17
Nodes (11): executions, database, down_script, down_sha256, evidence, executed_at, forward_sha256, migration (+3 more)

### Community 2589 - "version"
Cohesion: 0.27
Nodes (12): version, handlers, optional, required, version, type, version, type (+4 more)

### Community 2590 - "extract_fields"
Cohesion: 0.17
Nodes (12): content, discovered_files, file_count, file_path, file_size, modified_at, scan_timestamp, response_handling (+4 more)

### Community 2591 - "state_transitions"
Cohesion: 0.17
Nodes (12): major, minor, patch, state_transitions, description, error_states, initial_state, persistence_enabled (+4 more)

### Community 2592 - "io_config"
Cohesion: 0.17
Nodes (12): io_config, content-type, x-onex-correlation-id, x-onex-operation, acks, compression, handler_type, headers (+4 more)

### Community 2593 - "version"
Cohesion: 0.27
Nodes (12): version, handlers, optional, required, version, type, version, type (+4 more)

### Community 2594 - "planned_subscribe_topics"
Cohesion: 0.17
Nodes (12): planned_subscribe_topics, class, idempotent, input_schema, method, output_schema, description, handler (+4 more)

### Community 2595 - "states"
Cohesion: 0.17
Nodes (12): icon, idempotency_note, ui_color, states, description, entry_actions, exit_actions, is_recoverable (+4 more)

### Community 2596 - "state_transitions"
Cohesion: 0.17
Nodes (12): major, minor, patch, state_transitions, description, error_states, initial_state, persistence_enabled (+4 more)

### Community 2597 - "version"
Cohesion: 0.27
Nodes (12): handlers, optional, required, version, type, version, type, version (+4 more)

### Community 2598 - "extra_depends_on"
Cohesion: 0.17
Nodes (12): omnimarket-projection-delegation, omnimarket-projection-llm-cost, omnimarket-projection-run-trace, omnimarket-projection-tenant-credentials, projection-api, extra_depends_on, redpanda-local-sasl, redpanda-local-sasl (+4 more)

### Community 2599 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2600 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2601 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2602 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2603 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2604 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2605 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2606 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2607 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, application, login, login (+4 more)

### Community 2608 - "onex-api.yaml"
Cohesion: 0.17
Nodes (11): catalog_env, GATEWAY_BRIDGED_TENANT_SLUGS, container_name, description, image, layer, name, required_env (+3 more)

### Community 2609 - "services"
Cohesion: 0.17
Nodes (11): driver, name, name, name, networks, ci-bus-network, services, volumes (+3 more)

### Community 2610 - "ci-bus-labwork-user"
Cohesion: 0.17
Nodes (12): command, container_name, depends_on, entrypoint, image, labels, networks, restart (+4 more)

### Community 2611 - "ci-bus-scram-user"
Cohesion: 0.17
Nodes (12): command, container_name, depends_on, entrypoint, image, labels, networks, restart (+4 more)

### Community 2612 - "interval"
Cohesion: 0.53
Nodes (12): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+4 more)

### Community 2613 - "interval"
Cohesion: 0.50
Nodes (12): healthcheck, healthcheck, interval, retries, start_period, test, timeout, healthcheck (+4 more)

### Community 2614 - "services"
Cohesion: 0.17
Nodes (11): name, networks, omnibase-infra-network, driver, name, services, x-healthcheck-defaults, interval (+3 more)

### Community 2615 - "interval"
Cohesion: 0.53
Nodes (12): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+4 more)

### Community 2616 - "environment"
Cohesion: 0.17
Nodes (12): GIT_CONFIG_COUNT, GIT_CONFIG_KEY_0, GIT_CONFIG_KEY_1, GIT_CONFIG_KEY_2, GIT_CONFIG_KEY_3, GIT_CONFIG_VALUE_0, GIT_CONFIG_VALUE_1, GIT_CONFIG_VALUE_2 (+4 more)

### Community 2617 - "hp051-ci-mq-runner-1"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2618 - "hp051-ci-mq-runner-2"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2619 - "hp051-ci-mq-runner-3"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2620 - "hp051-ci-mq-runner-4"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2621 - "hp053-ci-runner-1"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2622 - "hp053-ci-runner-3"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2623 - "hp053-ci-runner-4"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2624 - "hp053-ci-runner-5"
Cohesion: 0.17
Nodes (12): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2625 - "omninode-deploy-runner"
Cohesion: 0.17
Nodes (12): container_name, cpus, entrypoint, extra_hosts, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2626 - "omninode-verify-runner-1"
Cohesion: 0.17
Nodes (12): container_name, cpus, entrypoint, extra_hosts, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2627 - "omninode-verify-runner-2"
Cohesion: 0.17
Nodes (12): container_name, cpus, entrypoint, extra_hosts, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2628 - "omninode-verify-runner-3"
Cohesion: 0.17
Nodes (12): container_name, cpus, entrypoint, extra_hosts, group_add, image, mem_limit, memswap_limit (+4 more)

### Community 2629 - "backends"
Cohesion: 0.17
Nodes (11): backends, backend_id, context_window, endpoint_url, max_tokens, parameter_count, served_model_id, timeout_ms (+3 more)

### Community 2630 - "env"
Cohesion: 0.17
Nodes (12): AWS_CLI_CONNECT_TIMEOUT, AWS_CLI_READ_TIMEOUT, AWS_REGION, BACKOFF_FACTOR, COMMIT_SHA, ECR_REGISTRY, ECR_REPOSITORY, FAIL_ON_MISSING (+4 more)

### Community 2631 - "env"
Cohesion: 0.17
Nodes (11): concurrency, cancel-in-progress, group, env, AWS_REGION, ECR_REGISTRY, ECR_REPOSITORY, PYTHON_VERSION (+3 more)

### Community 2632 - "build-and-push"
Cohesion: 0.17
Nodes (12): name, needs, outputs, permissions, runs-on, timeout-minutes, build-and-push, commit_sha (+4 more)

### Community 2633 - "outputs"
Cohesion: 0.18
Nodes (12): outputs, description, value, description, value, description, value, candidate-tag (+4 more)

### Community 2634 - "steps"
Cohesion: 0.18
Nodes (12): name, needs, runs-on, steps, timeout-minutes, c28-consumer-flow, steps, id (+4 more)

### Community 2635 - "env"
Cohesion: 0.17
Nodes (12): DISPATCH_PR_NUMBER, DISPATCH_REPO, DRY_RUN, EVENT_NAME, GH_TOKEN, INPUT_PR_NUMBER, INPUT_REPO, LINEAR_API_KEY (+4 more)

### Community 2636 - "env"
Cohesion: 0.17
Nodes (12): env, AWS_REGION, BOOT_WAIT_SECONDS, DB_WAIT_SECONDS, KIND_NODE_IMAGE, KIND_VERSION, LAB_DB_SERVICE, LAB_NAMESPACE (+4 more)

### Community 2637 - "compute-matrix"
Cohesion: 0.17
Nodes (11): name, outputs, runs-on, timeout-minutes, jobs, compute-matrix, name, on (+3 more)

### Community 2638 - "with"
Cohesion: 0.17
Nodes (12): with, fetch-depth, if-no-files-found, lanes-json, match, name, path, persist-credentials (+4 more)

### Community 2639 - "jobs"
Cohesion: 0.17
Nodes (11): concurrency, cancel-in-progress, group, if, name, needs, runs-on, timeout-minutes (+3 more)

### Community 2640 - "docs-validate.yml"
Cohesion: 0.17
Nodes (11): uses, concurrency, cancel-in-progress, group, jobs, call, name, on (+3 more)

### Community 2641 - "lockfile-cve-autobump.yml"
Cohesion: 0.17
Nodes (11): concurrency, cancel-in-progress, group, env, CACHE_VERSION, PYTHON_VERSION, UV_VERSION, jobs (+3 more)

### Community 2642 - "with"
Cohesion: 0.26
Nodes (12): steps, name, run, uses, with, steps, steps, cache-enabled (+4 more)

### Community 2643 - "post-release-checks"
Cohesion: 0.17
Nodes (11): concurrency, cancel-in-progress, group, jobs, post-release-checks, name, if, name (+3 more)

### Community 2644 - "env"
Cohesion: 0.17
Nodes (12): env, if, name, runs-on, timeout-minutes, DOCKER_COMPOSE_VERSION, LLM_ENDPOINT_CIDR_ALLOWLIST, ONEX_EVENT_BUS_TYPE (+4 more)

### Community 2654 - "venv_reconcile_lock.sh"
Cohesion: 0.29
Nodes (9): check-omnimarket-venv-drift.sh script, atomic(), install-node-skill-package.sh script, verify_install(), venv_reconcile_lock.sh script, VENV_LOCK_TIMEOUT_EXIT, venv_reconcile_lock_path(), venv_reconcile_run_locked() (+1 more)

### Community 2655 - "xml_etree_elementtree"
Cohesion: 0.26
Nodes (4): is_expired(), load_baseline(), main(), parse_junit()

### Community 2656 - "evaluate"
Cohesion: 0.17
Nodes (7): applicable_external_contexts(), check_run_event_index(), evaluate(), _verdict(), _is_allowlisted(), own_workflow_run_ids(), _report()

### Community 2657 - "gateway_restart_safety_proof.sh"
Cohesion: 0.33
Nodes (11): container_identity(), log_error(), log_info(), log_step(), log_warn(), main(), require_reachable(), gateway_restart_safety_proof.sh script (+3 more)

### Community 2658 - "git-maintenance.sh"
Cohesion: 0.33
Nodes (11): check_no_open_pr(), check_worktree_removable(), claim_keys(), is_regenerable_ignored(), is_secret_bearing_name(), open_claims_for(), open_pr_count(), refuse() (+3 more)

### Community 2659 - "_cooldown_read"
Cohesion: 0.21
Nodes (4): _backoff_seconds(), _cooldown_read(), FileTailer, JournalTailer

### Community 2660 - "resolve_omnimarket_pins.py"
Cohesion: 0.24
Nodes (5): main(), normalize_name(), read_dependencies(), requirement_name(), select_requirements()

### Community 2661 - "trigger-deploy.sh"
Cohesion: 0.55
Nodes (11): deploy_dev(), deploy_local(), deploy_prod(), log_error(), log_info(), log_step(), log_warn(), pull_latest() (+3 more)

### Community 2662 - "db/handler_contract.yaml"
Cohesion: 0.17
Nodes (11): activation, optional, description, handler_id, input_model, module, name, name (+3 more)

### Community 2663 - "descriptor"
Cohesion: 0.17
Nodes (12): enabled, failure_threshold, timeout_ms, descriptor, circuit_breaker, concurrency_policy, idempotent, isolation_policy (+4 more)

### Community 2664 - "http/handler_contract.yaml"
Cohesion: 0.17
Nodes (11): activation, optional, description, handler_id, input_model, module, name, name (+3 more)

### Community 2665 - "descriptor"
Cohesion: 0.17
Nodes (12): enabled, failure_threshold, timeout_ms, descriptor, circuit_breaker, concurrency_policy, idempotent, isolation_policy (+4 more)

### Community 2666 - "handler_routing"
Cohesion: 0.17
Nodes (12): module, name, handler_routing, handlers, routing_strategy, version, handler, handler_type (+4 more)

### Community 2667 - "deep_dive/models.py"
Cohesion: 0.17
Nodes (5): ActiveWorktree, CommitEntry, DriftReport, MergeEntry, sectionize_highlights()

### Community 2668 - "get_github_merged_prs"
Cohesion: 0.17
Nodes (6): GitHubMergedPR, classify_pr(), is_exempt_pr(), is_workflow_pr(), velocity_score_v2(), get_github_merged_prs()

### Community 2670 - "build_aiokafka_auth_kwargs_from_env"
Cohesion: 0.17
Nodes (3): build_aiokafka_auth_kwargs_from_env(), sanitize_bootstrap_servers(), _create_runtime_log_bridge_producer()

### Community 2675 - "endpoints"
Cohesion: 0.17
Nodes (11): endpoints, claude_code_settings_schema_url, github_api_base_url, github_web_base_url, gmail_api_base_url, gmail_oauth_token_url, linear_graphql_url, linear_oauth_token_url (+3 more)

### Community 2676 - "descriptor"
Cohesion: 0.17
Nodes (12): enabled, descriptor, circuit_breaker, concurrency_policy, idempotent, isolation_policy, node_archetype, observability_level (+4 more)

### Community 2678 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, handler_type, module, name, handler_routing, handlers, routing_strategy (+4 more)

### Community 2679 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2680 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+4 more)

### Community 2681 - "handlers"
Cohesion: 0.17
Nodes (12): description, module, name, module, name, handler_routing, handlers, routing_strategy (+4 more)

### Community 2682 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2683 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2684 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2685 - "handlers"
Cohesion: 0.17
Nodes (12): handler, class, input_model, module, name, handler_routing, handlers, routing_strategy (+4 more)

### Community 2686 - "io_config"
Cohesion: 0.17
Nodes (12): io_config, content-type, x-onex-correlation-id, x-onex-operation, acks, compression, handler_type, headers (+4 more)

### Community 2687 - "version"
Cohesion: 0.27
Nodes (12): version, handlers, optional, required, version, type, version, type (+4 more)

### Community 2688 - "topics"
Cohesion: 0.17
Nodes (12): class, idempotent, input_schema, method, output_schema, subscriptions, topics, description (+4 more)

### Community 2689 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2690 - "retention_days"
Cohesion: 0.17
Nodes (12): default_factory, description, item_type, type, properties, archive_labels, retention_days, default (+4 more)

### Community 2691 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2692 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2693 - "node_impact_analyzer_compute/contract.yaml"
Cohesion: 0.17
Nodes (11): capabilities, description, name, contract_name, description, event_bus, publish_topics, subscribe_topics (+3 more)

### Community 2694 - "node_lane_entry_point_census_effect/contract.yaml"
Cohesion: 0.17
Nodes (11): contract_name, description, event_bus, publish_topics, subscribe_topics, input, name, node_name (+3 more)

### Community 2695 - "node_ledger_write_effect/contract.yaml"
Cohesion: 0.17
Nodes (11): contract_name, description, descriptor, node_archetype, event_bus, publish_topics, subscribe_topics, name (+3 more)

### Community 2697 - "states"
Cohesion: 0.17
Nodes (12): icon, idempotency_note, ui_color, states, description, entry_actions, exit_actions, is_recoverable (+4 more)

### Community 2698 - "state_transitions"
Cohesion: 0.17
Nodes (12): major, minor, patch, state_transitions, description, error_states, initial_state, persistence_enabled (+4 more)

### Community 2699 - "node_placement_runtime_effect/contract.yaml"
Cohesion: 0.17
Nodes (11): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+3 more)

### Community 2700 - "coordination_rules"
Cohesion: 0.17
Nodes (12): checkpoint_enabled, checkpoint_interval_ms, execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, recovery_enabled (+4 more)

### Community 2702 - "state_machine"
Cohesion: 0.17
Nodes (12): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, entry_actions (+4 more)

### Community 2703 - "node_registry_api_effect/contract.yaml"
Cohesion: 0.17
Nodes (11): capabilities, description, name, contract_name, description, descriptor, node_archetype, purity (+3 more)

### Community 2704 - "metadata"
Cohesion: 0.17
Nodes (12): metadata, author, created, default_factory, description, key_type, license, tags (+4 more)

### Community 2705 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+4 more)

### Community 2706 - "ModelRsdDataFetchResult"
Cohesion: 0.20
Nodes (3): HandlerRsdDataFetch, ModelRsdDataFetchRequest, ModelRsdDataFetchResult

### Community 2707 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+4 more)

### Community 2708 - "HandlerRunnerFleetMaintainStart"
Cohesion: 0.20
Nodes (3): HandlerRunnerFleetMaintainStart, ModelRunnerFleetMaintainStartCommand, ModelRunnerFleetSnapshotGatherCommand

### Community 2709 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2710 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2711 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+4 more)

### Community 2712 - "tags"
Cohesion: 0.17
Nodes (12): description, type, type, properties, health_filter, service_name, tags, description (+4 more)

### Community 2713 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2714 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2715 - "node_update_plan_reducer/contract.yaml"
Cohesion: 0.17
Nodes (11): contract_name, description, error_handling, retry_policy, event_bus, publish_topics, subscribe_topics, name (+3 more)

### Community 2716 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, description (+4 more)

### Community 2717 - "node_validation_executor/contract.yaml"
Cohesion: 0.17
Nodes (11): contract_name, description, descriptor, node_archetype, name, node_name, node_type, published_events (+3 more)

### Community 2718 - "handlers"
Cohesion: 0.17
Nodes (12): module, name, module, name, handler_routing, handlers, routing_strategy, description (+4 more)

### Community 2719 - "node_validation_ledger_projection_compute/contract.yaml"
Cohesion: 0.17
Nodes (11): contract_name, description, descriptor, node_archetype, intent_consumption, intent_routing_table, validation_ledger.append, name (+3 more)

### Community 2720 - "input_fields"
Cohesion: 0.17
Nodes (12): correlation_id, envelope_bytes, envelope_hash, event_type, event_version, kafka_offset, kafka_partition, kafka_topic (+4 more)

### Community 2721 - "retry_policy"
Cohesion: 0.17
Nodes (12): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, exponential_base (+4 more)

### Community 2724 - "ProtocolAutoWiringManifestLike"
Cohesion: 0.18
Nodes (4): ProtocolAutoWiringManifestLike, select_kernel_nonwriting_projections(), _topic_declarers(), _describe_discovery_errors()

### Community 2731 - "managed_staging_canary_catalog_namespace.yaml"
Cohesion: 0.17
Nodes (11): candidate_contract_roots, default_partitions, default_replication_factor, description, epoch, group_prefix, group_start_policy, iam_group_patterns (+3 more)

### Community 2732 - "object_type"
Cohesion: 0.44
Nodes (12): grants, object_type, objects, privileges, schema, grants, grants, grants (+4 more)

### Community 2733 - "application"
Cohesion: 0.17
Nodes (12): checksum_ledger, checksum_ledgers, migration_stream, physical_name, checksum_column, domain_column, relation, schema (+4 more)

### Community 2734 - "omnibase_infra"
Cohesion: 0.17
Nodes (12): omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+4 more)

### Community 2735 - "services"
Cohesion: 0.23
Nodes (12): compose_service, host_port, local, mode, local, mode, local, mode (+4 more)

### Community 2736 - "object_type"
Cohesion: 0.44
Nodes (12): grants, object_type, objects, privileges, schema, grants, grants, grants (+4 more)

### Community 2737 - "omnibase_infra"
Cohesion: 0.17
Nodes (12): omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+4 more)

### Community 2738 - "omniintelligence"
Cohesion: 0.17
Nodes (12): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+4 more)

### Community 2739 - "object_type"
Cohesion: 0.44
Nodes (12): grants, object_type, objects, privileges, schema, grants, grants, grants (+4 more)

### Community 2740 - "omnibase_infra"
Cohesion: 0.17
Nodes (12): omnibase_infra, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+4 more)

### Community 2741 - "omniintelligence"
Cohesion: 0.17
Nodes (12): omniintelligence, checksum_ledger, checksum_ledgers, migration_stream, owners, physical_name, principals, login (+4 more)

### Community 2743 - "build_local_claim_interface"
Cohesion: 0.18
Nodes (3): ActionAuthorizationClaimOverlayError, build_local_claim_interface(), ModelActionAuthorizationClaimOverlay

### Community 2744 - "validator-requirements-baseline.yaml"
Cohesion: 0.18
Nodes (10): gaps, kind, repo, validator, repo, report_format, schema_version, major (+2 more)

### Community 2745 - "git_mirror"
Cohesion: 0.18
Nodes (11): git_mirror, active, bind_address, docker_network, kill_switch_env, mirror_root, port, refresh_interval_seconds (+3 more)

### Community 2746 - "hosts"
Cohesion: 0.18
Nodes (11): hosts, action_archive_host_root, arch, classes, expected_count, host, pools, runner_name_prefix (+3 more)

### Community 2747 - "effect_operations"
Cohesion: 0.18
Nodes (11): effect_operations, execution_mode, observability, subcontract_version, emit_metrics, log_request, log_response, trace_propagation (+3 more)

### Community 2748 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, is_critical, timeout_ms, transitions, actions, description, from_state (+3 more)

### Community 2749 - "major"
Cohesion: 0.44
Nodes (11): version, version, version, version, version, version, version, version (+3 more)

### Community 2750 - "wiring_config"
Cohesion: 0.18
Nodes (11): emit_on_success, payload_schema, topic, wiring_config, event_bus_type, fail_on_invalid_topic, max_concurrent_subscriptions, ready_event (+3 more)

### Community 2751 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, is_critical, timeout_ms, transitions, actions, description, from_state (+3 more)

### Community 2752 - "major"
Cohesion: 0.44
Nodes (11): version, version, version, version, version, version, version, version (+3 more)

### Community 2753 - "workflow_metadata"
Cohesion: 0.18
Nodes (11): workflow_metadata, author, description, documentation_url, execution_mode, tags, workflow_name, workflow_version (+3 more)

### Community 2754 - "foldable_services"
Cohesion: 0.18
Nodes (11): effects, projection-api, projection-writer-delegation, projection-writer-live-events, projection-writer-registration, projection-writer-savings, projection-writer-tenant-credentials, projection-writer-tenant-registry (+3 more)

### Community 2755 - "lab/developer-principal-grants.yaml"
Cohesion: 0.18
Nodes (10): allowed_login_domains, grants_from, lane, machine_grants_from, max_devices_per_login, principal_grants_from, dev-jonah-omnibook, principal_prefix (+2 more)

### Community 2756 - "local/developer-principal-grants.yaml"
Cohesion: 0.18
Nodes (10): allowed_login_domains, grants, name, operations, pattern, resource, lane, max_devices_per_login (+2 more)

### Community 2757 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2758 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2759 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2760 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2761 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2762 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2763 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2764 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2765 - "source"
Cohesion: 0.18
Nodes (10): databases, environment, schema_version, source, path, profile_catalog_path, profile_catalog_sha256, repository (+2 more)

### Community 2766 - "operational_defaults"
Cohesion: 0.18
Nodes (11): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+3 more)

### Community 2767 - "runtime-worker.yaml"
Cohesion: 0.18
Nodes (10): container_name, description, image, layer, name, ports, required_env, restart (+2 more)

### Community 2768 - "tenant-projection-writer.yaml"
Cohesion: 0.18
Nodes (10): container_name, description, image, layer, name, ports, required_env, restart (+2 more)

### Community 2769 - "redpanda-partition-cap"
Cohesion: 0.18
Nodes (11): REDPANDA_GROUP_OFFSET_RETENTION_SEC, REDPANDA_PROFILE_MEMORY, RPK_PASS, RPK_USER, file, service, container_name, environment (+3 more)

### Community 2770 - "volumes"
Cohesion: 0.40
Nodes (11): create_host_path, volumes, volumes, volumes, volumes, bind, read_only, source (+3 more)

### Community 2771 - "migration-gate"
Cohesion: 0.18
Nodes (11): forward-migration, condition, container_name, depends_on, entrypoint, image, networks, profiles (+3 more)

### Community 2772 - "valkey"
Cohesion: 0.18
Nodes (11): VALKEY_PASSWORD, valkey, command, container_name, environment, image, networks, ports (+3 more)

### Community 2773 - "gateway-runtime"
Cohesion: 0.20
Nodes (10): networks, external, name, external, ipv4_address, name, networks, gateway-lane-mirror-source (+2 more)

### Community 2774 - "valkey"
Cohesion: 0.18
Nodes (11): VALKEY_PASSWORD, valkey, command, container_name, environment, image, labels, networks (+3 more)

### Community 2775 - "migration-gate"
Cohesion: 0.18
Nodes (11): forward-migration, condition, container_name, depends_on, entrypoint, image, networks, profiles (+3 more)

### Community 2776 - "valkey"
Cohesion: 0.18
Nodes (11): VALKEY_PASSWORD, valkey, command, container_name, environment, image, networks, ports (+3 more)

### Community 2777 - "resources"
Cohesion: 0.18
Nodes (11): resources, deploy, cpus, memory, deploy, cpus, memory, limits (+3 more)

### Community 2778 - "valkey"
Cohesion: 0.18
Nodes (11): VALKEY_PASSWORD, valkey, command, container_name, environment, image, networks, ports (+3 more)

### Community 2779 - "omninode-runtime"
Cohesion: 0.18
Nodes (11): intelligence-migration, redpanda-partition-cap, condition, container_name, depends_on, image, labels, ports (+3 more)

### Community 2780 - "redpanda-partition-cap"
Cohesion: 0.18
Nodes (11): redpanda, condition, command, container_name, depends_on, entrypoint, image, labels (+3 more)

### Community 2781 - "hp051-ci-runner-1"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2782 - "hp051-ci-runner-2"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2783 - "hp051-ci-runner-3"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2784 - "hp051-ci-runner-4"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2785 - "hp053-ci-mq-runner-1"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2786 - "hp053-ci-mq-runner-2"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2787 - "hp053-ci-mq-runner-3"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2788 - "hp053-ci-mq-runner-4"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2789 - "interval"
Cohesion: 0.55
Nodes (11): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+3 more)

### Community 2790 - "bind"
Cohesion: 0.55
Nodes (11): volumes, volumes, volumes, volumes, volumes, bind, read_only, source (+3 more)

### Community 2791 - "omnipc2-ci-runner-1"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2792 - "omnipc2-ci-runner-2"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2793 - "omnipc2-ci-runner-3"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2794 - "omnipc2-ci-runner-4"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2795 - "omnipc2-ci-runner-7"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2796 - "omnipc2-ci-runner-8"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2797 - "omnipc2-ci-runner-9"
Cohesion: 0.18
Nodes (11): cgroup_parent, container_name, cpus, entrypoint, group_add, image, mem_limit, memswap_limit (+3 more)

### Community 2798 - "docker-compose.sim-202.yml"
Cohesion: 0.18
Nodes (10): name, networks, omnibase-infra-dogfood-network, name, x-lane-overlay-mount, read_only, source, target (+2 more)

### Community 2799 - "omninode-runtime"
Cohesion: 0.18
Nodes (11): intelligence-migration, redpanda-partition-cap, condition, container_name, depends_on, labels, networks, ports (+3 more)

### Community 2800 - "legacy-rds-fixture/prove.sh"
Cohesion: 0.42
Nodes (10): assert_pair(), create_fixture_database(), fail(), provision_action_authorization_claim_schema(), run_control_forward(), run_forward(), prove.sh script, signature() (+2 more)

### Community 2801 - "resolve-ecr-digest/action.yml"
Cohesion: 0.18
Nodes (10): author, description, name, runs, steps, using, id, name (+2 more)

### Community 2802 - "inputs"
Cohesion: 0.18
Nodes (11): description, required, description, required, inputs, commit-sha, ecr-registry, max-attempts (+3 more)

### Community 2803 - "no-cache"
Cohesion: 0.18
Nodes (11): no-cache, default, description, required, type, on, push, workflow_dispatch (+3 more)

### Community 2804 - "chain-canary-c12-provider-catalogue.yml"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, jobs, name, on, schedule, permissions (+2 more)

### Community 2805 - "jobs"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, name, needs, runs-on, timeout-minutes, jobs (+2 more)

### Community 2806 - "candidate-boot-gate"
Cohesion: 0.22
Nodes (11): permissions, permissions, name, needs, permissions, runs-on, timeout-minutes, candidate-boot-gate (+3 more)

### Community 2807 - "env"
Cohesion: 0.18
Nodes (11): APP_ID, APP_PRIVATE_KEY, BUMP_PACKAGE, CORE_LINEAGE, CORE_LINEAGE_RULES, EVIDENCE_SOURCE, GH_TOKEN, PACKAGE (+3 more)

### Community 2808 - "with"
Cohesion: 0.18
Nodes (11): with, app-id, cache-dependency-glob, dest, enable-cache, fetch-depth, overlay-repo, persist-credentials (+3 more)

### Community 2809 - "security-scan"
Cohesion: 0.20
Nodes (11): permissions, security-scan, contents, packages, security-events, if, name, needs (+3 more)

### Community 2810 - "hostile-review-judge.yml"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, jobs, name, on, workflow_run, permissions (+2 more)

### Community 2811 - "hostile-review-judge"
Cohesion: 0.18
Nodes (11): REVIEW_JUDGE_OVERLAY, REVIEW_VOTERS_OVERLAY, env, if, name, permissions, runs-on, timeout-minutes (+3 more)

### Community 2812 - "steps"
Cohesion: 0.18
Nodes (11): probe-hygiene, name, runs-on, steps, timeout-minutes, id, if, name (+3 more)

### Community 2813 - "with"
Cohesion: 0.18
Nodes (11): with, app-id, cache-dependency-glob, enable-cache, fetch-depth, owner, permission-contents, permission-metadata (+3 more)

### Community 2814 - "with"
Cohesion: 0.18
Nodes (11): with, app-id, fetch-depth, key, name, overlay-repo, path, private-key (+3 more)

### Community 2815 - "package"
Cohesion: 0.18
Nodes (11): package, version, on, workflow_dispatch, description, required, type, description (+3 more)

### Community 2816 - "env"
Cohesion: 0.18
Nodes (11): env, CACHE_VERSION, GIT_CONFIG_COUNT, GIT_CONFIG_KEY_0, GIT_CONFIG_VALUE_0, OMNI_CI_ENV_ROOT, OMNI_CI_SHARED_ENV_ENABLED, ONEX_EVENT_BUS_TYPE (+3 more)

### Community 2817 - "execute"
Cohesion: 0.18
Nodes (11): default, description, required, type, execute, ref, workflow_dispatch, default (+3 more)

### Community 2818 - "boot-catalog-local"
Cohesion: 0.18
Nodes (10): env, if, name, runs-on, timeout-minutes, LOCAL_PROJECT, jobs, boot-catalog-local (+2 more)

### Community 2819 - "inputs"
Cohesion: 0.18
Nodes (11): mode, real_ssh_user, description, required, type, workflow_call, default, description (+3 more)

### Community 2820 - "_execute_shell_check"
Cohesion: 0.29
Nodes (6): _build_result(), _build_sanitized_env(), _coerce_timeout_output(), _denied_command_reason(), execute_checks(), _execute_shell_check()

### Community 2827 - "check-env-reads.sh"
Cohesion: 0.35
Nodes (9): check_new_env_reads(), check_secret_name_declarations(), extract_env_names(), get_diff(), is_approved(), is_secret_name_check_target(), report_block(), resolve_base_commit() (+1 more)

### Community 2828 - "check_lane_attestation.py"
Cohesion: 0.29
Nodes (6): check_file(), check_staged(), _has_lane_attestation(), _has_verified_annotation(), _is_exempt(), main()

### Community 2829 - "ProbeInputError"
Cohesion: 0.20
Nodes (6): inject(), junit_outcomes(), parse_high_watermark(), probe_correlation_id(), ProbeInputError, run()

### Community 2830 - "instance_receipt_lanes.py"
Cohesion: 0.27
Nodes (4): InstanceReceiptLanesError, _instances(), receipt_lanes_for(), receipt_lanes_from_text()

### Community 2831 - "pr_trailers.py"
Cohesion: 0.20
Nodes (4): iter_prose_lines(), _mask_inline_code(), _trailer_on_line(), TrailerConflictError

### Community 2832 - "RestartWatcher"
Cohesion: 0.22
Nodes (4): _get_worker_state(), _restart_hwm_read(), _restart_hwm_write(), RestartWatcher

### Community 2833 - "pull-all.sh"
Cohesion: 0.31
Nodes (9): _branch_summary(), _checkout_and_ff(), _emit_summary(), _kill_descendants_leaf_first(), _leave_on_dev(), _on_exit(), _pull_one(), _run_bounded() (+1 more)

### Community 2834 - "semantic_mappings"
Cohesion: 0.20
Nodes (11): defaults, widget_type, group_by, show_address, show_capability_tags, show_port, defaults, widget_type (+3 more)

### Community 2835 - "security"
Cohesion: 0.18
Nodes (11): default_directory_mode, default_file_mode, preserve_on_overwrite, privilege_escalation, metadata, author, security, ticket (+3 more)

### Community 2836 - "strategies"
Cohesion: 0.18
Nodes (11): toctou_mitigation, description, implementation_note, note, strategy, follow_symlinks, max_symlink_depth, reject_external_targets (+3 more)

### Community 2837 - "pinned_model_identity"
Cohesion: 0.18
Nodes (10): contract_name, pinned_model_identity, architecture, artifact_model, artifact_ref, base_model, launch_profile, observed_model (+2 more)

### Community 2838 - "client"
Cohesion: 0.18
Nodes (10): client, consumer_group_prefix, lane, point_queries, queries, description, event_bus, publish_topics (+2 more)

### Community 2841 - "handler_manifest_persistence.contract.yaml"
Cohesion: 0.20
Nodes (10): contract_name, description, handler, handler_category, module, handler_name, handler_type, name (+2 more)

### Community 2842 - "optional"
Cohesion: 0.18
Nodes (11): configuration, optional, required, default, description, env_var, name, type (+3 more)

### Community 2850 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2851 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2852 - "workflow_metadata"
Cohesion: 0.18
Nodes (11): workflow_coordination, workflow_definition, execution_graph, workflow_metadata, description, execution_mode, workflow_name, workflow_version (+3 more)

### Community 2853 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2854 - "board_check_coverage.yaml"
Cohesion: 0.18
Nodes (10): contract_version, major, minor, patch, coverage, check_id, criteria, surface_class (+2 more)

### Community 2855 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2856 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2857 - "node_build_loop_write_effect/contract.yaml"
Cohesion: 0.18
Nodes (10): description, descriptor, node_archetype, event_bus, publish_topics, subscribe_topics, name, node_type (+2 more)

### Community 2858 - "retry_policy"
Cohesion: 0.18
Nodes (11): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, retry_policy, exponential_base, initial_delay_ms (+3 more)

### Community 2859 - "liveness"
Cohesion: 0.18
Nodes (11): liveness, degraded_after_seconds, drain_deadline_seconds, heartbeat_interval_seconds, inbound_topic_retry_seconds, lag_threshold_messages, lag_threshold_seconds, max_silence_window_seconds (+3 more)

### Community 2860 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2862 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, expression, required, transitions, actions, conditions, description (+3 more)

### Community 2863 - "state_machine"
Cohesion: 0.18
Nodes (11): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, entry_actions (+3 more)

### Community 2864 - "node_contract_resolver_bridge/contract.yaml"
Cohesion: 0.18
Nodes (10): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+2 more)

### Community 2865 - "metadata"
Cohesion: 0.18
Nodes (11): metadata, author, created, depends_on_ticket, description, license, replacement_path, tags (+3 more)

### Community 2866 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2867 - "wiring_config"
Cohesion: 0.18
Nodes (11): emit_on_success, payload_schema, topic, wiring_config, event_bus_type, fail_on_invalid_topic, max_concurrent_subscriptions, ready_event (+3 more)

### Community 2868 - "retry_policy"
Cohesion: 0.18
Nodes (11): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, retry_policy, exponential_base, initial_delay_ms (+3 more)

### Community 2869 - "node_hostile_review_fix_verifier_effect/contract.yaml"
Cohesion: 0.18
Nodes (10): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+2 more)

### Community 2870 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2872 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2873 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2874 - "node_ledger_projection_compute/contract.yaml"
Cohesion: 0.18
Nodes (10): contract_name, description, intent_consumption, intent_routing_table, ledger.append, name, node_name, node_type (+2 more)

### Community 2875 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, is_critical, timeout_ms, transitions, actions, description, from_state (+3 more)

### Community 2876 - "major"
Cohesion: 0.44
Nodes (11): version, version, version, version, version, version, version, version (+3 more)

### Community 2877 - "node_onex_api_image_build_effect/contract.yaml"
Cohesion: 0.18
Nodes (10): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+2 more)

### Community 2878 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2879 - "node_onex_api_image_import_check_effect/contract.yaml"
Cohesion: 0.18
Nodes (10): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+2 more)

### Community 2880 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2881 - "node_pattern_lifecycle_effect/contract.yaml"
Cohesion: 0.18
Nodes (10): capabilities, description, name, contract_name, description, descriptor, node_archetype, name (+2 more)

### Community 2882 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2883 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2884 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2886 - "node_registration_reducer/contract.yaml"
Cohesion: 0.18
Nodes (10): description, name, node_capabilities, processing, node_type, node_version, state_model, description (+2 more)

### Community 2887 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, expression, required, transitions, actions, conditions, description (+3 more)

### Community 2888 - "columns"
Cohesion: 0.18
Nodes (11): constraints, description, name, type, database_schema, columns, indexes, table_name (+3 more)

### Community 2889 - "node_runner_fleet_maintain_orchestrator/contract.yaml"
Cohesion: 0.18
Nodes (10): contract_version, major, minor, patch, description, name, node_type, node_version (+2 more)

### Community 2890 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2891 - "published_events"
Cohesion: 0.18
Nodes (11): cleanup.policy, retention.ms, published_events, description, event_type, message_category, topic, topic_config (+3 more)

### Community 2892 - "published_events"
Cohesion: 0.18
Nodes (11): cleanup.policy, retention.ms, published_events, description, event_type, message_category, topic, topic_config (+3 more)

### Community 2893 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, description (+3 more)

### Community 2894 - "handlers"
Cohesion: 0.18
Nodes (11): description, module, name, module, name, handler_routing, handlers, routing_strategy (+3 more)

### Community 2895 - "workflow_metadata"
Cohesion: 0.18
Nodes (11): workflow_metadata, author, description, documentation_url, execution_mode, tags, workflow_name, workflow_version (+3 more)

### Community 2896 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2898 - "ModelRegistrationResult"
Cohesion: 0.18
Nodes (11): ModelRegistrationResult, description, type, description, properties, required, type, error (+3 more)

### Community 2899 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, expression, required, transitions, actions, conditions, description (+3 more)

### Community 2900 - "state_machine"
Cohesion: 0.18
Nodes (11): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, entry_actions (+3 more)

### Community 2901 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2903 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, description (+3 more)

### Community 2904 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, expression, required, transitions, actions, conditions, description (+3 more)

### Community 2905 - "state_machine"
Cohesion: 0.18
Nodes (11): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, entry_actions (+3 more)

### Community 2906 - "transitions"
Cohesion: 0.18
Nodes (11): action_name, action_type, expression, required, transitions, actions, conditions, description (+3 more)

### Community 2907 - "state_machine"
Cohesion: 0.18
Nodes (11): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, entry_actions (+3 more)

### Community 2908 - "retry_policy"
Cohesion: 0.18
Nodes (11): error_handling, error_types, retry_policy, description, name, recoverable, exponential_base, initial_delay_ms (+3 more)

### Community 2909 - "output_fields"
Cohesion: 0.18
Nodes (11): io_operations, description, idempotent, operation, output_fields, duplicate, kafka_offset, kafka_partition (+3 more)

### Community 2910 - "workflow_metadata"
Cohesion: 0.18
Nodes (11): workflow_coordination, workflow_definition, execution_graph, workflow_metadata, description, execution_mode, workflow_name, workflow_version (+3 more)

### Community 2911 - "retry_policy"
Cohesion: 0.18
Nodes (11): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, retry_policy, exponential_base, initial_delay_ms (+3 more)

### Community 2912 - "handlers"
Cohesion: 0.18
Nodes (11): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+3 more)

### Community 2913 - "write_config_hybrid"
Cohesion: 0.18
Nodes (11): write_config_hybrid, AWS_REGION, CLOUD_PROVIDER, GCP_PROJECT, KAFKA_BOOTSTRAP_SERVERS, LLM_CODER_URL, ONEX_DEPLOYMENT_MODE, POSTGRES_HOST (+3 more)

### Community 2914 - "set_state"
Cohesion: 0.18
Nodes (11): condition, next, set_state, aws_region, gcp_project, llm_endpoint, selected_local_services, transitions (+3 more)

### Community 2920 - "overlay_from_env_dict"
Cohesion: 0.20
Nodes (4): _classify_key(), overlay_from_env_dict(), _contains_secret_keys(), OverlayWriter

### Community 2922 - "event_bus"
Cohesion: 0.18
Nodes (10): description, event_bus, circuit_breaker_threshold, environment, max_history, profile, type, group_id (+2 more)

### Community 2925 - "session_projector/projector.py"
Cohesion: 0.24
Nodes (4): _extract_ticket_from_branch(), project_session_ended(), project_session_started(), project_tool_executed()

### Community 2926 - "LedgerSinkError"
Cohesion: 0.22
Nodes (3): LedgerSinkClosedError, LedgerSinkError, LedgerSinkFullError

### Community 2929 - "c11_delegation_legs.yaml"
Cohesion: 0.20
Nodes (9): canary_surfaces, canary_value, corpus_ids, corpus_path, ladder, rung_1, rung_2, policy_refusal_attempts (+1 more)

### Community 2930 - "evidence_artifacts"
Cohesion: 0.20
Nodes (9): evidence_artifacts, artifact_name, id, job, producer_step, rationale, required_paths, workflow (+1 more)

### Community 2931 - "sources"
Cohesion: 0.20
Nodes (9): sources, branch, dispatch_title_contains, events, max_age_hours, name, no_verdict_conclusions, repo (+1 more)

### Community 2932 - "machine_classes"
Cohesion: 0.20
Nodes (9): machine_classes, description, infisical_path_prefix, name, overlay_file, shared, description, name (+1 more)

### Community 2933 - "runner_fleet.yaml"
Cohesion: 0.20
Nodes (9): burst_count, expected_count, github_org, network_pool_capacity, network_pool_warn_ratio, runner_group, runner_host, runner_name_prefix (+1 more)

### Community 2934 - "activation_gate"
Cohesion: 0.53
Nodes (10): capacity_budget, evidence_companion_fate_isolation, maintenance_roll_convergence, positive_control_acceptance, sustained_min_span_seconds, sustained_samples, activation_gate, activation_gate (+2 more)

### Community 2935 - "states"
Cohesion: 0.20
Nodes (10): icon, ui_color, states, description, entry_actions, exit_actions, is_terminal, metadata (+2 more)

### Community 2936 - "checks"
Cohesion: 0.20
Nodes (10): check_type, critical, description, name, threshold_ms, health_check, checks, enabled (+2 more)

### Community 2937 - "testing_guidance"
Cohesion: 0.20
Nodes (10): scenarios, verify, states, triggers, verify, testing_guidance, edge_cases, idempotency (+2 more)

### Community 2938 - "metadata"
Cohesion: 0.20
Nodes (10): alternatives_considered, decision, rationale, related_patterns, metadata, author, description, design_decisions (+2 more)

### Community 2939 - "main"
Cohesion: 0.20
Nodes (10): bifrost_verify_endpoints, capabilities, folded_profiles, omnimemory_enabled, omnimemory_memgraph_host, publish_introspection, retired_services, runtime_address (+2 more)

### Community 2940 - "bindings"
Cohesion: 0.20
Nodes (9): bindings, broker, consumer_group, direction, physical_topic, group_prefixes, broker, prefix (+1 more)

### Community 2941 - "sources"
Cohesion: 0.20
Nodes (9): required_connect_databases, schema_version, sources, path, purpose, repository, revision, sha256 (+1 more)

### Community 2942 - "local-lane-mirror"
Cohesion: 0.20
Nodes (10): lane-mirror-forwarder, redpanda-local-sasl, local-lane-mirror, description, extra_depends_on, includes, inject_env, inject_required_env (+2 more)

### Community 2943 - "dlq-replay-consumer.yaml"
Cohesion: 0.20
Nodes (9): container_name, description, image, layer, name, required_env, restart, stop_grace_period (+1 more)

### Community 2944 - "fault-inject-fixture.yaml"
Cohesion: 0.20
Nodes (9): container_name, description, image, layer, name, required_env, restart, stop_grace_period (+1 more)

### Community 2945 - "operational_defaults"
Cohesion: 0.20
Nodes (10): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+2 more)

### Community 2946 - "runtime-canary.yaml"
Cohesion: 0.20
Nodes (9): container_name, description, image, layer, name, required_env, restart, stop_grace_period (+1 more)

### Community 2947 - "operational_defaults"
Cohesion: 0.20
Nodes (10): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+2 more)

### Community 2948 - "operational_defaults"
Cohesion: 0.20
Nodes (10): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+2 more)

### Community 2949 - "operational_defaults"
Cohesion: 0.20
Nodes (10): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+2 more)

### Community 2950 - "operational_defaults"
Cohesion: 0.20
Nodes (10): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_DATABASE_TOPOLOGY_PROFILE, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL (+2 more)

### Community 2951 - "options"
Cohesion: 0.31
Nodes (10): logging, logging, logging, logging, driver, options, max-file, max-size (+2 more)

### Community 2952 - "environment"
Cohesion: 0.20
Nodes (10): redpanda-sasl-enable, REDPANDA_CONTROL_PLANE_TOPIC, REDPANDA_GROUP_OFFSET_RETENTION_SEC, REDPANDA_PROFILE_MEMORY, RPK_PASS, RPK_USER, depends_on, environment (+2 more)

### Community 2953 - "dogfood-delegation-fault-429"
Cohesion: 0.20
Nodes (10): command, container_name, image, networks, profiles, read_only, restart, tmpfs (+2 more)

### Community 2954 - "runtime"
Cohesion: 0.20
Nodes (10): context, dockerfile, build, container_name, labels, networks, ports, profiles (+2 more)

### Community 2955 - "depends_on"
Cohesion: 0.22
Nodes (10): infisical, postgres, redpanda-topic-manager, valkey, condition, depends_on, condition, condition (+2 more)

### Community 2956 - "redpanda-topic-manager"
Cohesion: 0.20
Nodes (10): redpanda, condition, command, container_name, depends_on, entrypoint, image, labels (+2 more)

### Community 2957 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2958 - "redpanda"
Cohesion: 0.20
Nodes (10): command, container_name, image, labels, mem_limit, mem_reservation, networks, ports (+2 more)

### Community 2959 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2960 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2961 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2962 - "volumes"
Cohesion: 0.56
Nodes (10): volumes, volumes, volumes, volumes, bind, read_only, source, target (+2 more)

### Community 2963 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2964 - "volumes"
Cohesion: 0.56
Nodes (10): volumes, volumes, volumes, volumes, bind, read_only, source, target (+2 more)

### Community 2965 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2966 - "volumes"
Cohesion: 0.56
Nodes (10): volumes, volumes, volumes, volumes, bind, read_only, source, target (+2 more)

### Community 2967 - "omninode-customer-plane-runner-1"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2968 - "omninode-customer-plane-runner-2"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2969 - "healthcheck"
Cohesion: 0.56
Nodes (10): interval, retries, start_period, test, timeout, healthcheck, healthcheck, healthcheck (+2 more)

### Community 2970 - "omninode-studio-ci-runner-1"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2971 - "omninode-studio-ci-runner-2"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2972 - "omninode-studio-ci-runner-3"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2973 - "omninode-studio-ci-runner-4"
Cohesion: 0.20
Nodes (10): container_name, cpus, entrypoint, image, mem_limit, memswap_limit, pids_limit, restart (+2 more)

### Community 2974 - "redpanda-partition-cap"
Cohesion: 0.20
Nodes (10): REDPANDA_PARTITIONS_PER_SHARD, REDPANDA_PROFILE_MEMORY, container_name, depends_on, entrypoint, environment, image, networks (+2 more)

### Community 2975 - "000_create_multiple_databases.sh"
Cohesion: 0.53
Nodes (9): create_database(), create_login_only_role(), create_role(), grant_database_create_to_corpus_applier(), grant_role_to_database(), revoke_cross_db_access(), 000_create_multiple_databases.sh script, validate_identifier() (+1 more)

### Community 2976 - "tables"
Cohesion: 0.20
Nodes (9): status, status, status, status, tables, agent_actions, agent_detection_failures, agent_transformation_events (+1 more)

### Community 2977 - "outputs"
Cohesion: 0.20
Nodes (10): description, value, description, value, description, value, outputs, digest (+2 more)

### Community 2978 - "outputs"
Cohesion: 0.20
Nodes (10): description, value, description, value, description, value, outputs, docker_runs_on (+2 more)

### Community 2979 - "digest"
Cohesion: 0.22
Nodes (10): outputs, description, value, description, value, workflow_call, digest, image-ref (+2 more)

### Community 2980 - "steps"
Cohesion: 0.20
Nodes (10): steps, COMMIT_SHA, IMAGE_TAG, IMAGE_URI, env, id, name, run (+2 more)

### Community 2981 - "chain-canary-c11-negative-paths.yml"
Cohesion: 0.20
Nodes (9): concurrency, cancel-in-progress, group, env, UV_VERSION, jobs, name, permissions (+1 more)

### Community 2982 - "steps"
Cohesion: 0.22
Nodes (10): steps, C12_CONTAINER, LANE_NAME, steps, env, id, if, name (+2 more)

### Community 2983 - "container"
Cohesion: 0.20
Nodes (10): default, description, required, default, description, required, container, deploy_wait_seconds (+2 more)

### Community 2984 - "steps"
Cohesion: 0.20
Nodes (10): environment, name, runs-on, steps, timeout-minutes, c16-receipt-identity, if, name (+2 more)

### Community 2985 - "chain-canary-c28-consumer-flow.yml"
Cohesion: 0.20
Nodes (9): concurrency, cancel-in-progress, group, env, UV_VERSION, jobs, name, permissions (+1 more)

### Community 2986 - "pr_number"
Cohesion: 0.20
Nodes (10): pr_number, repo, workflow_dispatch, description, required, type, description, required (+2 more)

### Community 2987 - "delegation-health"
Cohesion: 0.20
Nodes (9): name, runs-on, timeout-minutes, jobs, delegation-health, name, permissions, contents (+1 more)

### Community 2988 - "steps"
Cohesion: 0.38
Nodes (10): steps, steps, steps, steps, continue-on-error, id, if, name (+2 more)

### Community 2989 - "env"
Cohesion: 0.20
Nodes (10): CANDIDATE, CI_OVERLAY_REPO, GH_TOKEN, KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, OVERLAY_REPO, REASON (+2 more)

### Community 2990 - "dry_run"
Cohesion: 0.20
Nodes (10): default, description, required, type, dry_run, on, schedule, workflow_dispatch (+2 more)

### Community 2991 - "nightly-integration.yml"
Cohesion: 0.20
Nodes (9): concurrency, cancel-in-progress, group, name, runs-on, timeout-minutes, jobs, guard-live-host (+1 more)

### Community 2992 - "jobs"
Cohesion: 0.20
Nodes (9): concurrency, cancel-in-progress, group, name, runs-on, timeout-minutes, jobs, handler-contract-compliance (+1 more)

### Community 2993 - "steps"
Cohesion: 0.40
Nodes (10): steps, steps, steps, continue-on-error, id, name, run, uses (+2 more)

### Community 2994 - "verify-onex-api-delivered"
Cohesion: 0.20
Nodes (9): jobs, verify-onex-api-delivered, name, on, if, name, needs, runs-on (+1 more)

### Community 2995 - "steps"
Cohesion: 0.20
Nodes (10): update-dockerfile-pins, id, if, name, run, uses, name, runs-on (+2 more)

### Community 2996 - "steps"
Cohesion: 0.51
Nodes (10): steps, steps, steps, id, if, name, run, uses (+2 more)

### Community 2997 - "cut-tag"
Cohesion: 0.20
Nodes (10): if, name, outputs, permissions, runs-on, timeout-minutes, cut-tag, tag (+2 more)

### Community 2998 - "cut"
Cohesion: 0.20
Nodes (10): if, name, needs, runs-on, strategy, timeout-minutes, cut, decision (+2 more)

### Community 2999 - "repos"
Cohesion: 0.20
Nodes (10): repos, on, schedule, workflow_dispatch, default, description, required, type (+2 more)

### Community 3000 - "dry_run"
Cohesion: 0.20
Nodes (10): default, description, required, type, dry_run, on, pull_request, workflow_dispatch (+2 more)

### Community 3001 - "route"
Cohesion: 0.20
Nodes (9): jobs, route, name, on, permissions, contents, name, runs-on (+1 more)

### Community 3002 - "force"
Cohesion: 0.20
Nodes (10): default, description, required, type, force, workflow_path, inputs, description (+2 more)

### Community 3003 - "jobs"
Cohesion: 0.20
Nodes (9): concurrency, cancel-in-progress, group, jobs, name, on, pull_request, branches (+1 more)

### Community 3004 - "reemit-queued-receipts"
Cohesion: 0.20
Nodes (10): reemit-queued-receipts, candidate, if, name, needs, runs-on, strategy, timeout-minutes (+2 more)

### Community 3005 - "rerun-refused-deliveries"
Cohesion: 0.22
Nodes (10): rerun-refused-deliveries, actions, contents, permissions, if, name, needs, permissions (+2 more)

### Community 3006 - "steps"
Cohesion: 0.40
Nodes (10): steps, steps, continue-on-error, id, if, name, run, uses (+2 more)

### Community 3007 - "secrets"
Cohesion: 0.20
Nodes (10): required, required, workflow_call, required, required, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, ONEXBOT_OCC_APP_ID (+2 more)

### Community 3014 - "check_seed_provenance.py"
Cohesion: 0.31
Nodes (5): check_scripts(), _has_provenance(), _is_seed_or_demo(), main(), _publishes_events()

### Community 3015 - "ci_check_contract_discovery.py"
Cohesion: 0.24
Nodes (3): check_contract_discovery(), _derive_package_root(), main()

### Community 3016 - "enable-merge-queue.sh"
Cohesion: 0.42
Nodes (9): check_existing_mq_ruleset(), die(), enable_merge_queue(), get_repo_id(), log(), enable-merge-queue.sh script, usage(), validate_repo() (+1 more)

### Community 3017 - "gather_repo_facts"
Cohesion: 0.20
Nodes (5): _dev_ahead(), gather_repo_facts(), _latest_tag(), _pypi_latest(), RepoConfig

### Community 3018 - "cut-lab-ref.sh"
Cohesion: 0.33
Nodes (9): BUILD_SOURCE, compute_lab_tag_name(), cut_lab_tags(), DEPLOY_HOTPATCH, err(), log(), OMNIBASE_INFRA_COMPOSE_PROJECT, cut-lab-ref.sh script (+1 more)

### Community 3019 - "cut_release_train_tag.sh"
Cohesion: 0.33
Nodes (9): ANCHOR_GITHUB_REPO, ANCHOR_REPO, compute_tag_name(), err(), git_clone(), log(), cut_release_train_tag.sh script, SIBLING_FALLBACK_REF (+1 more)

### Community 3022 - "min_samples"
Cohesion: 0.20
Nodes (10): confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count, source (+2 more)

### Community 3023 - "sample_window_days"
Cohesion: 0.20
Nodes (10): confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count, source (+2 more)

### Community 3024 - "authoritative"
Cohesion: 0.20
Nodes (10): authoritative, qwen2.5-14b, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k (+2 more)

### Community 3025 - "generated_at"
Cohesion: 0.20
Nodes (10): generated_at, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+2 more)

### Community 3026 - "query"
Cohesion: 0.20
Nodes (10): query, qwen2.5-72b, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k (+2 more)

### Community 3027 - "sample_count"
Cohesion: 0.20
Nodes (10): sample_count, qwen3.8, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k (+2 more)

### Community 3028 - "defaults"
Cohesion: 0.20
Nodes (10): event.consume, filter_level, highlight_critical, max_items, show_source, defaults, widget_type, defaults (+2 more)

### Community 3029 - "descriptor"
Cohesion: 0.20
Nodes (10): enabled, failure_threshold, timeout_ms, descriptor, circuit_breaker, concurrency_policy, idempotent, node_archetype (+2 more)

### Community 3030 - "path_validation"
Cohesion: 0.20
Nodes (10): encoding_validation, enforce_max_file_size, content_validation, path_validation, max_filename_length, max_path_length, reject_control_chars, reject_null_bytes (+2 more)

### Community 3031 - "version"
Cohesion: 0.24
Nodes (10): expression, parameter_name, required, version, operation_bindings, global_bindings, version, major (+2 more)

### Community 3032 - "http.get"
Cohesion: 0.20
Nodes (10): http.get, expression, parameter_name, required, expression, parameter_name, required, operation_bindings (+2 more)

### Community 3037 - "ModelSubprocessOutcome"
Cohesion: 0.20
Nodes (3): lane_runtime_containers(), read_lane_presence(), ModelSubprocessOutcome

### Community 3038 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): enabled, failure_threshold, reset_timeout_ms, transport_type, error_handling, circuit_breaker, error_types, description (+2 more)

### Community 3042 - "evaluate_projection_liveness"
Cohesion: 0.22
Nodes (3): ModelFlowAttribution, _attribute_delta(), evaluate_projection_liveness()

### Community 3044 - "node_artifact_change_detector_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3045 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3047 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3048 - "node_broker_principal_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3049 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3050 - "topic_sets"
Cohesion: 0.20
Nodes (10): lane_mirror, declared_lanes, max_messages_per_poll, poll_timeout_ms, topic_sets, ci_run_failed, delegation, hook_edge (+2 more)

### Community 3053 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, event_model, operation (+2 more)

### Community 3054 - "node_chain_orchestrator/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_version, major, minor, patch, description, name, node_type, node_version (+1 more)

### Community 3055 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): failure_threshold, name, reset_timeout_ms, backends, enabled, failure_threshold, half_open_max_requests, per_backend (+2 more)

### Community 3056 - "error_handling"
Cohesion: 0.20
Nodes (10): error_handling, error_sanitization, error_types, enabled, sanitize_patterns, utility, description, name (+2 more)

### Community 3057 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, event_type, handler (+2 more)

### Community 3058 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): failure_threshold, name, reset_timeout_ms, backends, enabled, failure_threshold, half_open_max_requests, per_backend (+2 more)

### Community 3059 - "error_handling"
Cohesion: 0.20
Nodes (10): error_handling, error_sanitization, error_types, enabled, sanitize_patterns, utility, description, name (+2 more)

### Community 3060 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3061 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): failure_threshold, name, reset_timeout_ms, backends, enabled, failure_threshold, half_open_max_requests, per_backend (+2 more)

### Community 3062 - "error_handling"
Cohesion: 0.20
Nodes (10): error_handling, error_sanitization, error_types, enabled, sanitize_patterns, utility, description, name (+2 more)

### Community 3063 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): failure_threshold, name, reset_timeout_ms, backends, enabled, failure_threshold, half_open_max_requests, per_backend (+2 more)

### Community 3064 - "error_handling"
Cohesion: 0.20
Nodes (10): error_handling, error_sanitization, error_types, enabled, sanitize_patterns, utility, description, name (+2 more)

### Community 3065 - "node_developer_principal_plan_compute/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_name, description, descriptor, node_archetype, name, node_name, node_type, runtime_profiles (+1 more)

### Community 3066 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3067 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3068 - "ownership"
Cohesion: 0.20
Nodes (10): reclassification_owner, replay_eligibility_change_owner, retention_owner, ownership, responsibility, role, responsibility, role (+2 more)

### Community 3070 - "checks"
Cohesion: 0.20
Nodes (10): check_type, critical, description, name, threshold_ms, health_check, checks, enabled (+2 more)

### Community 3071 - "handlers"
Cohesion: 0.20
Nodes (10): handler, class, input_model, module, name, handlers, description, handler (+2 more)

### Community 3072 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3073 - "node_gateway_attach_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3074 - "handlers"
Cohesion: 0.20
Nodes (10): handler_type, module, name, handler_routing, handlers, routing_strategy, description, handler (+2 more)

### Community 3075 - "node_gateway_link_health_write_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): description, descriptor, node_archetype, event_bus, subscribe_topics, name, node_type, node_version (+1 more)

### Community 3076 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, input_model (+2 more)

### Community 3077 - "error_types"
Cohesion: 0.20
Nodes (10): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, description, name (+2 more)

### Community 3078 - "published_events"
Cohesion: 0.20
Nodes (10): cleanup.policy, retention.ms, published_events, description, event_type, topic, topic_config, kafka_config (+2 more)

### Community 3079 - "circuit_breaker"
Cohesion: 0.20
Nodes (10): failure_threshold, name, reset_timeout_ms, backends, enabled, failure_threshold, half_open_max_requests, per_backend (+2 more)

### Community 3080 - "error_handling"
Cohesion: 0.20
Nodes (10): error_handling, error_sanitization, error_types, enabled, sanitize_patterns, utility, description, name (+2 more)

### Community 3081 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+2 more)

### Community 3082 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3083 - "testing_guidance"
Cohesion: 0.20
Nodes (10): scenarios, verify, states, triggers, verify, testing_guidance, edge_cases, idempotency (+2 more)

### Community 3084 - "node_reconcile_workspace_venvs/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_name, description, name, node_name, node_type, reconcile_policy, git_timeout_seconds, sync_timeout_seconds (+1 more)

### Community 3085 - "projection_reader"
Cohesion: 0.22
Nodes (10): projection_reader, projection_reader, description, implementation, module, projections, protocol, shared (+2 more)

### Community 3086 - "published_events"
Cohesion: 0.20
Nodes (10): cleanup.policy, retention.ms, published_events, description, event_type, topic, topic_config, kafka_config (+2 more)

### Community 3087 - "error_types"
Cohesion: 0.20
Nodes (10): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, description, name (+2 more)

### Community 3088 - "dependencies"
Cohesion: 0.20
Nodes (10): dependencies, class_name, description, implementations, module, name, pluggable, type (+2 more)

### Community 3089 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, default_handler, handlers, routing_strategy, description, handler (+2 more)

### Community 3090 - "node_remote_agent_invoke_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3091 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_type, module, name, handlers, description, event_model (+2 more)

### Community 3092 - "published_events"
Cohesion: 0.20
Nodes (10): cleanup.policy, retention.ms, published_events, description, event_type, topic, topic_config, kafka_config (+2 more)

### Community 3093 - "node_routing_orchestrator/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_version, major, minor, patch, description, name, node_type, node_version (+1 more)

### Community 3094 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+2 more)

### Community 3095 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, module, name, handler_routing, handlers, routing_strategy, event_model (+2 more)

### Community 3096 - "node_runtime_manifest_reducer/contract.yaml"
Cohesion: 0.20
Nodes (9): description, event_bus, publish_topics, subscribe_topics, name, node_capabilities, processing, node_type (+1 more)

### Community 3097 - "metadata"
Cohesion: 0.20
Nodes (10): alternatives_considered, decision, rationale, related_patterns, metadata, author, description, design_decisions (+2 more)

### Community 3098 - "node_scope_workflow_orchestrator/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_version, major, minor, patch, description, name, node_type, node_version (+1 more)

### Community 3099 - "node_session_lifecycle_reducer/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_name, description, name, node_name, node_type, state_model, description, module (+1 more)

### Community 3100 - "cost_estimator_build"
Cohesion: 0.20
Nodes (10): config, cost_estimator_build, description, follow, gate, container, description, default_enabled (+2 more)

### Community 3101 - "node_sibling_clone_pin_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): contract_name, description, descriptor, node_archetype, name, node_name, node_type, node_version (+1 more)

### Community 3102 - "handlers"
Cohesion: 0.20
Nodes (10): handler, class, input_model, module, name, handlers, description, handler (+2 more)

### Community 3103 - "error_types"
Cohesion: 0.20
Nodes (10): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker, error_types, description, name (+2 more)

### Community 3104 - "nodes"
Cohesion: 0.20
Nodes (10): nodes, depends_on, description, node_id, node_type, step_config, adjudicator_node, effect_node (+2 more)

### Community 3105 - "node_venv_atomic_install/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3107 - "node_workspace_config_root_resolve_effect/contract.yaml"
Cohesion: 0.20
Nodes (9): capabilities, description, name, contract_name, description, name, node_name, node_type (+1 more)

### Community 3108 - "handlers"
Cohesion: 0.20
Nodes (10): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+2 more)

### Community 3109 - "interactive_onboarding.yaml"
Cohesion: 0.20
Nodes (9): description, max_estimated_minutes, policy_name, policy_type, target_capabilities, version, major, minor (+1 more)

### Community 3113 - "wire_shape.py"
Cohesion: 0.29
Nodes (3): parse_delegation_terminal_envelope(), status_for_topic(), WorkflowReceiptReplayEnvelopeError

### Community 3114 - "[0.11.0] - 2026-02-25"
Cohesion: 0.22
Nodes (9): [0.11.0] - 2026-02-25, Added, Changed, Deployment Safety, Documentation, Event Bus Registry, Runtime Observability, Tests (+1 more)

### Community 3115 - "OmniNode Architecture – Constraint Map (omnibase_infra)"
Cohesion: 0.22
Nodes (8): Core Principles, Layer Boundaries, Non-Goals (DO NOT), OmniNode Architecture – Constraint Map (omnibase_infra), Patterns to Avoid, Platform-Wide Rules, Rules the Agent Must Obey, This Repo Contains

### Community 3116 - "dev"
Cohesion: 0.22
Nodes (8): read_at, repo_specs, repo_specs_measured_at, source, specs_per_boot, lanes, dev, schema_version

### Community 3117 - "omnibase_core"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3118 - "omnibase_spi"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3119 - "omniclaude"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3120 - "omniintelligence"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3121 - "omnimarket"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3122 - "omnimemory"
Cohesion: 0.22
Nodes (9): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, mode_note, package, release_relevant_paths (+1 more)

### Community 3123 - "probe_placement_variables"
Cohesion: 0.22
Nodes (9): OMNI_OCC_AUTOBIND_RUNS_ON_JSON, OMNI_PUBLIC_PR_RUNS_ON_JSON, OMNI_TRUSTED_CI_RUNS_ON_JSON, CUSTOMER_MACHINE_RUNS_ON_JSON, OMNI_INFRA_FLEET_RUNS_ON_JSON, OMNI_TRUSTED_CI_RUNS_ON_JSON, probe_placement_variables, omnimarket (+1 more)

### Community 3124 - "dod_evidence"
Cohesion: 0.22
Nodes (9): check_type, check_value, cwd, dod_evidence, binds_ac, checks, description, id (+1 more)

### Community 3125 - "security"
Cohesion: 0.22
Nodes (9): security, deny_absolute_paths, deny_patterns, deny_symlink_escape, enabled, allowed_base_paths, path_validation, require_path_under_base (+1 more)

### Community 3126 - "metadata"
Cohesion: 0.22
Nodes (9): exclude_patterns, follow_symlinks, hot_reload_exclude_patterns, pattern, recursive, scan_mode, watch_debounce_ms, watch_mode (+1 more)

### Community 3127 - "dev"
Cohesion: 0.22
Nodes (9): boundary_dlq_enabled, compose_project, effects_port, main_port, secret_resolver_config_path, services, topic_provisioner_max_partitions, trajectory_evaluation_backend (+1 more)

### Community 3128 - "source"
Cohesion: 0.33
Nodes (9): secret_resolver_mappings, secret_resolver_mappings, secret_resolver_mappings, secret_resolver_mappings, logical_name, source, source_path, source_type (+1 more)

### Community 3129 - "dogfood"
Cohesion: 0.22
Nodes (9): boundary_dlq_enabled, compose_project, effects_port, main_port, secret_resolver_config_path, topic_provisioner_max_partitions, trajectory_evaluation_backend, profiles (+1 more)

### Community 3130 - "effects"
Cohesion: 0.22
Nodes (9): from_auto_offset_reset, from_instance_id, handover_rule, effects, workers, fold_handovers, from_auto_offset_reset, from_instance_id (+1 more)

### Community 3131 - "judge"
Cohesion: 0.22
Nodes (9): boundary_dlq_enabled, compose_project, effects_port, main_port, secret_resolver_config_path, services, topic_provisioner_max_partitions, trajectory_evaluation_backend (+1 more)

### Community 3132 - "lakshman"
Cohesion: 0.22
Nodes (9): boundary_dlq_enabled, compose_project, effects_port, main_port, secret_resolver_config_path, services, topic_provisioner_max_partitions, trajectory_evaluation_backend (+1 more)

### Community 3133 - "tenant-projection"
Cohesion: 0.22
Nodes (9): tenant-projection, services, bifrost_verify_endpoints, capabilities, omnimemory_enabled, omnimemory_memgraph_host, publish_introspection, runtime_address (+1 more)

### Community 3134 - "worker"
Cohesion: 0.22
Nodes (9): worker, bifrost_verify_endpoints, capabilities, omnimemory_enabled, omnimemory_memgraph_host, publish_introspection, replicas, runtime_address (+1 more)

### Community 3135 - "runtime-entry-points.yaml"
Cohesion: 0.22
Nodes (8): allowed_commands, entry_point_families, label, pattern, launcher_marks_runtime_image, launcher_prefix, runtime_image_patterns, schema_version

### Community 3136 - "units"
Cohesion: 0.22
Nodes (8): units, hosts, installed, installer, installer_args, name, render_registry_root_env, tracked

### Community 3137 - "memgraph"
Cohesion: 0.22
Nodes (9): OMNIMEMORY_ENABLED, OMNIMEMORY_MEMGRAPH_HOST, OMNIMEMORY_MEMGRAPH_PORT, memgraph, description, includes, inject_env, inject_required_env (+1 more)

### Community 3138 - "operational_defaults"
Cohesion: 0.22
Nodes (9): operational_defaults, CORS_ORIGINS, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_ENVIRONMENT, ONEX_LOG_LEVEL, ONEX_SERVICE_CLIENT_ID (+1 more)

### Community 3139 - "operational_defaults"
Cohesion: 0.22
Nodes (9): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL, ONEX_SERVICE_CLIENT_ID (+1 more)

### Community 3140 - "operational_defaults"
Cohesion: 0.22
Nodes (9): operational_defaults, KEYCLOAK_ADMIN_CLIENT_ID, KEYCLOAK_ISSUER, KEYCLOAK_REALM, ONEX_ENVIRONMENT, ONEX_GROUP_ID, ONEX_LOG_LEVEL, ONEX_SERVICE_CLIENT_ID (+1 more)

### Community 3141 - "volumes"
Cohesion: 0.22
Nodes (9): name, name, name, name, volumes, postgres_data, redpanda_data, runtime_logs (+1 more)

### Community 3142 - "volumes"
Cohesion: 0.22
Nodes (9): name, name, name, name, volumes, hp051-ci-mq-runner-1-creds, hp051-ci-mq-runner-2-creds, hp051-ci-mq-runner-3-creds (+1 more)

### Community 3143 - "options"
Cohesion: 0.33
Nodes (9): logging, logging, logging, logging, driver, options, max-file, max-size (+1 more)

### Community 3144 - "x-runner-base"
Cohesion: 0.22
Nodes (9): x-runner-base, cgroup_parent, cpus, group_add, image, mem_limit, memswap_limit, pids_limit (+1 more)

### Community 3145 - "options"
Cohesion: 0.33
Nodes (9): logging, logging, logging, logging, driver, options, max-file, max-size (+1 more)

### Community 3146 - "x-runner-base"
Cohesion: 0.22
Nodes (9): x-runner-base, cgroup_parent, cpus, group_add, image, mem_limit, memswap_limit, pids_limit (+1 more)

### Community 3147 - "volumes"
Cohesion: 0.22
Nodes (9): name, name, name, name, volumes, hp053-ci-mq-runner-1-creds, hp053-ci-mq-runner-2-creds, hp053-ci-mq-runner-3-creds (+1 more)

### Community 3148 - "options"
Cohesion: 0.33
Nodes (9): logging, logging, logging, logging, driver, options, max-file, max-size (+1 more)

### Community 3149 - "x-runner-base"
Cohesion: 0.22
Nodes (9): x-runner-base, cgroup_parent, cpus, group_add, image, mem_limit, memswap_limit, pids_limit (+1 more)

### Community 3150 - "x-runner-base"
Cohesion: 0.22
Nodes (9): x-runner-base, cgroup_parent, cpus, group_add, image, mem_limit, memswap_limit, pids_limit (+1 more)

### Community 3151 - "x-runner-base"
Cohesion: 0.22
Nodes (8): services, x-runner-base, cpus, image, mem_limit, memswap_limit, pids_limit, restart

### Community 3152 - "options"
Cohesion: 0.33
Nodes (9): driver, options, logging, logging, logging, logging, max-file, max-size (+1 more)

### Community 3153 - "volumes"
Cohesion: 0.22
Nodes (9): name, name, name, name, volumes, omninode-studio-ci-runner-1-creds, omninode-studio-ci-runner-2-creds, omninode-studio-ci-runner-3-creds (+1 more)

### Community 3154 - "lane_mirror_source_bus"
Cohesion: 0.22
Nodes (8): lane_mirror_buses, lane_mirror_source_bus, auto_offset_reset, bootstrap_servers, enable_auto_commit, environment, security_protocol, timeout_seconds

### Community 3155 - "dev"
Cohesion: 0.22
Nodes (9): auto_offset_reset, bootstrap_servers, enable_auto_commit, environment, sasl_credential_ref, sasl_mechanism, security_protocol, timeout_seconds (+1 more)

### Community 3156 - "local_bus"
Cohesion: 0.22
Nodes (9): local_bus, auto_offset_reset, bootstrap_servers, enable_auto_commit, environment, sasl_credential_ref, sasl_mechanism, security_protocol (+1 more)

### Community 3158 - "inputs"
Cohesion: 0.22
Nodes (9): default, description, required, inputs, cache-enabled, shared-env-root, default, description (+1 more)

### Community 3159 - "pins"
Cohesion: 0.22
Nodes (8): pins, omnibase_core, omniclaude, omnidash, omniintelligence, omnimarket, omnimemory, omninode_infra

### Community 3160 - "build-and-push-migrate-image.yml"
Cohesion: 0.22
Nodes (8): concurrency, cancel-in-progress, group, jobs, name, permissions, contents, id-token

### Community 3161 - "steps"
Cohesion: 0.22
Nodes (9): name, runs-on, steps, timeout-minutes, c11-negative-paths, if, name, run (+1 more)

### Community 3162 - "inputs"
Cohesion: 0.22
Nodes (9): other_tenant_slug, on, schedule, workflow_dispatch, default, description, required, cron (+1 more)

### Community 3163 - "contractor-integration-note.yml"
Cohesion: 0.22
Nodes (8): concurrency, cancel-in-progress, group, jobs, name, permissions, contents, pull-requests

### Community 3164 - "open-bump-pr"
Cohesion: 0.22
Nodes (9): open-bump-pr, repo, name, needs, runs-on, strategy, timeout-minutes, fail-fast (+1 more)

### Community 3165 - "env"
Cohesion: 0.22
Nodes (9): CROSS_REPO_PAT, GH_TOKEN, LANE_PROBE_HOST, ONEX_LANE_KEEPALIVE, ROUTE_REASON, RUNNER_FLEET_STATUS_TOKEN, SLACK_BOT_TOKEN, SLACK_CHANNEL_ID (+1 more)

### Community 3166 - "dlq-depth-monitor.yml"
Cohesion: 0.22
Nodes (8): concurrency, cancel-in-progress, group, env, PYTHON_VERSION, UV_VERSION, jobs, name

### Community 3167 - "inputs"
Cohesion: 0.22
Nodes (9): window_seconds, on, schedule, workflow_dispatch, cron, default, description, required (+1 more)

### Community 3168 - "evidence-autoclose-sweep"
Cohesion: 0.22
Nodes (9): SWEEP_DISPATCH_APPLY, SWEEP_TRIGGER, env, name, permissions, runs-on, timeout-minutes, evidence-autoclose-sweep (+1 more)

### Community 3169 - "steps"
Cohesion: 0.36
Nodes (9): steps, steps, steps, id, if, name, run, uses (+1 more)

### Community 3170 - "apply"
Cohesion: 0.22
Nodes (9): default, description, type, apply, on, schedule, workflow_dispatch, cron (+1 more)

### Community 3171 - "refresh_after_days"
Cohesion: 0.22
Nodes (9): refresh_after_days, on, schedule, workflow_dispatch, default, description, required, cron (+1 more)

### Community 3172 - "steps"
Cohesion: 0.22
Nodes (9): if, runs-on, steps, timeout-minutes, autobump, id, name, run (+1 more)

### Community 3173 - "env"
Cohesion: 0.22
Nodes (9): DEFAULT_BRANCH, DRY_RUN, EVENT_NAME, GH_TOKEN, OSV_SCANNER_SHA256, OSV_SCANNER_URL, REPOSITORY, RUN_REF (+1 more)

### Community 3174 - "steps"
Cohesion: 0.31
Nodes (9): SLACK_BOT_TOKEN, steps, steps, steps, env, if, name, run (+1 more)

### Community 3175 - "success_threshold"
Cohesion: 0.22
Nodes (9): success_threshold, on, schedule, workflow_dispatch, cron, default, description, required (+1 more)

### Community 3176 - "trigger-delivery"
Cohesion: 0.22
Nodes (9): trigger-delivery, correlation_id, published, runtime_lane, source_sha, name, outputs, runs-on (+1 more)

### Community 3177 - "secrets"
Cohesion: 0.22
Nodes (9): required, required, required, required, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, ONEXBOT_OCC_APP_ID, ONEXBOT_OCC_PRIVATE_KEY (+1 more)

### Community 3178 - "with"
Cohesion: 0.22
Nodes (9): with, app-id, owner, permission-contents, permission-pull-requests, private-key, python-version, repositories (+1 more)

### Community 3179 - "detect"
Cohesion: 0.22
Nodes (8): name, runs-on, timeout-minutes, jobs, detect, name, permissions, contents

### Community 3180 - "apply"
Cohesion: 0.22
Nodes (9): default, description, type, apply, on, schedule, workflow_dispatch, cron (+1 more)

### Community 3181 - "release-train-lab.yml"
Cohesion: 0.22
Nodes (8): concurrency, cancel-in-progress, group, jobs, name, on, push, tags

### Community 3182 - "steps"
Cohesion: 0.31
Nodes (9): steps, steps, id, if, name, run, uses, with (+1 more)

### Community 3183 - "env"
Cohesion: 0.22
Nodes (9): BASE_BRANCH, CANDIDATE_SHA, GH_TOKEN, NEEDS_BUMP, SELECTED, TARGET_REPO, TICKET, VERSION (+1 more)

### Community 3184 - "env"
Cohesion: 0.22
Nodes (9): CROSS_REPO_PAT, EVENT_NAME, GITHUB_TOKEN, HEAD_REPO, PUBLIC_JSON, REPOSITORY, RUNNER_FLEET_STATUS_TOKEN, SEAM_JSON (+1 more)

### Community 3185 - "with"
Cohesion: 0.22
Nodes (9): with, cache-key-prefix, if-no-files-found, install-args, name, path, persist-credentials, python-version (+1 more)

### Community 3186 - "steps"
Cohesion: 0.39
Nodes (9): steps, continue-on-error, id, if, name, run, uses, steps (+1 more)

### Community 3187 - "with"
Cohesion: 0.22
Nodes (9): with, category, config-file, fetch-depth, languages, persist-credentials, queries, upload (+1 more)

### Community 3198 - "entries"
Cohesion: 0.22
Nodes (8): entries, expires, kind, path, symbol, target, ticket, schema_version

### Community 3199 - "_apply_overlays"
Cohesion: 0.31
Nodes (4): _apply_overlays(), _apply_single_overlay(), _merge_dict_field(), _selector_matches()

### Community 3202 - "ModelLockfileCveVerdict"
Cohesion: 0.33
Nodes (3): _format_report(), ModelLockfileCveVerdict, ModelLockfileFinding

### Community 3203 - "_api_get"
Cohesion: 0.28
Nodes (4): _api_get(), _is_rate_limited_403(), _is_transient_http_status(), _rate_limit_backoff_seconds()

### Community 3204 - "integration_skip_guard.yaml"
Cohesion: 0.22
Nodes (8): allowed_optional_skip_patterns, curated_test_paths, missing_skip_patterns, provisioned_by, require_executed_min, required_services, postgres, silent_skip_allowed

### Community 3206 - "runner_fleet_canary.sh"
Cohesion: 0.47
Nodes (8): config_field(), declared_action_pools(), fail(), fetch_page(), log(), runner_fleet_canary.sh script, slack_alert(), summary()

### Community 3207 - "test_selection_adjacency.yaml"
Cohesion: 0.22
Nodes (8): boundary_integration_tests, src/omnibase_infra/event_bus/kafka_auth.py, src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py, schema_version, shared_modules, test_infrastructure_paths, thresholds, modules_changed_for_full_suite

### Community 3208 - "edge_delegation_worker/cli.py"
Cohesion: 0.31
Nodes (4): _amain(), build_arg_parser(), main(), build_kafka_channel()

### Community 3209 - "monitor_alert_contract.yaml"
Cohesion: 0.22
Nodes (8): contract_version, major, minor, patch, description, event_bus, publish_topics, name

### Community 3210 - "compose_files.sh"
Cohesion: 0.39
Nodes (7): _compose_files_log_error(), export_lane_overlay_dir(), _lane_overlay_dir_candidate(), resolve_compose_file_args(), resolve_lane_overlay_dir(), resolve_lane_overlay_filename(), compose_files.sh script

### Community 3215 - "qwen2.5-7b"
Cohesion: 0.22
Nodes (9): qwen2.5-7b, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3216 - "qwen2.5-coder-14b"
Cohesion: 0.22
Nodes (9): qwen2.5-coder-14b, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3217 - "qwen2-vl"
Cohesion: 0.22
Nodes (9): qwen2-vl, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3218 - "qwen3.6-27b-mtp-iq4xs"
Cohesion: 0.22
Nodes (9): qwen3.6-27b-mtp-iq4xs, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3219 - "qwen3.6-35b-a3b"
Cohesion: 0.22
Nodes (9): qwen3.6-35b-a3b, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3220 - "text-embedding-gte"
Cohesion: 0.22
Nodes (9): text-embedding-gte, confidence, effective_date, evidence, input_cost_per_1k, note, output_cost_per_1k, sample_count (+1 more)

### Community 3221 - "defaults"
Cohesion: 0.22
Nodes (9): event.emit, health.heartbeat, auto_scroll, show_last_seen, show_timestamp, defaults, widget_type, defaults (+1 more)

### Community 3222 - "db.execute"
Cohesion: 0.22
Nodes (9): db.execute, db.query, expression, parameter_name, required, expression, parameter_name, required (+1 more)

### Community 3227 - "input_fields"
Cohesion: 0.22
Nodes (9): correlation_id, created_after, created_before, limit, manifest, manifest_id, metadata_only, node_id (+1 more)

### Community 3228 - "operations"
Cohesion: 0.22
Nodes (9): operations, complexity, description, idempotent, notes, operation, output_model, module (+1 more)

### Community 3233 - "settings"
Cohesion: 0.22
Nodes (8): settings, bind_all_address, local_host, local_kafka_bootstrap_servers, local_valkey_port, loopback_address, loopback_host_names, runtime_health_port

### Community 3234 - "ModelProjectionContractRef"
Cohesion: 0.25
Nodes (3): ModelProjectionContractRef, _declared_projection_refs(), select_projection_contracts()

### Community 3235 - "intent_targets"
Cohesion: 0.22
Nodes (8): intent_targets, build_loop_runs, contracts, event_ledger, gateway_link_health, node_registrations, topics, validation_event_ledger

### Community 3241 - "node_api_fixture_record_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_name, description, name, node_name, node_type, runtime_profiles, tags, terminal_event

### Community 3242 - "recording_policy"
Cohesion: 0.22
Nodes (9): config, recording_policy, cleanup, count_resources, credentials, expected_route_count, output, response_headers (+1 more)

### Community 3243 - "node_architecture_validator/handlers/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, description, handler_id, name, node_type, tags

### Community 3244 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, operation (+1 more)

### Community 3245 - "coordination_rules"
Cohesion: 0.22
Nodes (9): checkpoint_enabled, execution_mode, failure_recovery_strategy, max_retries, parallel_execution_allowed, rollback_enabled, state_persistence_enabled, timeout_ms (+1 more)

### Community 3246 - "nodes"
Cohesion: 0.22
Nodes (9): nodes, depends_on, description, node_id, node_type, step_config, handler, skip_condition (+1 more)

### Community 3248 - "node_baseline_comparison_compute/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3249 - "node_broker_disk_watermark_compute/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3250 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3251 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3252 - "event_bus"
Cohesion: 0.22
Nodes (9): event_bus, auto_offset_reset, consumer_group, consumer_purpose, subscribe_topics, version, major, minor (+1 more)

### Community 3253 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, operation (+1 more)

### Community 3254 - "health_check"
Cohesion: 0.22
Nodes (9): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds (+1 more)

### Community 3255 - "io_operations"
Cohesion: 0.22
Nodes (9): io_operations, description, idempotent, operation, output_fields, id, run_id, success (+1 more)

### Community 3256 - "cloud_leg"
Cohesion: 0.22
Nodes (9): acl_provisioner_ref, broker_provider_id, cloud_auth_ref, cloud_broker_ref, msk_region_ref, sasl_mechanism, security_protocol, transport (+1 more)

### Community 3257 - "handlers"
Cohesion: 0.22
Nodes (9): handler_type, module, name, handler_routing, handlers, routing_strategy, description, handler (+1 more)

### Community 3259 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3260 - "node_coding_agent_invoke_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): description, event_bus, publish_topics, subscribe_topics, name, node_type, runtime_profiles, terminal_event

### Community 3261 - "node_coding_agent_orchestrator/contract.yaml"
Cohesion: 0.22
Nodes (8): description, event_bus, publish_topics, subscribe_topics, name, node_type, runtime_profiles, terminal_event

### Community 3262 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3263 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3264 - "operations"
Cohesion: 0.22
Nodes (9): operations, description, idempotent, operation_name, operation_timeout_ms, response_handling, extraction_engine, fail_on_empty (+1 more)

### Community 3265 - "security"
Cohesion: 0.22
Nodes (9): security, deny_absolute_paths, deny_patterns, deny_symlink_escape, enabled, allowed_base_paths, path_validation, require_path_under_base (+1 more)

### Community 3266 - "rules"
Cohesion: 0.22
Nodes (9): error_code, error_message, field, invalid_error_code, required, valid_values, validation, enabled (+1 more)

### Community 3267 - "node_contract_validate_compute/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_name, description, name, node_name, node_type, runtime_profiles, status, terminal_event

### Community 3268 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3269 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3270 - "node_decision_store_query_compute/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3271 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, default_handler, handlers, routing_strategy, description, handler (+1 more)

### Community 3272 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3273 - "node_delegation_dispatch_terminal_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3274 - "node_deploy_job_reading_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_name, description, descriptor, node_archetype, name, node_name, node_type, runtime_profiles

### Community 3275 - "node_dlq_depth_monitor_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_version, major, minor, patch, description, name, node_type, node_version

### Community 3276 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3277 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3278 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3279 - "event_bus"
Cohesion: 0.22
Nodes (9): event_bus, auto_offset_reset, consumer_group, consumer_purpose, subscribe_topics, version, major, minor (+1 more)

### Community 3280 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, operation (+1 more)

### Community 3281 - "health_check"
Cohesion: 0.22
Nodes (9): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds (+1 more)

### Community 3282 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3283 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3284 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3285 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3286 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3287 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, author, created, description, license, tags, ticket, topic_registry_source (+1 more)

### Community 3288 - "node_in_progress_probe_hygiene_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): description, name, node_type, node_version, output_model, description, module, name

### Community 3289 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3291 - "node_lab_proof_run_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3292 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3293 - "event_bus"
Cohesion: 0.22
Nodes (9): event_bus, auto_offset_reset, consumer_group, consumer_purpose, subscribe_topics, version, major, minor (+1 more)

### Community 3294 - "health_check"
Cohesion: 0.22
Nodes (9): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds (+1 more)

### Community 3295 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3296 - "node_llm_embedding_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, description, name, node_type, node_version, runtime_profiles

### Community 3297 - "node_node_broker_resolve_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_name, description, descriptor, node_archetype, name, node_name, node_type, runtime_profiles

### Community 3298 - "inmemory"
Cohesion: 0.22
Nodes (9): campaign_interval_s, fence_tick_s, keepalive_interval_s, margin_s, min_serving, ttl_s, placement_runtime, profiles (+1 more)

### Community 3299 - "rules"
Cohesion: 0.22
Nodes (9): error_code, error_message, field, invalid_error_code, required, valid_values, validation, enabled (+1 more)

### Community 3300 - "health_check"
Cohesion: 0.22
Nodes (9): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds (+1 more)

### Community 3301 - "handlers"
Cohesion: 0.25
Nodes (9): handler_routing, handlers, routing_strategy, input_model, operation, input_model, description, module (+1 more)

### Community 3302 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, author, created, description, license, tags, ticket, transport_type (+1 more)

### Community 3303 - "state_machine"
Cohesion: 0.22
Nodes (9): state_machine, initial_state, persistence_enabled, state_machine_name, states, terminal_states, description, required_data (+1 more)

### Community 3304 - "node_runner_usage_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3305 - "consumes"
Cohesion: 0.22
Nodes (9): description, schema, topic, description, schema, topic, topics, consumes (+1 more)

### Community 3306 - "node_runtime_error_triage_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): contract_name, description, escalation, new_fingerprint_severity, occurrence_threshold, name, node_name, node_type

### Community 3307 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3308 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3309 - "config"
Cohesion: 0.22
Nodes (9): config, counterfactual_policy, description, module, name, fallback_model, rules, counterfactual_model (+1 more)

### Community 3310 - "node_session_state_effect/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3311 - "io_operations"
Cohesion: 0.22
Nodes (9): context, correlation_id, index, run_id, io_operations, description, idempotent, input_fields (+1 more)

### Community 3312 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, operation (+1 more)

### Community 3313 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3314 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3315 - "health_check"
Cohesion: 0.22
Nodes (9): check_type, description, env_var, name, health_check, checks, enabled, endpoint (+1 more)

### Community 3316 - "handlers"
Cohesion: 0.22
Nodes (9): module, name, handler_routing, handlers, routing_strategy, description, handler, operation (+1 more)

### Community 3317 - "event_bus"
Cohesion: 0.22
Nodes (9): event_bus, auto_offset_reset, consumer_group, consumer_purpose, subscribe_topics, version, major, minor (+1 more)

### Community 3318 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, author, created, description, license, related_tickets, tags, ticket (+1 more)

### Community 3319 - "health_check"
Cohesion: 0.22
Nodes (9): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds (+1 more)

### Community 3320 - "coordination_rules"
Cohesion: 0.22
Nodes (9): checkpoint_enabled, execution_mode, failure_recovery_strategy, max_retries, parallel_execution_allowed, rollback_enabled, state_persistence_enabled, timeout_ms (+1 more)

### Community 3323 - "node_waste_detection_compute/contract.yaml"
Cohesion: 0.22
Nodes (8): capabilities, description, name, contract_name, description, name, node_name, node_type

### Community 3324 - "consumes"
Cohesion: 0.22
Nodes (9): description, schema, topic, description, schema, topic, topics, consumes (+1 more)

### Community 3327 - "steps"
Cohesion: 0.22
Nodes (9): steps, action, condition, id, options, produces_capabilities, prompt, required (+1 more)

### Community 3330 - "load_and_validate_contract_yaml"
Cohesion: 0.25
Nodes (3): _dispatch_contract_model(), load_and_validate_contract_yaml(), ModelContractNodeType

### Community 3331 - "select_nonprojection_group_infixes"
Cohesion: 0.22
Nodes (4): _contract_group_infix(), select_nonprojection_group_infixes(), select_projection_group_suffixes(), _unambiguous_group_infixes()

### Community 3334 - "create_app"
Cohesion: 0.28
Nodes (3): create_app(), create_mounted_app(), lifespan()

### Community 3339 - "ProducerZeroOutputError"
Cohesion: 0.28
Nodes (3): assert_producer_emitted(), ProducerZeroOutputError, require_producer_preconditions()

### Community 3342 - "omnibase_infra"
Cohesion: 0.25
Nodes (8): default_branch, dependency_refresh, lab_evidence, lab_evidence_note, mode, package, release_relevant_paths, omnibase_infra

### Community 3343 - "model_review"
Cohesion: 0.25
Nodes (8): model_review, active, credential_reference_id, endpoint_reference_id, healthcheck_reference_id, max_observation_age_seconds, runner_group, runner_label

### Community 3344 - "pypi_cache"
Cohesion: 0.25
Nodes (8): pypi_cache, active, canary_runners, fallback_index_url, host, port, simple_index_url, target_hit_rate

### Community 3345 - "omnibase_infra"
Cohesion: 0.25
Nodes (8): LAB_PROBE_RUNS_ON_JSON, LANE_CENSUS_RUNS_ON_JSON, OMNI_DOCKER_CI_RUNS_ON_JSON, OMNI_PUBLIC_PR_RUNS_ON_JSON, OMNI_REQUIRED_CI_RUNS_ON_JSON, OMNI_SECURITY_SCAN_RUNS_ON_JSON, OMNI_TRUSTED_CI_RUNS_ON_JSON, omnibase_infra

### Community 3346 - "saturation_alert"
Cohesion: 0.25
Nodes (8): route, hosted_workflows, saturation_alert, busy_fraction_threshold, lab_load_ratio_threshold, min_lab_free_mem_mib, sustained_min_span_seconds, sustained_samples

### Community 3347 - "allowed_commands"
Cohesion: 0.25
Nodes (8): allowed_commands, argv, argv_sha256, command_hint, field, match, reason, services

### Community 3348 - "OMN-19404.yaml"
Cohesion: 0.25
Nodes (7): interface_change, interfaces_touched, is_seam_ticket, schema_version, summary, ticket_id, title

### Community 3349 - "dod-omn19404-ac3-slot-topics-accepted"
Cohesion: 0.25
Nodes (8): dod-omn19404-ac3-slot-topics-accepted, context, accepted_ac_bindings, accepted_at, accepted_by, criterion_hash, label, proposed_by

### Community 3350 - "default_retry_policy"
Cohesion: 0.25
Nodes (8): backoff_strategy, base_delay_ms, enabled, jitter_factor, max_delay_ms, max_retries, retryable_errors, default_retry_policy

### Community 3351 - "io_config"
Cohesion: 0.25
Nodes (8): atomic, create_dirs, encoding, file_path_template, handler_type, operation, timeout_ms, io_config

### Community 3352 - "retry_policy"
Cohesion: 0.25
Nodes (8): retry_policy, backoff_strategy, base_delay_ms, enabled, jitter_factor, max_delay_ms, max_retries, retryable_errors

### Community 3353 - "methodology"
Cohesion: 0.25
Nodes (8): cpu_cores, memory_gb, storage, baseline_environment, measurement_percentile, minimum_sample_size, warmup_iterations, methodology

### Community 3354 - "feature_flags"
Cohesion: 0.25
Nodes (7): description, feature_flags, default, description, env_var, name, name

### Community 3355 - "feature_flags"
Cohesion: 0.25
Nodes (7): description, feature_flags, default, description, env_var, name, name

### Community 3356 - "prod"
Cohesion: 0.25
Nodes (8): boundary_dlq_enabled, compose_project, effects_port, main_port, services, topic_provisioner_max_partitions, trajectory_evaluation_backend, prod

### Community 3357 - "stability-test"
Cohesion: 0.25
Nodes (8): stability-test, boundary_dlq_enabled, compose_project, effects_port, main_port, secret_resolver_config_path, topic_provisioner_max_partitions, trajectory_evaluation_backend

### Community 3358 - "omninode-host-maintenance-sync.sh"
Cohesion: 0.46
Nodes (7): die(), install_blob(), installed_sha(), ref_blob_exists(), ref_blob_sha(), omninode-host-maintenance-sync.sh script, sha_of_stdin()

### Community 3359 - "core"
Cohesion: 0.25
Nodes (8): core, description, includes, inject_env, inject_required_env, services, INFISICAL_ADDR, ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS

### Community 3360 - "environment"
Cohesion: 0.36
Nodes (8): environment, environment, environment, environment, CI_BUS_KAFKA_SASL_PASSWORD, CI_BUS_KAFKA_SASL_USERNAME, CI_BUS_LABWORK_SASL_PASSWORD, CI_BUS_LABWORK_SASL_USERNAME

### Community 3361 - "ci-bus-topics"
Cohesion: 0.25
Nodes (8): command, container_name, entrypoint, image, labels, networks, restart, ci-bus-topics

### Community 3362 - "postgres"
Cohesion: 0.25
Nodes (8): container_name, image, networks, ports, profiles, restart, volumes, postgres

### Community 3363 - "postgres"
Cohesion: 0.25
Nodes (8): command, container_name, image, labels, networks, ports, volumes, postgres

### Community 3364 - "valkey"
Cohesion: 0.25
Nodes (8): valkey, command, container_name, image, labels, networks, ports, volumes

### Community 3365 - "volumes"
Cohesion: 0.54
Nodes (8): volumes, volumes, volumes, bind, read_only, source, target, type

### Community 3366 - "environment"
Cohesion: 0.43
Nodes (8): BIFROST_LANE_OVERLAY_PATH, ONEX_BOX_ID, ONEX_RUNTIME_ADDRESS, ONEX_RUNTIME_LANE, OTEL_SERVICE_NAME, environment, environment, environment

### Community 3367 - "projection-delegation-writer"
Cohesion: 0.25
Nodes (8): command, container_name, labels, networks, profiles, restart, stop_grace_period, projection-delegation-writer

### Community 3368 - "projection-tenant-registry-writer"
Cohesion: 0.25
Nodes (8): command, container_name, labels, networks, profiles, restart, stop_grace_period, projection-tenant-registry-writer

### Community 3369 - "toolcache-seed.sh"
Cohesion: 0.46
Nodes (6): cmd_report(), cmd_restore(), cmd_snapshot(), container_entries(), toolcache-seed.sh script, usage()

### Community 3370 - "steps"
Cohesion: 0.25
Nodes (8): steps, id, if, name, run, shell, uses, working-directory

### Community 3371 - "PULL_REQUEST_TEMPLATE.md"
Cohesion: 0.25
Nodes (7): Changes, Related issues, Runtime startup gate, Strict-gate ordering, Summary, Test plan, Type safety checklist

### Community 3372 - "resolve-lane"
Cohesion: 0.25
Nodes (8): resolve-lane, docker_runs_on, lane, lane_json, name, outputs, runs-on, timeout-minutes

### Community 3373 - "on"
Cohesion: 0.25
Nodes (8): on, merge_group, pull_request, push, workflow_dispatch, branches, types, branches

### Community 3374 - "build-runtime-candidate"
Cohesion: 0.25
Nodes (8): name, secrets, uses, with, build-runtime-candidate, pinned_repo, pinned_revision, sibling_ref

### Community 3375 - "steps"
Cohesion: 0.50
Nodes (8): steps, steps, steps, id, if, name, run, uses

### Community 3376 - "lab-pass-gate"
Cohesion: 0.25
Nodes (8): lab-pass-gate, name, needs, permissions, runs-on, timeout-minutes, actions, pull-requests

### Community 3377 - "env"
Cohesion: 0.25
Nodes (8): BUS_LANE, KAFKA_SASL_PASSWORD, KAFKA_SASL_USERNAME, ONEX_ALLOW_OMNIMARKET_DRIFT, SUPPRESS_ALERT_EXIT, TOPIC_PREFIX, WINDOW_SECONDS, env

### Community 3378 - "in-progress-probe-hygiene.yml"
Cohesion: 0.25
Nodes (7): env, PYTHON_VERSION, UV_VERSION, jobs, name, permissions, contents

### Community 3379 - "env"
Cohesion: 0.25
Nodes (8): CONTRACT_REPOS, CONTRACT_REPOS_PRIVATE, GH_TOKEN, LINEAR_API_KEY, MAX_COMMENTS, PROJECT_ID, SWEEP_APPLY, env

### Community 3380 - "lane-census-refresh.yml"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, jobs, name, permissions, contents

### Community 3381 - "plugin-pin-cascade.yml"
Cohesion: 0.25
Nodes (7): env, PLUGIN_PIN_CASCADE_STANDING_TICKET, jobs, name, permissions, contents, pull-requests

### Community 3382 - "product-readiness-shadow.yml"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, jobs, name, permissions, contents

### Community 3383 - "release"
Cohesion: 0.25
Nodes (8): release, evidence_source, ticket, version, if, outputs, runs-on, timeout-minutes

### Community 3384 - "sync-main"
Cohesion: 0.25
Nodes (8): sync-main, permissions, contents, if, name, permissions, runs-on, timeout-minutes

### Community 3385 - "deploy"
Cohesion: 0.25
Nodes (8): env, if, name, needs, runs-on, timeout-minutes, TRIGGERING_TAG, deploy

### Community 3386 - "release-train-nightly.yml"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, jobs, name, permissions, contents

### Community 3387 - "steps"
Cohesion: 0.36
Nodes (8): steps, steps, continue-on-error, id, if, name, run, uses

### Community 3388 - "steps"
Cohesion: 0.36
Nodes (8): steps, steps, id, if, name, run, uses, working-directory

### Community 3389 - "with"
Cohesion: 0.25
Nodes (8): with, enable-cache, if-no-files-found, name, path, python-version, retention-days, version

### Community 3390 - "steps"
Cohesion: 0.36
Nodes (8): steps, steps, steps, steps, id, name, run, uses

### Community 3391 - "route-inline"
Cohesion: 0.25
Nodes (8): route-inline, decision, labels, reason, name, outputs, runs-on, timeout-minutes

### Community 3392 - "route-reusable"
Cohesion: 0.25
Nodes (8): route-reusable, name, secrets, uses, with, CROSS_REPO_PAT, RUNNER_FLEET_STATUS_TOKEN, workflow_path

### Community 3393 - "verify-lane-converged"
Cohesion: 0.25
Nodes (8): verify-lane-converged, reemit_candidates, if, name, needs, outputs, runs-on, timeout-minutes

### Community 3394 - "security-scan.yml"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, env, TRUFFLEHOG_IMAGE, jobs, name

### Community 3395 - "codeql"
Cohesion: 0.25
Nodes (8): if, name, permissions, runs-on, timeout-minutes, codeql, actions, security-events

### Community 3396 - "on"
Cohesion: 0.25
Nodes (8): on, pull_request, push, schedule, branches, types, branches, cron

### Community 3410 - "gate_runner_supervisor.sh"
Cohesion: 0.36
Nodes (4): die(), heartbeat_loop(), gate_runner_supervisor.sh script, write_receipt()

### Community 3411 - "diagnose_e2e.sh"
Cohesion: 0.54
Nodes (7): check_required_log(), log_error(), log_info(), log_success(), log_warning(), section_header(), diagnose_e2e.sh script

### Community 3412 - "render_desired_state"
Cohesion: 0.25
Nodes (3): render_desired_state(), dispatch(), import_model_cls()

### Community 3414 - "runtime_health_wait.sh"
Cohesion: 0.29
Nodes (4): runtime_boot_window_seconds(), runtime_health_budget_seconds(), runtime_started_at(), runtime_health_wait.sh script

### Community 3415 - "ModelCiBusOverlay"
Cohesion: 0.25
Nodes (3): ModelCiBusOverlay, resolve_ci_bus_broker(), resolve_ci_bus_security()

### Community 3416 - "packages"
Cohesion: 0.25
Nodes (7): packages, omnibase-core, omnibase-infra, omnibase-spi, repos, label, name

### Community 3417 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-haiku-4-5

### Community 3418 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-opus-4-5

### Community 3419 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-opus-4-6

### Community 3420 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-opus-5-5

### Community 3421 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-sonnet-4-20250514

### Community 3422 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-sonnet-4-6

### Community 3423 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, claude-sonnet-5-5

### Community 3424 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, gemini-2.5-flash

### Community 3425 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, gemini-2.5-flash-lite

### Community 3426 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, gpt-4-turbo

### Community 3427 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, gpt-4o

### Community 3428 - "evidence"
Cohesion: 0.25
Nodes (8): confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source, gpt-4o-mini

### Community 3429 - "evidence"
Cohesion: 0.25
Nodes (8): o1, confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source

### Community 3430 - "evidence"
Cohesion: 0.25
Nodes (8): o3-mini, confidence, effective_date, evidence, input_cost_per_1k, output_cost_per_1k, sample_count, source

### Community 3431 - "event.transform"
Cohesion: 0.25
Nodes (8): event.transform, workflow.orchestrate, show_duration, show_steps, defaults, widget_type, defaults, widget_type

### Community 3432 - "handler_routing"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, supported_operations

### Community 3433 - "WORKSPACE_ROOT"
Cohesion: 0.25
Nodes (8): environment_variables, initialization_policy, WORKSPACE_ROOT, configuration_requirements, description, missing_behavior, required, validation

### Community 3434 - "handler_routing"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, supported_operations

### Community 3435 - "handler_routing"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, handler, handler_type, supported_operations

### Community 3441 - "ModelLaneEntryPointCensus"
Cohesion: 0.25
Nodes (3): render_table(), ModelContainerEntryPoint, ModelLaneEntryPointCensus

### Community 3442 - "kafka_publish_headers.py"
Cohesion: 0.32
Nodes (3): headers_for_unheadered_publish(), _uuid_or_none(), prepare()

### Community 3447 - "settings"
Cohesion: 0.25
Nodes (7): settings, bind_all_address, local_host, loopback_address, postgres_port, tcp_port_max, tcp_port_min

### Community 3450 - "error_types"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, max_retries

### Community 3451 - "error_types"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, description, name, recoverable, retry_strategy, max_retries

### Community 3452 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3453 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3456 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3457 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3458 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3459 - "node_broker_grant_derive_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, descriptor, node_archetype, name, node_name, node_type

### Community 3460 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3461 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3462 - "node_build_loop_projection_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): description, intent_consumption, intent_routing_table, build_loop.append, name, node_type, node_version

### Community 3463 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3464 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3465 - "input_fields"
Cohesion: 0.25
Nodes (8): correlation_id, event_type, id, payload, run_id, terminal_event_at, workflow_name, input_fields

### Community 3466 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3467 - "https_ingest"
Cohesion: 0.25
Nodes (8): https_ingest, idempotency_key, ingest_auth_ref, ingest_url_ref, max_batch_records, request_timeout_seconds, retry_initial_seconds, retry_max_seconds

### Community 3470 - "event_bus"
Cohesion: 0.25
Nodes (8): event_bus, dlq_topics, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3471 - "nodes"
Cohesion: 0.25
Nodes (8): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph

### Community 3477 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3479 - "node_coding_agent_fsm_reducer/contract.yaml"
Cohesion: 0.25
Nodes (7): description, event_bus, publish_topics, subscribe_topics, name, node_type, runtime_profiles

### Community 3480 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3481 - "node_coding_agent_workspace_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): description, event_bus, publish_topics, subscribe_topics, name, node_type, runtime_profiles

### Community 3482 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3483 - "node_consumer_coordinator_rejoin_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, descriptor, node_archetype, name, node_name, node_type

### Community 3484 - "extract_fields"
Cohesion: 0.25
Nodes (8): content, discovered_files, file_count, file_path, file_size, modified_at, scan_timestamp, extract_fields

### Community 3485 - "io_config"
Cohesion: 0.25
Nodes (8): atomic, create_dirs, encoding, file_path_template, handler_type, operation, timeout_ms, io_config

### Community 3486 - "metadata"
Cohesion: 0.25
Nodes (8): exclude_patterns, follow_symlinks, pattern, recursive, scan_mode, watch_debounce_ms, watch_mode, metadata

### Community 3487 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3488 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3489 - "config"
Cohesion: 0.25
Nodes (8): config, cors_allow_credentials, cors_allow_headers, cors_allow_methods, emit_kafka_events, host, port, request_timeout_seconds

### Community 3490 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3491 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3492 - "node_declared_state_diff_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, descriptor, node_archetype, name, node_name, node_type

### Community 3493 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, note, tags, ticket, updated

### Community 3495 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3496 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3497 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3498 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3499 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3500 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3501 - "node_dlq_replay_effect/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, descriptor, node_archetype, name, node_name, node_type

### Community 3503 - "node_event_provenance_stamp_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, descriptor, node_archetype, name, node_name, node_type

### Community 3504 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, gap_comment_fingerprint_version, tags, transport_type, updated

### Community 3505 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3506 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3507 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3508 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, known_gaps, tags, transport_type, updated

### Community 3509 - "node_gateway_link_health_projection_compute/contract.yaml"
Cohesion: 0.25
Nodes (7): description, intent_consumption, intent_routing_table, gateway_link_health.upsert, name, node_type, node_version

### Community 3510 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3511 - "io_operations"
Cohesion: 0.25
Nodes (8): io_operations, description, idempotent, operation, output_fields, success, tenant_id, was_insert

### Community 3512 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3513 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3514 - "dependencies"
Cohesion: 0.25
Nodes (8): dependencies, description, env_var, library, name, protocol, required, type

### Community 3515 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3516 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3517 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3518 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3519 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3520 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3521 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3522 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3523 - "handler"
Cohesion: 0.25
Nodes (8): handler_type, module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3525 - "dependencies"
Cohesion: 0.25
Nodes (8): dependencies, class_name, description, env_var, module, name, required, type

### Community 3526 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, related_tickets, tags, updated

### Community 3527 - "node_merge_provenance_observe_effect/contract.yaml"
Cohesion: 0.25
Nodes (7): contract_name, description, github_api, base_url, name, node_name, node_type

### Community 3528 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3529 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3530 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3531 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3532 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3533 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3534 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3535 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3536 - "catalog_service"
Cohesion: 0.25
Nodes (8): description, module, protocol, required, catalog_service, handler_routing, handler_dependencies, routing_strategy

### Community 3537 - "event_bus"
Cohesion: 0.25
Nodes (8): event_bus, dlq_topics, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3538 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3540 - "health_check"
Cohesion: 0.25
Nodes (8): check_type, name, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3541 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3542 - "event_bus"
Cohesion: 0.25
Nodes (8): event_bus, dlq_topics, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3543 - "nodes"
Cohesion: 0.25
Nodes (8): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph

### Community 3544 - "node_rsd_orchestrator/contract.yaml"
Cohesion: 0.25
Nodes (7): description, descriptor, node_archetype, name, node_type, node_version, runtime_profiles

### Community 3545 - "event_bus"
Cohesion: 0.25
Nodes (8): event_bus, dlq_topics, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3546 - "nodes"
Cohesion: 0.25
Nodes (8): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph

### Community 3547 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3548 - "nodes"
Cohesion: 0.25
Nodes (8): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph

### Community 3549 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, ticket, updated

### Community 3550 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3551 - "methodology"
Cohesion: 0.25
Nodes (8): cpu_cores, memory_gb, storage, baseline_environment, measurement_percentile, minimum_sample_size, warmup_iterations, methodology

### Community 3552 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3553 - "produces"
Cohesion: 0.25
Nodes (8): description, topic, description, schema, topic, topics, consumes, produces

### Community 3554 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3555 - "event_bus"
Cohesion: 0.25
Nodes (8): event_bus, dlq_topics, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3556 - "nodes"
Cohesion: 0.25
Nodes (8): nodes, depends_on, description, node_id, node_type, workflow_coordination, workflow_definition, execution_graph

### Community 3557 - "definitions"
Cohesion: 0.25
Nodes (8): definitions, ModelDiscoveryQuery, ModelServiceInfo, description, type, description, required, type

### Community 3558 - "metadata"
Cohesion: 0.25
Nodes (8): metadata, author, created, description, license, tags, type, metadata

### Community 3559 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3560 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3562 - "build"
Cohesion: 0.25
Nodes (8): branch, build_timeout_seconds, clone_timeout_seconds, image, read_timeout_seconds, repository_env, repository_url_env, build

### Community 3563 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3564 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3565 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3566 - "intents"
Cohesion: 0.25
Nodes (8): intent_emission, enabled, intents, description, intent_type, payload_model, payload_module, target_pattern

### Community 3567 - "health_check"
Cohesion: 0.25
Nodes (8): critical, name, type, health_check, checks, enabled, endpoint, interval_seconds

### Community 3569 - "handler"
Cohesion: 0.25
Nodes (8): handler_type, module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3571 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3572 - "handlers"
Cohesion: 0.25
Nodes (8): module, name, handler_routing, handlers, routing_strategy, description, handler, operation

### Community 3573 - "retry_policy"
Cohesion: 0.25
Nodes (8): error_handling, error_types, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3574 - "write_config_local"
Cohesion: 0.25
Nodes (8): write_config_local, KAFKA_BOOTSTRAP_SERVERS, LLM_CODER_URL, ONEX_DEPLOYMENT_MODE, POSTGRES_HOST, POSTGRES_PORT, VALKEY_HOST, VALKEY_PORT

### Community 3581 - "graph_queries.py"
Cohesion: 0.25
Nodes (3): build_dependency_chain_query(), build_file_conflict_query(), build_lineage_tree_query()

### Community 3583 - "classification"
Cohesion: 0.29
Nodes (7): classification, no-hardcoded-topics, self-gating-workflows, ci_workflow, tracking, pre_commit, tracking

### Community 3584 - "adapter"
Cohesion: 0.29
Nodes (7): deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs, adapter

### Community 3585 - "classification"
Cohesion: 0.29
Nodes (7): deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs, classification

### Community 3586 - "internal_catalog"
Cohesion: 0.29
Nodes (7): internal_catalog, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3587 - "one_database"
Cohesion: 0.29
Nodes (7): one_database, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3588 - "role_acl"
Cohesion: 0.29
Nodes (7): role_acl, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3589 - "schema_qualification"
Cohesion: 0.29
Nodes (7): schema_qualification, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3590 - "tenant_rls"
Cohesion: 0.29
Nodes (7): tenant_rls, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3591 - "topology_parity"
Cohesion: 0.29
Nodes (7): topology_parity, deployment_blockers, deployment_enforcement, seeded_red_controls, source_enforcement, source_proof_paths, source_proofs

### Community 3592 - "predecessor_pins"
Cohesion: 0.29
Nodes (7): predecessor_pins, omnibase_core#1529, omnibase_infra#2547, omnibase_infra#2548, omnibase_infra#2558, omnimarket#1956, omninode_infra#771

### Community 3593 - "pinned_backends"
Cohesion: 0.29
Nodes (6): pinned_backends, free_cloud, local, paid, row_query, version

### Community 3594 - "machines"
Cohesion: 0.29
Nodes (6): machines, hostname, machine_id, omni_home, resolved_home_dir, role

### Community 3595 - "dns_cache"
Cohesion: 0.29
Nodes (7): dns_cache, active, host, port, target_dns_failure_count, target_hit_rate, upstream_forwarders

### Community 3596 - "response_handling"
Cohesion: 0.29
Nodes (7): response_handling, subscribed_at, subscription_id, extract_fields, extraction_engine, fail_on_empty, success_codes

### Community 3597 - "publish_topics"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, description, payload_schema, provisioning_priority, topic

### Community 3598 - "topic_validation"
Cohesion: 0.29
Nodes (7): allowed_pattern, deny_patterns, enabled, max_subscriptions_per_node, max_topic_length, reserved_prefixes, topic_validation

### Community 3599 - "install-host-maintenance.sh"
Cohesion: 0.57
Nodes (6): install_log_monitor(), retire_legacy_cron(), install-host-maintenance.sh script, status_human(), status_json(), usage()

### Community 3600 - "omninode-runner-tree-converge.sh"
Cohesion: 0.62
Nodes (6): busy_state(), clone_is_converged(), emit_verdict(), precondition(), say(), omninode-runner-tree-converge.sh script

### Community 3601 - "tracing"
Cohesion: 0.29
Nodes (7): OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_TRACES_EXPORTER, tracing, description, includes, inject_env, services

### Community 3602 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3603 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3604 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3605 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3606 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3607 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3608 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3609 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3610 - "canonical"
Cohesion: 0.29
Nodes (7): checksum_column, domain_column, relation, schema, stream_column, version_column, canonical

### Community 3611 - "build"
Cohesion: 0.29
Nodes (7): BUILD_SOURCE, EXPECTED_BUILD_SOURCE, args, context, dockerfile, build, build

### Community 3612 - "environment"
Cohesion: 0.29
Nodes (7): AWS_CONFIG_FILE, AWS_DEFAULT_REGION, AWS_PROFILE, AWS_REGION, AWS_SDK_LOAD_CONFIG, ONEX_TOPIC_ENFORCEMENT_MODE, environment

### Community 3613 - "healthcheck"
Cohesion: 0.48
Nodes (7): healthcheck, healthcheck, interval, retries, start_period, test, timeout

### Community 3614 - "environment"
Cohesion: 0.29
Nodes (7): ROLE_OMNIDASH_PASSWORD, ROLE_OMNINODE_PASSWORD, container_name, depends_on, environment, profiles, forward-migration

### Community 3615 - "projection-api"
Cohesion: 0.29
Nodes (7): container_name, depends_on, labels, ports, profiles, restart, projection-api

### Community 3616 - "lane.dev.kafka.scram"
Cohesion: 0.29
Nodes (6): lane.dev.kafka.scram, password, username, lane.dev-local.kafka.scram, password, username

### Community 3617 - "env"
Cohesion: 0.29
Nodes (7): LAB_LANES_JSON, LAB_LANES_VAR, LANE_DEPLOY_WAIT_SECONDS, LANE_DEPLOY_WAIT_WHILE_BUSY, LANE_MATCH, LANE_REQUIRE, env

### Community 3618 - "inputs"
Cohesion: 0.29
Nodes (7): inputs, lanes-json, require, description, required, description, required

### Community 3619 - "with"
Cohesion: 0.29
Nodes (7): with, enable-cache, key, path, python-version, restore-keys, version

### Community 3620 - "attest-source-hash"
Cohesion: 0.29
Nodes (7): name, needs, uses, with, attest-source-hash, expected_commit_sha, runtime_source_hash

### Community 3621 - "steps"
Cohesion: 0.29
Nodes (7): steps, id, if, name, run, shell, uses

### Community 3622 - "env"
Cohesion: 0.29
Nodes (7): AWS_CLI_CONNECT_TIMEOUT, AWS_CLI_READ_TIMEOUT, PUSH_BACKOFF_FACTOR, PUSH_INITIAL_DELAY, PUSH_MAX_ATTEMPTS, PUSH_MAX_DELAY, env

### Community 3623 - "resolve-lane"
Cohesion: 0.29
Nodes (7): resolve-lane, docker_runs_on, lane, name, outputs, runs-on, timeout-minutes

### Community 3624 - "demo-loop-gate"
Cohesion: 0.29
Nodes (7): if, name, needs, runs-on, steps, timeout-minutes, demo-loop-gate

### Community 3625 - "docs-only-marker"
Cohesion: 0.29
Nodes (7): if, name, needs, runs-on, steps, timeout-minutes, docs-only-marker

### Community 3626 - "repo_key"
Cohesion: 0.29
Nodes (7): repo_key, on, workflow_call, description, required, type, inputs

### Community 3627 - "steps"
Cohesion: 0.38
Nodes (7): steps, steps, id, if, name, run, uses

### Community 3628 - "onexbot-occ-app-id"
Cohesion: 0.29
Nodes (7): description, required, description, required, onexbot-occ-app-id, onexbot-occ-private-key, secrets

### Community 3629 - "resolve-lane"
Cohesion: 0.29
Nodes (7): resolve-lane, docker_runs_on, lane, name, outputs, runs-on, timeout-minutes

### Community 3630 - "attest-source-hash"
Cohesion: 0.29
Nodes (7): name, needs, uses, with, attest-source-hash, expected_commit_sha, runtime_source_hash

### Community 3631 - "docker-build"
Cohesion: 0.29
Nodes (7): name, needs, outputs, runs-on, timeout-minutes, docker-build, image_id

### Community 3632 - "docker-integration-tests"
Cohesion: 0.29
Nodes (7): continue-on-error, if, name, needs, runs-on, timeout-minutes, docker-integration-tests

### Community 3633 - "steps"
Cohesion: 0.29
Nodes (7): steps, id, if, name, run, shell, uses

### Community 3634 - "steps"
Cohesion: 0.29
Nodes (7): steps, id, if, name, run, uses, working-directory

### Community 3635 - "hostile-review"
Cohesion: 0.29
Nodes (7): REVIEW_VOTERS_OVERLAY, env, if, name, runs-on, timeout-minutes, hostile-review

### Community 3636 - "steps"
Cohesion: 0.29
Nodes (7): steps, continue-on-error, id, if, name, run, uses

### Community 3637 - "env"
Cohesion: 0.29
Nodes (7): env, OMNI_CI_ENV_ROOT, OMNI_CI_SHARED_ENV_ENABLED, PYTHON_VERSION, REPO_NAME, UV_HTTP_TIMEOUT, UV_VERSION

### Community 3638 - "on"
Cohesion: 0.29
Nodes (7): on, merge_group, pull_request, push, workflow_dispatch, branches, branches

### Community 3639 - "steps"
Cohesion: 0.43
Nodes (7): id, if, name, run, uses, steps, steps

### Community 3640 - "nonrequired-check-failure-rate"
Cohesion: 0.29
Nodes (7): cancel-in-progress, group, nonrequired-check-failure-rate, concurrency, name, runs-on, timeout-minutes

### Community 3641 - "steps"
Cohesion: 0.48
Nodes (7): steps, steps, id, if, name, run, uses

### Community 3642 - "decide"
Cohesion: 0.29
Nodes (7): name, outputs, runs-on, timeout-minutes, decide, cut_count, cuts

### Community 3643 - "steps"
Cohesion: 0.29
Nodes (7): steps, continue-on-error, id, if, name, run, uses

### Community 3655 - "bootstrap-infisical.sh"
Cohesion: 0.52
Nodes (6): log_error(), log_info(), log_step(), log_warn(), run_cmd(), bootstrap-infisical.sh script

### Community 3657 - "violations"
Cohesion: 0.29
Nodes (6): count, generated_at, violations, key, path, violation

### Community 3658 - "run_post_verdicts"
Cohesion: 0.33
Nodes (4): _flat(), _read_status(), run_post_verdicts(), verdict_body()

### Community 3659 - "ensure_ci_env.sh"
Cohesion: 0.52
Nodes (6): link_workspace_venv(), publish_env(), ready(), ensure_ci_env.sh script, write_project_metadata(), write_uv_wrapper()

### Community 3660 - "runner_broker_dispatch_wedge_rerun.sh"
Cohesion: 0.52
Nodes (6): already_rerun(), api_get(), api_post_rerun(), log(), record_rerun(), runner_broker_dispatch_wedge_rerun.sh script

### Community 3661 - "resolve_host"
Cohesion: 0.33
Nodes (3): HostDeclarationError, resolve_host(), validate_hosts()

### Community 3663 - "omnidash-lifecycle.sh"
Cohesion: 0.71
Nodes (6): cmd_restart(), cmd_start(), cmd_status(), cmd_stop(), _resolve_omnidash_dir(), omnidash-lifecycle.sh script

### Community 3664 - "reconcile_privilege_lib.sh"
Cohesion: 0.38
Nodes (5): rp_plan_privileges(), rp_surface_owner(), rp_user_home(), RUN_AS, reconcile_privilege_lib.sh script

### Community 3669 - "allowlist"
Cohesion: 0.29
Nodes (6): allowlist, expires_at, owner_repo, planned_subscriber, reason, topic

### Community 3670 - "operation_bindings"
Cohesion: 0.29
Nodes (7): expression, parameter_name, required, operation_bindings, global_bindings, max_expression_length, max_path_segments

### Community 3671 - "future_considerations"
Cohesion: 0.29
Nodes (7): backpressure_signal, burst_allowance, per_operation_limits, deferred_to, enabled, future_considerations, rate_limiting

### Community 3672 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, capabilities, handler_category, handler_type, resilience_features, security_features, transport_type

### Community 3673 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, handler_category, handler_type, notes, resilience_features, security_features, transport_type

### Community 3681 - "handler_placement_epoch_guard/contract.yaml"
Cohesion: 0.29
Nodes (6): descriptor, node_archetype, handler, module, name, node_type

### Community 3685 - "DiscoveredCapabilitiesCacheDict"
Cohesion: 0.29
Nodes (3): DiscoveredCapabilitiesCacheDict, IntrospectionCacheDict, PerformanceMetricsCacheDict

### Community 3694 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3695 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, ticket

### Community 3696 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, ticket

### Community 3697 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, library, name, protocol, required, type

### Community 3698 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3699 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3700 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3701 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3702 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3703 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3704 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3705 - "retry_policy"
Cohesion: 0.29
Nodes (7): error_handling, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3706 - "node_chain_canary_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): description, health_check, enabled, name, node_type, node_version

### Community 3707 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3708 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3709 - "handler"
Cohesion: 0.29
Nodes (7): handler, class, handler_type, input_model, module, name, handler

### Community 3710 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3711 - "coordination_rules"
Cohesion: 0.29
Nodes (7): execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, timeout_ms, coordination_rules

### Community 3712 - "workflow_metadata"
Cohesion: 0.29
Nodes (7): workflow_metadata, description, workflow_name, workflow_version, major, minor, patch

### Community 3713 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3714 - "descriptor"
Cohesion: 0.29
Nodes (7): agent_credential_home, agent_invocation, idempotent, node_archetype, purity, timeout_ms, descriptor

### Community 3715 - "descriptor"
Cohesion: 0.29
Nodes (7): descriptor, agent_invocation, allowed_workspace_roots, idempotent, node_archetype, purity, timeout_ms

### Community 3716 - "retry_policy"
Cohesion: 0.29
Nodes (7): error_handling, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3717 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3718 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3720 - "retry_policy"
Cohesion: 0.29
Nodes (7): retry_policy, backoff_strategy, base_delay_ms, enabled, jitter_factor, max_delay_ms, max_retries

### Community 3721 - "health_check"
Cohesion: 0.29
Nodes (7): check_type, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3722 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, related_tickets, tags, updated

### Community 3723 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, class_name, description, module, name, optional, type

### Community 3724 - "io_operations"
Cohesion: 0.29
Nodes (7): io_operations, description, input_fields, input_model, operation, output_fields, output_model

### Community 3725 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3726 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3727 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3728 - "health_check"
Cohesion: 0.29
Nodes (7): check_type, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3729 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3730 - "retry_policy"
Cohesion: 0.29
Nodes (7): error_handling, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3731 - "health_check"
Cohesion: 0.29
Nodes (7): check_type, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3732 - "health_check"
Cohesion: 0.29
Nodes (7): check_type, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3733 - "handler_routing"
Cohesion: 0.29
Nodes (7): module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3735 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3736 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3737 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3738 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3739 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3740 - "quarantine"
Cohesion: 0.29
Nodes (7): quarantine, purpose, re_entry_path, topic, description, invariant, steps

### Community 3742 - "response_handling"
Cohesion: 0.29
Nodes (7): response_handling, subscribed_at, subscription_id, extract_fields, extraction_engine, fail_on_empty, success_codes

### Community 3743 - "topic_validation"
Cohesion: 0.29
Nodes (7): allowed_pattern, deny_patterns, enabled, max_subscriptions_per_node, max_topic_length, reserved_prefixes, topic_validation

### Community 3745 - "node_event_forward_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): description, backend_url, name, node_type, node_version, descriptor

### Community 3746 - "handler_routing"
Cohesion: 0.29
Nodes (7): module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3747 - "retry_policy"
Cohesion: 0.29
Nodes (7): error_handling, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3748 - "node_evidence_autoclose_sweep_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): description, handler_routing, routing_strategy, name, node_type, node_version

### Community 3749 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3750 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3751 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3752 - "input_fields"
Cohesion: 0.29
Nodes (7): lag_messages, lag_seconds, last_seen_at, local_transport_flavor, principal_id, tenant_id, input_fields

### Community 3753 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, name, protocol, required, type

### Community 3754 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3755 - "handlers"
Cohesion: 0.29
Nodes (7): handler_routing, handlers, routing_strategy, description, operation, output_fields, supported_operations

### Community 3756 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3757 - "retry_policy"
Cohesion: 0.29
Nodes (7): error_handling, dead_letter_queue, retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries

### Community 3758 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3759 - "node_invariant_evaluate_compute/contract.yaml"
Cohesion: 0.29
Nodes (6): contract_name, description, name, node_name, node_type, status

### Community 3762 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3763 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3764 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, class_name, description, module, name, optional, type

### Community 3765 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, ticket

### Community 3766 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3767 - "handler_routing"
Cohesion: 0.29
Nodes (7): module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3768 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3769 - "health_check"
Cohesion: 0.29
Nodes (7): check_type, timeout_ms, health_check, backends, enabled, endpoint, interval_seconds

### Community 3770 - "config"
Cohesion: 0.29
Nodes (7): config, linear_api_key, linear_team_id, default, description, default, description

### Community 3771 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3772 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3773 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3774 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3775 - "redpanda"
Cohesion: 0.29
Nodes (7): redpanda, campaign_interval_s, fence_tick_s, keepalive_interval_s, margin_s, min_serving, ttl_s

### Community 3776 - "consumed_events"
Cohesion: 0.29
Nodes (7): consumed_events, description, direct_handler, event_type, internal, message_category, topic

### Community 3777 - "feature_flags"
Cohesion: 0.29
Nodes (7): feature_flags, category, default_value, description, env_var, name, owner

### Community 3778 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, related_tickets, tags, updated

### Community 3779 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, related_tickets, tags, updated

### Community 3781 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, class_name, description, module, name, optional, type

### Community 3782 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3783 - "coordination_rules"
Cohesion: 0.29
Nodes (7): execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, timeout_ms, coordination_rules

### Community 3784 - "workflow_metadata"
Cohesion: 0.29
Nodes (7): workflow_metadata, description, workflow_name, workflow_version, major, minor, patch

### Community 3785 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish, subscribe, description, topic, description, topic

### Community 3786 - "handler_routing"
Cohesion: 0.29
Nodes (7): module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3788 - "coordination_rules"
Cohesion: 0.29
Nodes (7): execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, timeout_ms, coordination_rules

### Community 3789 - "workflow_metadata"
Cohesion: 0.29
Nodes (7): workflow_metadata, description, workflow_name, workflow_version, major, minor, patch

### Community 3790 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3791 - "coordination_rules"
Cohesion: 0.29
Nodes (7): execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, timeout_ms, coordination_rules

### Community 3792 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3793 - "workflow_metadata"
Cohesion: 0.29
Nodes (7): workflow_metadata, description, workflow_name, workflow_version, major, minor, patch

### Community 3794 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3795 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3796 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3797 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3798 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, related_tickets, tags, updated

### Community 3799 - "env_dependencies"
Cohesion: 0.29
Nodes (7): env_dependencies, KAFKA_ENVIRONMENT, ONEX_ENVIRONMENT, description, required, description, required

### Community 3800 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3801 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3802 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3803 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3804 - "coordination_rules"
Cohesion: 0.29
Nodes (7): execution_mode, failure_recovery_strategy, max_parallel_branches, max_retries, parallel_execution_allowed, timeout_ms, coordination_rules

### Community 3805 - "workflow_metadata"
Cohesion: 0.29
Nodes (7): workflow_metadata, description, workflow_name, workflow_version, major, minor, patch

### Community 3806 - "node_secret_seed_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): description, health_check, enabled, name, node_type, node_version

### Community 3807 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3808 - "handler"
Cohesion: 0.29
Nodes (7): handler, class, handler_type, input_model, module, name, handler

### Community 3809 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3810 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3811 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3812 - "overlay_revision_render"
Cohesion: 0.29
Nodes (7): overlay_revision_render, clone_env, description, overlay_dir, read_timeout_seconds, rendered_dir_name, store_root_env

### Community 3813 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3814 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3815 - "severity"
Cohesion: 0.29
Nodes (7): severity, default, description, enum, enum_class, module, type

### Community 3816 - "node_sync_revert_watchdog_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): description, handler_routing, routing_strategy, name, node_type, node_version

### Community 3817 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, description, env_var, library, name, required, type

### Community 3818 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, tags, transport_type, updated

### Community 3819 - "node_topic_migration_executor_effect/contract.yaml"
Cohesion: 0.29
Nodes (6): contract_name, description, name, node_name, node_type, runtime_profiles

### Community 3820 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3821 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3822 - "node_topic_migration_projection/contract.yaml"
Cohesion: 0.29
Nodes (6): contract_name, dependencies, description, name, node_name, node_type

### Community 3823 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3824 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, updated

### Community 3825 - "handler_routing"
Cohesion: 0.29
Nodes (7): module, name, handler_routing, handlers, routing_strategy, handler, operation

### Community 3826 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3827 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3828 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3829 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3830 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3831 - "dependencies"
Cohesion: 0.29
Nodes (7): dependencies, class_name, description, module, name, optional, type

### Community 3832 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, tags, ticket

### Community 3833 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3834 - "metadata"
Cohesion: 0.29
Nodes (7): metadata, author, created, description, license, related_tickets, tags

### Community 3836 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3839 - "event_bus"
Cohesion: 0.29
Nodes (7): event_bus, publish_topics, subscribe_topics, version, major, minor, patch

### Community 3844 - "owners"
Cohesion: 0.29
Nodes (7): owners, login, login, login, owner_omninode_internal, owner_onex_tenant, owner_platform_catalog

### Community 3845 - "service_owned"
Cohesion: 0.29
Nodes (7): service_owned, checksum_column, domain_column, relation, schema, stream_column, version_column

### Community 3846 - "service_owned"
Cohesion: 0.29
Nodes (7): service_owned, checksum_column, domain_column, relation, schema, stream_column, version_column

### Community 3847 - "checks"
Cohesion: 0.29
Nodes (6): checks, check_code, command, handler, label, severity

### Community 3849 - "entries"
Cohesion: 0.33
Nodes (5): entries, content_sha1, family, path, schema_version

### Community 3850 - "action_archive_cache"
Cohesion: 0.33
Nodes (6): action_archive_cache, active, container_path, local_owner, refresh_interval_seconds, staging_root

### Community 3851 - "tool_cache"
Cohesion: 0.33
Nodes (6): tool_cache, durable, path, recreate_procedure, seed_script, snapshot_root

### Community 3852 - "docker_ci_runner_variable"
Cohesion: 0.33
Nodes (6): docker_ci_runner_variable, expected_json, name, org_scope, repositories, revert_when

### Community 3853 - "omninode_infra_fleet_runner_variable"
Cohesion: 0.33
Nodes (6): omninode_infra_fleet_runner_variable, expected_json, name, owner, repositories, scope

### Community 3854 - "private_repo_hosted_exception"
Cohesion: 0.33
Nodes (6): private_repo_hosted_exception, job, path, reason, repository, ruling

### Community 3855 - "security_scan_runner_variable"
Cohesion: 0.33
Nodes (6): security_scan_runner_variable, expected_json, name, org_scope, repositories, revert_when

### Community 3856 - "test-failure-baseline.yaml"
Cohesion: 0.33
Nodes (5): clusters, generated_at, generated_by, repo, ttl_days

### Community 3857 - "requirements"
Cohesion: 0.33
Nodes (6): id, statement, requirements, acceptance, id, statement

### Community 3858 - "dod-omn19404-ac1-slot-roles-accepted"
Cohesion: 0.33
Nodes (6): dod-omn19404-ac1-slot-roles-accepted, accepted_at, accepted_by, criterion_hash, label, proposed_by

### Community 3859 - "dod-omn19404-ac2-slot-migration-accepted"
Cohesion: 0.33
Nodes (6): dod-omn19404-ac2-slot-migration-accepted, accepted_at, accepted_by, criterion_hash, label, proposed_by

### Community 3860 - "circuit_breaker"
Cohesion: 0.33
Nodes (6): enabled, failure_threshold, half_open_requests, success_threshold, timeout_ms, circuit_breaker

### Community 3861 - "conditions"
Cohesion: 0.33
Nodes (6): condition_name, condition_type, description, expression, required, conditions

### Community 3862 - "conditions"
Cohesion: 0.33
Nodes (6): condition_name, condition_type, description, expression, required, conditions

### Community 3863 - "secret_resolver_namespaces"
Cohesion: 0.53
Nodes (6): secret_resolver_namespaces, secret_resolver_namespaces, namespace, ref_pattern, source_path_template, source_type

### Community 3864 - "bindings"
Cohesion: 0.33
Nodes (5): bindings, broker, direction, physical_topic, principal

### Community 3865 - "canary-s7-override.yml"
Cohesion: 0.33
Nodes (5): ONEX_CORE_RUNTIME_TOPICS, name, environment, services, omninode-runtime

### Community 3866 - "canary"
Cohesion: 0.33
Nodes (6): canary, description, includes, inject_env, inject_required_env, services

### Community 3867 - "fault-injection"
Cohesion: 0.33
Nodes (6): fault-injection, description, includes, inject_env, inject_required_env, services

### Community 3868 - "omnimemory"
Cohesion: 0.33
Nodes (6): omnimemory, description, includes, inject_env, inject_required_env, services

### Community 3869 - "runtime"
Cohesion: 0.33
Nodes (6): runtime, description, includes, inject_env, inject_required_env, services

### Community 3870 - "runtime-core"
Cohesion: 0.33
Nodes (6): runtime-core, description, includes, inject_env, inject_required_env, services

### Community 3871 - "runtime-infrastructure"
Cohesion: 0.33
Nodes (6): runtime-infrastructure, description, includes, inject_env, inject_required_env, services

### Community 3872 - "runtime-integrations"
Cohesion: 0.33
Nodes (6): runtime-integrations, description, includes, inject_env, inject_required_env, services

### Community 3873 - "runtime-observability-projections"
Cohesion: 0.33
Nodes (6): runtime-observability-projections, description, includes, inject_env, inject_required_env, services

### Community 3874 - "secrets"
Cohesion: 0.33
Nodes (6): secrets, description, includes, inject_env, inject_required_env, services

### Community 3875 - "tenant-path"
Cohesion: 0.33
Nodes (6): tenant-path, description, includes, inject_env, inject_required_env, services

### Community 3876 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3877 - "labels"
Cohesion: 0.33
Nodes (6): labels, com.omninode.layer, com.omninode.service, com.omninode.ticket, com.omninode.transitional, com.omninode.version

### Community 3878 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3879 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3880 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3881 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3882 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3883 - "labels"
Cohesion: 0.33
Nodes (6): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.ticket, com.omninode.version

### Community 3884 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3885 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3886 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3887 - "healthcheck"
Cohesion: 0.33
Nodes (6): healthcheck, interval_s, retries, start_period_s, test, timeout_s

### Community 3888 - "labels"
Cohesion: 0.33
Nodes (6): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.ticket, com.omninode.version

### Community 3889 - "healthcheck"
Cohesion: 0.33
Nodes (6): interval, retries, start_period, test, timeout, healthcheck

### Community 3890 - "gateway-dns-bastion"
Cohesion: 0.33
Nodes (6): GATEWAY_BASTION_IP, container_name, environment, labels, restart, gateway-dns-bastion

### Community 3891 - "projection-live-events-writer"
Cohesion: 0.33
Nodes (6): container_name, depends_on, labels, profiles, restart, projection-live-events-writer

### Community 3892 - "projection-savings-writer"
Cohesion: 0.33
Nodes (6): container_name, depends_on, labels, profiles, restart, projection-savings-writer

### Community 3893 - "projection-tenant-credentials-writer"
Cohesion: 0.33
Nodes (6): container_name, depends_on, labels, profiles, restart, projection-tenant-credentials-writer

### Community 3894 - "projection-tenant-registry-writer"
Cohesion: 0.33
Nodes (6): container_name, depends_on, labels, profiles, restart, projection-tenant-registry-writer

### Community 3895 - "runtime-worker"
Cohesion: 0.33
Nodes (6): container_name, depends_on, labels, profiles, restart, runtime-worker

### Community 3896 - "tenant-projection-writer"
Cohesion: 0.33
Nodes (6): tenant-projection-writer, container_name, depends_on, labels, profiles, restart

### Community 3897 - "redpanda"
Cohesion: 0.33
Nodes (6): command, container_name, image, labels, ports, redpanda

### Community 3898 - "entries"
Cohesion: 0.33
Nodes (5): entries, citation, connect_target, disposition, file

### Community 3899 - "shape_reconciliation_exemptions"
Cohesion: 0.33
Nodes (5): shape_reconciliation_exemptions, frozen_by, id, residual, ticket

### Community 3900 - "resolve-lab-lane/action.yml"
Cohesion: 0.33
Nodes (5): author, description, name, runs, using

### Community 3901 - "setup-python-uv/action.yml"
Cohesion: 0.33
Nodes (5): author, description, name, runs, using

### Community 3902 - "detectors"
Cohesion: 0.33
Nodes (5): detectors, keywords, name, regex, token

### Community 3903 - "build-and-push"
Cohesion: 0.33
Nodes (6): env, name, runs-on, timeout-minutes, TARGET_SHA, build-and-push

### Community 3904 - "with"
Cohesion: 0.33
Nodes (6): with, aws-region, python-version, ref, role-session-name, role-to-assume

### Community 3905 - "steps"
Cohesion: 0.33
Nodes (6): steps, id, if, name, run, uses

### Community 3906 - "exposed-identifier-gate"
Cohesion: 0.33
Nodes (6): if, name, runs-on, steps, timeout-minutes, exposed-identifier-gate

### Community 3907 - "lockfile-registry-allowlist"
Cohesion: 0.33
Nodes (6): lockfile-registry-allowlist, if, name, runs-on, steps, timeout-minutes

### Community 3908 - "topic-enum-drift"
Cohesion: 0.33
Nodes (6): topic-enum-drift, if, name, runs-on, steps, timeout-minutes

### Community 3909 - "topic-naming-lint"
Cohesion: 0.33
Nodes (6): topic-naming-lint, if, name, runs-on, steps, timeout-minutes

### Community 3910 - "on"
Cohesion: 0.33
Nodes (6): on, pull_request, repository_dispatch, branches, types, types

### Community 3911 - "steps"
Cohesion: 0.33
Nodes (6): steps, id, if, name, run, uses

### Community 3912 - "build-migrate-bundle"
Cohesion: 0.33
Nodes (6): name, secrets, uses, with, build-migrate-bundle, git-ref

### Community 3913 - "saturation-record"
Cohesion: 0.33
Nodes (6): saturation-record, if, name, needs, runs-on, timeout-minutes

### Community 3914 - "dlq-depth-monitor"
Cohesion: 0.33
Nodes (6): name, permissions, runs-on, timeout-minutes, dlq-depth-monitor, contents

### Community 3915 - "steps"
Cohesion: 0.33
Nodes (6): steps, id, if, name, run, uses

### Community 3916 - "suppress_alert_exit"
Cohesion: 0.33
Nodes (6): suppress_alert_exit, default, description, options, required, type

### Community 3917 - "build-summary"
Cohesion: 0.33
Nodes (6): if, name, needs, runs-on, timeout-minutes, build-summary

### Community 3918 - "env"
Cohesion: 0.33
Nodes (6): env, DOCKER_COMPOSE_VERSION, IMAGE_NAME, PYTHON_VERSION, REGISTRY, UV_VERSION

### Community 3919 - "notify-on-failure"
Cohesion: 0.33
Nodes (6): notify-on-failure, if, name, needs, runs-on, timeout-minutes

### Community 3920 - "omni-standards-gate"
Cohesion: 0.33
Nodes (6): omni-standards-gate, if, name, needs, runs-on, timeout-minutes

### Community 3921 - "env"
Cohesion: 0.33
Nodes (6): APP_TOKEN, DO_APPLY, GH_TOKEN, SLACK_BOT_TOKEN, SLACK_CHANNEL_ID, env

### Community 3922 - "steps"
Cohesion: 0.33
Nodes (6): HEAD_SHA, LINT_CONCLUSION, TESTS_CONCLUSION, TYPECHECK_CONCLUSION, steps, env

### Community 3923 - "reason-graph"
Cohesion: 0.33
Nodes (6): reason-graph, if, name, needs, runs-on, timeout-minutes

### Community 3924 - "post-release-dev-bump"
Cohesion: 0.33
Nodes (6): post-release-dev-bump, if, name, needs, runs-on, timeout-minutes

### Community 3925 - "trigger-runtime-rebuild"
Cohesion: 0.33
Nodes (6): trigger-runtime-rebuild, if, name, needs, runs-on, timeout-minutes

### Community 3926 - "env"
Cohesion: 0.33
Nodes (6): EXECUTE, GH_TOKEN, LANE, REF, TAG, env

### Community 3927 - "route-wiring-proof"
Cohesion: 0.33
Nodes (6): route-wiring-proof, if, name, needs, runs-on, timeout-minutes

### Community 3928 - "steps"
Cohesion: 0.40
Nodes (6): steps, steps, if, name, run, uses

### Community 3929 - "env"
Cohesion: 0.33
Nodes (6): BASE, EVENT_NAME, HEAD, PR_BASE_SHA, PR_HEAD_SHA, env

### Community 3930 - "secret-scan"
Cohesion: 0.33
Nodes (6): secret-scan, contents, name, permissions, runs-on, timeout-minutes

### Community 3946 - ".public-repo-hygiene.yaml"
Cohesion: 0.33
Nodes (5): allowed_top_level, enforce_classes, enforce_scope, mode, suppression_registry

### Community 3948 - "carries_failure_conclusion"
Cohesion: 0.33
Nodes (3): carries_failure_conclusion(), dedup_latest(), _state_severity()

### Community 3949 - "_capture_render_inputs"
Cohesion: 0.47
Nodes (4): _capture_render_inputs(), run(), source(), _cmd_render()

### Community 3950 - "lane-census-check.sh"
Cohesion: 0.80
Nodes (5): broker_produce(), finish(), log(), memory_pass(), lane-census-check.sh script

### Community 3952 - "render_dev_lane_tenant_path_env.sh"
Cohesion: 0.53
Nodes (4): current_value(), generate(), render_dev_lane_tenant_path_env.sh script, usage()

### Community 3955 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, handler_category, handler_type, resilience_features, security_features, transport_type

### Community 3956 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, backoff_strategy, base_delay_ms, enabled, max_delay_ms, max_retries

### Community 3957 - "descriptor"
Cohesion: 0.33
Nodes (6): descriptor, concurrency_policy, idempotent, node_archetype, purity, timeout_ms

### Community 3958 - "authentication"
Cohesion: 0.33
Nodes (6): additional_headers, api_key_header, methods, multi_key_support, required, authentication

### Community 3959 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, backoff_strategy, base_delay_ms, enabled, max_delay_ms, max_retries

### Community 3960 - "feature_flags"
Cohesion: 0.33
Nodes (6): feature_flags, category, default_value, description, env_var, name

### Community 3978 - "datetime_handling"
Cohesion: 0.33
Nodes (6): datetime_handling, iso_parsing, naive_datetime_behavior, notes, timezone_awareness, z_suffix_handling

### Community 3979 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 3980 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, license, tags, ticket

### Community 4000 - "masking_rules"
Cohesion: 0.33
Nodes (6): credential, identifier, identity, private_text, volatile, masking_rules

### Community 4001 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, description, module, name, required, type

### Community 4004 - "consumer_flow_topics"
Cohesion: 0.33
Nodes (6): consumer_flow_topics, applied, audit_projection, exposure, publishing_reducer, seam_dlq

### Community 4005 - "handler"
Cohesion: 0.33
Nodes (6): handler, class, input_model, module, name, handler

### Community 4007 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4008 - "version"
Cohesion: 0.60
Nodes (6): version, version, version, major, minor, patch

### Community 4011 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4013 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4014 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4015 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, tags, updated

### Community 4019 - "descriptor"
Cohesion: 0.33
Nodes (6): descriptor, agent_invocation, idempotent, node_archetype, purity, timeout_ms

### Community 4020 - "descriptor"
Cohesion: 0.33
Nodes (6): descriptor, agent_invocation, idempotent, node_archetype, purity, timeout_ms

### Community 4021 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4022 - "node_consumer_health_triage_effect/contract.yaml"
Cohesion: 0.33
Nodes (5): contract_name, description, name, node_name, node_type

### Community 4023 - "circuit_breaker"
Cohesion: 0.33
Nodes (6): enabled, failure_threshold, half_open_requests, success_threshold, timeout_ms, circuit_breaker

### Community 4025 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4026 - "error_codes"
Cohesion: 0.33
Nodes (6): backend, code, description, operation, retriable, error_codes

### Community 4027 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4028 - "version"
Cohesion: 0.33
Nodes (6): event_bus, subscribe_topics, version, major, minor, patch

### Community 4029 - "idempotency"
Cohesion: 0.33
Nodes (6): idempotency, derivation, description, enabled, strategy, tracking_fields

### Community 4030 - "version"
Cohesion: 0.33
Nodes (6): event_bus, subscribe_topics, version, major, minor, patch

### Community 4031 - "produced_events"
Cohesion: 0.33
Nodes (6): module, name, produced_events, description, model, name

### Community 4032 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, description, module, name, required, type

### Community 4033 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4034 - "error_codes"
Cohesion: 0.33
Nodes (6): backend, code, description, operation, retriable, error_codes

### Community 4035 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4036 - "node_capabilities"
Cohesion: 0.33
Nodes (6): node_capabilities, database, postgres, read, transactions, write

### Community 4038 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4039 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4041 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4042 - "error_codes"
Cohesion: 0.33
Nodes (6): backend, code, description, operation, retriable, error_codes

### Community 4043 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4044 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4045 - "error_codes"
Cohesion: 0.33
Nodes (6): backend, code, description, operation, retriable, error_codes

### Community 4046 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4048 - "node_dlq_depth_evaluate_compute/contract.yaml"
Cohesion: 0.33
Nodes (5): dependencies, description, name, node_type, node_version

### Community 4049 - "error_types"
Cohesion: 0.33
Nodes (6): error_handling, error_types, description, name, recoverable, retry_strategy

### Community 4050 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4051 - "handler"
Cohesion: 0.33
Nodes (6): handler, class, input_model, module, name, handler

### Community 4052 - "handlers"
Cohesion: 0.33
Nodes (6): handler_routing, handlers, routing_strategy, description, operation, output_fields

### Community 4053 - "error_types"
Cohesion: 0.33
Nodes (6): error_handling, error_types, description, name, recoverable, retry_strategy

### Community 4054 - "circuit_breaker"
Cohesion: 0.33
Nodes (6): enabled, failure_threshold, half_open_requests, success_threshold, timeout_ms, circuit_breaker

### Community 4056 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4057 - "version"
Cohesion: 0.33
Nodes (6): event_bus, publish_topics, version, major, minor, patch

### Community 4058 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, tags, transport_type

### Community 4059 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4060 - "fault_injection"
Cohesion: 0.33
Nodes (6): dead_letter_topic_override, dead_letter_topic_provisioned, how_to_drive_double_failure, purpose, runbook, fault_injection

### Community 4063 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4064 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4065 - "handler"
Cohesion: 0.33
Nodes (6): handler, class, input_model, module, name, handler

### Community 4066 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, license, tags

### Community 4068 - "handler"
Cohesion: 0.33
Nodes (6): handler, class, input_model, module, name, handler

### Community 4069 - "handlers"
Cohesion: 0.33
Nodes (6): handler_routing, handlers, routing_strategy, description, operation, output_fields

### Community 4070 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, description, module, name, required, type

### Community 4075 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4076 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, license, tags

### Community 4077 - "node_llm_completion_effect/contract.yaml"
Cohesion: 0.33
Nodes (5): description, name, node_type, node_version, runtime_profiles

### Community 4078 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4079 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4080 - "version"
Cohesion: 0.33
Nodes (6): event_bus, subscribe_topics, version, major, minor, patch

### Community 4081 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, license, tags

### Community 4082 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4083 - "error_codes"
Cohesion: 0.33
Nodes (6): backend, code, description, operation, retriable, error_codes

### Community 4084 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4086 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4087 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4089 - "conditions"
Cohesion: 0.33
Nodes (6): condition_name, condition_type, description, expression, required, conditions

### Community 4091 - "version"
Cohesion: 0.33
Nodes (6): event_bus, publish_topics, version, major, minor, patch

### Community 4094 - "idempotency"
Cohesion: 0.33
Nodes (6): idempotency, derivation, description, enabled, hash_fields, strategy

### Community 4095 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4096 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, license, tags

### Community 4097 - "node_capabilities"
Cohesion: 0.33
Nodes (6): node_capabilities, database, postgres, read, transactions, write

### Community 4098 - "handler"
Cohesion: 0.40
Nodes (6): cost_api_binding, handler, handler_type, module, name, handler

### Community 4099 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4100 - "tags"
Cohesion: 0.33
Nodes (6): type, tags, default_factory, description, items, type

### Community 4101 - "timestamp"
Cohesion: 0.33
Nodes (6): timestamp, description, type, validation, description, name

### Community 4102 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, license, tags

### Community 4107 - "node_runner_fleet_health_compute/contract.yaml"
Cohesion: 0.33
Nodes (5): description, name, node_type, node_version, runtime_profiles

### Community 4108 - "topic_config"
Cohesion: 0.33
Nodes (6): cleanup.policy, retention.ms, topic_config, kafka_config, partitions, replication_factor

### Community 4111 - "node_runtime_source_attestor_effect/contract.yaml"
Cohesion: 0.33
Nodes (5): contract_name, description, name, node_name, node_type

### Community 4114 - "error_types"
Cohesion: 0.33
Nodes (6): error_handling, error_types, description, name, recoverable, retry_strategy

### Community 4115 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4116 - "handlers"
Cohesion: 0.33
Nodes (6): handler_routing, handlers, routing_strategy, description, operation, output_fields

### Community 4121 - "llm_endpoint_env_render"
Cohesion: 0.33
Nodes (6): llm_endpoint_env_render, contract_path, description, lane_overlay_glob, rendered_dir_name, state_dir_env

### Community 4122 - "overlay_home_lane_env"
Cohesion: 0.33
Nodes (6): overlay_home_lane_env, clone_env, description, overlay_path, read_timeout_seconds, ref

### Community 4127 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, tags, ticket

### Community 4128 - "title"
Cohesion: 0.33
Nodes (6): title, default, description, max_length, nullable, type

### Community 4129 - "dependencies"
Cohesion: 0.33
Nodes (6): dependencies, class_name, description, module, name, type

### Community 4130 - "retry_policy"
Cohesion: 0.33
Nodes (6): retry_policy, exponential_base, initial_delay_ms, max_delay_ms, max_retries, retry_on

### Community 4132 - "node_vector_store_effect/contract.yaml"
Cohesion: 0.33
Nodes (5): description, name, node_type, node_version, runtime_profiles

### Community 4133 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, created, description, tags, transport_type

### Community 4134 - "config"
Cohesion: 0.33
Nodes (6): config, description, module, name, premium_model_policy, markers

### Community 4135 - "metadata"
Cohesion: 0.33
Nodes (6): metadata, author, description, license, related_tickets, tags

### Community 4138 - "write_config_cloud"
Cohesion: 0.33
Nodes (6): env_output, write_config_cloud, AWS_REGION, CLOUD_PROVIDER, GCP_PROJECT, ONEX_DEPLOYMENT_MODE

### Community 4139 - "setup.yaml"
Cohesion: 0.33
Nodes (5): description, max_estimated_minutes, policy_name, policy_version, target_capabilities

### Community 4140 - "standalone_quickstart.yaml"
Cohesion: 0.33
Nodes (5): description, max_estimated_minutes, policy_name, policy_version, target_capabilities

### Community 4146 - "_load_overlay_entries"
Cohesion: 0.47
Nodes (3): _load_overlay_entries(), load_scope(), _resolve_overlay_path()

### Community 4151 - "instances/local.yaml"
Cohesion: 0.33
Nodes (5): active_preset, databases, presets, application-local, schema_version

### Community 4152 - "omninode_runtime_service"
Cohesion: 0.33
Nodes (6): omninode_runtime_service, bindings, bindings, database_ref, dsn_env, principal

### Community 4153 - "omninode_runtime_service"
Cohesion: 0.33
Nodes (6): omninode_runtime_service, bindings, bindings, database_ref, dsn_env, principal

### Community 4159 - "allowlisted_handlers"
Cohesion: 0.40
Nodes (4): allowlisted_handlers, path, ticket, violations

### Community 4160 - "v0.31.0 (2026-03-31)"
Cohesion: 0.40
Nodes (5): Added, Changed, Dependencies, Fixed, v0.31.0 (2026-03-31)

### Community 4161 - "v0.23.0 (2026-03-20)"
Cohesion: 0.40
Nodes (5): Added, Changed, Fixed, Tests, v0.23.0 (2026-03-20)

### Community 4162 - "tag_clone_consumed_paths"
Cohesion: 0.60
Nodes (5): tag_clone_consumed_paths, tag_clone_consumed_paths, tag_clone_consumed_paths, consumers, path

### Community 4163 - "known_entrypointless"
Cohesion: 0.40
Nodes (4): known_entrypointless, contract, handler, module

### Community 4164 - "observability"
Cohesion: 0.40
Nodes (5): observability, emit_metrics, log_request, log_response, trace_propagation

### Community 4165 - "disk-gc/install-worktree-prune.sh"
Cohesion: 0.80
Nodes (4): die(), preflight(), resolve_python(), install-worktree-prune.sh script

### Community 4166 - "omnidash"
Cohesion: 0.40
Nodes (5): omnidash, description, includes, inject_required_env, services

### Community 4167 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4168 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4169 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4170 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4171 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4172 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4173 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4174 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4175 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4176 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4177 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4178 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4179 - "optional_directory_bind_mounts"
Cohesion: 0.40
Nodes (5): optional_directory_bind_mounts, absent_source, container_path, read_only, source_env

### Community 4180 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4181 - "labels"
Cohesion: 0.40
Nodes (5): labels, autoheal, com.omninode.layer, com.omninode.service, com.omninode.version

### Community 4182 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4183 - "resources"
Cohesion: 0.40
Nodes (5): resources, cpus, cpus_reservation, memory, memory_reservation

### Community 4184 - "redpanda-partition-cap"
Cohesion: 0.40
Nodes (5): REDPANDA_PROFILE_MEMORY, container_name, environment, image, redpanda-partition-cap

### Community 4185 - "redpanda"
Cohesion: 0.40
Nodes (5): command, container_name, image, ports, redpanda

### Community 4186 - "redpanda-partition-cap"
Cohesion: 0.40
Nodes (5): REDPANDA_PROFILE_MEMORY, container_name, environment, image, redpanda-partition-cap

### Community 4187 - "projection-api"
Cohesion: 0.40
Nodes (5): container_name, labels, ports, restart, projection-api

### Community 4188 - "projection-api"
Cohesion: 0.40
Nodes (5): container_name, image, labels, ports, projection-api

### Community 4189 - "redpanda"
Cohesion: 0.40
Nodes (5): command, container_name, image, ports, redpanda

### Community 4190 - "runtime-effects"
Cohesion: 0.40
Nodes (5): container_name, image, labels, ports, runtime-effects

### Community 4191 - "redpanda"
Cohesion: 0.40
Nodes (4): command, image, services, redpanda

### Community 4192 - "run-cloud-migrations.sh"
Cohesion: 0.50
Nodes (4): PGOPTIONS, psql_db(), PSQL_HISTORY, run-cloud-migrations.sh script

### Community 4193 - "broker_readiness_probe.sh"
Cohesion: 0.60
Nodes (4): not_ready(), ready(), RPK_BROKERS, broker_readiness_probe.sh script

### Community 4195 - "steps"
Cohesion: 0.40
Nodes (5): steps, id, name, run, shell

### Community 4196 - "on"
Cohesion: 0.40
Nodes (5): on, push, workflow_dispatch, branches, paths

### Community 4197 - "lineage-guard"
Cohesion: 0.40
Nodes (5): lineage-guard, name, uses, with, enforce_lineage

### Community 4198 - "pinned_repo"
Cohesion: 0.40
Nodes (5): pinned_repo, default, description, required, type

### Community 4199 - "check_delegation_legs"
Cohesion: 0.40
Nodes (5): default, description, required, type, check_delegation_legs

### Community 4200 - "env"
Cohesion: 0.40
Nodes (5): CHAIN_CANARY_GATEWAY_API_KEY, GATEWAY_URL_OVERRIDE, OTHER_TENANT_SLUG, PLANT_CANARY, env

### Community 4201 - "plant_canary_in_answer"
Cohesion: 0.40
Nodes (5): plant_canary_in_answer, default, description, required, type

### Community 4202 - "c12-provider-catalogue"
Cohesion: 0.40
Nodes (5): name, needs, runs-on, timeout-minutes, c12-provider-catalogue

### Community 4203 - "steps"
Cohesion: 0.40
Nodes (5): name, runs-on, steps, timeout-minutes, canonical-handler-shape-gate

### Community 4204 - "dry_run"
Cohesion: 0.40
Nodes (5): default, description, required, type, dry_run

### Community 4205 - "secrets"
Cohesion: 0.40
Nodes (5): required, required, ONEXBOT_OCC_APP_ID, ONEXBOT_OCC_PRIVATE_KEY, secrets

### Community 4206 - "drill_ticket"
Cohesion: 0.40
Nodes (5): default, description, required, type, drill_ticket

### Community 4207 - "dev-lane-liveness"
Cohesion: 0.40
Nodes (5): name, needs, runs-on, timeout-minutes, dev-lane-liveness

### Community 4208 - "env"
Cohesion: 0.40
Nodes (5): COMPOSE_PROJECT_NAME, DOCKER_BUILDKIT, OMNI_DOCKER_BUILD_TIMEOUT_SECONDS, ONEX_EVENT_BUS_TYPE, env

### Community 4209 - "apply"
Cohesion: 0.40
Nodes (5): default, description, required, type, apply

### Community 4210 - "hostile-review-gate"
Cohesion: 0.40
Nodes (5): if, name, needs, runs-on, hostile-review-gate

### Community 4211 - "integration-tests"
Cohesion: 0.40
Nodes (5): name, needs, runs-on, timeout-minutes, integration-tests

### Community 4212 - "on"
Cohesion: 0.40
Nodes (5): on, merge_group, pull_request, branches, types

### Community 4213 - "lane"
Cohesion: 0.40
Nodes (5): lane, description, options, required, type

### Community 4214 - "compose_pg_port"
Cohesion: 0.40
Nodes (5): default, description, required, type, compose_pg_port

### Community 4215 - "pg_port"
Cohesion: 0.40
Nodes (5): pg_port, default, description, required, type

### Community 4216 - "runtime_host"
Cohesion: 0.40
Nodes (5): runtime_host, default, description, required, type

### Community 4217 - "consume-inline"
Cohesion: 0.40
Nodes (5): name, needs, runs-on, timeout-minutes, consume-inline

### Community 4218 - "with"
Cohesion: 0.40
Nodes (5): with, cache-key-prefix, install-args, persist-credentials, python-version

### Community 4219 - "secrets"
Cohesion: 0.40
Nodes (5): required, required, CROSS_REPO_PAT, RUNNER_FLEET_STATUS_TOKEN, secrets

### Community 4220 - "dry_run"
Cohesion: 0.40
Nodes (5): default, description, required, type, dry_run

### Community 4221 - "infra_ref"
Cohesion: 0.40
Nodes (5): default, description, required, type, infra_ref

### Community 4222 - "labels"
Cohesion: 0.40
Nodes (5): labels, default, description, required, type

### Community 4223 - "allowlisted_handlers"
Cohesion: 0.40
Nodes (4): allowlisted_handlers, path, ticket, violations

### Community 4257 - "audit-merge-method-on-main.sh"
Cohesion: 0.70
Nodes (4): allowlisted_two_parent_reason(), build_query(), audit-merge-method-on-main.sh script, usage()

### Community 4258 - "buildx-orphan-sweep.sh"
Cohesion: 0.70
Nodes (4): is_idle(), is_registered(), log(), buildx-orphan-sweep.sh script

### Community 4259 - "check_shell_hygiene.sh"
Cohesion: 0.60
Nodes (3): is_shell_file(), severity_for(), check_shell_hygiene.sh script

### Community 4261 - "validate-state.sh"
Cohesion: 0.90
Nodes (4): log(), log_error(), validate-state.sh script, validate_file()

### Community 4264 - "setup-infisical-identity.sh"
Cohesion: 0.70
Nodes (4): log_error(), log_info(), log_warn(), setup-infisical-identity.sh script

### Community 4270 - "summary_stats"
Cohesion: 0.40
Nodes (5): compare_to_previous, show_trend, summary_stats, defaults, widget_type

### Community 4271 - "defaults"
Cohesion: 0.40
Nodes (5): show_state_transitions, show_timestamps, defaults, widget_type, registration_timeline

### Community 4272 - "version"
Cohesion: 0.60
Nodes (5): version, version, major, minor, patch

### Community 4273 - "audit_logging"
Cohesion: 0.40
Nodes (5): include_correlation_id, never_log, required_events, security_event_level, audit_logging

### Community 4274 - "version"
Cohesion: 0.60
Nodes (5): version, version, major, minor, patch

### Community 4275 - "security"
Cohesion: 0.40
Nodes (5): security, atomic_writes, file_size_validation, idempotent_storage, notes

### Community 4307 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, tags, ticket

### Community 4309 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, optional, type

### Community 4310 - "config"
Cohesion: 0.40
Nodes (5): config, contract_glob, debounce_seconds, source_repo, watch_root

### Community 4311 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4315 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4316 - "board_checks"
Cohesion: 0.40
Nodes (5): board_checks, check_id, description, operation, surface_class

### Community 4317 - "published_events"
Cohesion: 0.40
Nodes (5): published_events, description, event_type, message_category, topic

### Community 4319 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4320 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4321 - "transitions"
Cohesion: 0.40
Nodes (5): transitions, from_state, to_state, transition_name, trigger

### Community 4322 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4323 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4325 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4326 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4329 - "canary"
Cohesion: 0.40
Nodes (5): cadence_seconds, produce_deadline_seconds, readback_deadline_seconds, topic, canary

### Community 4330 - "dedupe"
Cohesion: 0.40
Nodes (5): key, retention_hours, store, store_path_required, dedupe

### Community 4333 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4334 - "published_events"
Cohesion: 0.40
Nodes (5): published_events, description, event_type, message_category, topic

### Community 4336 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4338 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4343 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, library, name, type, version

### Community 4344 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, related_tickets, transport_type

### Community 4345 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, library, name, type, version

### Community 4346 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, related_tickets, transport_type

### Community 4348 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, library, name, type, version

### Community 4349 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, related_tickets, transport_type

### Community 4351 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, library, name, type, version

### Community 4352 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, related_tickets, transport_type

### Community 4354 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4356 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, required, type

### Community 4357 - "feature_flags"
Cohesion: 0.40
Nodes (5): feature_flags, default, description, env_var, name

### Community 4358 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4360 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, documentation_url, tags

### Community 4361 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4362 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4364 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4365 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4367 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, tags, ticket

### Community 4369 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4371 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4372 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4374 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4376 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4380 - "chain_topology"
Cohesion: 0.40
Nodes (5): chain_topology, alternatives, parent, reroute_parents, topic

### Community 4381 - "node_capabilities"
Cohesion: 0.40
Nodes (5): node_capabilities, database, postgres, read, write

### Community 4384 - "descriptor"
Cohesion: 0.40
Nodes (5): descriptor, idempotent, node_archetype, purity, timeout_ms

### Community 4385 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4386 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4387 - "handler_routing"
Cohesion: 0.40
Nodes (5): handler_routing, aggregation_strategy, execution_mode, partial_failure_handling, routing_strategy

### Community 4388 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4390 - "handler_routing"
Cohesion: 0.40
Nodes (5): handler_routing, aggregation_strategy, execution_mode, partial_failure_handling, routing_strategy

### Community 4391 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4393 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, tags, ticket

### Community 4399 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4400 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4401 - "alert_policy"
Cohesion: 0.40
Nodes (5): alert_policy, default_max_arrivals_per_window, max_retained_depth, overrides, window_seconds

### Community 4402 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4405 - "alert_policy"
Cohesion: 0.40
Nodes (5): alert_policy, default_max_arrivals_per_window, max_retained_depth, overrides, window_seconds

### Community 4406 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4408 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, required, type

### Community 4409 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4410 - "replay_policy"
Cohesion: 0.40
Nodes (5): replay_policy, broadcast_topics, target_replay_bound, max_replays, window_seconds

### Community 4412 - "observability"
Cohesion: 0.40
Nodes (5): observability, emit_metrics, log_request, log_response, trace_propagation

### Community 4413 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, documentation_url, tags

### Community 4414 - "circuit_breaker"
Cohesion: 0.40
Nodes (5): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker

### Community 4415 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4417 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4420 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4421 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4424 - "errors"
Cohesion: 0.40
Nodes (5): default_factory, description, item_type, type, errors

### Community 4426 - "events_published"
Cohesion: 0.40
Nodes (5): default, description, ge, type, events_published

### Community 4427 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4428 - "messages_archived"
Cohesion: 0.40
Nodes (5): default, description, ge, type, messages_archived

### Community 4430 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4431 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4433 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4434 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4436 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_model, operation, output_model

### Community 4437 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, tags, ticket

### Community 4439 - "output_model"
Cohesion: 0.50
Nodes (5): output_model, output_model, description, module, name

### Community 4440 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4441 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4442 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4443 - "published_events"
Cohesion: 0.40
Nodes (5): published_events, description, event_type, message_category, topic

### Community 4444 - "consumed_events"
Cohesion: 0.40
Nodes (5): consumed_events, description, event_type, message_category, topic

### Community 4445 - "node_capabilities"
Cohesion: 0.40
Nodes (5): node_capabilities, database, postgres, transactions, write

### Community 4446 - "circuit_breaker"
Cohesion: 0.40
Nodes (5): enabled, failure_threshold, reset_timeout_ms, error_handling, circuit_breaker

### Community 4447 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4448 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4449 - "backends"
Cohesion: 0.40
Nodes (5): failure_threshold, half_open_max_requests, name, reset_timeout_ms, backends

### Community 4450 - "consumed_events"
Cohesion: 0.40
Nodes (5): consumed_events, description, event_type, message_category, topic

### Community 4452 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4453 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4459 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4460 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4461 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, documentation_url, tags

### Community 4463 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4465 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4468 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, tags, ticket

### Community 4470 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4471 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, license, related_tickets

### Community 4473 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4474 - "intent_consumption"
Cohesion: 0.40
Nodes (5): intent_consumption, intent_routing_table, subscribed_intents, postgres.update_registration, postgres.upsert_registration

### Community 4475 - "mcp"
Cohesion: 0.40
Nodes (5): mcp, description, expose, timeout_seconds, tool_name

### Community 4476 - "time_injection"
Cohesion: 0.40
Nodes (5): time_injection, description, enabled, field, source

### Community 4478 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4480 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4481 - "backend_id"
Cohesion: 0.40
Nodes (5): default, description, nullable, type, backend_id

### Community 4482 - "service_name"
Cohesion: 0.40
Nodes (5): service_name, default, description, nullable, type

### Community 4486 - "consumed_events"
Cohesion: 0.40
Nodes (5): consumed_events, description, event_type, message_category, topic

### Community 4487 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4489 - "node_routing_score_reducer/contract.yaml"
Cohesion: 0.40
Nodes (4): description, name, node_type, node_version

### Community 4490 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4491 - "transitions"
Cohesion: 0.40
Nodes (5): transitions, description, from_state, to_state, trigger

### Community 4498 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4499 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4502 - "published_events"
Cohesion: 0.40
Nodes (5): published_events, description, event_type, message_category, topic

### Community 4506 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4507 - "node_runner_health_snapshot_effect/contract.yaml"
Cohesion: 0.40
Nodes (4): description, name, node_type, runtime_profiles

### Community 4509 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4511 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, required, type

### Community 4512 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4513 - "published_events"
Cohesion: 0.40
Nodes (5): published_events, description, event_type, message_category, topic

### Community 4515 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4517 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, required, type

### Community 4518 - "feature_flags"
Cohesion: 0.40
Nodes (5): feature_flags, default, description, env_var, name

### Community 4519 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4522 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4526 - "descriptor"
Cohesion: 0.40
Nodes (5): descriptor, idempotent, node_archetype, purity, timeout_ms

### Community 4527 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, created, description, tags

### Community 4529 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4530 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4531 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4534 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4535 - "output_model"
Cohesion: 0.50
Nodes (5): output_model, output_model, description, module, name

### Community 4536 - "source_configuration"
Cohesion: 0.40
Nodes (5): source_configuration, description, fragment_directory, fragment_schema, home

### Community 4537 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, name, output_fields

### Community 4538 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4540 - "duration_ms"
Cohesion: 0.40
Nodes (5): default, description, ge, type, duration_ms

### Community 4541 - "message"
Cohesion: 0.40
Nodes (5): description, max_length, min_length, type, message

### Community 4542 - "retry_count"
Cohesion: 0.40
Nodes (5): retry_count, default, description, ge, type

### Community 4545 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4548 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, name, required, type

### Community 4549 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4551 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4553 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4555 - "dependencies"
Cohesion: 0.40
Nodes (5): dependencies, description, module, name, type

### Community 4558 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4561 - "node_capabilities"
Cohesion: 0.40
Nodes (5): node_capabilities, database, postgres, transactions, write

### Community 4562 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4563 - "descriptor"
Cohesion: 0.40
Nodes (5): idempotent, node_archetype, purity, qdrant_url, descriptor

### Community 4564 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4565 - "metadata"
Cohesion: 0.40
Nodes (5): metadata, author, description, license, related_tickets

### Community 4568 - "input_model"
Cohesion: 0.50
Nodes (5): input_model, input_model, description, module, name

### Community 4570 - "input_model"
Cohesion: 0.60
Nodes (5): input_model, input_model, description, module, name

### Community 4571 - "io_operations"
Cohesion: 0.40
Nodes (5): io_operations, description, input_fields, operation, output_fields

### Community 4575 - "contributor_cloud.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4576 - "contributor_hybrid.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4577 - "contributor_local.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4578 - "full_platform.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4579 - "gcp"
Cohesion: 0.40
Nodes (5): set_state, next, set_state, gcp, cloud_provider

### Community 4580 - "new_employee.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4581 - "omnimarket_quickstart.yaml"
Cohesion: 0.40
Nodes (4): description, max_estimated_minutes, policy_name, target_capabilities

### Community 4607 - "arch-002-no-transport-imports"
Cohesion: 0.50
Nodes (4): ci_workflow, pre_commit, tracking, arch-002-no-transport-imports

### Community 4608 - "bandit-security"
Cohesion: 0.50
Nodes (4): ci_workflow, pre_commit, tracking, bandit-security

### Community 4609 - "banned-compose-env-vars"
Cohesion: 0.50
Nodes (4): ci_workflow, pre_commit, tracking, banned-compose-env-vars

### Community 4610 - "detect-secrets"
Cohesion: 0.50
Nodes (4): detect-secrets, ci_workflow, pre_commit, tracking

### Community 4611 - "enum-governance"
Cohesion: 0.50
Nodes (4): enum-governance, ci_workflow, pre_commit, tracking

### Community 4612 - "hardcoded-local-paths"
Cohesion: 0.50
Nodes (4): hardcoded-local-paths, ci_workflow, pre_commit, tracking

### Community 4613 - "naming-conventions"
Cohesion: 0.50
Nodes (4): naming-conventions, ci_workflow, pre_commit, tracking

### Community 4614 - "pydantic-patterns"
Cohesion: 0.50
Nodes (4): pydantic-patterns, ci_workflow, pre_commit, tracking

### Community 4615 - "single-class-per-file"
Cohesion: 0.50
Nodes (4): single-class-per-file, ci_workflow, pre_commit, tracking

### Community 4616 - "v0.24.0 (2026-03-22)"
Cohesion: 0.50
Nodes (4): Added, Changed, Fixed, v0.24.0 (2026-03-22)

### Community 4617 - "v0.27.0 (2026-03-25)"
Cohesion: 0.50
Nodes (4): Dependencies, Fixed, Tests, v0.27.0 (2026-03-25)

### Community 4618 - "gates"
Cohesion: 0.50
Nodes (3): gates, schema_version, ticket

### Community 4619 - "entries"
Cohesion: 0.50
Nodes (3): entries, finding, removed_by

### Community 4620 - "check-dependency-pins"
Cohesion: 0.50
Nodes (4): reason, status, target, check-dependency-pins

### Community 4621 - "check-doc-content-scan-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-doc-content-scan-compute

### Community 4622 - "check-duplicate-registry-ids"
Cohesion: 0.50
Nodes (4): reason, status, target, check-duplicate-registry-ids

### Community 4623 - "check-fsm-handler-drift"
Cohesion: 0.50
Nodes (4): reason, status, target, check-fsm-handler-drift

### Community 4624 - "check-github-token-env-reads"
Cohesion: 0.50
Nodes (4): reason, status, target, check-github-token-env-reads

### Community 4625 - "check-imperative-contracts"
Cohesion: 0.50
Nodes (4): reason, status, target, check-imperative-contracts

### Community 4626 - "check-invariant-analysis"
Cohesion: 0.50
Nodes (4): reason, status, target, check-invariant-analysis

### Community 4627 - "check-local-paths-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-local-paths-compute

### Community 4628 - "check-local-paths"
Cohesion: 0.50
Nodes (4): reason, status, target, check-local-paths

### Community 4629 - "check-localhost-url-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-localhost-url-compute

### Community 4630 - "check-merge-queue-parity"
Cohesion: 0.50
Nodes (4): reason, status, target, check-merge-queue-parity

### Community 4631 - "check-no-backward-compat"
Cohesion: 0.50
Nodes (4): reason, status, target, check-no-backward-compat

### Community 4632 - "check-no-faked-boundary"
Cohesion: 0.50
Nodes (4): reason, status, target, check-no-faked-boundary

### Community 4633 - "check-pin-hygiene-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-pin-hygiene-compute

### Community 4634 - "check-private-ip-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-private-ip-compute

### Community 4635 - "check-todo-fixme-marker-compute"
Cohesion: 0.50
Nodes (4): reason, status, target, check-todo-fixme-marker-compute

### Community 4636 - "omni-agent"
Cohesion: 0.50
Nodes (4): reason, status, target, omni-agent

### Community 4637 - "omninode-runtime-host-dev"
Cohesion: 0.50
Nodes (4): omninode-runtime-host-dev, reason, status, target

### Community 4638 - "onex"
Cohesion: 0.50
Nodes (4): onex, reason, status, target

### Community 4639 - "onex-validate-links"
Cohesion: 0.50
Nodes (4): onex-validate-links, reason, status, target

### Community 4640 - "onex-validate-topics"
Cohesion: 0.50
Nodes (4): onex-validate-topics, reason, status, target

### Community 4641 - "onex-work-ledger"
Cohesion: 0.50
Nodes (4): onex-work-ledger, reason, status, target

### Community 4642 - "resolve-private-vocabulary"
Cohesion: 0.50
Nodes (4): resolve-private-vocabulary, reason, status, target

### Community 4643 - "validate-string-versions"
Cohesion: 0.50
Nodes (4): validate-string-versions, reason, status, target

### Community 4644 - "onex-gateway-forwarder"
Cohesion: 0.50
Nodes (4): onex-gateway-forwarder, reason, status, target

### Community 4645 - "onex-linear-relay"
Cohesion: 0.50
Nodes (4): onex-linear-relay, reason, status, target

### Community 4646 - "onex-status"
Cohesion: 0.50
Nodes (4): onex-status, reason, status, target

### Community 4647 - "emergency_bypass"
Cohesion: 0.50
Nodes (4): emergency_bypass, enabled, follow_up_ticket_id, justification

### Community 4648 - "evidence_requirements"
Cohesion: 0.50
Nodes (4): evidence_requirements, command, description, kind

### Community 4649 - "install-lane-census.sh"
Cohesion: 0.83
Nodes (3): render_unit(), resolve_unit_dir(), install-lane-census.sh script

### Community 4650 - "provision-vllm-0.27-venv.sh"
Cohesion: 1.00
Nodes (3): ensure_symlink(), fail(), provision-vllm-0.27-venv.sh script

### Community 4651 - "nofile"
Cohesion: 0.50
Nodes (4): hard, soft, ulimits, nofile

### Community 4652 - "labels"
Cohesion: 0.50
Nodes (4): labels, com.omninode.layer, com.omninode.service, com.omninode.ticket

### Community 4653 - "operational_defaults"
Cohesion: 0.50
Nodes (4): operational_defaults, GATEWAY_P0B_PER_TENANT_CREDENTIALS, RATE_LIMIT_TRUSTED_PROXY_CIDRS, TRUST_PROXY_HEADERS

### Community 4654 - "nofile"
Cohesion: 0.50
Nodes (4): hard, soft, ulimits, nofile

### Community 4655 - "nofile"
Cohesion: 0.50
Nodes (4): hard, soft, ulimits, nofile

### Community 4656 - "nofile"
Cohesion: 0.50
Nodes (4): hard, soft, ulimits, nofile

### Community 4657 - "phoenix"
Cohesion: 0.50
Nodes (4): container_name, ports, profiles, phoenix

### Community 4658 - "phoenix"
Cohesion: 0.50
Nodes (4): container_name, ports, profiles, phoenix

### Community 4659 - "runtime-effects"
Cohesion: 0.50
Nodes (4): container_name, networks, ports, runtime-effects

### Community 4660 - "environment"
Cohesion: 0.50
Nodes (4): KC_HOSTNAME, KC_HOSTNAME_BACKCHANNEL_DYNAMIC, environment, keycloak

### Community 4661 - "forward-migration"
Cohesion: 0.50
Nodes (4): ONEX_MIGRATION_LANE, environment, labels, forward-migration

### Community 4662 - "intelligence-migration"
Cohesion: 0.50
Nodes (4): container_name, labels, restart, intelligence-migration

### Community 4663 - "valkey"
Cohesion: 0.50
Nodes (4): valkey, container_name, labels, ports

### Community 4664 - "lane_mirror_binding"
Cohesion: 0.50
Nodes (4): lane_mirror_binding, mirror_lanes, source_lane, topic_sets

### Community 4665 - "tenant_identity"
Cohesion: 0.50
Nodes (4): tenant_identity, principal_id, tenant_id, tenant_slug

### Community 4666 - "cutover-proof/prove.sh"
Cohesion: 0.83
Nodes (3): fail_cutover(), prove.sh script, sql_value_cutover()

### Community 4667 - "fenced_node_migrations"
Cohesion: 0.50
Nodes (3): fenced_node_migrations, id, ticket

### Community 4668 - "grandfathered_force_rls_migrations"
Cohesion: 0.50
Nodes (3): grandfathered_force_rls_migrations, id, ticket

### Community 4669 - "git-mirror-refresh.sh"
Cohesion: 0.83
Nodes (3): apply_mirror_serving_config(), maintain_mirror_packs(), git-mirror-refresh.sh script

### Community 4670 - "playwright-smoke.sh"
Cohesion: 0.67
Nodes (3): fail(), PLAYWRIGHT_BROWSERS_PATH, playwright-smoke.sh script

### Community 4672 - "aws-region"
Cohesion: 0.50
Nodes (4): default, description, required, aws-region

### Community 4673 - "backoff-factor"
Cohesion: 0.50
Nodes (4): default, description, required, backoff-factor

### Community 4674 - "fail-on-missing"
Cohesion: 0.50
Nodes (4): default, description, required, fail-on-missing

### Community 4675 - "initial-delay"
Cohesion: 0.50
Nodes (4): default, description, required, initial-delay

### Community 4676 - "max-delay"
Cohesion: 0.50
Nodes (4): max-delay, default, description, required

### Community 4677 - "deploy-wait-seconds"
Cohesion: 0.50
Nodes (4): default, description, required, deploy-wait-seconds

### Community 4678 - "lanes-var"
Cohesion: 0.50
Nodes (4): lanes-var, default, description, required

### Community 4679 - "match"
Cohesion: 0.50
Nodes (4): match, default, description, required

### Community 4680 - "wait-while-busy"
Cohesion: 0.50
Nodes (4): wait-while-busy, default, description, required

### Community 4681 - "cache-key-prefix"
Cohesion: 0.50
Nodes (4): default, description, required, cache-key-prefix

### Community 4682 - "cache-version"
Cohesion: 0.50
Nodes (4): default, description, required, cache-version

### Community 4683 - "github-token"
Cohesion: 0.50
Nodes (4): default, description, required, github-token

### Community 4684 - "install-args"
Cohesion: 0.50
Nodes (4): install-args, default, description, required

### Community 4685 - "lock-file-path"
Cohesion: 0.50
Nodes (4): lock-file-path, default, description, required

### Community 4686 - "python-version"
Cohesion: 0.50
Nodes (4): python-version, default, description, required

### Community 4687 - "shared-env-enabled"
Cohesion: 0.50
Nodes (4): shared-env-enabled, default, description, required

### Community 4688 - "shared-env-install-args"
Cohesion: 0.50
Nodes (4): shared-env-install-args, default, description, required

### Community 4689 - "skip-install"
Cohesion: 0.50
Nodes (4): skip-install, default, description, required

### Community 4690 - "sync-attempts"
Cohesion: 0.50
Nodes (4): sync-attempts, default, description, required

### Community 4691 - "sync-retry-delay-seconds"
Cohesion: 0.50
Nodes (4): sync-retry-delay-seconds, default, description, required

### Community 4692 - "uv-version"
Cohesion: 0.50
Nodes (4): uv-version, default, description, required

### Community 4693 - "working-directory"
Cohesion: 0.50
Nodes (4): working-directory, default, description, required

### Community 4694 - "codeql-config.yml"
Cohesion: 0.50
Nodes (3): name, paths, paths-ignore

### Community 4695 - "env"
Cohesion: 0.50
Nodes (4): env, AWS_REGION, ECR_REGISTRY, ECR_REPOSITORY

### Community 4696 - "git-ref"
Cohesion: 0.50
Nodes (4): description, required, type, git-ref

### Community 4697 - "bus_lane"
Cohesion: 0.50
Nodes (4): default, description, required, bus_lane

### Community 4698 - "deploy_wait_seconds"
Cohesion: 0.50
Nodes (4): default, description, required, deploy_wait_seconds

### Community 4699 - "gateway_url"
Cohesion: 0.50
Nodes (4): default, description, required, gateway_url

### Community 4700 - "check_pinned_backends"
Cohesion: 0.50
Nodes (4): default, description, required, check_pinned_backends

### Community 4701 - "pinned_backend_override_free_cloud"
Cohesion: 0.50
Nodes (4): pinned_backend_override_free_cloud, default, description, required

### Community 4702 - "pinned_backend_override_local"
Cohesion: 0.50
Nodes (4): pinned_backend_override_local, default, description, required

### Community 4703 - "pinned_backend_override_paid"
Cohesion: 0.50
Nodes (4): pinned_backend_override_paid, default, description, required

### Community 4704 - "env"
Cohesion: 0.50
Nodes (4): C28_BASE_URL, LANE_COMPOSE_PROJECT, LANE_NAME, env

### Community 4705 - "deploy_wait_seconds"
Cohesion: 0.50
Nodes (4): default, description, required, deploy_wait_seconds

### Community 4706 - "gateway_url"
Cohesion: 0.50
Nodes (4): default, description, required, gateway_url

### Community 4707 - "quarantine_bootstrap_servers"
Cohesion: 0.50
Nodes (4): quarantine_bootstrap_servers, default, description, required

### Community 4708 - "task_type"
Cohesion: 0.50
Nodes (4): task_type, default, description, required

### Community 4709 - "on"
Cohesion: 0.50
Nodes (4): on, schedule, workflow_dispatch, cron

### Community 4710 - "version"
Cohesion: 0.50
Nodes (4): version, description, required, type

### Community 4711 - "bus_lane"
Cohesion: 0.50
Nodes (4): default, description, required, bus_lane

### Community 4712 - "topic_prefix"
Cohesion: 0.50
Nodes (4): topic_prefix, default, description, required

### Community 4713 - "lookback_hours"
Cohesion: 0.50
Nodes (4): lookback_hours, default, description, required

### Community 4714 - "max_comments_per_run"
Cohesion: 0.50
Nodes (4): max_comments_per_run, default, description, type

### Community 4715 - "project_id"
Cohesion: 0.50
Nodes (4): project_id, default, description, type

### Community 4716 - "refresh"
Cohesion: 0.50
Nodes (4): refresh, name, runs-on, timeout-minutes

### Community 4717 - "ci-naming-convention"
Cohesion: 0.50
Nodes (4): name, runs-on, timeout-minutes, ci-naming-convention

### Community 4718 - "type-safety"
Cohesion: 0.50
Nodes (4): type-safety, name, runs-on, timeout-minutes

### Community 4719 - "type-union-check"
Cohesion: 0.50
Nodes (4): type-union-check, name, runs-on, timeout-minutes

### Community 4720 - "changed_files"
Cohesion: 0.50
Nodes (4): description, required, type, changed_files

### Community 4721 - "source_sha"
Cohesion: 0.50
Nodes (4): source_sha, description, required, type

### Community 4722 - "ticket"
Cohesion: 0.50
Nodes (4): ticket, description, required, type

### Community 4723 - "lint-shadow"
Cohesion: 0.50
Nodes (4): lint-shadow, name, runs-on, timeout-minutes

### Community 4724 - "tests-shadow"
Cohesion: 0.50
Nodes (4): tests-shadow, name, runs-on, timeout-minutes

### Community 4725 - "typecheck-shadow"
Cohesion: 0.50
Nodes (4): typecheck-shadow, name, runs-on, timeout-minutes

### Community 4726 - "source_sha"
Cohesion: 0.50
Nodes (4): source_sha, description, required, type

### Community 4765 - "node_boundary_import_baseline.yaml"
Cohesion: 0.50
Nodes (3): edges, gate, schema_version

### Community 4770 - "resolve_modern_bash.sh"
Cohesion: 0.83
Nodes (3): bash_major_version(), resolve_modern_bash.sh script, try_candidate()

### Community 4771 - "git-gc-auto.sh"
Cohesion: 0.83
Nodes (3): active_op_reason(), repo_blocked_reason(), git-gc-auto.sh script

### Community 4772 - "infra-watchdog.sh"
Cohesion: 0.83
Nodes (3): container_is_running(), log(), infra-watchdog.sh script

### Community 4774 - "sync-node-migrations.sh"
Cohesion: 0.83
Nodes (3): is_legacy_declared(), resolve_omnimarket_src(), sync-node-migrations.sh script

### Community 4775 - "check_no_infra_inmemory_import.sh"
Cohesion: 0.83
Nodes (3): _is_allowlisted(), _normalize_path(), check_no_infra_inmemory_import.sh script

### Community 4776 - "verify-plugin-cache.sh"
Cohesion: 0.83
Nodes (3): build_file_manifest(), compute_content_hash(), verify-plugin-cache.sh script

### Community 4777 - "compute.validate"
Cohesion: 0.50
Nodes (4): compute.validate, defaults, widget_type, show_pass_rate

### Community 4778 - "kafka.stream"
Cohesion: 0.50
Nodes (4): kafka.stream, show_offset, defaults, widget_type

### Community 4779 - "postgres.query"
Cohesion: 0.50
Nodes (4): postgres.query, show_query_time, defaults, widget_type

### Community 4780 - "workflow.reduce"
Cohesion: 0.50
Nodes (4): workflow.reduce, show_progress, defaults, widget_type

### Community 4781 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4782 - "retry_policy"
Cohesion: 0.50
Nodes (4): retry_policy, backoff_ms, enabled, max_retries

### Community 4783 - "graph.delete_node"
Cohesion: 0.50
Nodes (4): graph.delete_node, expression, parameter_name, required

### Community 4784 - "graph.delete_relationship"
Cohesion: 0.50
Nodes (4): graph.delete_relationship, expression, parameter_name, required

### Community 4785 - "graph.execute_query_batch"
Cohesion: 0.50
Nodes (4): graph.execute_query_batch, expression, parameter_name, required

### Community 4786 - "graph.traverse"
Cohesion: 0.50
Nodes (4): graph.traverse, expression, parameter_name, required

### Community 4787 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4788 - "http.post"
Cohesion: 0.50
Nodes (4): http.post, expression, parameter_name, required

### Community 4789 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4790 - "future_considerations"
Cohesion: 0.50
Nodes (4): burst_allowance, client_tracking, per_tool_limits, future_considerations

### Community 4791 - "input_validation"
Cohesion: 0.50
Nodes (4): max_arguments_size, max_nesting_depth, validate_schema, input_validation

### Community 4796 - "omnibase_infra/handlers/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4797 - "handler_broker_grant_derive/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4798 - "handler_consumer_coordinator_rejoin/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4799 - "handler_declared_state_diff/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4800 - "handler_developer_principal_plan/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4801 - "handler_event_provenance_stamp/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4804 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4805 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4806 - "storage"
Cohesion: 0.50
Nodes (4): storage, example, partitioning, pattern

### Community 4807 - "version"
Cohesion: 0.50
Nodes (4): version, major, minor, patch

### Community 4808 - "handler_registration_from_announcement/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4809 - "handler_validation_adjudicate/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4810 - "handler_validator_event_ledger_project/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4811 - "hook_venv_reconcile/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4812 - "venv_atomic_install/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 4813 - "settings"
Cohesion: 0.50
Nodes (3): settings, default_contracts_dir, llm_endpoint_env_keys

### Community 4814 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4815 - "descriptor"
Cohesion: 0.50
Nodes (4): descriptor, idempotent, node_archetype, purity

### Community 4816 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4817 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4818 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4819 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4820 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4821 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4822 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4823 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4824 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4825 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4826 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4827 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4828 - "consumed_events"
Cohesion: 0.50
Nodes (4): consumed_events, description, event_type, topic

### Community 4829 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4830 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4831 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4832 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4833 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4834 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4835 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4837 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4838 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4839 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4840 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4841 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4842 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4843 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4844 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4845 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4846 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4847 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4848 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4849 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4850 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4851 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4852 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 4853 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4854 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4855 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4856 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 4857 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4858 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4859 - "node_capabilities"
Cohesion: 0.50
Nodes (4): node_capabilities, database, postgres, write

### Community 4860 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4861 - "config"
Cohesion: 0.50
Nodes (4): config, description, module, name

### Community 4862 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4863 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4864 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4865 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4866 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4867 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4868 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4869 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4870 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4871 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4872 - "event_model"
Cohesion: 0.50
Nodes (4): description, module, name, event_model

### Community 4873 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4874 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4875 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4876 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4877 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4878 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4879 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4880 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4881 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4882 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4883 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4884 - "runtime_dispatch"
Cohesion: 0.50
Nodes (4): runtime_dispatch, command_topic, external_trigger, terminal_events

### Community 4885 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4886 - "event_model"
Cohesion: 0.50
Nodes (4): description, module, name, event_model

### Community 4887 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4888 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4889 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4890 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4891 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4892 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4893 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4894 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4895 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4896 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4897 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4898 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4899 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 4900 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4901 - "staleness"
Cohesion: 0.50
Nodes (4): staleness, description, env_var, threshold_seconds

### Community 4902 - "state_model"
Cohesion: 0.50
Nodes (4): state_model, description, module, name

### Community 4903 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4904 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 4905 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4906 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4907 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4908 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4909 - "event_bus"
Cohesion: 0.50
Nodes (4): event_bus, dlq_topics, publish_topics, subscribe_topics

### Community 4910 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4911 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4912 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4913 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4914 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4915 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4916 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4917 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4918 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4919 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4920 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4921 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4922 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4923 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4924 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4925 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4926 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4927 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4928 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4929 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4930 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4931 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4932 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4933 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4934 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4935 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4936 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4937 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4938 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4939 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4940 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4941 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4942 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4943 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4944 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4945 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4946 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4947 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4948 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4950 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4951 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4952 - "dependencies"
Cohesion: 0.50
Nodes (4): dependencies, description, name, type

### Community 4953 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4954 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4955 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4956 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4957 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4958 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4959 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4960 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4961 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4962 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4963 - "config"
Cohesion: 0.50
Nodes (4): config, description, module, name

### Community 4964 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4965 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4966 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4967 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4968 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 4969 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4970 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4971 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4972 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 4973 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4974 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4975 - "node_capabilities"
Cohesion: 0.50
Nodes (4): node_capabilities, database, postgres, write

### Community 4976 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4977 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4978 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4979 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4980 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4981 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4982 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4983 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4984 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4985 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4986 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4987 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4988 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4989 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4990 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4991 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4992 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 4993 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4994 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 4995 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4996 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4997 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 4998 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 4999 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5000 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5001 - "descriptor"
Cohesion: 0.50
Nodes (4): descriptor, idempotent, node_archetype, purity

### Community 5002 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5003 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5004 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 5005 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5006 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5007 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5008 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 5009 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5010 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5011 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5012 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5013 - "published_events"
Cohesion: 0.50
Nodes (4): published_events, description, event_type, topic

### Community 5014 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5015 - "dependencies"
Cohesion: 0.50
Nodes (4): dependencies, description, name, type

### Community 5016 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5017 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5018 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5019 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5020 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5021 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5022 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5023 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5024 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5025 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5026 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5027 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5028 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5029 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5030 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5031 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5032 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5033 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5034 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5035 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5036 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5037 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5038 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5039 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5040 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5041 - "io_operations"
Cohesion: 0.50
Nodes (4): io_operations, description, idempotent, operation

### Community 5042 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5043 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5044 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5045 - "event_bus"
Cohesion: 0.50
Nodes (4): event_bus, plugin_managed, publish_topics, subscribe_topics

### Community 5046 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5047 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5048 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5049 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5050 - "descriptor"
Cohesion: 0.50
Nodes (4): descriptor, idempotent, node_archetype, purity

### Community 5051 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5052 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5053 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5054 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5055 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5056 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5057 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5058 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5059 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5060 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5061 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5062 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5063 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5064 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5065 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5066 - "handler_routing"
Cohesion: 0.50
Nodes (4): handler_routing, execution_mode, max_timeout_seconds, routing_strategy

### Community 5067 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5068 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5069 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5070 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5071 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5072 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5073 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5074 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5075 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5076 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5077 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5078 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5079 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5080 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5081 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5082 - "runtime_dispatch"
Cohesion: 0.50
Nodes (4): runtime_dispatch, command_topic, external_trigger, terminal_events

### Community 5083 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5084 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5085 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5086 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5087 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5088 - "metadata"
Cohesion: 0.50
Nodes (4): metadata, description, related_tickets, tags

### Community 5089 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5090 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5091 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5092 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5093 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5094 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5095 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5096 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5097 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5098 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5099 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5100 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5101 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5102 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5103 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5104 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5105 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5106 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5107 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5108 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5109 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5110 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5111 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5112 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5113 - "idempotency"
Cohesion: 0.50
Nodes (4): idempotency, description, enabled, strategy

### Community 5114 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5115 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5116 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5117 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5118 - "dependencies"
Cohesion: 0.50
Nodes (4): dependencies, description, name, type

### Community 5119 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5120 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5121 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 5122 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5123 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5124 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5125 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5126 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5127 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5128 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5129 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5130 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5131 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5132 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5133 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5134 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5135 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5136 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5137 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5138 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5139 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5140 - "idempotency"
Cohesion: 0.50
Nodes (4): idempotency, description, enabled, strategy

### Community 5141 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5142 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5143 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5144 - "published_events"
Cohesion: 0.50
Nodes (4): published_events, description, event_type, topic

### Community 5145 - "state_model"
Cohesion: 0.50
Nodes (4): state_model, description, module, name

### Community 5146 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5147 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5148 - "idempotency"
Cohesion: 0.50
Nodes (4): idempotency, description, enabled, strategy

### Community 5149 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5150 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5151 - "state_model"
Cohesion: 0.50
Nodes (4): state_model, description, module, name

### Community 5152 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5153 - "io_operations"
Cohesion: 0.50
Nodes (4): io_operations, description, idempotent, operation

### Community 5154 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5155 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5156 - "capabilities"
Cohesion: 0.50
Nodes (4): capabilities, description, name, version

### Community 5157 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5158 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5159 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5160 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5161 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5162 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5163 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5164 - "dependencies"
Cohesion: 0.50
Nodes (4): dependencies, description, name, type

### Community 5165 - "health_check"
Cohesion: 0.50
Nodes (4): health_check, enabled, endpoint, interval_seconds

### Community 5166 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5167 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5168 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5169 - "dependencies"
Cohesion: 0.50
Nodes (4): dependencies, description, name, type

### Community 5170 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5171 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5172 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5173 - "descriptor"
Cohesion: 0.50
Nodes (4): descriptor, idempotent, node_archetype, purity

### Community 5174 - "input_model"
Cohesion: 0.50
Nodes (4): input_model, description, module, name

### Community 5175 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5176 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5177 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5178 - "metadata"
Cohesion: 0.50
Nodes (4): metadata, consumer_group_id, description, related_tickets

### Community 5179 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5180 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5181 - "contract_version"
Cohesion: 0.50
Nodes (4): contract_version, major, minor, patch

### Community 5182 - "node_version"
Cohesion: 0.50
Nodes (4): node_version, major, minor, patch

### Community 5183 - "output_model"
Cohesion: 0.50
Nodes (4): output_model, description, module, name

### Community 5184 - "auto_wiring/contract.yaml"
Cohesion: 0.50
Nodes (3): descriptor, node_archetype, node_type

### Community 5187 - "tracing_contract.yaml"
Cohesion: 0.50
Nodes (3): otel_exporter_otlp_endpoint, name, descriptor

### Community 5190 - "no-untracked-todos"
Cohesion: 0.67
Nodes (3): no-untracked-todos, ci_workflow, tracking

### Community 5191 - "stub-implementations"
Cohesion: 0.67
Nodes (3): stub-implementations, ci_workflow, tracking

### Community 5197 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5198 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5199 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5200 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5201 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5202 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5203 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5204 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5205 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5206 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5207 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5208 - "ports"
Cohesion: 0.67
Nodes (3): ports, external, internal

### Community 5209 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5210 - "depends_on"
Cohesion: 0.67
Nodes (3): depends_on, condition, service

### Community 5211 - "omninode-contract-resolver"
Cohesion: 0.67
Nodes (3): container_name, ports, omninode-contract-resolver

### Community 5212 - "postgres"
Cohesion: 0.67
Nodes (3): container_name, ports, postgres

### Community 5213 - "redpanda-sasl-enable"
Cohesion: 0.67
Nodes (3): container_name, image, redpanda-sasl-enable

### Community 5214 - "agent-actions-consumer"
Cohesion: 0.67
Nodes (3): container_name, ports, agent-actions-consumer

### Community 5215 - "context-audit-consumer"
Cohesion: 0.67
Nodes (3): container_name, ports, context-audit-consumer

### Community 5216 - "redpanda-sasl-enable"
Cohesion: 0.67
Nodes (3): container_name, image, redpanda-sasl-enable

### Community 5217 - "redpanda-scram-user"
Cohesion: 0.67
Nodes (3): container_name, image, redpanda-scram-user

### Community 5218 - "gateway-dns-bastion"
Cohesion: 0.67
Nodes (3): gateway-dns-bastion, condition, depends_on

### Community 5219 - "volumes"
Cohesion: 0.67
Nodes (3): name, volumes, gateway-delivery-state

### Community 5220 - "baselines_breakdown"
Cohesion: 0.67
Nodes (3): reason, status, baselines_breakdown

### Community 5221 - "baselines_comparisons"
Cohesion: 0.67
Nodes (3): reason, status, baselines_comparisons

### Community 5222 - "baselines_trend"
Cohesion: 0.67
Nodes (3): reason, status, baselines_trend

### Community 5223 - "pr_state"
Cohesion: 0.67
Nodes (3): reason, status, pr_state

### Community 5224 - "savings_estimates"
Cohesion: 0.67
Nodes (3): reason, status, savings_estimates

### Community 5225 - "session_outcomes"
Cohesion: 0.67
Nodes (3): reason, status, session_outcomes

### Community 5228 - "ecr-repository"
Cohesion: 0.67
Nodes (3): description, required, ecr-repository

### Community 5448 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5449 - "event_bus"
Cohesion: 0.67
Nodes (3): event_bus, publish_topics, subscribe_topics

### Community 5450 - "supported_rules"
Cohesion: 0.67
Nodes (3): supported_rules, description, rule_id

### Community 5451 - "event_bus"
Cohesion: 0.67
Nodes (3): event_bus, publish_topics, subscribe_topics

### Community 5452 - "input_subscriptions"
Cohesion: 0.67
Nodes (3): input_subscriptions, operation, topic

### Community 5453 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5454 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5455 - "http_mount"
Cohesion: 0.67
Nodes (3): http_mount, factory, path_prefix

### Community 5457 - "health_check"
Cohesion: 0.67
Nodes (3): health_check, description, enabled

### Community 5458 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5459 - "writer"
Cohesion: 0.67
Nodes (3): writer, settle_attempts, settle_delay_ms

### Community 5461 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5462 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5464 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5465 - "health_check"
Cohesion: 0.67
Nodes (3): health_check, description, enabled

### Community 5466 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5467 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5468 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5469 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

### Community 5471 - "capabilities"
Cohesion: 0.67
Nodes (3): capabilities, description, name

## Knowledge Gaps
- **132 isolated node(s):** `PSQL_HISTORY`, `PGOPTIONS`, `RPK_BROKERS`, `PLAYWRIGHT_BROWSERS_PATH`, `_C2_REWRITE_ENV_LINES` (+127 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 66858 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1386 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EnumHandlerTypeCategory` connect `EnumHandlerTypeCategory` to `ModelInfraErrorContext`, `.with_correlation`, `HandlerServiceValidate`, `.handler_category`, `HandlerFaultInjectFixture`, `handler_scope_extract.py`, `ModelBackendResult`, `ProtocolEventBusLike`, `HandlerTopicMigrationExecutor`, `EventBusKafka`, `HandlerBrokerGrantDerive`, `HandlerGitFixVerifier`, `HandlerLLMCompletion`, `.compute_savings`, `HandlerPlacementEtcd`, `sanitize_error_message`, `HandlerHttpRest`, `HandlerWorkspaceRuntimeConfigMaterializer`, `ModelSessionStateResult`, `_parse_contract`, `HandlerInfisicalFullSetup`, `ProtocolArchitectureRule`, `ModelCheckpoint`, `EnumHandlerType`, `ModelUpdatePlan`, `HandlerRsdInitiate`, `HandlerCILiveContact`, `ModelWorkspaceConfigRootRequest`, `HandlerPlacementElection`, `HandlerBootFailureLedger`, `HandlerPlacementSession`, `ModelEventMessage`, `HandlerGatewayHeartbeat`, `ModelRegistryApiRequest`, `HandlerOnexApiImageImportCheck`, `HandlerBrokerDiskWatermark`, `HandlerDlqDepthMonitor`, `HandlerPostgresHeartbeat`, `handler_a2a_task.py`, `HandlerBuildLoopAppend`, `HandlerSetupOrchestrator`, `HandlerRsdScoreCalculate`, `HandlerContractSubscribePreflight`, `ModelWasteFinding`, `handler_reward_binder.py`, `ModelDevLaneCodeRefPin`, `ModelRuleCheckResult`, `HandlerManifestPersistence`, `ModelRsdDataFetchResult`, `handler_reconcile_workspace_venvs.py`, `HandlerRunnerFleetMaintainStart`, `HandlerRuntimeErrorTriage`, `ModelTopicCatalogResponse`, `ContractFileWatcher`, `HandlerMCP`, `handler_standalone_compose_recreate.py`, `ModelLocalProvisionEffectOutput`, `runtime_cost_estimator_build.py`, `handler_dev_lane_pin.py`, `HandlerPlacementEpochGuard`, `ModelHandlerContract`, `HandlerRuntimeTargetCollect`, `ModelRuntimeAttachReadiness`, `HandlerQueryDecisions`, `HandlerEventForward`, `HandlerRunnerHookMountDrift`, `CLAUDE.md - Omnibase Infrastructure`, `HandlerLlmInferenceCommand`, `HandlerDelegationChainLedger`, `HandlerMergeProvenance`, `HandlerRunnerFleetSnapshot`, `ModelSecretSeedRequest`, `HandlerScopeCheckInitiate`, `handler_api_fixture_record.py`, `HandlerLaneEntryPointCensus`, `HandlerEventProvenanceStamp`, `ProtocolPostgresAdapter`, `ProtocolBrokerPrincipalAdmin`, `ProtocolChainVectorClient`, `HandlerGmailApi`, `HandlerVectorSearch`, `ModelBrokerAclBinding`, `HandlerActionArchiveCacheFill`, `ModelAdjudicatorState`, `ModelSecurityValidationResult`, `HandlerLabProofVerdict`, `HandlerRequestAdjudication`, `HandlerChainCanary`, `HandlerMetricsPrometheus`, `ModelRunnerFleetSnapshot`, `ModelRunnerFleetHealthVerdict`, `in_progress_demotion.py`, `HandlerSourceAttestation`, `HandlerDockerForwarderStateReader`, `HandlerReleaseIdentity`, `HandlerValidationLedgerProjection`, `HandlerReleaseNotesApply`, `HandlerServiceDiscoveryMock`, `HandlerMergeGroupRunReadGithub`, `HandlerProbeHealth`, `HandlerAuthGate`, `HandlerBaselinesBatchCompute`, `HandlerSavingsEstimation`, `HandlerDeclaredStateApply`, `HandlerDeclaredStateReconcile`, `HandlerDeployRunners`, `HandlerVenvAtomicInstall`, `CollectorRunnerHealth`, `.handle`, `HandlerOnexApiImageBuild`, `HandlerSyncRevertWatchdog`, `HandlerSavingsCorrelation`, `RuntimeDelegationDispatchPort`, `HandlerCodingAgentInvoke`, `handler_invariant_evaluate.py`, `handler_c16_pinned_backend.py`, `HandlerDeclaredStateLeaseLocal`, `HandlerCostApiDependencies`, `HandlerCodingAgentOrchestrator`, `handler_update_scores.py`, `HandlerLabProofRun`, `HandlerRepoStateCollect`, `HandlerPluginLoader`, `HandlerGatewayLinkHealthUpsert`, `ModelRoutingCommand`, `HandlerRunnerFleetHealthEvaluate`, `HandlerPlacementRuntime`, `HandlerSlackWebhook`, `HandlerBaselineCapture`, `HandlerInProgressProbeHygiene`, `HandlerBaselineComparison`, `HandlerCatalogRequest`, `ConsumerTopicVisitor`, `HandlerBrokerPrincipalVerify`, `HandlerDeclaredStateDiff`, `handler_developer_principal_plan/handler.py`, `HandlerRegistrationStorageMock`, `HandlerWriteDecision`, `HandlerMergeProvenanceObserve`, `handler_hotpatch_ledger_reconcile.py`, `HandlerGmailIntentPoll`, `ModelGatewayEnvelope`, `ModelWorkspaceValidateCommand`, `HandlerToolchainCollect`, `HandlerPostgresTopicUpdate`, `HandlerKafkaReplay`, `HandlerWorkLedgerQuery`, `HandlerConsumerHealthTriage`, `expand_contract_env_refs`, `_run_errors`, `HandlerLabProofPlan`, `handler_score_models.py`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `PSQL_HISTORY`, `PGOPTIONS`, `RPK_BROKERS` to the rest of the system?**
  _132 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `typing` be split into smaller, more focused modules?**
  _Cohesion score 0.0034219481178782716 - nodes in this community are weakly interconnected._
- **Why does `ModelInfraErrorContext` connect `ModelInfraErrorContext` to `RegistryCompute`, `PayloadRegistryError`, `.with_correlation`, `DispatchResultApplier`, `ModelInjectedRequest`, `ModelLogContext`, `HandlerManifestPersistence`, `ModelDiscoveryResult`, `ModelRegistrationProjection`, `model_infra_error_context.py`, `gateway_control_plane.py`, `.create_app`, `RuntimeHostProcess`, `ServiceContractPublisher`, `ModelChainViolation`, `HandlerDb`, `RepositoryExecutionError`, `ProtocolEventBusLike`, `credential_loader.py`, `worker_cycle.py`, `HandlerMCP`, `ModelCreateTopicRefusal`, `RegistrationReducerService`, `apply_topic_namespace`, `sanitize_error_message`, `MixinAsyncCircuitBreaker`, `ModelRuntimeAttachReadiness`, `SecretResolver`, `HandlerHttpRest`, `ModelCoverageMetrics`, `ServiceRegistration`, `TopicResolver`, `TopicProvisioner`, `MixinNodeIntrospection`, `InvocationSecurityEnforcer`, `RemoteTaskStateRepository`, `InfraUnavailableError`, `quote_identifier`, `AdapterTicketLinear`, `resolve_embedded_runtime_config`, `TransitionNotificationOutbox`, `HandlerGraph`, `compute_schema_fingerprint`, `bootstrap`, `MixinLlmHttpTransport`, `HandlerPluginLoader`, `RuntimeLocalIngressServer`, `ServiceRetryWorker`, `handler_a2a_task.py`, `ModelSnapshotTopicConfig`, `sanitize_error_string`, `ModelValidationOutcome`, `.graphql_transport`, `RegistryError`, `kafka_contract_source.py`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **Should `pydantic` be split into smaller, more focused modules?**
  _Cohesion score 0.002396266357958478 - nodes in this community are weakly interconnected._
- **Why does `Changelog` connect `Changelog` to `MixinNodeIntrospection`, `Added`, `[0.11.0] - 2026-02-25`, `registry_infra_ledger_write.py`, `CHANGELOG.md`, `Added`, `ConfigSessionStorage`, `update_env_file`, `RepositoryExecutionError`, `EventBusSubcontractWiring`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **Should `ModelInfraErrorContext` be split into smaller, more focused modules?**
  _Cohesion score 0.007034936208629295 - nodes in this community are weakly interconnected._