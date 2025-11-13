# Enterprise Integration Patterns

**Extracted from:** PR #16 (feature/onextree-phase4-enterprise-features)
**Date:** 2025-10-15
**Status:** Reference patterns for future production implementations

## Overview

This document preserves valuable enterprise architecture patterns from PR #16, which added comprehensive enterprise-grade features to the MetadataStampingService. While these features were not merged due to architectural misalignment, the patterns and approaches are valuable for future production SaaS services.

## Core Enterprise Patterns

### 1. Multi-Tenancy Architecture Pattern

**Pattern:** Complete tenant isolation with configurable isolation levels

**Isolation Levels:**

```
┌────────────────────────────────────────────────────────┐
│         ISOLATION LEVEL 1: SHARED                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Single Database                                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │  │
│  │  │ Tenant A    │ │ Tenant B    │ │ Tenant C   │ │  │
│  │  │ (tenant_id) │ │ (tenant_id) │ │(tenant_id) │ │  │
│  │  └─────────────┘ └─────────────┘ └────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: Lowest cost, highest density                │
│  Drawbacks: Shared resources, noisy neighbor           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│         ISOLATION LEVEL 2: DATABASE                    │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │  Database A  │ │  Database B  │ │  Database C   │  │
│  │  (Tenant A)  │ │  (Tenant B)  │ │  (Tenant C)   │  │
│  └──────────────┘ └──────────────┘ └───────────────┘  │
│  Benefits: Better isolation, independent scaling       │
│  Drawbacks: Higher cost, more management               │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│         ISOLATION LEVEL 3: INSTANCE                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Dedicated Instance (Tenant A)                   │  │
│  │  Database + Application + Resources              │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Dedicated Instance (Tenant B)                   │  │
│  │  Database + Application + Resources              │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: Complete isolation, regulatory compliance   │
│  Drawbacks: Highest cost, most operational overhead    │
└────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
class MultiTenancyManager:
    def __init__(self, default_isolation: IsolationLevel):
        self.default_isolation = default_isolation
        self.tenant_configs = {}

    async def provision_tenant(
        self,
        name: str,
        tier: TenantTier,
        isolation_level: IsolationLevel
    ) -> Tenant:
        """Provision new tenant with isolation level."""

        if isolation_level == IsolationLevel.SHARED:
            # Use shared database with tenant_id filtering
            db_connection = await self.get_shared_db_connection()

        elif isolation_level == IsolationLevel.DATABASE:
            # Provision dedicated database
            db_connection = await self.provision_tenant_database(name)

        elif isolation_level == IsolationLevel.INSTANCE:
            # Provision dedicated infrastructure
            instance = await self.provision_dedicated_instance(name)
            db_connection = instance.db_connection

        # Create tenant record
        tenant = Tenant(
            name=name,
            tier=tier,
            isolation_level=isolation_level,
            db_connection=db_connection,
            config=self.get_tier_config(tier)
        )

        await self.register_tenant(tenant)
        return tenant

    async def get_tenant_context(self, tenant_id: str) -> TenantContext:
        """Get tenant context for request processing."""
        tenant = await self.get_tenant(tenant_id)

        return TenantContext(
            tenant_id=tenant.id,
            tenant_name=tenant.name,
            tier=tenant.tier,
            db_connection=tenant.db_connection,
            config=tenant.config,
            quotas=tenant.quotas
        )
```

**Tenant Tier System:**

| Tier | Isolation | Quotas | Features |
|------|-----------|--------|----------|
| **Basic** | Shared | 10K stamps/day, 10GB storage | Core features |
| **Professional** | Shared | 100K stamps/day, 100GB storage | + Advanced analytics |
| **Enterprise** | Database | 1M stamps/day, 1TB storage | + Custom integrations, SLA |
| **Premium** | Instance | Unlimited | + White-label, dedicated support |

**Benefits:**
- Flexible isolation for different customer needs
- Cost optimization (pay for isolation level needed)
- Regulatory compliance (dedicated instances for sensitive data)
- Independent scaling per tenant

**Considerations:**
- Database sharding strategy for shared databases
- Cross-tenant data leakage prevention
- Tenant-specific configuration management
- Resource quota enforcement

### 2. Role-Based Access Control (RBAC) Pattern

**Pattern:** Fine-grained permission system with hierarchical roles

**RBAC Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                  RBAC System                            │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Roles (with inheritance)                        │  │
│  │  - Admin (inherits all)                          │  │
│  │    └─ Developer (inherits operator)              │  │
│  │        └─ Operator (inherits readonly)           │  │
│  │            └─ ReadOnly (base permissions)        │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Permissions (Resource + Action)                 │  │
│  │  - metadata_stamp:create                         │  │
│  │  - metadata_stamp:read                           │  │
│  │  - metadata_stamp:update                         │  │
│  │  - metadata_stamp:delete                         │  │
│  │  - system:admin                                  │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Policy Evaluation                               │  │
│  │  check_permission(user, action, resource)        │  │
│  │  → Allow / Deny                                  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class RBACManager:
    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self.roles = {}
        self.permissions_cache = LRUCache(maxsize=10000)

    async def check_permission(
        self,
        user_id: str,
        action: ActionType,
        resource: ResourceType,
        tenant_id: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if user has permission for action on resource."""

        # Check cache first
        cache_key = f"{user_id}:{tenant_id}:{action}:{resource}:{resource_id}"
        cached_result = self.permissions_cache.get(cache_key)

        if cached_result is not None:
            return cached_result

        # Get user roles for tenant
        roles = await self.get_user_roles(user_id, tenant_id)

        # Check each role for permission
        has_permission = False
        for role in roles:
            role_obj = await self.get_role(role.role_name)

            # Check direct permissions
            if self._has_permission(role_obj, action, resource):
                has_permission = True
                break

            # Check resource-level permissions
            if resource_id and await self._has_resource_permission(
                user_id, resource_id, action
            ):
                has_permission = True
                break

        # Cache result
        self.permissions_cache.set(cache_key, has_permission, ttl=self.cache_ttl)

        # Log access attempt for audit
        await self.audit_logger.log_access_check(
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            allowed=has_permission
        )

        return has_permission

    def _has_permission(
        self,
        role: Role,
        action: ActionType,
        resource: ResourceType
    ) -> bool:
        """Check if role has permission."""
        for perm in role.permissions:
            if perm.resource_type == resource and action in perm.actions:
                return True

        # Check inherited roles
        for parent_role_name in role.inherits_from:
            parent_role = self.roles.get(parent_role_name)
            if parent_role and self._has_permission(parent_role, action, resource):
                return True

        return False
```

**Pre-defined Roles:**

```python
PREDEFINED_ROLES = {
    "admin": Role(
        name="admin",
        permissions=[
            Permission(resource=ResourceType.ALL, actions=[ActionType.ALL])
        ]
    ),
    "developer": Role(
        name="developer",
        inherits_from=["operator"],
        permissions=[
            Permission(resource=ResourceType.METADATA_STAMP, actions=[
                ActionType.CREATE, ActionType.READ, ActionType.UPDATE, ActionType.DELETE
            ]),
            Permission(resource=ResourceType.API_KEY, actions=[
                ActionType.CREATE, ActionType.READ, ActionType.DELETE
            ])
        ]
    ),
    "operator": Role(
        name="operator",
        inherits_from=["readonly"],
        permissions=[
            Permission(resource=ResourceType.METADATA_STAMP, actions=[
                ActionType.CREATE, ActionType.READ
            ]),
            Permission(resource=ResourceType.SYSTEM, actions=[
                ActionType.READ  # Can view system metrics
            ])
        ]
    ),
    "readonly": Role(
        name="readonly",
        permissions=[
            Permission(resource=ResourceType.METADATA_STAMP, actions=[ActionType.READ]),
            Permission(resource=ResourceType.SYSTEM, actions=[ActionType.READ])
        ]
    )
}
```

**Benefits:**
- Fine-grained access control
- Role inheritance reduces duplication
- Permission caching for performance (<5ms checks)
- Audit logging for compliance
- Time-based access with expiration

**Considerations:**
- Cache invalidation when roles change
- Permission granularity (too many permissions increase complexity)
- Role explosion (avoid creating too many roles)
- Tenant-specific role customization

### 3. Audit Logging Pattern

**Pattern:** Tamper-proof audit trail with compliance reporting

**Audit Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│              Audit Logging System                       │
│                                                         │
│  User Action → Audit Event Creation                     │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Audit Event                                     │  │
│  │  - event_id (UUID)                               │  │
│  │  - timestamp                                     │  │
│  │  - user_id, tenant_id                            │  │
│  │  - action, resource, resource_id                 │  │
│  │  - result (success/failure)                      │  │
│  │  - context (IP, user_agent, etc.)                │  │
│  │  - cryptographic_signature (tamper-proof)        │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Audit Storage                                   │  │
│  │  - PostgreSQL (primary)                          │  │
│  │  - S3/GCS (long-term archival)                   │  │
│  │  - Elasticsearch (search and analysis)           │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Compliance Reporting                            │  │
│  │  - SOC2 reports                                  │  │
│  │  - HIPAA audit logs                              │  │
│  │  - GDPR data access logs                         │  │
│  │  - PCI-DSS compliance reports                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class AuditLogger:
    def __init__(self, signing_key: bytes):
        self.signing_key = signing_key
        self.retention_policy = {
            AuditEventType.SECURITY: timedelta(days=2555),  # 7 years
            AuditEventType.DATA_ACCESS: timedelta(days=1095),  # 3 years
            AuditEventType.DEFAULT: timedelta(days=365)  # 1 year
        }

    async def log_event(
        self,
        user_id: str,
        tenant_id: str,
        action: ActionType,
        resource: ResourceType,
        resource_id: Optional[str],
        result: AuditResult,
        context: Dict[str, Any]
    ) -> AuditEvent:
        """Log audit event with cryptographic signature."""

        event = AuditEvent(
            event_id=uuid4(),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            result=result,
            context=context
        )

        # Generate tamper-proof signature
        event.signature = self._sign_event(event)

        # Store in multiple backends for durability
        await asyncio.gather(
            self._store_in_postgres(event),
            self._store_in_elasticsearch(event),
            self._archive_to_s3(event)  # Async archival
        )

        # Real-time alerting for critical events
        if self._is_critical_event(event):
            await self.alert_manager.send_security_alert(event)

        return event

    def _sign_event(self, event: AuditEvent) -> str:
        """Generate HMAC signature for tamper detection."""
        message = f"{event.event_id}:{event.timestamp}:{event.user_id}:{event.action}:{event.resource}"
        signature = hmac.new(
            self.signing_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def verify_event_integrity(self, event: AuditEvent) -> bool:
        """Verify event has not been tampered with."""
        expected_signature = self._sign_event(event)
        return hmac.compare_digest(expected_signature, event.signature)

    async def generate_compliance_report(
        self,
        tenant_id: str,
        compliance_standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        """Generate compliance report (SOC2, HIPAA, GDPR, PCI-DSS)."""

        # Query relevant audit events
        events = await self._query_audit_events(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        # Generate report based on compliance standard
        if compliance_standard == ComplianceStandard.SOC2:
            return self._generate_soc2_report(events, start_date, end_date)
        elif compliance_standard == ComplianceStandard.HIPAA:
            return self._generate_hipaa_report(events, start_date, end_date)
        elif compliance_standard == ComplianceStandard.GDPR:
            return self._generate_gdpr_report(events, start_date, end_date)
        elif compliance_standard == ComplianceStandard.PCI_DSS:
            return self._generate_pci_dss_report(events, start_date, end_date)
```

**Audit Event Types:**

```python
class AuditEventType(Enum):
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    MFA_ENABLED = "mfa_enabled"

    # Data Access
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"

    # Security
    PERMISSION_DENIED = "permission_denied"
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # Configuration
    CONFIG_CHANGED = "config_changed"
    ROLE_ASSIGNED = "role_assigned"
    PERMISSION_GRANTED = "permission_granted"

    # Compliance
    DATA_EXPORT = "data_export"
    DATA_ANONYMIZATION = "data_anonymization"
    RETENTION_POLICY_APPLIED = "retention_policy_applied"
```

**Benefits:**
- Tamper-proof audit trail (cryptographic signatures)
- Compliance reporting (SOC2, HIPAA, GDPR, PCI-DSS)
- Real-time security alerting
- Event correlation for investigation
- Retention policies with automated archival

**Considerations:**
- Storage costs for audit logs
- Performance impact of async logging
- Elasticsearch for fast search and analysis
- Long-term archival strategy (S3 Glacier)
- GDPR right to erasure (audit log exceptions)

### 4. Billing and Metering Pattern

**Pattern:** Usage-based billing with real-time metering

**Billing Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│              Billing & Metering System                  │
│                                                         │
│  Usage Event → Metering Engine                          │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Usage Tracking                                  │  │
│  │  - Hash operations                               │  │
│  │  - Stamps created                                │  │
│  │  - API calls                                     │  │
│  │  - Storage (GB)                                  │  │
│  │  - Bandwidth (GB)                                │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Usage Aggregation                               │  │
│  │  - Per tenant                                    │  │
│  │  - Per billing period                            │  │
│  │  - Per usage type                                │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Cost Calculation                                │  │
│  │  - Pricing model (per-operation, tiered, volume) │  │
│  │  - Discounts and credits                         │  │
│  │  - Tax calculation                               │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Invoice Generation                              │  │
│  │  - PDF generation                                │  │
│  │  - Email delivery                                │  │
│  │  - Payment integration                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class BillingManager:
    def __init__(self):
        self.pricing_models = {}
        self.usage_aggregator = UsageAggregator()

    async def track_usage(
        self,
        tenant_id: str,
        usage_type: UsageType,
        quantity: int,
        metadata: Dict[str, Any]
    ):
        """Track usage event for billing."""

        usage_event = UsageEvent(
            event_id=uuid4(),
            tenant_id=tenant_id,
            usage_type=usage_type,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )

        # Store usage event
        await self.usage_aggregator.record_usage(usage_event)

        # Check quota limits
        current_usage = await self.get_current_usage(tenant_id, usage_type)
        quota = await self.get_tenant_quota(tenant_id, usage_type)

        if current_usage + quantity > quota:
            raise QuotaExceededError(
                f"Quota exceeded for {usage_type}: {current_usage + quantity}/{quota}"
            )

        # Real-time cost tracking
        await self.update_running_cost(tenant_id, usage_event)

    async def calculate_bill(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Bill:
        """Calculate bill for billing period."""

        # Get usage for period
        usage = await self.usage_aggregator.get_usage(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        # Get pricing model for tenant
        pricing_model = await self.get_pricing_model(tenant_id)

        # Calculate cost per usage type
        line_items = []
        for usage_type, quantity in usage.items():
            cost = pricing_model.calculate_cost(usage_type, quantity)

            line_items.append(LineItem(
                description=f"{usage_type.value} usage",
                quantity=quantity,
                unit_price=pricing_model.get_unit_price(usage_type),
                total=cost
            ))

        # Apply discounts and credits
        subtotal = sum(item.total for item in line_items)
        discounts = await self.calculate_discounts(tenant_id, subtotal)
        credits = await self.get_available_credits(tenant_id)

        total = subtotal - discounts - credits

        # Apply tax
        tax = await self.calculate_tax(tenant_id, total)

        return Bill(
            tenant_id=tenant_id,
            billing_period_start=start_date,
            billing_period_end=end_date,
            line_items=line_items,
            subtotal=subtotal,
            discounts=discounts,
            credits=credits,
            tax=tax,
            total=total + tax
        )

    async def forecast_cost(
        self,
        tenant_id: str,
        forecast_days: int = 30
    ) -> CostForecast:
        """Forecast future costs based on historical usage."""

        # Get historical usage
        historical_usage = await self.get_historical_usage(
            tenant_id=tenant_id,
            days=90
        )

        # Predict future usage (simple moving average or ML model)
        predicted_usage = self._predict_usage(historical_usage, forecast_days)

        # Calculate predicted cost
        pricing_model = await self.get_pricing_model(tenant_id)
        predicted_cost = {}

        for usage_type, quantity in predicted_usage.items():
            cost = pricing_model.calculate_cost(usage_type, quantity)
            predicted_cost[usage_type] = cost

        total_predicted_cost = sum(predicted_cost.values())

        return CostForecast(
            tenant_id=tenant_id,
            forecast_period_days=forecast_days,
            predicted_usage=predicted_usage,
            predicted_cost=predicted_cost,
            total_predicted_cost=total_predicted_cost
        )
```

**Pricing Models:**

```python
# Per-operation pricing
per_operation_pricing = PricingModel(
    name="per_operation",
    rates={
        UsageType.HASH_OPERATION: 0.001,  # $0.001 per hash
        UsageType.STAMP_CREATED: 0.01,    # $0.01 per stamp
        UsageType.API_CALL: 0.0001,       # $0.0001 per API call
        UsageType.STORAGE_GB: 0.10,       # $0.10 per GB/month
    }
)

# Tiered pricing (volume discounts)
tiered_pricing = TieredPricingModel(
    name="tiered",
    tiers=[
        Tier(range=(0, 10000), rates={UsageType.HASH_OPERATION: 0.001}),
        Tier(range=(10000, 100000), rates={UsageType.HASH_OPERATION: 0.0008}),
        Tier(range=(100000, 1000000), rates={UsageType.HASH_OPERATION: 0.0005}),
        Tier(range=(1000000, float('inf')), rates={UsageType.HASH_OPERATION: 0.0003})
    ]
)

# Volume-based pricing (all-or-nothing tiers)
volume_pricing = VolumePricingModel(
    name="volume",
    tiers=[
        VolumeTier(max_volume=10000, price=10),       # $10 for up to 10K operations
        VolumeTier(max_volume=100000, price=80),      # $80 for up to 100K operations
        VolumeTier(max_volume=1000000, price=600),    # $600 for up to 1M operations
        VolumeTier(max_volume=float('inf'), price=5000)  # $5000 for unlimited
    ]
)
```

**Benefits:**
- Accurate usage tracking
- Flexible pricing models (per-operation, tiered, volume)
- Cost forecasting for customers
- Budget alerts
- Automated invoice generation
- Credit system for promotions

**Considerations:**
- Real-time vs batch metering (tradeoff: latency vs accuracy)
- Data retention for billing disputes
- Handling refunds and credits
- Payment integration (Stripe, PayPal)
- Tax calculation (varies by jurisdiction)
- Currency support for international customers

### 5. API Gateway Pattern

**Pattern:** Unified API gateway with routing, auth, and rate limiting

**API Gateway Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                    API Gateway                          │
│                                                         │
│  Client Request                                         │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  1. Authentication                               │  │
│  │     - API Key validation                         │  │
│  │     - JWT token verification                     │  │
│  │     - OAuth2/OpenID Connect                      │  │
│  │     - mTLS certificate validation                │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  2. Rate Limiting                                │  │
│  │     - Per-tenant rate limits                     │  │
│  │     - Per-user rate limits                       │  │
│  │     - Per-endpoint rate limits                   │  │
│  │     - Distributed rate limiting (Redis)          │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  3. Authorization (RBAC)                         │  │
│  │     - Check permissions                          │  │
│  │     - Resource-level access control              │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  4. Request Transformation                       │  │
│  │     - Protocol translation (HTTP/gRPC/WebSocket) │  │
│  │     - Request/response mapping                   │  │
│  │     - API versioning                             │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  5. Routing & Load Balancing                     │  │
│  │     - Path-based routing                         │  │
│  │     - Header-based routing                       │  │
│  │     - Round-robin / least-connections            │  │
│  │     - Circuit breaker                            │  │
│  └──────────────────────────────────────────────────┘  │
│      ↓                                                  │
│  Backend Services                                       │
│  - MetadataStampingService                              │
│  - OnexTreeService                                      │
│  - BillingService                                       │
│  - AnalyticsService                                     │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class APIGateway:
    def __init__(
        self,
        auth_manager: AuthManager,
        rate_limiter: RateLimiter,
        rbac_manager: RBACManager,
        router: RequestRouter
    ):
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.rbac_manager = rbac_manager
        self.router = router

    async def handle_request(self, request: Request) -> Response:
        """Handle incoming API request through gateway pipeline."""

        try:
            # 1. Authentication
            auth_context = await self.auth_manager.authenticate(request)

            # 2. Rate Limiting
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                tenant_id=auth_context.tenant_id,
                user_id=auth_context.user_id,
                endpoint=request.path
            )

            if not rate_limit_result.allowed:
                return Response(
                    status_code=429,
                    headers={
                        "X-RateLimit-Limit": str(rate_limit_result.limit),
                        "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                        "X-RateLimit-Reset": str(rate_limit_result.reset_at),
                        "Retry-After": str(rate_limit_result.retry_after_seconds)
                    },
                    body={"error": "Rate limit exceeded"}
                )

            # 3. Authorization (RBAC)
            has_permission = await self.rbac_manager.check_permission(
                user_id=auth_context.user_id,
                tenant_id=auth_context.tenant_id,
                action=self._extract_action(request),
                resource=self._extract_resource(request)
            )

            if not has_permission:
                return Response(status_code=403, body={"error": "Permission denied"})

            # 4. Request Transformation
            transformed_request = await self.transform_request(request, auth_context)

            # 5. Route to Backend Service
            backend_response = await self.router.route_request(
                transformed_request,
                auth_context
            )

            # 6. Response Transformation
            final_response = await self.transform_response(backend_response)

            # 7. Analytics & Logging
            await self.log_request(request, final_response, auth_context)

            return final_response

        except AuthenticationError as e:
            return Response(status_code=401, body={"error": str(e)})
        except AuthorizationError as e:
            return Response(status_code=403, body={"error": str(e)})
        except Exception as e:
            logger.exception(f"Gateway error: {e}")
            return Response(status_code=500, body={"error": "Internal server error"})
```

**Request Router:**

```python
class RequestRouter:
    def __init__(self):
        self.routes = []
        self.service_registry = {}
        self.circuit_breakers = {}

    def register_route(
        self,
        pattern: str,
        service_name: str,
        methods: List[str] = ["GET", "POST", "PUT", "DELETE"],
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ):
        """Register route to backend service."""
        self.routes.append(Route(
            pattern=pattern,
            service_name=service_name,
            methods=methods,
            load_balancing_strategy=load_balancing_strategy
        ))

    async def route_request(
        self,
        request: Request,
        auth_context: AuthContext
    ) -> Response:
        """Route request to appropriate backend service."""

        # Find matching route
        route = self._find_route(request.path, request.method)

        if not route:
            return Response(status_code=404, body={"error": "Not found"})

        # Get backend instances
        instances = await self.service_registry.get_instances(route.service_name)

        if not instances:
            return Response(status_code=503, body={"error": "Service unavailable"})

        # Select instance based on load balancing strategy
        instance = self._select_instance(instances, route.load_balancing_strategy)

        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(route.service_name)

        if circuit_breaker and circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for {route.service_name}")
            return Response(status_code=503, body={"error": "Service temporarily unavailable"})

        # Forward request to backend
        try:
            response = await self._forward_request(instance, request)
            circuit_breaker.record_success() if circuit_breaker else None
            return response

        except Exception as e:
            circuit_breaker.record_failure() if circuit_breaker else None
            raise
```

**Benefits:**
- Centralized authentication and authorization
- Rate limiting to prevent abuse
- Request routing and load balancing
- Protocol translation (HTTP ↔ gRPC ↔ WebSocket)
- Circuit breaker for resilience
- API versioning support
- Request/response transformation
- Analytics and monitoring

**Considerations:**
- Single point of failure (mitigate with HA deployment)
- Latency overhead (~10ms additional latency)
- Complex configuration management
- Service discovery integration (Consul, Kubernetes)
- WebSocket support for real-time features

### 6. Advanced Rate Limiting Pattern

**Pattern:** Multi-algorithm distributed rate limiting

**Rate Limiting Algorithms:**

```
┌─────────────────────────────────────────────────────────┐
│         1. Token Bucket Algorithm                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Bucket (capacity=1000 tokens)                   │  │
│  │  Refill rate: 100 tokens/second                  │  │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┐      │  │
│  │  │ ●● │ ●● │ ●● │ ●● │ ●● │ ●● │ ●● │ ●● │      │  │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┘      │  │
│  │  Request → Remove token → Allow (if available)   │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: Handles bursts, smooth traffic            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         2. Sliding Window Algorithm                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Window (1 minute)                               │  │
│  │  │←────────────── 60s ──────────────→│          │  │
│  │  ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┤          │  │
│  │  │10│ 8│12│ 5│ 9│11│ 7│ 6│10│ 8│ 9│11│ = 106    │  │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘          │  │
│  │  Request → Count in window → Allow (if < limit)  │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: More accurate than fixed window           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         3. Leaky Bucket Algorithm                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Queue (max size=1000)                           │  │
│  │  Leak rate: 100 requests/second                  │  │
│  │  ┌────────────────────────────────────┐          │  │
│  │  │  Request → Add to queue            │          │  │
│  │  │  │                                 │          │  │
│  │  │  │←── Process at fixed rate ──────●          │  │
│  │  │  │                                            │  │
│  │  └────────────────────────────────────┘          │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: Smooth constant output rate               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         4. Fixed Window Algorithm                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Window 1 (60s): 95 requests                     │  │
│  │  Window 2 (60s): 103 requests                    │  │
│  │  Window 3 (60s): 88 requests                     │  │
│  │  ├────────┤├────────┤├────────┤                 │  │
│  │  0s     60s     120s    180s                     │  │
│  │  Request → Count in current window → Allow       │  │
│  └──────────────────────────────────────────────────┘  │
│  Benefits: Simple, fast, low memory                  │
│  Drawbacks: Burst at window boundaries               │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class TokenBucketRateLimiter:
    def __init__(
        self,
        capacity: int,
        refill_rate: int,
        scope: RateLimitScope,
        redis_client: RedisClient
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.scope = scope
        self.redis = redis_client

    async def allow_request(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> RateLimitResult:
        """Check if request is allowed under rate limit."""

        # Construct key based on scope
        key = self._construct_key(tenant_id, user_id, endpoint, ip_address)

        # Execute atomic Lua script in Redis
        tokens_remaining = await self.redis.eval(
            RATE_LIMIT_LUA_SCRIPT,
            keys=[key],
            args=[self.capacity, self.refill_rate, time.time()]
        )

        allowed = tokens_remaining >= 1

        if allowed:
            tokens_remaining -= 1

        # Calculate reset time
        refill_time = (self.capacity - tokens_remaining) / self.refill_rate
        reset_at = time.time() + refill_time

        return RateLimitResult(
            allowed=allowed,
            limit=self.capacity,
            remaining=int(tokens_remaining),
            reset_at=int(reset_at),
            retry_after_seconds=int(refill_time) if not allowed else 0
        )

    def _construct_key(
        self,
        tenant_id: Optional[str],
        user_id: Optional[str],
        endpoint: Optional[str],
        ip_address: Optional[str]
    ) -> str:
        """Construct Redis key based on rate limit scope."""

        if self.scope == RateLimitScope.GLOBAL:
            return "rate_limit:global"
        elif self.scope == RateLimitScope.TENANT:
            return f"rate_limit:tenant:{tenant_id}"
        elif self.scope == RateLimitScope.USER:
            return f"rate_limit:user:{user_id}"
        elif self.scope == RateLimitScope.ENDPOINT:
            return f"rate_limit:endpoint:{endpoint}"
        elif self.scope == RateLimitScope.IP_ADDRESS:
            return f"rate_limit:ip:{ip_address}"
        elif self.scope == RateLimitScope.API_KEY:
            return f"rate_limit:api_key:{user_id}"
```

**Rate Limit Scopes:**

| Scope | Description | Use Case |
|-------|-------------|----------|
| **Global** | System-wide rate limit | Prevent system overload |
| **Tenant** | Per-tenant rate limit | Ensure fair resource sharing |
| **User** | Per-user rate limit | Prevent individual abuse |
| **Endpoint** | Per-endpoint rate limit | Protect specific endpoints |
| **IP Address** | Per-IP rate limit | Prevent DDoS attacks |
| **API Key** | Per-API-key rate limit | Track third-party integrations |

**Distributed Rate Limiting with Redis:**

```lua
-- Lua script for atomic token bucket rate limiting in Redis
-- KEYS[1]: rate limit key
-- ARGV[1]: capacity
-- ARGV[2]: refill_rate
-- ARGV[3]: current_time

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

-- Get current token count and last refill time
local tokens = redis.call('hget', key, 'tokens')
local last_refill = redis.call('hget', key, 'last_refill')

if tokens == false then
    tokens = capacity
    last_refill = current_time
else
    tokens = tonumber(tokens)
    last_refill = tonumber(last_refill)
end

-- Calculate tokens to add based on time elapsed
local elapsed = current_time - last_refill
local tokens_to_add = elapsed * refill_rate

-- Refill tokens (up to capacity)
tokens = math.min(capacity, tokens + tokens_to_add)

-- Update state
redis.call('hset', key, 'tokens', tokens)
redis.call('hset', key, 'last_refill', current_time)
redis.call('expire', key, 3600)  -- Expire key after 1 hour of inactivity

return tokens
```

**Benefits:**
- Multiple rate limiting algorithms for different use cases
- Distributed rate limiting across multiple gateway instances
- Atomic operations with Redis Lua scripts
- Multi-scope rate limiting (global, tenant, user, endpoint, IP)
- Dynamic limit adjustment based on system load
- Rate limit analytics and monitoring

**Considerations:**
- Redis dependency for distributed rate limiting
- Network latency to Redis (~1-2ms)
- Redis memory usage (optimize with key expiration)
- Algorithm selection based on traffic patterns
- Burst handling strategies

## Additional Enterprise Patterns

### 7. API Analytics Pattern

**Metrics Tracked:**
- Request count, latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Traffic patterns (time-of-day, day-of-week)
- User behavior (most used endpoints, user cohorts)
- Performance profiling per endpoint
- Geolocation analysis

### 8. Lifecycle Management Pattern

**Lifecycle Stages:**
1. **Provisioning**: Automated tenant and resource provisioning
2. **Monitoring**: Health checks, metrics collection, alerting
3. **Backup/Recovery**: Automated backups, point-in-time recovery
4. **Upgrade**: Rolling upgrades, A/B testing, rollback
5. **Decommissioning**: Data retention, archival, deletion

### 9. API Key Management Pattern

**Features:**
- API key generation with cryptographic randomness
- Key rotation with grace periods
- Key scoping (limit permissions per key)
- Key expiration and renewal
- Audit logging for key usage
- Rate limiting per API key

### 10. Multi-Factor Authentication (MFA) Pattern

**MFA Methods:**
- TOTP (Time-based One-Time Password) - Google Authenticator, Authy
- SMS-based OTP
- Email-based OTP
- Hardware tokens (YubiKey, U2F)
- Biometric authentication (fingerprint, face recognition)

## Key Takeaways

### For Production SaaS Implementations

1. **Multi-Tenancy**: Choose isolation level based on customer needs and regulatory requirements
2. **RBAC**: Implement fine-grained permissions with role inheritance and caching
3. **Audit Logging**: Tamper-proof audit trail with cryptographic signatures for compliance
4. **Billing**: Accurate usage metering with flexible pricing models and cost forecasting
5. **API Gateway**: Centralized authentication, authorization, rate limiting, and routing
6. **Rate Limiting**: Distributed rate limiting with multiple scopes and algorithms

### Critical Success Factors

- **Security First**: Zero-trust architecture, encryption, mTLS, security monitoring
- **Compliance**: SOC2, HIPAA, GDPR, PCI-DSS compliance from day one
- **Scalability**: Design for horizontal scaling and multi-region deployment
- **Observability**: Comprehensive logging, metrics, tracing, and alerting
- **Cost Optimization**: Usage-based pricing, resource quotas, cost forecasting

### Common Pitfalls to Avoid

- **Cross-Tenant Data Leakage**: Enforce strict tenant isolation at all layers
- **RBAC Complexity**: Avoid creating too many roles (role explosion)
- **Audit Log Performance**: Use async logging and batching
- **Rate Limiting False Positives**: Implement graceful degradation and user feedback
- **Billing Disputes**: Ensure accurate metering and transparent invoicing

## References

### Key Technologies (from PR #16)

- **JWT**: JSON Web Tokens for authentication
- **Redis**: Distributed rate limiting and caching
- **PostgreSQL**: RBAC, audit logging, billing data
- **Elasticsearch**: Audit log search and analysis
- **Stripe/PayPal**: Payment integration

### Further Reading

- **Multi-Tenancy**: "Multi-Tenancy Architecture" by Microsoft
- **RBAC**: "Role-Based Access Control" by NIST
- **API Gateway**: "Building Microservices" by Sam Newman
- **SaaS Metrics**: "SaaS Metrics 2.0" by David Skok

---

**Note**: These patterns were extracted from PR #16 and represent valuable enterprise architecture approaches. They should be adapted and customized for specific production requirements and compliance needs.
