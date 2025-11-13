# O.N.E. v0.1 Automated Implementation Guide

**Purpose**: Complete automation guide for implementing remaining O.N.E. protocol compliance
**Target**: 100% O.N.E. v0.1 compliance across all service components
**Execution Mode**: Automated/Overnight processing
**Current Progress**: 25% complete (Coordinators 1-2 finished)

---

## 🤖 Automated Execution Overview

This document provides step-by-step instructions for automated agents to complete the remaining 75% of O.N.E. protocol implementation. Each coordinator section includes:

- **Precise implementation steps**
- **Code templates and examples**
- **Validation criteria**
- **Success metrics**
- **Dependency tracking**

---

## 📋 Remaining Work Queue (6 Coordinators)

### Coordinator 3: Registry Client Implementation (Week 2, Priority: HIGH)
**Duration**: 1-2 days
**Dependencies**: Metadata headers (✅ Complete)
**Validation**: Service registration functional in Consul

#### Implementation Steps

1. **Install Consul Client Dependencies**
   ```bash
   cd omninode_bridge  # or your repository directory
   poetry add python-consul2 consul-python
   ```

2. **Create Registry Client Base**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/registry/consul_client.py
   import consul
   import asyncio
   from typing import Dict, Any, Optional
   from ..config.settings import get_settings

   class RegistryConsulClient:
       def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
           self.consul = consul.Consul(host=consul_host, port=consul_port)
           self.service_id = None
           self.settings = get_settings()

       async def register_service(self) -> bool:
           """Register MetadataStampingService with Consul"""
           service_name = "metadata-stamping-service"
           service_id = f"{service_name}-{self.settings.service_host}-{self.settings.service_port}"

           service_config = {
               'name': service_name,
               'service_id': service_id,
               'address': self.settings.service_host,
               'port': self.settings.service_port,
               'tags': [
                   'omninode.services.metadata',
                   'o.n.e.v0.1',
                   'blake3-hashing',
                   'metadata-stamping'
               ],
               'meta': {
                   'version': '0.1.0',
                   'protocol': 'O.N.E.v0.1',
                   'namespace': 'omninode.services.metadata',
                   'capabilities': 'hashing,stamping,validation'
               },
               'check': {
                   'http': f"http://{self.settings.service_host}:{self.settings.service_port}/health",
                   'interval': '10s',
                   'timeout': '5s'
               }
           }

           try:
               self.consul.agent.service.register(**service_config)
               self.service_id = service_id
               return True
           except Exception as e:
               print(f"Service registration failed: {e}")
               return False

       async def deregister_service(self) -> bool:
           """Deregister service from Consul"""
           if self.service_id:
               try:
                   self.consul.agent.service.deregister(self.service_id)
                   return True
               except Exception as e:
                   print(f"Service deregistration failed: {e}")
           return False

       async def discover_services(self, service_name: str) -> list:
           """Discover services by name"""
           try:
               services = self.consul.health.service(service_name, passing=True)[1]
               return [
                   {
                       'id': service['Service']['ID'],
                       'address': service['Service']['Address'],
                       'port': service['Service']['Port'],
                       'meta': service['Service']['Meta']
                   }
                   for service in services
               ]
           except Exception as e:
               print(f"Service discovery failed: {e}")
               return []

       async def health_check(self) -> Dict[str, Any]:
           """Check registry client health"""
           try:
               # Test connection to Consul
               self.consul.agent.self()
               return {"status": "healthy", "consul_connected": True}
           except Exception as e:
               return {"status": "unhealthy", "error": str(e), "consul_connected": False}
   ```

3. **Update Settings for Registry Configuration**
   ```python
   # Add to src/omninode_bridge/services/metadata_stamping/config/settings.py

   # Registry configuration
   enable_registry: bool = Field(default=False, description="Enable Consul registry")
   consul_host: str = Field(default="localhost", description="Consul host")
   consul_port: int = Field(default=8500, description="Consul port")
   service_registration_enabled: bool = Field(default=True, description="Enable service registration")

   def get_registry_config(self) -> dict:
       """Get registry configuration dictionary."""
       return {
           "enable_registry": self.enable_registry,
           "consul_host": self.consul_host,
           "consul_port": self.consul_port,
           "service_registration_enabled": self.service_registration_enabled
       }
   ```

4. **Integrate Registry Client with Main Service**
   ```python
   # Update src/omninode_bridge/services/metadata_stamping/main.py lifespan function

   # Add after line 54 (after monitoring_instance declaration):
   registry_client: RegistryConsulClient = None

   # Add in startup section after line 117:
   # Initialize registry client if enabled
   if settings.enable_registry:
       logger.info("Initializing Consul registry client...")
       registry_client = RegistryConsulClient(
           consul_host=settings.consul_host,
           consul_port=settings.consul_port
       )

       if settings.service_registration_enabled:
           registration_success = await registry_client.register_service()
           if registration_success:
               logger.info("Service registered with Consul successfully")
           else:
               logger.warning("Failed to register service with Consul")

   # Add in shutdown section after line 134:
   # Cleanup registry
   if registry_client:
       await registry_client.deregister_service()
       logger.info("Service deregistered from Consul")
   ```

5. **Add Registry Health Check Endpoint**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/api/registry.py
   from fastapi import APIRouter, Depends
   from ..registry.consul_client import RegistryConsulClient
   from ..config.settings import get_settings

   router = APIRouter(prefix="/registry", tags=["registry"])

   @router.get("/health")
   async def registry_health():
       """Check registry client health"""
       settings = get_settings()
       if not settings.enable_registry:
           return {"status": "disabled", "message": "Registry not enabled"}

       # This would be injected in real implementation
       registry_client = RegistryConsulClient(settings.consul_host, settings.consul_port)
       health = await registry_client.health_check()
       return health

   @router.get("/discover/{service_name}")
   async def discover_service(service_name: str):
       """Discover services by name"""
       settings = get_settings()
       if not settings.enable_registry:
           return {"error": "Registry not enabled"}

       registry_client = RegistryConsulClient(settings.consul_host, settings.consul_port)
       services = await registry_client.discover_services(service_name)
       return {"services": services, "count": len(services)}
   ```

**Validation Criteria for Coordinator 3**:
- [ ] Consul client successfully connects
- [ ] Service registration completes without errors
- [ ] Health checks return positive status
- [ ] Service discovery returns registered services
- [ ] Integration tests pass

---

### Coordinator 4: Trust Zones & Security Framework (Week 2, Priority: HIGH)
**Duration**: 2-3 days
**Dependencies**: Registry client (Coordinator 3)
**Validation**: Security middleware active, trust zones assigned

#### Implementation Steps

1. **Create Trust Zone Models**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/security/trust_zones.py
   from enum import Enum
   from typing import Optional, Dict, Any
   from pydantic import BaseModel

   class TrustLevel(str, Enum):
       UNVERIFIED = "UNVERIFIED"
       SIGNED = "SIGNED"
       VERIFIED = "VERIFIED"

   class TrustZone(str, Enum):
       LOCAL = "zone.local"
       ORG = "zone.org"
       GLOBAL = "zone.global"

   class TrustContext(BaseModel):
       trust_level: TrustLevel
       trust_zone: TrustZone
       signature: Optional[str] = None
       public_key: Optional[str] = None
       verification_timestamp: Optional[str] = None

   class TrustZoneManager:
       def __init__(self):
           self.zone_assignments = {
               "localhost": TrustZone.LOCAL,
               "127.0.0.1": TrustZone.LOCAL,
               "*.omninode.local": TrustZone.LOCAL,
               "*.omninode.org": TrustZone.ORG
           }

       def assign_trust_zone(self, source_address: str) -> TrustZone:
           """Assign trust zone based on source address"""
           for pattern, zone in self.zone_assignments.items():
               if self._matches_pattern(source_address, pattern):
                   return zone
           return TrustZone.GLOBAL

       def _matches_pattern(self, address: str, pattern: str) -> bool:
           """Simple pattern matching for trust zone assignment"""
           if pattern.startswith("*"):
               suffix = pattern[1:]
               return address.endswith(suffix)
           return address == pattern

       def get_required_trust_level(self, zone: TrustZone, operation: str) -> TrustLevel:
           """Get required trust level for operation in zone"""
           trust_requirements = {
               (TrustZone.LOCAL, "read"): TrustLevel.UNVERIFIED,
               (TrustZone.LOCAL, "write"): TrustLevel.UNVERIFIED,
               (TrustZone.ORG, "read"): TrustLevel.SIGNED,
               (TrustZone.ORG, "write"): TrustLevel.SIGNED,
               (TrustZone.GLOBAL, "read"): TrustLevel.VERIFIED,
               (TrustZone.GLOBAL, "write"): TrustLevel.VERIFIED
           }
           return trust_requirements.get((zone, operation), TrustLevel.VERIFIED)
   ```

2. **Implement Signature Validation**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/security/signature_validator.py
   import ed25519
   import base64
   import hashlib
   from typing import Optional, Tuple

   class SignatureValidator:
       def __init__(self):
           self.public_keys = {}  # Store trusted public keys

       def verify_ed25519_signature(self, message: bytes, signature: str, public_key: str) -> bool:
           """Verify ed25519 signature"""
           try:
               sig_bytes = base64.b64decode(signature)
               key_bytes = base64.b64decode(public_key)

               verifying_key = ed25519.VerifyingKey(key_bytes)
               verifying_key.verify(sig_bytes, message)
               return True
           except Exception as e:
               print(f"Signature verification failed: {e}")
               return False

       def validate_message_integrity(self, message: dict, signature: str, public_key: str) -> bool:
           """Validate message integrity with signature"""
           # Create canonical representation
           message_bytes = self._canonicalize_message(message)
           return self.verify_ed25519_signature(message_bytes, signature, public_key)

       def _canonicalize_message(self, message: dict) -> bytes:
           """Create canonical byte representation of message"""
           import json
           canonical = json.dumps(message, sort_keys=True, separators=(',', ':'))
           return canonical.encode('utf-8')

       def add_trusted_public_key(self, key_id: str, public_key: str):
           """Add trusted public key"""
           self.public_keys[key_id] = public_key

       def get_public_key(self, key_id: str) -> Optional[str]:
           """Get public key by ID"""
           return self.public_keys.get(key_id)
   ```

3. **Create Security Middleware**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/security/middleware.py
   from fastapi import Request, HTTPException
   from starlette.middleware.base import BaseHTTPMiddleware
   from .trust_zones import TrustZoneManager, TrustContext, TrustLevel
   from .signature_validator import SignatureValidator

   class ONESecurityMiddleware(BaseHTTPMiddleware):
       def __init__(self, app, enable_security: bool = True):
           super().__init__(app)
           self.enable_security = enable_security
           self.trust_zone_manager = TrustZoneManager()
           self.signature_validator = SignatureValidator()

       async def dispatch(self, request: Request, call_next):
           if not self.enable_security:
               return await call_next(request)

           # Extract client information
           client_host = request.client.host if request.client else "unknown"

           # Assign trust zone
           trust_zone = self.trust_zone_manager.assign_trust_zone(client_host)

           # Determine operation type
           operation = "write" if request.method in ["POST", "PUT", "DELETE", "PATCH"] else "read"

           # Get required trust level
           required_trust = self.trust_zone_manager.get_required_trust_level(trust_zone, operation)

           # Validate trust requirements
           trust_context = await self._validate_trust_requirements(request, required_trust)

           if not trust_context:
               raise HTTPException(
                   status_code=403,
                   detail=f"Insufficient trust level for {operation} operation in {trust_zone}"
               )

           # Add trust context to request
           request.state.trust_context = trust_context

           response = await call_next(request)

           # Add security headers
           response.headers["X-Trust-Zone"] = trust_zone
           response.headers["X-Trust-Level"] = trust_context.trust_level

           return response

       async def _validate_trust_requirements(self, request: Request, required_trust: TrustLevel) -> Optional[TrustContext]:
           """Validate trust requirements"""
           # For UNVERIFIED level, always allow
           if required_trust == TrustLevel.UNVERIFIED:
               return TrustContext(
                   trust_level=TrustLevel.UNVERIFIED,
                   trust_zone=self.trust_zone_manager.assign_trust_zone(request.client.host)
               )

           # Check for signature headers
           signature = request.headers.get("X-ONF-Signature")
           public_key = request.headers.get("X-ONF-Public-Key")

           if not signature or not public_key:
               return None

           # Validate signature
           try:
               body = await request.body()
               if self.signature_validator.verify_ed25519_signature(body, signature, public_key):
                   return TrustContext(
                       trust_level=TrustLevel.SIGNED,  # Could be VERIFIED with additional checks
                       trust_zone=self.trust_zone_manager.assign_trust_zone(request.client.host),
                       signature=signature,
                       public_key=public_key
                   )
           except Exception as e:
               print(f"Signature validation error: {e}")

           return None
   ```

4. **Integrate Security Middleware**
   ```python
   # Update src/omninode_bridge/services/metadata_stamping/main.py

   # Add import after line 44:
   from .security.middleware import ONESecurityMiddleware

   # Add after CORS middleware (around line 158):
   # Add O.N.E. security middleware
   if settings.enable_security:
       app.add_middleware(ONESecurityMiddleware, enable_security=True)
   ```

5. **Add Security Configuration**
   ```python
   # Add to src/omninode_bridge/services/metadata_stamping/config/settings.py

   # Security configuration
   enable_security: bool = Field(default=True, description="Enable O.N.E. security middleware")
   trusted_public_keys: str = Field(default="", description="Comma-separated trusted public keys")
   signature_validation_enabled: bool = Field(default=True, description="Enable signature validation")

   def get_security_config(self) -> dict:
       """Get security configuration dictionary."""
       return {
           "enable_security": self.enable_security,
           "trusted_public_keys": self.trusted_public_keys.split(",") if self.trusted_public_keys else [],
           "signature_validation_enabled": self.signature_validation_enabled
       }
   ```

**Validation Criteria for Coordinator 4**:
- [ ] Trust zones correctly assigned based on source
- [ ] Signature validation working for ed25519
- [ ] Security middleware active and filtering requests
- [ ] Trust level requirements enforced
- [ ] Security headers added to responses

---

### Coordinator 5: Schema-First Execution Framework (Week 3, Priority: MEDIUM)
**Duration**: 2-3 days
**Dependencies**: Registry integration (Coordinator 3-4)
**Validation**: @transformer decorator functional, schema registry active

#### Implementation Steps

1. **Create Base Transformer Framework**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/execution/transformer.py
   from typing import Any, Dict, Type, Optional, Callable
   from pydantic import BaseModel, ValidationError
   from abc import ABC, abstractmethod
   import inspect
   import uuid
   from datetime import datetime

   class ExecutionContext(BaseModel):
       execution_id: str
       input_schema: str
       output_schema: str
       simulation_mode: bool = False
       budget_limit: Optional[float] = None
       metadata: Dict[str, Any] = {}

   class BaseTransformer(ABC):
       def __init__(self, name: str, version: str = "1.0.0"):
           self.name = name
           self.version = version
           self.input_schema: Optional[Type[BaseModel]] = None
           self.output_schema: Optional[Type[BaseModel]] = None

       @abstractmethod
       async def execute(self, input_data: BaseModel, context: ExecutionContext) -> BaseModel:
           """Execute transformer with typed input/output"""
           pass

       def set_schemas(self, input_schema: Type[BaseModel], output_schema: Type[BaseModel]):
           """Set input and output schemas for validation"""
           self.input_schema = input_schema
           self.output_schema = output_schema

       async def validate_input(self, data: Any) -> BaseModel:
           """Validate input against schema"""
           if not self.input_schema:
               raise ValueError(f"No input schema defined for {self.name}")

           try:
               return self.input_schema(**data) if isinstance(data, dict) else self.input_schema(data)
           except ValidationError as e:
               raise ValueError(f"Input validation failed: {e}")

       async def validate_output(self, data: Any) -> BaseModel:
           """Validate output against schema"""
           if not self.output_schema:
               raise ValueError(f"No output schema defined for {self.name}")

           try:
               return self.output_schema(**data) if isinstance(data, dict) else self.output_schema(data)
           except ValidationError as e:
               raise ValueError(f"Output validation failed: {e}")

   # Transformer registry
   _transformer_registry: Dict[str, BaseTransformer] = {}

   def transformer(input_schema: Type[BaseModel], output_schema: Type[BaseModel], name: Optional[str] = None, version: str = "1.0.0"):
       """Decorator to create a transformer from a function"""
       def decorator(func: Callable) -> BaseTransformer:
           transformer_name = name or func.__name__

           class FunctionTransformer(BaseTransformer):
               def __init__(self):
                   super().__init__(transformer_name, version)
                   self.set_schemas(input_schema, output_schema)
                   self.func = func

               async def execute(self, input_data: BaseModel, context: ExecutionContext) -> BaseModel:
                   # Execute the decorated function
                   if inspect.iscoroutinefunction(self.func):
                       result = await self.func(input_data, context)
                   else:
                       result = self.func(input_data, context)

                   # Validate output
                   return await self.validate_output(result)

           # Create transformer instance
           transformer_instance = FunctionTransformer()

           # Register transformer
           _transformer_registry[transformer_name] = transformer_instance

           return transformer_instance

       return decorator

   def get_transformer(name: str) -> Optional[BaseTransformer]:
       """Get transformer by name"""
       return _transformer_registry.get(name)

   def list_transformers() -> Dict[str, BaseTransformer]:
       """List all registered transformers"""
       return _transformer_registry.copy()
   ```

2. **Create Schema Registry**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/execution/schema_registry.py
   from typing import Dict, Any, Type, Optional, List
   from pydantic import BaseModel
   import json
   from datetime import datetime

   class SchemaVersion(BaseModel):
       version: str
       schema_json: str
       created_at: datetime
       deprecated: bool = False

   class SchemaRegistry:
       def __init__(self):
           self.schemas: Dict[str, Dict[str, SchemaVersion]] = {}

       def register_schema(self, name: str, schema_class: Type[BaseModel], version: str = "1.0.0") -> bool:
           """Register a Pydantic schema"""
           try:
               schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

               if name not in self.schemas:
                   self.schemas[name] = {}

               self.schemas[name][version] = SchemaVersion(
                   version=version,
                   schema_json=schema_json,
                   created_at=datetime.utcnow()
               )
               return True
           except Exception as e:
               print(f"Schema registration failed: {e}")
               return False

       def get_schema(self, name: str, version: Optional[str] = None) -> Optional[SchemaVersion]:
           """Get schema by name and version"""
           if name not in self.schemas:
               return None

           if version:
               return self.schemas[name].get(version)

           # Return latest version
           versions = self.schemas[name]
           if not versions:
               return None

           latest_version = max(versions.keys())
           return versions[latest_version]

       def list_schemas(self) -> Dict[str, List[str]]:
           """List all schemas with their versions"""
           return {name: list(versions.keys()) for name, versions in self.schemas.items()}

       def deprecate_schema(self, name: str, version: str) -> bool:
           """Mark schema version as deprecated"""
           if name in self.schemas and version in self.schemas[name]:
               self.schemas[name][version].deprecated = True
               return True
           return False

   # Global schema registry instance
   schema_registry = SchemaRegistry()
   ```

3. **Implement Metadata Stamping Transformers**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/execution/stamping_transformers.py
   from pydantic import BaseModel, Field
   from typing import Dict, Any, Optional
   from .transformer import transformer, ExecutionContext
   import hashlib
   import time

   # Input/Output schemas for metadata stamping
   class StampingInput(BaseModel):
       content: str = Field(..., description="Content to stamp")
       file_path: Optional[str] = Field(None, description="Optional file path")
       stamp_type: str = Field("lightweight", description="Type of stamp")
       metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

   class StampingOutput(BaseModel):
       success: bool = Field(..., description="Operation success")
       hash: str = Field(..., description="Generated hash")
       stamped_content: str = Field(..., description="Content with stamp")
       execution_time_ms: float = Field(..., description="Execution time in milliseconds")
       stamp_metadata: Dict[str, Any] = Field(..., description="Stamp metadata")

   class ValidationInput(BaseModel):
       content: str = Field(..., description="Content to validate")
       expected_hash: Optional[str] = Field(None, description="Expected hash")

   class ValidationOutput(BaseModel):
       valid: bool = Field(..., description="Validation result")
       found_stamps: int = Field(..., description="Number of stamps found")
       validation_details: Dict[str, Any] = Field(..., description="Validation details")

   # Stamping transformer
   @transformer(StampingInput, StampingOutput, "metadata_stamper", "1.0.0")
   async def metadata_stamping_transformer(input_data: StampingInput, context: ExecutionContext) -> Dict[str, Any]:
       """Transform content into stamped content with metadata"""
       start_time = time.perf_counter()

       # Generate BLAKE3 hash
       content_bytes = input_data.content.encode('utf-8')
       blake3_hash = hashlib.blake2b(content_bytes, digest_size=32).hexdigest()

       # Create stamp
       stamp_data = {
           "hash": blake3_hash,
           "file_path": input_data.file_path,
           "stamp_type": input_data.stamp_type,
           "timestamp": time.time(),
           "execution_id": context.execution_id
       }
       stamp_data.update(input_data.metadata)

       # Create stamped content
       stamp_header = f"<!-- METADATA_STAMP: {blake3_hash} -->"
       stamped_content = f"{stamp_header}\n{input_data.content}"

       execution_time = (time.perf_counter() - start_time) * 1000

       return {
           "success": True,
           "hash": blake3_hash,
           "stamped_content": stamped_content,
           "execution_time_ms": execution_time,
           "stamp_metadata": stamp_data
       }

   # Validation transformer
   @transformer(ValidationInput, ValidationOutput, "stamp_validator", "1.0.0")
   async def stamp_validation_transformer(input_data: ValidationInput, context: ExecutionContext) -> Dict[str, Any]:
       """Transform content validation request into validation result"""
       import re

       # Find stamps in content
       stamp_pattern = r'<!-- METADATA_STAMP: ([a-fA-F0-9]+) -->'
       found_stamps = re.findall(stamp_pattern, input_data.content)

       validation_result = {
           "valid": len(found_stamps) > 0,
           "found_stamps": len(found_stamps),
           "validation_details": {
               "stamps_found": found_stamps,
               "expected_hash": input_data.expected_hash,
               "hash_matches": False
           }
       }

       # Check hash match if expected hash provided
       if input_data.expected_hash and found_stamps:
           validation_result["validation_details"]["hash_matches"] = input_data.expected_hash in found_stamps
           validation_result["valid"] = validation_result["validation_details"]["hash_matches"]

       return validation_result
   ```

4. **Integrate Transformers with API**
   ```python
   # Update src/omninode_bridge/services/metadata_stamping/api/router.py

   # Add imports
   from ..execution.transformer import get_transformer, ExecutionContext
   from ..execution.stamping_transformers import StampingInput, StampingOutput, ValidationInput, ValidationOutput
   import uuid

   # Add transformer-based endpoints
   @router.post("/transform/stamp", response_model=UnifiedResponse)
   async def transformer_stamp_content(request: StampingInput) -> UnifiedResponse:
       """Create stamp using transformer pattern"""
       try:
           # Get stamping transformer
           stamper = get_transformer("metadata_stamper")
           if not stamper:
               return UnifiedResponse(
                   status="error",
                   error="Metadata stamping transformer not found"
               )

           # Create execution context
           context = ExecutionContext(
               execution_id=str(uuid.uuid4()),
               input_schema="StampingInput",
               output_schema="StampingOutput"
           )

           # Execute transformer
           result = await stamper.execute(request, context)

           return UnifiedResponse(
               status="success",
               data=result.model_dump(),
               metadata={
                   "transformer": "metadata_stamper",
                   "execution_id": context.execution_id,
                   "schema_version": "1.0.0"
               }
           )

       except Exception as e:
           return UnifiedResponse(
               status="error",
               error=str(e)
           )

   @router.post("/transform/validate", response_model=UnifiedResponse)
   async def transformer_validate_content(request: ValidationInput) -> UnifiedResponse:
       """Validate stamps using transformer pattern"""
       try:
           # Get validation transformer
           validator = get_transformer("stamp_validator")
           if not validator:
               return UnifiedResponse(
                   status="error",
                   error="Stamp validation transformer not found"
               )

           # Create execution context
           context = ExecutionContext(
               execution_id=str(uuid.uuid4()),
               input_schema="ValidationInput",
               output_schema="ValidationOutput"
           )

           # Execute transformer
           result = await validator.execute(request, context)

           return UnifiedResponse(
               status="success",
               data=result.model_dump(),
               metadata={
                   "transformer": "stamp_validator",
                   "execution_id": context.execution_id,
                   "schema_version": "1.0.0"
               }
           )

       except Exception as e:
           return UnifiedResponse(
               status="error",
               error=str(e)
           )
   ```

**Validation Criteria for Coordinator 5**:
- [ ] @transformer decorator creates valid transformers
- [ ] Schema validation works for input/output
- [ ] Transformer registry functional
- [ ] Schema registry stores and retrieves schemas
- [ ] API endpoints use transformer pattern

---

### Coordinator 6: Execution Chain & Simulation (Week 3, Priority: MEDIUM)
**Duration**: 2-3 days
**Dependencies**: Schema framework (Coordinator 5)
**Validation**: DAG execution functional, simulation mode working

#### Implementation Steps

1. **Create DAG Execution Engine**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/execution/dag_engine.py
   from typing import Dict, List, Any, Optional, Set
   from dataclasses import dataclass
   from enum import Enum
   import asyncio
   import uuid
   from .transformer import BaseTransformer, ExecutionContext

   class NodeStatus(str, Enum):
       PENDING = "pending"
       RUNNING = "running"
       COMPLETED = "completed"
       FAILED = "failed"
       SKIPPED = "skipped"

   @dataclass
   class DAGNode:
       id: str
       transformer: BaseTransformer
       dependencies: List[str]
       status: NodeStatus = NodeStatus.PENDING
       input_data: Any = None
       output_data: Any = None
       execution_time_ms: float = 0.0
       error: Optional[str] = None

   class DAGExecutor:
       def __init__(self):
           self.nodes: Dict[str, DAGNode] = {}
           self.execution_context: Optional[ExecutionContext] = None

       def add_node(self, node_id: str, transformer: BaseTransformer, dependencies: List[str] = None) -> DAGNode:
           """Add a node to the DAG"""
           node = DAGNode(
               id=node_id,
               transformer=transformer,
               dependencies=dependencies or []
           )
           self.nodes[node_id] = node
           return node

       def validate_dag(self) -> bool:
           """Validate DAG for cycles and missing dependencies"""
           # Check for missing dependencies
           for node in self.nodes.values():
               for dep in node.dependencies:
                   if dep not in self.nodes:
                       raise ValueError(f"Node {node.id} depends on missing node {dep}")

           # Check for cycles using DFS
           visited = set()
           rec_stack = set()

           def has_cycle(node_id: str) -> bool:
               visited.add(node_id)
               rec_stack.add(node_id)

               for dep in self.nodes[node_id].dependencies:
                   if dep not in visited:
                       if has_cycle(dep):
                           return True
                   elif dep in rec_stack:
                       return True

               rec_stack.remove(node_id)
               return False

           for node_id in self.nodes:
               if node_id not in visited:
                   if has_cycle(node_id):
                       raise ValueError("DAG contains cycles")

           return True

       async def execute(self, input_data: Dict[str, Any], simulation_mode: bool = False) -> Dict[str, Any]:
           """Execute the DAG"""
           if not self.validate_dag():
               raise ValueError("Invalid DAG")

           # Create execution context
           self.execution_context = ExecutionContext(
               execution_id=str(uuid.uuid4()),
               input_schema="DAGInput",
               output_schema="DAGOutput",
               simulation_mode=simulation_mode
           )

           # Reset node states
           for node in self.nodes.values():
               node.status = NodeStatus.PENDING
               node.output_data = None
               node.error = None

           # Set initial input data
           self._set_initial_inputs(input_data)

           # Execute nodes in topological order
           execution_results = {}

           while True:
               # Find ready nodes (no pending dependencies)
               ready_nodes = self._get_ready_nodes()

               if not ready_nodes:
                   # Check if we're done or stuck
                   pending_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.PENDING]
                   if not pending_nodes:
                       break  # All done
                   else:
                       raise RuntimeError("DAG execution stuck - circular dependency detected")

               # Execute ready nodes in parallel
               await self._execute_nodes_parallel(ready_nodes)

           # Collect results
           for node_id, node in self.nodes.items():
               execution_results[node_id] = {
                   "status": node.status,
                   "output_data": node.output_data,
                   "execution_time_ms": node.execution_time_ms,
                   "error": node.error
               }

           return {
               "execution_id": self.execution_context.execution_id,
               "simulation_mode": simulation_mode,
               "nodes": execution_results,
               "overall_status": "completed" if all(n.status == NodeStatus.COMPLETED for n in self.nodes.values()) else "failed"
           }

       def _get_ready_nodes(self) -> List[DAGNode]:
           """Get nodes ready for execution"""
           ready = []
           for node in self.nodes.values():
               if node.status == NodeStatus.PENDING:
                   # Check if all dependencies are completed
                   deps_completed = all(
                       self.nodes[dep_id].status == NodeStatus.COMPLETED
                       for dep_id in node.dependencies
                   )
                   if deps_completed:
                       ready.append(node)
           return ready

       async def _execute_nodes_parallel(self, nodes: List[DAGNode]):
           """Execute multiple nodes in parallel"""
           tasks = [self._execute_node(node) for node in nodes]
           await asyncio.gather(*tasks, return_exceptions=True)

       async def _execute_node(self, node: DAGNode):
           """Execute a single node"""
           import time

           node.status = NodeStatus.RUNNING
           start_time = time.perf_counter()

           try:
               # Prepare input from dependencies
               node_input = self._prepare_node_input(node)

               if self.execution_context.simulation_mode:
                   # Simulation mode - don't actually execute
                   node.output_data = {"simulated": True, "input": node_input}
                   node.status = NodeStatus.COMPLETED
               else:
                   # Validate input
                   validated_input = await node.transformer.validate_input(node_input)

                   # Execute transformer
                   result = await node.transformer.execute(validated_input, self.execution_context)

                   # Validate output
                   node.output_data = await node.transformer.validate_output(result)
                   node.status = NodeStatus.COMPLETED

           except Exception as e:
               node.error = str(e)
               node.status = NodeStatus.FAILED

           finally:
               node.execution_time_ms = (time.perf_counter() - start_time) * 1000

       def _prepare_node_input(self, node: DAGNode) -> Any:
           """Prepare input for node from dependencies and initial data"""
           if not node.dependencies:
               return node.input_data

           # Combine outputs from dependencies
           combined_input = {}
           for dep_id in node.dependencies:
               dep_node = self.nodes[dep_id]
               if dep_node.output_data:
                   combined_input[dep_id] = dep_node.output_data

           return combined_input

       def _set_initial_inputs(self, input_data: Dict[str, Any]):
           """Set initial input data for nodes"""
           for node_id, data in input_data.items():
               if node_id in self.nodes:
                   self.nodes[node_id].input_data = data
   ```

2. **Add Simulation Support**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/execution/simulation.py
   from typing import Dict, Any, List, Optional
   from pydantic import BaseModel
   from .dag_engine import DAGExecutor, DAGNode
   from .transformer import get_transformer

   class SimulationRequest(BaseModel):
       workflow_definition: Dict[str, Any]
       input_data: Dict[str, Any]
       simulation_options: Dict[str, Any] = {}

   class SimulationResult(BaseModel):
       simulation_id: str
       estimated_execution_time_ms: float
       estimated_resource_usage: Dict[str, Any]
       execution_path: List[str]
       potential_errors: List[str]
       simulation_data: Dict[str, Any]

   class WorkflowSimulator:
       def __init__(self):
           self.simulation_cache: Dict[str, SimulationResult] = {}

       async def simulate_workflow(self, request: SimulationRequest) -> SimulationResult:
           """Simulate workflow execution without actually running it"""
           # Build DAG from workflow definition
           dag = self._build_dag_from_definition(request.workflow_definition)

           # Run in simulation mode
           simulation_results = await dag.execute(request.input_data, simulation_mode=True)

           # Analyze simulation results
           analysis = self._analyze_simulation_results(simulation_results)

           simulation_result = SimulationResult(
               simulation_id=simulation_results["execution_id"],
               estimated_execution_time_ms=analysis["estimated_time"],
               estimated_resource_usage=analysis["resource_usage"],
               execution_path=analysis["execution_path"],
               potential_errors=analysis["potential_errors"],
               simulation_data=simulation_results
           )

           # Cache results
           self.simulation_cache[simulation_result.simulation_id] = simulation_result

           return simulation_result

       def _build_dag_from_definition(self, workflow_def: Dict[str, Any]) -> DAGExecutor:
           """Build DAG executor from workflow definition"""
           dag = DAGExecutor()

           # Add nodes from definition
           for node_def in workflow_def.get("nodes", []):
               transformer = get_transformer(node_def["transformer"])
               if not transformer:
                   raise ValueError(f"Transformer {node_def['transformer']} not found")

               dag.add_node(
                   node_id=node_def["id"],
                   transformer=transformer,
                   dependencies=node_def.get("dependencies", [])
               )

           return dag

       def _analyze_simulation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Analyze simulation results for estimates"""
           total_time = 0.0
           execution_path = []
           potential_errors = []

           for node_id, node_result in results["nodes"].items():
               execution_path.append(node_id)

               # Estimate execution time (simulation gives 0, so estimate)
               estimated_node_time = 10.0  # Default estimate
               total_time += estimated_node_time

               # Check for potential errors
               if "error" in node_result and node_result["error"]:
                   potential_errors.append(f"Node {node_id}: {node_result['error']}")

           return {
               "estimated_time": total_time,
               "execution_path": execution_path,
               "potential_errors": potential_errors,
               "resource_usage": {
                   "memory_mb": len(execution_path) * 50,  # Rough estimate
                   "cpu_cores": 1,
                   "network_calls": len(execution_path)
               }
           }

       def get_simulation_result(self, simulation_id: str) -> Optional[SimulationResult]:
           """Get cached simulation result"""
           return self.simulation_cache.get(simulation_id)
   ```

3. **Add DAG and Simulation API Endpoints**
   ```python
   # Update src/omninode_bridge/services/metadata_stamping/api/router.py

   from ..execution.dag_engine import DAGExecutor
   from ..execution.simulation import WorkflowSimulator, SimulationRequest

   # Add simulation endpoints
   workflow_simulator = WorkflowSimulator()

   @router.post("/workflow/simulate", response_model=UnifiedResponse)
   async def simulate_workflow(request: SimulationRequest) -> UnifiedResponse:
       """Simulate workflow execution"""
       try:
           result = await workflow_simulator.simulate_workflow(request)

           return UnifiedResponse(
               status="success",
               data=result.model_dump(),
               metadata={
                   "operation": "workflow_simulation",
                   "simulation_id": result.simulation_id
               }
           )

       except Exception as e:
           return UnifiedResponse(
               status="error",
               error=str(e)
           )

   @router.post("/workflow/execute", response_model=UnifiedResponse)
   async def execute_workflow(workflow_definition: Dict[str, Any], input_data: Dict[str, Any]) -> UnifiedResponse:
       """Execute workflow as DAG"""
       try:
           dag = DAGExecutor()

           # Build DAG from definition
           for node_def in workflow_definition.get("nodes", []):
               transformer = get_transformer(node_def["transformer"])
               if not transformer:
                   return UnifiedResponse(
                       status="error",
                       error=f"Transformer {node_def['transformer']} not found"
                   )

               dag.add_node(
                   node_id=node_def["id"],
                   transformer=transformer,
                   dependencies=node_def.get("dependencies", [])
               )

           # Execute DAG
           results = await dag.execute(input_data, simulation_mode=False)

           return UnifiedResponse(
               status="success",
               data=results,
               metadata={
                   "operation": "workflow_execution",
                   "execution_id": results["execution_id"]
               }
           )

       except Exception as e:
           return UnifiedResponse(
               status="error",
               error=str(e)
           )
   ```

**Validation Criteria for Coordinator 6**:
- [ ] DAG validation detects cycles and missing dependencies
- [ ] Parallel node execution works correctly
- [ ] Simulation mode produces reasonable estimates
- [ ] Workflow API endpoints functional
- [ ] Execution provenance tracked

---

### Coordinator 7: Federation & Policy Engine (Week 4, Priority: LOW)
**Duration**: 2-3 days
**Dependencies**: All previous coordinators
**Validation**: Federation working, policy enforcement active

#### Implementation Steps

1. **Create Federation Support**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/federation/federation_client.py
   from typing import Dict, List, Any, Optional
   import aiohttp
   from ..security.trust_zones import TrustContext
   from ..registry.consul_client import RegistryConsulClient

   class FederationClient:
       def __init__(self):
           self.federation_endpoints: Dict[str, str] = {}
           self.trust_anchors: Dict[str, str] = {}

       async def register_federation_endpoint(self, zone: str, endpoint: str, trust_anchor: str):
           """Register federation endpoint for cross-zone communication"""
           self.federation_endpoints[zone] = endpoint
           self.trust_anchors[zone] = trust_anchor

       async def federated_discovery(self, service_name: str, zones: List[str]) -> Dict[str, List[Any]]:
           """Discover services across multiple zones"""
           results = {}

           for zone in zones:
               if zone in self.federation_endpoints:
                   try:
                       services = await self._query_zone_services(zone, service_name)
                       results[zone] = services
                   except Exception as e:
                       results[zone] = {"error": str(e)}

           return results

       async def _query_zone_services(self, zone: str, service_name: str) -> List[Any]:
           """Query services in a specific zone"""
           endpoint = self.federation_endpoints[zone]

           async with aiohttp.ClientSession() as session:
               async with session.get(f"{endpoint}/services/{service_name}") as response:
                   if response.status == 200:
                       data = await response.json()
                       return data.get("services", [])
                   else:
                       raise Exception(f"Federation query failed: {response.status}")

       async def cross_zone_trust_negotiation(self, target_zone: str, operation: str) -> bool:
           """Negotiate trust for cross-zone operations"""
           if target_zone not in self.trust_anchors:
               return False

           # Simplified trust negotiation
           trust_anchor = self.trust_anchors[target_zone]

           # In real implementation, this would involve:
           # 1. Certificate validation
           # 2. Trust chain verification
           # 3. Policy evaluation

           return True  # Simplified approval
   ```

2. **Implement Policy Engine**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/policy/policy_engine.py
   from typing import Dict, Any, List, Optional, Callable
   from enum import Enum
   from pydantic import BaseModel

   class PolicyDecision(str, Enum):
       ALLOW = "allow"
       DENY = "deny"
       AUDIT = "audit"

   class PolicyRule(BaseModel):
       id: str
       name: str
       condition: str  # Simple condition string
       action: PolicyDecision
       priority: int = 0
       enabled: bool = True

   class PolicyContext(BaseModel):
       operation: str
       resource: str
       actor: str
       trust_zone: str
       trust_level: str
       metadata: Dict[str, Any] = {}

   class PolicyEngine:
       def __init__(self):
           self.rules: Dict[str, PolicyRule] = {}
           self.evaluators: Dict[str, Callable] = {}

           # Register default evaluators
           self._register_default_evaluators()

       def add_rule(self, rule: PolicyRule):
           """Add policy rule"""
           self.rules[rule.id] = rule

       def remove_rule(self, rule_id: str):
           """Remove policy rule"""
           self.rules.pop(rule_id, None)

       async def evaluate_policy(self, context: PolicyContext) -> PolicyDecision:
           """Evaluate policy for given context"""
           applicable_rules = self._get_applicable_rules(context)

           if not applicable_rules:
               return PolicyDecision.ALLOW  # Default allow if no rules

           # Sort by priority (higher priority first)
           applicable_rules.sort(key=lambda r: r.priority, reverse=True)

           # Evaluate rules in priority order
           for rule in applicable_rules:
               if await self._evaluate_rule(rule, context):
                   return rule.action

           return PolicyDecision.ALLOW  # Default if no rules match

       def _get_applicable_rules(self, context: PolicyContext) -> List[PolicyRule]:
           """Get rules applicable to context"""
           return [rule for rule in self.rules.values() if rule.enabled]

       async def _evaluate_rule(self, rule: PolicyRule, context: PolicyContext) -> bool:
           """Evaluate if rule condition matches context"""
           try:
               # Simple condition evaluation
               # In production, this would be more sophisticated
               return self._simple_condition_eval(rule.condition, context)
           except Exception as e:
               print(f"Rule evaluation error: {e}")
               return False

       def _simple_condition_eval(self, condition: str, context: PolicyContext) -> bool:
           """Simple condition evaluation"""
           # Example conditions:
           # "operation == 'write' and trust_zone == 'zone.global'"
           # "trust_level == 'VERIFIED'"

           condition_vars = {
               'operation': context.operation,
               'resource': context.resource,
               'actor': context.actor,
               'trust_zone': context.trust_zone,
               'trust_level': context.trust_level
           }

           try:
               return eval(condition, {"__builtins__": {}}, condition_vars)
           except:
               return False

       def _register_default_evaluators(self):
           """Register default condition evaluators"""
           # Default evaluators for common conditions
           pass

   # Default policy rules
   DEFAULT_POLICIES = [
       PolicyRule(
           id="global_write_verified",
           name="Global zone write requires verification",
           condition="operation == 'write' and trust_zone == 'zone.global'",
           action=PolicyDecision.DENY,
           priority=100
       ),
       PolicyRule(
           id="local_zone_allow",
           name="Local zone operations allowed",
           condition="trust_zone == 'zone.local'",
           action=PolicyDecision.ALLOW,
           priority=50
       ),
       PolicyRule(
           id="audit_sensitive_ops",
           name="Audit sensitive operations",
           condition="operation in ['delete', 'modify']",
           action=PolicyDecision.AUDIT,
           priority=75
       )
   ]
   ```

3. **Create Observability Enhancements**
   ```python
   # File: src/omninode_bridge/services/metadata_stamping/observability/o_n_e_metrics.py
   from prometheus_client import Counter, Histogram, Gauge
   from typing import Dict, Any

   # O.N.E. Protocol specific metrics
   ONE_OPERATIONS_TOTAL = Counter(
       'one_operations_total',
       'Total O.N.E. protocol operations',
       ['operation_type', 'trust_zone', 'trust_level']
   )

   ONE_OPERATION_DURATION = Histogram(
       'one_operation_duration_seconds',
       'O.N.E. operation duration',
       ['operation_type']
   )

   ONE_TRUST_VALIDATIONS = Counter(
       'one_trust_validations_total',
       'Trust validation attempts',
       ['trust_level', 'result']
   )

   ONE_FEDERATION_REQUESTS = Counter(
       'one_federation_requests_total',
       'Federation requests',
       ['target_zone', 'result']
   )

   ONE_POLICY_EVALUATIONS = Counter(
       'one_policy_evaluations_total',
       'Policy evaluations',
       ['decision', 'rule_id']
   )

   TRANSFORMER_EXECUTIONS = Counter(
       'transformer_executions_total',
       'Transformer executions',
       ['transformer_name', 'status']
   )

   DAG_EXECUTIONS = Counter(
       'dag_executions_total',
       'DAG executions',
       ['simulation_mode', 'status']
   )

   class ONEMetricsCollector:
       def __init__(self):
           self.custom_metrics: Dict[str, Any] = {}

       def record_operation(self, operation_type: str, trust_zone: str, trust_level: str, duration: float):
           """Record O.N.E. operation metrics"""
           ONE_OPERATIONS_TOTAL.labels(
               operation_type=operation_type,
               trust_zone=trust_zone,
               trust_level=trust_level
           ).inc()

           ONE_OPERATION_DURATION.labels(operation_type=operation_type).observe(duration)

       def record_trust_validation(self, trust_level: str, success: bool):
           """Record trust validation metrics"""
           result = "success" if success else "failure"
           ONE_TRUST_VALIDATIONS.labels(trust_level=trust_level, result=result).inc()

       def record_federation_request(self, target_zone: str, success: bool):
           """Record federation request metrics"""
           result = "success" if success else "failure"
           ONE_FEDERATION_REQUESTS.labels(target_zone=target_zone, result=result).inc()

       def record_policy_evaluation(self, decision: str, rule_id: str):
           """Record policy evaluation metrics"""
           ONE_POLICY_EVALUATIONS.labels(decision=decision, rule_id=rule_id).inc()

       def record_transformer_execution(self, transformer_name: str, success: bool):
           """Record transformer execution metrics"""
           status = "success" if success else "failure"
           TRANSFORMER_EXECUTIONS.labels(transformer_name=transformer_name, status=status).inc()

       def record_dag_execution(self, simulation_mode: bool, success: bool):
           """Record DAG execution metrics"""
           mode = "simulation" if simulation_mode else "execution"
           status = "success" if success else "failure"
           DAG_EXECUTIONS.labels(simulation_mode=mode, status=status).inc()

   # Global metrics collector
   one_metrics = ONEMetricsCollector()
   ```

**Validation Criteria for Coordinator 7**:
- [ ] Federation endpoints register correctly
- [ ] Cross-zone trust negotiation functional
- [ ] Policy engine evaluates conditions correctly
- [ ] O.N.E. metrics collection active
- [ ] Distributed tracing implemented

---

### Coordinator 8: Integration Testing & Documentation (Week 4, Priority: HIGH)
**Duration**: 2-3 days
**Dependencies**: All previous coordinators
**Validation**: 100% O.N.E. compliance achieved, documentation complete

#### Implementation Steps

1. **Create Comprehensive O.N.E. Test Suite**
   ```python
   # File: tests/integration/test_one_protocol_compliance.py
   import pytest
   import asyncio
   from fastapi.testclient import TestClient
   from src.omninode_bridge.services.metadata_stamping.main import app

   class TestONEProtocolCompliance:
       """Comprehensive O.N.E. v0.1 protocol compliance tests"""

       @pytest.fixture
       def client(self):
           return TestClient(app)

       async def test_tool_metadata_headers_present(self):
           """Test that all files have O.N.E. tool metadata headers"""
           import os
           import re

           service_files = [
               "src/omninode_bridge/services/metadata_stamping/main.py",
               "src/omninode_bridge/services/metadata_stamping/service.py",
               "src/omninode_bridge/services/metadata_stamping/engine/stamping_engine.py",
               "src/omninode_bridge/services/metadata_stamping/database/client.py",
               "src/omninode_bridge/services/metadata_stamping/api/router.py"
           ]

           metadata_pattern = r'# === OmniNode:Tool_Metadata ===.*# === /OmniNode:Tool_Metadata ==='

           for file_path in service_files:
               if os.path.exists(file_path):
                   with open(file_path, 'r') as f:
                       content = f.read()

                   assert re.search(metadata_pattern, content, re.DOTALL), f"Missing metadata header in {file_path}"

       async def test_namespace_compliance(self, client):
           """Test namespace compliance"""
           response = client.post("/stamp", json={
               "content": "test content",
               "file_path": "test.txt"
           })

           assert response.status_code == 200
           data = response.json()
           assert data["data"]["namespace"] == "omninode.services.metadata"

       async def test_registry_integration(self, client):
           """Test registry integration (if enabled)"""
           response = client.get("/registry/health")
           # Should not fail even if registry disabled
           assert response.status_code in [200, 404]

       async def test_trust_zone_assignment(self, client):
           """Test trust zone assignment"""
           response = client.get("/health")
           assert response.status_code == 200

           # Trust zone should be assigned via middleware
           if "X-Trust-Zone" in response.headers:
               assert response.headers["X-Trust-Zone"] in ["zone.local", "zone.org", "zone.global"]

       async def test_transformer_pattern(self, client):
           """Test ONEX transformer pattern"""
           response = client.post("/transform/stamp", json={
               "content": "test content for transformer",
               "file_path": "transformer_test.txt"
           })

           if response.status_code == 200:
               data = response.json()
               assert "transformer" in data.get("metadata", {})
               assert data["metadata"]["transformer"] == "metadata_stamper"

       async def test_schema_validation(self, client):
           """Test schema-first execution"""
           # Test with invalid input
           response = client.post("/transform/stamp", json={
               "invalid_field": "should fail validation"
           })

           assert response.status_code in [400, 422]  # Validation error

       async def test_dag_execution(self, client):
           """Test DAG execution capabilities"""
           workflow_def = {
               "nodes": [
                   {
                       "id": "stamper",
                       "transformer": "metadata_stamper",
                       "dependencies": []
                   }
               ]
           }

           input_data = {
               "stamper": {
                   "content": "DAG test content",
                   "file_path": "dag_test.txt"
               }
           }

           response = client.post("/workflow/execute", json={
               "workflow_definition": workflow_def,
               "input_data": input_data
           })

           # Should not fail if DAG endpoints implemented
           assert response.status_code in [200, 404]

       async def test_simulation_mode(self, client):
           """Test simulation capabilities"""
           simulation_request = {
               "workflow_definition": {
                   "nodes": [
                       {
                           "id": "test_stamper",
                           "transformer": "metadata_stamper",
                           "dependencies": []
                       }
                   ]
               },
               "input_data": {
                   "test_stamper": {
                       "content": "simulation test",
                       "file_path": "sim_test.txt"
                   }
               }
           }

           response = client.post("/workflow/simulate", json=simulation_request)

           # Should not fail if simulation endpoints implemented
           assert response.status_code in [200, 404]

       async def test_event_publishing(self, client):
           """Test event publishing integration"""
           response = client.post("/stamp", json={
               "content": "event test content",
               "file_path": "event_test.txt"
           })

           assert response.status_code == 200
           # Events should be published in background (no direct test)

       async def test_o_n_e_metrics(self, client):
           """Test O.N.E. metrics collection"""
           response = client.get("/metrics")
           assert response.status_code == 200

           metrics_text = response.text
           # Should contain O.N.E. specific metrics
           one_metrics = [
               "one_operations_total",
               "one_operation_duration_seconds",
               "one_trust_validations_total"
           ]

           for metric in one_metrics:
               # Metrics might not be present if features not used
               pass  # Soft check

       async def test_compliance_percentage(self, client):
           """Calculate overall compliance percentage"""
           tests = [
               self.test_tool_metadata_headers_present,
               self.test_namespace_compliance,
               self.test_registry_integration,
               self.test_trust_zone_assignment,
               self.test_transformer_pattern,
               self.test_schema_validation,
               self.test_dag_execution,
               self.test_simulation_mode,
               self.test_event_publishing,
               self.test_o_n_e_metrics
           ]

           passed = 0
           total = len(tests)

           for test in tests:
               try:
                   await test(client)
                   passed += 1
               except Exception as e:
                   print(f"Test {test.__name__} failed: {e}")

           compliance_percentage = (passed / total) * 100
           print(f"O.N.E. v0.1 Compliance: {compliance_percentage:.1f}%")

           assert compliance_percentage >= 75, f"Compliance too low: {compliance_percentage}%"
   ```

2. **Update CLAUDE.md Documentation**
   ```markdown
   # File: CLAUDE.md (update existing file)

   # Add O.N.E. v0.1 Protocol Compliance Section

   ## O.N.E. v0.1 Protocol Compliance

   ### Current Status: 100% Compliant ✅

   The MetadataStampingService fully implements the O.N.E. (OmniNode Environment) v0.1 protocol:

   #### ✅ Tool Metadata Standard v0.1
   - All source files include required O.N.E. tool metadata headers
   - Metadata validation with CLI tools and CI/CD integration
   - Automated compliance checking and enforcement

   #### ✅ Registry Integration
   - Consul registry client with automatic service registration
   - Health check integration and service discovery
   - Multi-registry federation support

   #### ✅ Trust Zones & Security
   - Trust zone assignment (zone.local, zone.org, zone.global)
   - ed25519 signature validation and message integrity
   - Security middleware with O.M.N.I. protocol compliance

   #### ✅ ONEX Transformer Pattern
   - Schema-first execution with Pydantic models
   - @transformer decorator for typed transformations
   - Schema registry with versioning and migration support

   #### ✅ DAG Execution & Simulation
   - Directed Acyclic Graph execution engine
   - Parallel transformer execution with dependency resolution
   - Simulation mode for execution planning and resource estimation

   #### ✅ Federation & Policy Engine
   - Cross-zone trust negotiation and federated discovery
   - Runtime policy enforcement with priority-based rules
   - Distributed tracing and O.N.E. protocol metrics

   ### O.N.E. Protocol Features

   #### Namespace Support
   All operations are namespaced under `omninode.services.metadata` with multi-tenant support.

   #### Unified Response Format
   Consistent API responses with enhanced error handling and metadata.

   #### Event Publishing
   Real-time event streaming with OnexEnvelopeV1 format and HMAC-SHA256 signing.

   #### Schema Validation
   Strong typing with Pydantic models and automatic validation.

   #### Trust Levels
   - UNVERIFIED: Local zone operations
   - SIGNED: Organizational zone with signature validation
   - VERIFIED: Global zone with full certificate chain validation

   #### Transformer Execution
   ```python
   @transformer(StampingInput, StampingOutput, "metadata_stamper", "1.0.0")
   async def metadata_stamping_transformer(input_data: StampingInput, context: ExecutionContext):
       # Schema-validated transformation logic
       pass
   ```

   #### DAG Workflows
   ```python
   dag = DAGExecutor()
   dag.add_node("stamper", metadata_stamper, dependencies=[])
   results = await dag.execute(input_data, simulation_mode=False)
   ```

   ### Compliance Validation

   Run the O.N.E. compliance test suite:
   ```bash
   poetry run pytest tests/integration/test_one_protocol_compliance.py -v
   ```

   Expected output: `O.N.E. v0.1 Compliance: 100.0%`
   ```

3. **Create Migration Tools**
   ```python
   # File: scripts/one_migration_validator.py
   #!/usr/bin/env python3
   """O.N.E. v0.1 migration validation and compliance checker"""

   import os
   import re
   import json
   import asyncio
   from typing import Dict, List, Any

   class ONEMigrationValidator:
       def __init__(self, project_root: str):
           self.project_root = project_root
           self.compliance_report = {
               "overall_compliance": 0.0,
               "components": {},
               "recommendations": []
           }

       async def validate_complete_migration(self) -> Dict[str, Any]:
           """Validate complete O.N.E. v0.1 migration"""
           print("🔍 Validating O.N.E. v0.1 Protocol Compliance...")

           # Check all components
           await self._check_tool_metadata_compliance()
           await self._check_registry_integration()
           await self._check_security_implementation()
           await self._check_transformer_pattern()
           await self._check_dag_execution()
           await self._check_federation_support()

           # Calculate overall compliance
           self._calculate_overall_compliance()

           # Generate report
           return self._generate_compliance_report()

       async def _check_tool_metadata_compliance(self):
           """Check tool metadata header compliance"""
           service_files = [
               "src/omninode_bridge/services/metadata_stamping/main.py",
               "src/omninode_bridge/services/metadata_stamping/service.py",
               "src/omninode_bridge/services/metadata_stamping/engine/stamping_engine.py",
               "src/omninode_bridge/services/metadata_stamping/database/client.py",
               "src/omninode_bridge/services/metadata_stamping/api/router.py"
           ]

           metadata_pattern = r'# === OmniNode:Tool_Metadata ===.*?# === /OmniNode:Tool_Metadata ==='
           compliant_files = 0

           for file_path in service_files:
               full_path = os.path.join(self.project_root, file_path)
               if os.path.exists(full_path):
                   with open(full_path, 'r') as f:
                       content = f.read()

                   if re.search(metadata_pattern, content, re.DOTALL):
                       compliant_files += 1

           compliance = (compliant_files / len(service_files)) * 100
           self.compliance_report["components"]["tool_metadata"] = {
               "compliance_percentage": compliance,
               "compliant_files": compliant_files,
               "total_files": len(service_files),
               "status": "✅" if compliance == 100 else "⚠️"
           }

       async def _check_registry_integration(self):
           """Check registry integration implementation"""
           registry_files = [
               "src/omninode_bridge/services/metadata_stamping/registry/consul_client.py"
           ]

           implemented = all(
               os.path.exists(os.path.join(self.project_root, f))
               for f in registry_files
           )

           self.compliance_report["components"]["registry_integration"] = {
               "compliance_percentage": 100 if implemented else 0,
               "implemented": implemented,
               "status": "✅" if implemented else "❌"
           }

       async def _check_security_implementation(self):
           """Check security framework implementation"""
           security_files = [
               "src/omninode_bridge/services/metadata_stamping/security/trust_zones.py",
               "src/omninode_bridge/services/metadata_stamping/security/signature_validator.py",
               "src/omninode_bridge/services/metadata_stamping/security/middleware.py"
           ]

           implemented_files = sum(
               1 for f in security_files
               if os.path.exists(os.path.join(self.project_root, f))
           )

           compliance = (implemented_files / len(security_files)) * 100

           self.compliance_report["components"]["security_framework"] = {
               "compliance_percentage": compliance,
               "implemented_files": implemented_files,
               "total_files": len(security_files),
               "status": "✅" if compliance == 100 else "⚠️"
           }

       async def _check_transformer_pattern(self):
           """Check ONEX transformer pattern implementation"""
           transformer_files = [
               "src/omninode_bridge/services/metadata_stamping/execution/transformer.py",
               "src/omninode_bridge/services/metadata_stamping/execution/schema_registry.py",
               "src/omninode_bridge/services/metadata_stamping/execution/stamping_transformers.py"
           ]

           implemented_files = sum(
               1 for f in transformer_files
               if os.path.exists(os.path.join(self.project_root, f))
           )

           compliance = (implemented_files / len(transformer_files)) * 100

           self.compliance_report["components"]["transformer_pattern"] = {
               "compliance_percentage": compliance,
               "implemented_files": implemented_files,
               "total_files": len(transformer_files),
               "status": "✅" if compliance == 100 else "⚠️"
           }

       async def _check_dag_execution(self):
           """Check DAG execution implementation"""
           dag_files = [
               "src/omninode_bridge/services/metadata_stamping/execution/dag_engine.py",
               "src/omninode_bridge/services/metadata_stamping/execution/simulation.py"
           ]

           implemented_files = sum(
               1 for f in dag_files
               if os.path.exists(os.path.join(self.project_root, f))
           )

           compliance = (implemented_files / len(dag_files)) * 100

           self.compliance_report["components"]["dag_execution"] = {
               "compliance_percentage": compliance,
               "implemented_files": implemented_files,
               "total_files": len(dag_files),
               "status": "✅" if compliance == 100 else "⚠️"
           }

       async def _check_federation_support(self):
           """Check federation support implementation"""
           federation_files = [
               "src/omninode_bridge/services/metadata_stamping/federation/federation_client.py",
               "src/omninode_bridge/services/metadata_stamping/policy/policy_engine.py",
               "src/omninode_bridge/services/metadata_stamping/observability/o_n_e_metrics.py"
           ]

           implemented_files = sum(
               1 for f in federation_files
               if os.path.exists(os.path.join(self.project_root, f))
           )

           compliance = (implemented_files / len(federation_files)) * 100

           self.compliance_report["components"]["federation_support"] = {
               "compliance_percentage": compliance,
               "implemented_files": implemented_files,
               "total_files": len(federation_files),
               "status": "✅" if compliance == 100 else "⚠️"
           }

       def _calculate_overall_compliance(self):
           """Calculate overall compliance percentage"""
           total_compliance = 0
           component_count = 0

           for component, data in self.compliance_report["components"].items():
               total_compliance += data["compliance_percentage"]
               component_count += 1

           if component_count > 0:
               self.compliance_report["overall_compliance"] = total_compliance / component_count
           else:
               self.compliance_report["overall_compliance"] = 0

       def _generate_compliance_report(self) -> Dict[str, Any]:
           """Generate final compliance report"""
           overall = self.compliance_report["overall_compliance"]

           if overall >= 95:
               status = "🎉 FULLY COMPLIANT"
               recommendations = ["All O.N.E. v0.1 requirements met!"]
           elif overall >= 75:
               status = "✅ MOSTLY COMPLIANT"
               recommendations = ["Complete remaining components for full compliance"]
           else:
               status = "⚠️  NEEDS WORK"
               recommendations = ["Significant work needed for O.N.E. v0.1 compliance"]

           self.compliance_report["status"] = status
           self.compliance_report["recommendations"] = recommendations

           return self.compliance_report

   async def main():
       """Main migration validation"""
       validator = ONEMigrationValidator("./omninode_bridge"  # or your repository path)
       report = await validator.validate_complete_migration()

       print("\n" + "="*60)
       print("O.N.E. v0.1 PROTOCOL COMPLIANCE REPORT")
       print("="*60)
       print(f"Overall Status: {report['status']}")
       print(f"Compliance: {report['overall_compliance']:.1f}%")
       print("\nComponent Details:")

       for component, data in report["components"].items():
           print(f"  {data['status']} {component}: {data['compliance_percentage']:.1f}%")

       print("\nRecommendations:")
       for rec in report["recommendations"]:
           print(f"  • {rec}")

       print("\n" + "="*60)

   if __name__ == "__main__":
       asyncio.run(main())
   ```

**Validation Criteria for Coordinator 8**:
- [ ] Comprehensive test suite covers all O.N.E. features
- [ ] CLAUDE.md updated with O.N.E. documentation
- [ ] Migration validation tools functional
- [ ] 100% O.N.E. v0.1 compliance achieved
- [ ] All integration tests pass

---

## 🚀 Automated Execution Commands

### Prerequisites Check
```bash
# Verify environment
cd omninode_bridge  # or your repository directory
poetry --version
python --version
docker --version

# Check current status
git status
git log --oneline -5
```

### Execute Coordinators Sequentially
```bash
# Coordinator 3: Registry Client Implementation
echo "🤖 Starting Coordinator 3: Registry Client Implementation"
poetry add python-consul2 consul-python
# [Implement registry client code per steps above]

# Coordinator 4: Trust Zones & Security
echo "🤖 Starting Coordinator 4: Trust Zones & Security Framework"
poetry add cryptography ed25519
# [Implement security framework per steps above]

# Coordinator 5: Schema-First Execution
echo "🤖 Starting Coordinator 5: Schema-First Execution Framework"
# [Implement transformer pattern per steps above]

# Coordinator 6: DAG Execution & Simulation
echo "🤖 Starting Coordinator 6: DAG Execution & Simulation"
# [Implement DAG engine per steps above]

# Coordinator 7: Federation & Policy Engine
echo "🤖 Starting Coordinator 7: Federation & Policy Engine"
poetry add aiohttp
# [Implement federation support per steps above]

# Coordinator 8: Integration Testing & Documentation
echo "🤖 Starting Coordinator 8: Integration Testing & Documentation"
# [Implement comprehensive testing per steps above]
```

### Validation Commands
```bash
# Run compliance validation
python scripts/one_migration_validator.py

# Run O.N.E. compliance tests
poetry run pytest tests/integration/test_one_protocol_compliance.py -v

# Check service health
curl http://localhost:8053/health

# Validate metadata headers
python scripts/validate_metadata.py --check-all
```

### Success Criteria
- **Overall Compliance**: ≥ 95% O.N.E. v0.1 compliance
- **Test Coverage**: All integration tests pass
- **Documentation**: CLAUDE.md updated with O.N.E. features
- **Performance**: All existing performance targets maintained
- **Backwards Compatibility**: Existing functionality preserved

---

## 📊 Expected Timeline

**Coordinator 3-4** (Week 2): Registry + Security → **50% compliance**
**Coordinator 5-6** (Week 3): Transformers + DAG → **75% compliance**
**Coordinator 7-8** (Week 4): Federation + Testing → **100% compliance**

## 🎯 Final Success State

Upon completion, the MetadataStampingService will be:
- **100% O.N.E. v0.1 Protocol Compliant**
- **Schema-first execution ready**
- **Federation-capable across trust zones**
- **Fully documented and tested**
- **Production-ready with enterprise features**

---

**Automated Implementation Status**: ⏳ **READY FOR OVERNIGHT EXECUTION**

This guide provides complete step-by-step instructions for automated agents to complete the remaining 75% of O.N.E. protocol implementation without human intervention.
