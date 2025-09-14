"""Kafka producer pool EFFECT node implementation."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, AsyncIterator
from uuid import uuid4

import aiokafka
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaConnectionError

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.nodes.base.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer

from omnibase_infra.models.kafka.model_kafka_producer_config import ModelKafkaProducerConfig
from omnibase_infra.models.kafka.model_kafka_producer_pool_stats import (
    ModelKafkaProducerPoolStats,
    ModelKafkaProducerStats,
    ModelKafkaTopicStats
)
from omnibase_infra.models.kafka.model_kafka_health_response import ModelKafkaHealthResponse

from .models import (
    ModelKafkaProducerPoolInput,
    ModelKafkaProducerPoolOutput,
    ModelSendMessageInput,
    ModelSendMessageOutput,
    ModelGetPoolStatsInput,
    ModelGetPoolStatsOutput,
    ModelGetHealthInput,
    ModelGetHealthOutput,
)


@dataclass
class ProducerInstance:
    """Internal representation of a producer instance in the pool."""
    producer_id: str
    producer: AIOKafkaProducer
    created_at: datetime
    last_activity: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0
    bytes_sent: int = 0
    is_active: bool = True
    last_error: Optional[str] = None


"""Kafka producer pool EFFECT node implementation."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, AsyncIterator
from uuid import uuid4

import aiokafka
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaConnectionError




@dataclass
class ProducerInstance:
    """Internal representation of a producer instance in the pool."""
    producer_id: str
    producer: AIOKafkaProducer
    created_at: datetime
    last_activity: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0
    bytes_sent: int = 0
    is_active: bool = True
    last_error: Optional[str] = None


class NodeKafkaProducerPoolEffect(NodeEffectService):
    """Kafka producer pool EFFECT node providing enterprise-grade producer connection pooling and message publishing."""

    def __init__(self, container: ModelONEXContainer):
        """Initialize Kafka producer pool EFFECT node.
        
        Args:
            container: ONEX container for dependency injection
        """
        super().__init__(container)
        self.config: Optional[ModelKafkaProducerConfig] = None
        self.pool_name: str = "infrastructure"
        self.min_pool_size: int = 2
        self.max_pool_size: int = 10
        
        # Internal state
        self.producers: Dict[str, ProducerInstance] = {}
        self.idle_producers: List[str] = []
        self.active_producers: List[str] = []
        self.failed_producers: List[str] = []
        
        # Statistics
        self.created_at = datetime.now()
        self.total_messages_sent = 0
        self.total_messages_failed = 0
        self.total_bytes_sent = 0
        self.topic_stats: Dict[str, Dict[str, any]] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        self.is_initialized = False
        
        # Backpressure and retry configuration
        self.max_wait_time = 30.0
        self.retry_delay = 0.1
        self.max_retry_delay = 2.0
        self.pending_requests: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.NodeKafkaProducerPoolEffect")

    async def _setup(self) -> None:
        """Set up the Kafka producer pool."""
        try:
            # Get configuration from container
            self.config = await self._get_config_from_container()
            
            # Create initial pool of producers
            async with self._lock:
                for i in range(self.min_pool_size):
                    producer_id = f"{self.pool_name}_producer_{i+1}_{uuid4().hex[:8]}"
                    await self._create_producer(producer_id)
                
                self.is_initialized = True
                self.logger.info(f"Kafka producer pool '{self.pool_name}' initialized with {len(self.producers)} producers")
                
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.NODE_SETUP_ERROR,
                message="Failed to initialize Kafka producer pool EFFECT node",
                details={"error": str(e)},
            ) from e

    async def _teardown(self) -> None:
        """Clean up the Kafka producer pool."""
        try:
            async with self._lock:
                # Stop all producers
                for producer_instance in list(self.producers.values()):
                    try:
                        await producer_instance.producer.stop()
                    except Exception as e:
                        self.logger.warning(f"Error stopping producer {producer_instance.producer_id}: {e}")
                
                # Clear all state
                self.producers.clear()
                self.idle_producers.clear()
                self.active_producers.clear()
                self.failed_producers.clear()
                
                self.is_initialized = False
                self.logger.info(f"Kafka producer pool '{self.pool_name}' shut down")
                
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.NODE_TEARDOWN_ERROR,
                message="Failed to teardown Kafka producer pool EFFECT node",
                details={"error": str(e)},
            ) from e

    async def _get_config_from_container(self) -> ModelKafkaProducerConfig:
        """Get Kafka producer configuration from container."""
        try:
            config = self.container.get_service("ModelKafkaProducerConfig")
            if config is None:
                raise OnexError(
                    code=CoreErrorCode.CONFIGURATION_ERROR,
                    message="Kafka producer configuration not found in container",
                )
            return config
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message="Failed to get Kafka producer configuration",
                details={"error": str(e)},
            ) from e

    async def _create_producer(self, producer_id: str) -> ProducerInstance:
        """Create a new producer instance."""
        try:
            producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=producer_id,
                acks=self.config.acks,
                retries=self.config.retries,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                buffer_memory=self.config.buffer_memory,
                compression_type=self.config.compression_type or "none",
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
                delivery_timeout_ms=self.config.delivery_timeout_ms,
                max_in_flight_requests_per_connection=self.config.max_in_flight_requests_per_connection,
                enable_idempotence=self.config.enable_idempotence
            )
            
            await producer.start()
            
            instance = ProducerInstance(
                producer_id=producer_id,
                producer=producer,
                created_at=datetime.now()
            )
            
            self.producers[producer_id] = instance
            self.idle_producers.append(producer_id)
            
            self.logger.debug(f"Created producer {producer_id}")
            return instance
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE,
                message=f"Failed to create Kafka producer {producer_id}: {str(e)}",
            ) from e

    @asynccontextmanager
    async def _acquire_producer(self) -> AsyncIterator[ProducerInstance]:
        """Acquire a producer from the pool."""
        producer_instance = None
        producer_id = None
        
        try:
            async with self._lock:
                # Try to get an idle producer
                if self.idle_producers:
                    producer_id = self.idle_producers.pop(0)
                    producer_instance = self.producers[producer_id]
                    self.active_producers.append(producer_id)
                    
                # If no idle producers, create a new one if under max limit
                elif len(self.producers) < self.max_pool_size:
                    producer_id = f"{self.pool_name}_producer_{len(self.producers)+1}_{uuid4().hex[:8]}"
                    producer_instance = await self._create_producer(producer_id)
                    self.active_producers.append(producer_id)
                    
                else:
                    # Pool is at capacity, wait for an available producer
                    raise OnexError(
                        code=CoreErrorCode.RESOURCE_EXHAUSTED,
                        message="Kafka producer pool at capacity",
                    )
            
            producer_instance.last_activity = datetime.now()
            yield producer_instance
            
        finally:
            if producer_id and producer_instance:
                async with self._lock:
                    if producer_id in self.active_producers:
                        self.active_producers.remove(producer_id)
                        self.idle_producers.append(producer_id)

    # io_operations implementation

    async def send_message(self, input_data: ModelSendMessageInput) -> ModelSendMessageOutput:
        """Send a message to a Kafka/RedPanda topic with producer pool management."""
        if not self.is_initialized:
            raise OnexError(
                code=CoreErrorCode.NODE_NOT_INITIALIZED,
                message="Kafka producer pool not initialized",
            )

        start_time = time.time()
        
        try:
            async with self._acquire_producer() as producer_instance:
                # Prepare message data
                message_value = input_data.value.encode('utf-8') if isinstance(input_data.value, str) else input_data.value
                message_key = input_data.key.encode('utf-8') if input_data.key else None
                
                # Send message
                record_metadata = await producer_instance.producer.send_and_wait(
                    topic=input_data.topic,
                    value=message_value,
                    key=message_key,
                    headers=input_data.headers,
                    partition=input_data.partition,
                    timestamp=input_data.timestamp
                )
                
                # Update statistics
                latency_ms = (time.time() - start_time) * 1000
                producer_instance.message_count += 1
                producer_instance.bytes_sent += len(message_value or b'')
                
                self.total_messages_sent += 1
                self.total_bytes_sent += len(message_value or b'')
                
                # Update topic statistics
                if input_data.topic not in self.topic_stats:
                    self.topic_stats[input_data.topic] = {
                        "message_count": 0,
                        "bytes_sent": 0,
                        "last_message_time": None,
                    }
                
                topic_stats = self.topic_stats[input_data.topic]
                topic_stats["message_count"] += 1
                topic_stats["bytes_sent"] += len(message_value or b'')
                topic_stats["last_message_time"] = datetime.now()
                
                return ModelSendMessageOutput(
                    success=True,
                    topic=record_metadata.topic,
                    partition=record_metadata.partition,
                    offset=record_metadata.offset,
                    timestamp=datetime.fromtimestamp(record_metadata.timestamp / 1000),
                    latency_ms=latency_ms,
                )

        except Exception as e:
            self.total_messages_failed += 1
            if producer_instance:
                producer_instance.error_count += 1
                producer_instance.last_error = str(e)
                
            raise OnexError(
                code=CoreErrorCode.INTEGRATION_SERVICE_ERROR,
                message=f"Failed to send message to topic {input_data.topic}: {str(e)}",
                details={"topic": input_data.topic},
            ) from e

    async def get_pool_stats(self, input_data: ModelGetPoolStatsInput) -> ModelGetPoolStatsOutput:
        """Get comprehensive statistics for the Kafka producer pool."""
        try:
            # Calculate overall stats
            total_producers = len(self.producers)
            active_producers = len(self.active_producers)
            idle_producers = len(self.idle_producers)
            failed_producers = len(self.failed_producers)
            
            # Get individual producer stats
            producer_stats = []
            for producer_instance in self.producers.values():
                producer_stat = ModelKafkaProducerStats(
                    producer_id=producer_instance.producer_id,
                    created_at=producer_instance.created_at,
                    last_activity=producer_instance.last_activity,
                    message_count=producer_instance.message_count,
                    error_count=producer_instance.error_count,
                    bytes_sent=producer_instance.bytes_sent,
                    is_active=producer_instance.is_active,
                    last_error=producer_instance.last_error,
                )
                producer_stats.append(producer_stat)
            
            # Get topic stats
            topic_stats = []
            for topic, stats in self.topic_stats.items():
                topic_stat = ModelKafkaTopicStats(
                    topic_name=topic,
                    message_count=stats["message_count"],
                    bytes_sent=stats["bytes_sent"],
                    last_message_time=stats["last_message_time"],
                )
                topic_stats.append(topic_stat)
            
            pool_stats = ModelKafkaProducerPoolStats(
                pool_name=self.pool_name,
                total_producers=total_producers,
                active_producers=active_producers,
                idle_producers=idle_producers,
                failed_producers=failed_producers,
                total_messages_sent=self.total_messages_sent,
                total_messages_failed=self.total_messages_failed,
                total_bytes_sent=self.total_bytes_sent,
                created_at=self.created_at,
                producer_stats=producer_stats,
                topic_stats=topic_stats,
            )
            
            return ModelGetPoolStatsOutput(
                success=True,
                stats=pool_stats,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.METRICS_COLLECTION_ERROR,
                message=f"Failed to get Kafka producer pool stats: {str(e)}",
            ) from e

    async def get_health(self, input_data: ModelGetHealthInput) -> ModelGetHealthOutput:
        """Get comprehensive health status of the Kafka producer pool."""
        try:
            if not self.is_initialized:
                health_data = ModelKafkaHealthResponse(
                    is_connected=False,
                    cluster_id="Unknown",
                    bootstrap_servers=self.config.bootstrap_servers if self.config else "Unknown",
                    total_producers=0,
                    active_producers=0,
                    failed_producers=0,
                    last_error="Producer pool not initialized",
                )
                
                return ModelGetHealthOutput(
                    success=False,
                    health_data=health_data,
                    timestamp=datetime.now(),
                )

            # Test basic connectivity
            total_producers = len(self.producers)
            active_producers = len(self.active_producers)
            failed_producers = len(self.failed_producers)
            
            # Check if we have at least one working producer
            has_working_producers = any(
                producer_instance.is_active and producer_instance.error_count == 0
                for producer_instance in self.producers.values()
            )
            
            health_data = ModelKafkaHealthResponse(
                is_connected=has_working_producers,
                cluster_id="RedPanda",  # Default for RedPanda integration
                bootstrap_servers=self.config.bootstrap_servers,
                total_producers=total_producers,
                active_producers=active_producers,
                failed_producers=failed_producers,
            )
            
            return ModelGetHealthOutput(
                success=has_working_producers,
                health_data=health_data,
                timestamp=datetime.now(),
            )

        except Exception as e:
            health_data = ModelKafkaHealthResponse(
                is_connected=False,
                cluster_id="Unknown",
                bootstrap_servers=self.config.bootstrap_servers if self.config else "Unknown",
                total_producers=0,
                active_producers=0,
                failed_producers=0,
                last_error=str(e),
            )
            
            return ModelGetHealthOutput(
                success=False,
                health_data=health_data,
                timestamp=datetime.now(),
            )