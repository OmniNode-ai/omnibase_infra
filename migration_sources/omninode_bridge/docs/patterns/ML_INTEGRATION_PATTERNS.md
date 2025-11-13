# ML Integration Patterns

**Extracted from:** PR #17 (feature/onextree-phase5-ai-ml-analytics)
**Date:** 2025-10-15
**Status:** Reference patterns for future production implementations

## Overview

This document preserves valuable ML/AI integration patterns from PR #17, which added comprehensive machine learning capabilities to the MetadataStampingService. While these features were not merged due to architectural misalignment, the patterns and approaches are valuable for future production services.

## Architecture Patterns

### 1. ML Model Server Pattern

**Pattern:** Centralized model serving with lifecycle management

**Key Components:**
```
┌─────────────────────────────────────────────────┐
│              ML Model Server                    │
│  ┌───────────────────────────────────────────┐  │
│  │  Model Registry                           │  │
│  │  - Version management                     │  │
│  │  - Lifecycle states (load/warm/active)    │  │
│  │  - A/B testing support                    │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Inference Engine                         │  │
│  │  - Batch prediction                       │  │
│  │  - Model caching                          │  │
│  │  - GPU acceleration support               │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Performance Monitoring                   │  │
│  │  - Latency tracking                       │  │
│  │  - Model drift detection                  │  │
│  │  - Health checks                          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Lifecycle States:**
1. **LOAD**: Model loading from disk/registry
2. **WARM**: Model warming (initial predictions)
3. **ACTIVE**: Serving predictions
4. **DEGRADED**: Performance degradation detected
5. **ERROR**: Model errors, fallback to previous version

**Benefits:**
- Centralized model management
- Hot swapping without downtime
- Performance monitoring per model
- Graceful degradation strategies

**Implementation Considerations:**
- Model versioning strategy
- Storage for model artifacts (S3, GCS, local)
- Caching strategy for predictions
- Resource management (CPU/GPU)

### 2. Multi-Algorithm Anomaly Detection

**Pattern:** Multiple detection algorithms with ensemble scoring

**Key Components:**
```
Input Data → Feature Extraction
              ↓
      ┌──────┴──────┬──────────┬──────────┬──────────┐
      ▼             ▼          ▼          ▼          ▼
Isolation      Autoencoder   LOF     Z-score    Moving
Forest                                           Average
      │             │          │          │          │
      └──────┬──────┴──────────┴──────────┴──────────┘
             ▼
      Ensemble Scoring
      (Weighted average)
             ▼
      Anomaly Classification
      (Statistical, Behavioral, Temporal, etc.)
             ▼
      Alert Generation
      (Severity: Low, Medium, High, Critical)
```

**Algorithms:**
1. **Isolation Forest**: Outlier detection for statistical anomalies
2. **Autoencoder**: Deep learning-based detection for complex patterns
3. **Local Outlier Factor (LOF)**: Density-based anomaly detection
4. **Z-score**: Statistical deviation detection
5. **Moving Average**: Temporal pattern deviation

**Anomaly Types:**
- Statistical: Outliers in metric distributions
- Behavioral: Unusual usage patterns
- Temporal: Time-based pattern deviations
- Structural: Architectural anomalies
- Content: Data content anomalies
- Performance: Performance metric anomalies
- Security: Security-related anomalies

**Benefits:**
- Multi-algorithm approach reduces false positives
- Different algorithms catch different anomaly types
- Ensemble scoring provides confidence levels
- Adaptive thresholds reduce alert fatigue

**Implementation Considerations:**
- Algorithm selection based on data characteristics
- Training data requirements (normal vs anomalous)
- Real-time vs batch detection
- False positive reduction techniques

### 3. Predictive Analytics Pipeline

**Pattern:** Time series forecasting with adaptive model selection

**Key Components:**
```
Historical Data → Feature Engineering → Model Selection
                                             ↓
                                    ┌────────┴────────┐
                                    │                 │
                              Linear Regression   ARIMA
                                    │                 │
                                    │        LSTM / Prophet
                                    │                 │
                                    │      Gradient Boosting
                                    │                 │
                                    └────────┬────────┘
                                             ▼
                                    Model Evaluation
                                             ▼
                                    Best Model Selection
                                             ▼
                                    Prediction Generation
                                    (with confidence intervals)
                                             ▼
                                    What-If Scenario Analysis
```

**Models:**
1. **Linear Regression**: Simple trend analysis
2. **ARIMA**: Time series patterns (seasonality, trend)
3. **LSTM**: Complex temporal dependencies (deep learning)
4. **Prophet**: Seasonal patterns with holidays
5. **Gradient Boosting**: Feature-rich predictions (XGBoost, LightGBM)

**Prediction Types:**
- Performance forecasting (latency, throughput)
- Resource utilization prediction (CPU, memory, disk)
- Capacity planning (scaling decisions)
- Error rate prediction
- System health scoring
- Bottleneck detection

**Benefits:**
- Adaptive model selection based on data characteristics
- Confidence intervals for uncertainty quantification
- What-if scenario simulation for planning
- Feature importance analysis for insights

**Implementation Considerations:**
- Lookback window size (how much history to use)
- Forecast horizon (how far to predict)
- Model retraining frequency
- Handling missing data and outliers
- Feature engineering strategies

### 4. Computer Vision Integration

**Pattern:** Multi-modal image analysis with pre-trained models

**Key Components:**
```
Image Input → Pre-processing
               ↓
      ┌────────┴────────┬────────────┬──────────┐
      ▼                 ▼            ▼          ▼
Image               Object      Segmentation  OCR
Classification      Detection                (Tesseract)
(ResNet, VGG)      (YOLO, R-CNN)
      │                 │            │          │
      └────────┬────────┴────────────┴──────────┘
               ▼
      Feature Extraction
               ↓
      Quality Assessment
               ↓
      Content Moderation
               ↓
      Metadata Enrichment
```

**Tasks:**
1. **Image Classification**: Pre-trained models (ResNet, VGG, EfficientNet)
2. **Object Detection**: YOLO, Faster R-CNN for object localization
3. **Image Segmentation**: Semantic segmentation for detailed analysis
4. **OCR**: Text extraction from images (Tesseract, EasyOCR)
5. **Feature Extraction**: Embeddings for similarity search
6. **Quality Assessment**: Blur detection, resolution analysis
7. **Content Moderation**: Safety checks, NSFW detection

**Benefits:**
- Pre-trained models reduce training requirements
- Multi-task capability for comprehensive analysis
- Feature extraction enables similarity search
- Quality assessment ensures input validity

**Implementation Considerations:**
- Model selection (accuracy vs speed tradeoff)
- GPU acceleration requirements
- Batch processing for throughput
- Custom model fine-tuning for domain-specific tasks
- Image preprocessing (resizing, normalization)

### 5. NLP Processing Pipeline

**Pattern:** Transformer-based NLP with multiple tasks

**Key Components:**
```
Text Input → Tokenization
              ↓
      Pre-trained Transformer Models
      (BERT, RoBERTa, DistilBERT)
              ↓
      ┌────────┴────────┬────────┬────────┬────────┐
      ▼                 ▼        ▼        ▼        ▼
Text          Named Entity  Sentiment  Topic    Keyword
Classification Recognition  Analysis   Modeling  Extraction
      │                 │        │        │        │
      └────────┬────────┴────────┴────────┴────────┘
               ▼
      Semantic Embeddings
               ↓
      Summarization
               ↓
      Language Detection
               ↓
      Intent Classification
```

**Tasks:**
1. **Text Classification**: Category assignment (spam, sentiment, topic)
2. **Named Entity Recognition (NER)**: Extract entities (person, org, location)
3. **Sentiment Analysis**: Fine-grained sentiment scoring
4. **Topic Modeling**: Unsupervised topic discovery (LDA, NMF)
5. **Keyword Extraction**: Importance ranking (TF-IDF, RAKE, TextRank)
6. **Text Summarization**: Extractive and abstractive summaries
7. **Language Detection**: Multi-language support
8. **Semantic Similarity**: Document/sentence similarity
9. **Intent Classification**: Query intent detection

**Benefits:**
- Transformer models provide state-of-the-art accuracy
- Pre-trained models reduce training data requirements
- Multi-task capability for comprehensive analysis
- Embeddings enable semantic search

**Implementation Considerations:**
- Model size vs speed tradeoff (BERT vs DistilBERT)
- GPU requirements for transformer models
- Token length limitations (512 tokens for BERT)
- Language-specific models for non-English text
- Fine-tuning for domain-specific tasks

### 6. Analytics Dashboard Pattern

**Pattern:** Real-time metrics aggregation with visualization

**Key Components:**
```
┌──────────────────────────────────────────────┐
│         Data Ingestion Layer                 │
│  - Metadata stamps                           │
│  - ML predictions                            │
│  - Anomaly events                            │
│  - System metrics                            │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│      Real-Time Aggregation Engine            │
│  - Time-series aggregation                   │
│  - Moving window calculations                │
│  - Percentile calculations                   │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│         Visualization Layer                  │
│  ┌────────────────────────────────────────┐  │
│  │  ML Model Performance                  │  │
│  │  - Accuracy trends                     │  │
│  │  - Latency distribution                │  │
│  │  - Model comparison                    │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Anomaly Detection                     │  │
│  │  - Timeline view                       │  │
│  │  - Severity heatmap                    │  │
│  │  - Root cause analysis                 │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Predictive Analytics                  │  │
│  │  - Forecast accuracy                   │  │
│  │  - Capacity planning                   │  │
│  │  - Resource predictions                │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**Metrics Tracked:**
1. **ML Model Performance**:
   - Accuracy, precision, recall, F1-score
   - Inference latency (p50, p95, p99)
   - Model comparison charts
   - Drift detection metrics

2. **Anomaly Detection**:
   - Anomaly timeline with events
   - Severity distribution (low, medium, high, critical)
   - Root cause analysis graphs
   - False positive rate tracking

3. **Predictive Analytics**:
   - Forecast accuracy trends
   - Capacity planning charts (CPU, memory, disk)
   - Resource utilization predictions
   - Cost forecasts

4. **System Insights**:
   - Performance trends (throughput, latency)
   - Bottleneck identification
   - Error analysis and debugging
   - User behavior patterns

**Benefits:**
- Real-time visibility into ML system performance
- Early detection of model drift and degradation
- Data-driven decision making for scaling
- Historical trend analysis for planning

**Implementation Considerations:**
- Data retention policies (raw vs aggregated)
- Aggregation window sizes (1min, 5min, 1hour)
- Real-time update frequency
- Alert integration for critical events
- Custom report generation

## ML Infrastructure Patterns

### 7. Model Registry Pattern

**Pattern:** Centralized model artifact management

**Components:**
- **Model Storage**: S3/GCS for model files
- **Metadata Store**: PostgreSQL for model metadata
- **Version Control**: Semantic versioning (1.0.0, 1.1.0, 2.0.0)
- **Lineage Tracking**: Training data, parameters, metrics
- **Access Control**: RBAC for model deployment

**Metadata Schema:**
```json
{
  "model_id": "anomaly_detector_v2",
  "version": "2.0.0",
  "model_type": "isolation_forest",
  "framework": "scikit-learn",
  "trained_at": "2025-10-01T10:00:00Z",
  "training_data": {
    "dataset_id": "training_data_v1",
    "samples": 100000,
    "features": 15
  },
  "hyperparameters": {
    "n_estimators": 100,
    "contamination": 0.1,
    "max_samples": 256
  },
  "metrics": {
    "precision": 0.92,
    "recall": 0.89,
    "f1_score": 0.90
  },
  "artifact_uri": "s3://models/anomaly_detector_v2.pkl",
  "status": "production"
}
```

### 8. Feature Store Pattern

**Pattern:** Centralized feature engineering and storage

**Components:**
- **Feature Extraction**: Consistent feature engineering logic
- **Feature Storage**: Redis/PostgreSQL for feature caching
- **Feature Versioning**: Track feature schema changes
- **Feature Monitoring**: Data quality and drift monitoring

**Benefits:**
- Consistent feature engineering across training and serving
- Reduce feature computation latency
- Enable feature reuse across models
- Track feature lineage and impact

### 9. Experiment Tracking Pattern

**Pattern:** MLflow-style experiment tracking

**Components:**
- **Run Tracking**: Log model training runs
- **Parameter Logging**: Hyperparameters and configurations
- **Metric Logging**: Training and validation metrics
- **Artifact Storage**: Model checkpoints and outputs

**Integration Points:**
- Model registry for production deployment
- Dashboard for visualization
- A/B testing framework for model comparison

### 10. A/B Testing Pattern

**Pattern:** Multi-variant model testing in production

**Components:**
- **Traffic Splitting**: Route % traffic to each model variant
- **Metric Collection**: Compare performance across variants
- **Statistical Significance**: Confidence intervals, p-values
- **Automated Rollout**: Promote winning variant

**Implementation:**
```python
async def predict_with_ab_testing(features):
    variant = select_variant(user_id, experiment_id)

    if variant == "A":
        model = model_server.get_model("anomaly_v1")
    else:
        model = model_server.get_model("anomaly_v2")

    prediction = await model.predict(features)

    # Log for A/B analysis
    await log_prediction(
        experiment_id=experiment_id,
        variant=variant,
        features=features,
        prediction=prediction,
        latency=prediction.latency
    )

    return prediction
```

## Performance Optimization Patterns

### 11. Model Quantization

**Pattern:** Reduce model size and inference latency

**Techniques:**
- **INT8 Quantization**: 32-bit floats → 8-bit integers (4x compression)
- **Dynamic Quantization**: Quantize weights, keep activations in float
- **Quantization-Aware Training**: Train with quantization simulation

**Benefits:**
- 2-4x faster inference
- 4x smaller model size
- Lower memory usage
- Enable edge deployment

### 12. Batch Prediction

**Pattern:** Process multiple predictions in a single batch

**Implementation:**
```python
async def batch_predict(inputs: List[Features], batch_size: int = 32):
    batches = chunk_list(inputs, batch_size)
    results = []

    for batch in batches:
        # GPU-accelerated batch prediction
        batch_results = await model.predict_batch(batch)
        results.extend(batch_results)

    return results
```

**Benefits:**
- 3-5x throughput improvement
- Better GPU utilization
- Amortize model loading overhead

### 13. Model Caching

**Pattern:** Cache predictions for frequent inputs

**Implementation:**
```python
@cache(ttl=3600, key_func=lambda x: hash(x.features))
async def predict_with_cache(features: Features):
    return await model.predict(features)
```

**Benefits:**
- Sub-millisecond latency for cached predictions
- Reduce GPU/CPU usage
- Handle traffic spikes

**Considerations:**
- Cache invalidation strategy
- Memory usage vs hit rate tradeoff
- Stale prediction tolerance

## Data Pipeline Patterns

### 14. Feature Extraction from Metadata Stamps

**Pattern:** Extract ML features from metadata stamps

**Feature Categories:**
1. **Temporal Features**:
   - Timestamp features (hour, day, day_of_week)
   - Time since last stamp
   - Stamp frequency

2. **Content Features**:
   - File size (raw, log-scaled)
   - File type encoding
   - Content hash patterns

3. **Performance Features**:
   - Hash generation time
   - API latency
   - Concurrent requests

4. **Behavioral Features**:
   - User stamp patterns
   - Namespace patterns
   - Stamp clustering

5. **Context Features**:
   - Geographical features (if available)
   - Client type
   - API version

**Implementation:**
```python
def extract_features(stamp: MetadataStamp) -> Features:
    return Features(
        # Temporal
        hour=stamp.created_at.hour,
        day_of_week=stamp.created_at.weekday(),
        time_since_last_stamp=calculate_time_delta(stamp),

        # Content
        file_size_log=math.log(stamp.file_size + 1),
        file_type=encode_file_type(stamp.file_type),

        # Performance
        hash_time_ms=stamp.hash_generation_time_ms,
        api_latency_ms=stamp.api_latency_ms,

        # Behavioral
        stamps_per_hour=calculate_stamp_frequency(stamp),
        namespace_entropy=calculate_namespace_entropy(stamp)
    )
```

### 15. Data Preprocessing Pipeline

**Pattern:** Consistent data preprocessing for training and serving

**Steps:**
1. **Data Validation**: Check for missing values, outliers
2. **Data Cleaning**: Handle missing data, remove duplicates
3. **Feature Scaling**: Standardization or normalization
4. **Feature Encoding**: One-hot, label encoding for categorical
5. **Feature Selection**: Remove low-importance features
6. **Data Augmentation**: Generate synthetic training data

**Implementation:**
```python
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def fit(self, data: DataFrame):
        self.scaler.fit(data[numeric_features])
        self.encoder.fit(data[categorical_features])

    def transform(self, data: DataFrame) -> Features:
        numeric_scaled = self.scaler.transform(data[numeric_features])
        categorical_encoded = self.encoder.transform(data[categorical_features])

        return combine_features(numeric_scaled, categorical_encoded)
```

### 16. Streaming Data Processing

**Pattern:** Real-time feature extraction from Kafka streams

**Architecture:**
```
Kafka Topic → Feature Extractor → Feature Store
     ↓                                  ↓
   (Stamps)                        (Features)
     ↓                                  ↓
Anomaly Detector ←──────────────────────┘
     ↓
Alert Manager
```

**Implementation:**
```python
async def process_stream():
    async for message in kafka_consumer:
        stamp = parse_metadata_stamp(message.value)

        # Extract features
        features = extract_features(stamp)

        # Store in feature store
        await feature_store.write(stamp.id, features)

        # Real-time anomaly detection
        anomaly = await anomaly_detector.detect_real_time(features)

        if anomaly.is_anomalous:
            await alert_manager.send_alert(anomaly)
```

**Benefits:**
- Real-time ML predictions
- Low latency (<100ms)
- Scalable with Kafka partitions

## Integration Patterns

### 17. Intelligent Caching with ML

**Pattern:** Predict cache access patterns and pre-warm cache

**Use Case:**
```python
# Predict which stamps will be accessed in next hour
predictions = await analytics.predict_cache_access_patterns(
    lookback_window=timedelta(days=7),
    forecast_horizon=timedelta(hours=1)
)

# Pre-warm cache with predicted stamps
for stamp_id in predictions.top_stamps:
    await cache_manager.warm_cache(stamp_id)
```

**Benefits:**
- Reduced cache misses
- Improved API latency
- Better resource utilization

### 18. Anomaly-Driven Alerting

**Pattern:** Real-time anomaly detection with intelligent alerting

**Implementation:**
```python
# Detect unusual patterns in hash operations
anomalies = await anomaly_system.detect_real_time({
    "hash_generation_time_ms": 150,  # Usually ~2ms
    "file_size_bytes": 1024000,
    "concurrent_requests": 500
})

if anomalies.is_anomalous:
    # Enrich anomaly with context
    anomaly_context = await gather_anomaly_context(anomalies)

    # Route to appropriate responder
    if anomalies.severity == "CRITICAL":
        await pager_duty.send_alert(anomaly_context)
    else:
        await slack.send_notification(anomaly_context)
```

**Benefits:**
- Proactive issue detection
- Reduce MTTR (mean time to resolution)
- Context-aware alerting reduces noise

### 19. Performance Prediction for Scaling

**Pattern:** Predict system load and scale proactively

**Use Case:**
```python
# Predict system load for next 7 days
forecast = await analytics.forecast_system_load(
    metric="requests_per_second",
    horizon=timedelta(days=7)
)

# Check if predicted load exceeds capacity
if forecast.predicted_max > capacity_threshold:
    # Calculate required replicas
    required_replicas = calculate_required_replicas(
        current_replicas=5,
        current_capacity=1000,
        predicted_load=forecast.predicted_max
    )

    # Trigger auto-scaling
    await scaling_manager.scale_up(
        replicas=required_replicas,
        reason=f"Predicted load: {forecast.predicted_max} RPS"
    )
```

**Benefits:**
- Proactive scaling before load spikes
- Cost optimization (scale down when predicted low load)
- Avoid service degradation

### 20. Content Analysis for Metadata Enrichment

**Pattern:** Use CV/NLP to enrich metadata stamps

**Use Case:**
```python
# Analyze image content
vision_result = await vision_system.analyze_image(
    image_data=image_bytes,
    tasks=["classification", "object_detection", "ocr"]
)

# Update metadata stamp with ML-extracted metadata
await metadata_service.enrich_stamp(
    stamp_id=stamp.id,
    enrichment={
        "image_labels": vision_result.labels,
        "detected_objects": vision_result.objects,
        "extracted_text": vision_result.ocr_text,
        "image_quality_score": vision_result.quality_score
    }
)
```

**Benefits:**
- Automatic metadata enrichment
- Enable content-based search
- Improve stamp quality

## Deployment Patterns

### 21. Blue-Green Model Deployment

**Pattern:** Zero-downtime model updates

**Steps:**
1. Deploy new model version (green)
2. Run shadow testing (green predictions without serving)
3. Compare green vs blue performance
4. Switch traffic from blue to green
5. Monitor for regressions
6. Keep blue as fallback

**Implementation:**
```python
async def deploy_model_blue_green(new_model_path: str):
    # Load new model (green)
    green_model = await model_server.load_model(new_model_path, version="green")

    # Shadow testing (100% traffic to blue, log green predictions)
    await run_shadow_testing(
        blue_model=model_server.get_model("blue"),
        green_model=green_model,
        duration=timedelta(hours=1)
    )

    # Compare performance
    comparison = await compare_model_performance("blue", "green")

    if comparison.green_is_better:
        # Switch traffic to green
        await model_server.set_active_model("green")

        # Monitor for regressions
        await monitor_model_performance(
            model="green",
            duration=timedelta(hours=24)
        )
    else:
        await model_server.remove_model("green")
```

### 22. Canary Model Deployment

**Pattern:** Gradual traffic shift to new model

**Steps:**
1. Deploy new model
2. Route 5% traffic to new model
3. Monitor metrics (accuracy, latency, errors)
4. Gradually increase to 10%, 25%, 50%, 100%
5. Rollback if issues detected

**Traffic Split Configuration:**
```python
CANARY_STAGES = [
    {"percentage": 5, "duration": timedelta(hours=1)},
    {"percentage": 10, "duration": timedelta(hours=2)},
    {"percentage": 25, "duration": timedelta(hours=4)},
    {"percentage": 50, "duration": timedelta(hours=8)},
    {"percentage": 100, "duration": None}
]
```

### 23. Model Fallback Strategy

**Pattern:** Graceful degradation when model fails

**Fallback Hierarchy:**
1. **Primary Model**: Latest production model
2. **Secondary Model**: Previous stable model
3. **Rule-Based Fallback**: Simple heuristics
4. **Default Prediction**: Conservative default

**Implementation:**
```python
async def predict_with_fallback(features: Features):
    try:
        # Try primary model
        return await primary_model.predict(features)
    except ModelError:
        logger.warning("Primary model failed, falling back to secondary")

        try:
            # Try secondary model
            return await secondary_model.predict(features)
        except ModelError:
            logger.error("Secondary model failed, using rule-based fallback")

            # Rule-based fallback
            return rule_based_predictor.predict(features)
```

## Monitoring and Observability Patterns

### 24. ML Model Observability

**Metrics to Track:**

1. **Prediction Metrics**:
   - Prediction latency (p50, p95, p99)
   - Prediction throughput (predictions/sec)
   - Batch prediction efficiency

2. **Accuracy Metrics**:
   - Online accuracy (when ground truth available)
   - Precision, recall, F1-score
   - Confusion matrix

3. **Data Quality Metrics**:
   - Input data distribution drift
   - Feature value ranges
   - Missing feature rates

4. **Model Health Metrics**:
   - Model loading time
   - Memory usage
   - GPU utilization
   - Error rates

5. **Business Metrics**:
   - Impact on key business metrics
   - Cost per prediction
   - Value delivered by ML

**Monitoring Dashboard:**
```
┌─────────────────────────────────────────────┐
│  Model: anomaly_detector_v2                 │
│  Status: ✅ HEALTHY                         │
│  Version: 2.0.0                             │
│  Uptime: 7 days                             │
├─────────────────────────────────────────────┤
│  Prediction Latency                         │
│  p50: 12ms  p95: 45ms  p99: 78ms           │
│  ──────────────────────────────────────     │
│  Throughput: 1,250 predictions/sec          │
│  ──────────────────────────────────────     │
│  Accuracy: 91.2% (↑ 1.5% vs baseline)      │
│  ──────────────────────────────────────     │
│  Data Drift: ⚠️ DETECTED (feature_7)       │
│  ──────────────────────────────────────     │
│  Memory: 2.4 GB / 4.0 GB (60%)             │
│  GPU: 45%                                   │
└─────────────────────────────────────────────┘
```

### 25. Data Drift Detection

**Pattern:** Monitor input data distribution changes

**Techniques:**
1. **Statistical Tests**: Kolmogorov-Smirnov test, Chi-squared test
2. **Distribution Comparison**: KL divergence, Wasserstein distance
3. **Feature Drift**: Track feature value distributions over time

**Implementation:**
```python
async def detect_data_drift(current_data: DataFrame, reference_data: DataFrame):
    drift_report = {}

    for feature in current_data.columns:
        # Calculate statistical distance
        ks_statistic, p_value = ks_2samp(
            reference_data[feature],
            current_data[feature]
        )

        # Detect drift (p-value < 0.05 indicates drift)
        drift_detected = p_value < 0.05

        drift_report[feature] = {
            "drift_detected": drift_detected,
            "ks_statistic": ks_statistic,
            "p_value": p_value
        }

        if drift_detected:
            logger.warning(f"Data drift detected in feature {feature}")
            await alert_manager.send_drift_alert(feature, drift_report[feature])

    return drift_report
```

**Benefits:**
- Early detection of model degradation
- Trigger model retraining
- Investigate root cause of data changes

## Cost Optimization Patterns

### 26. Model Compression

**Techniques:**
1. **Pruning**: Remove unimportant weights
2. **Knowledge Distillation**: Train smaller student model from large teacher
3. **Low-Rank Factorization**: Decompose weight matrices
4. **Quantization**: Reduce precision (covered earlier)

**Implementation:**
```python
# Knowledge distillation
teacher_model = load_large_model("anomaly_v2_large")
student_model = SmallModel()

for batch in training_data:
    # Get teacher predictions (soft targets)
    teacher_predictions = teacher_model.predict(batch)

    # Train student to match teacher
    loss = distillation_loss(
        student_model.predict(batch),
        teacher_predictions,
        temperature=3.0
    )

    student_model.update(loss)

# Deploy compressed student model
await model_server.deploy_model(student_model, version="v2_compressed")
```

**Benefits:**
- 10-100x smaller model size
- 2-10x faster inference
- Lower infrastructure costs
- Enable edge deployment

### 27. Batch Prediction for Cost Efficiency

**Pattern:** Batch non-urgent predictions for better resource utilization

**Use Cases:**
- Daily report generation
- Bulk data processing
- Non-real-time analytics

**Implementation:**
```python
# Schedule batch prediction jobs
@schedule(cron="0 2 * * *")  # Run at 2 AM daily
async def batch_prediction_job():
    # Fetch data to process
    pending_data = await fetch_pending_predictions()

    # Batch predict with larger batch sizes
    results = await model.predict_batch(
        pending_data,
        batch_size=256  # Larger batch for better GPU utilization
    )

    # Store results
    await store_predictions(results)
```

**Benefits:**
- Better GPU/CPU utilization (80%+ vs 30-40% for real-time)
- Lower cost per prediction
- Predictable resource usage

## Testing Patterns

### 28. ML Model Testing

**Test Types:**

1. **Unit Tests**: Test individual model components
   ```python
   def test_feature_extraction():
       stamp = create_test_stamp()
       features = extract_features(stamp)
       assert features.file_size_log > 0
       assert 0 <= features.hour < 24
   ```

2. **Integration Tests**: Test model serving pipeline
   ```python
   async def test_model_prediction_pipeline():
       features = create_test_features()
       prediction = await model_server.predict(features)
       assert prediction.confidence > 0
       assert prediction.latency_ms < 100
   ```

3. **Performance Tests**: Validate latency requirements
   ```python
   async def test_prediction_latency():
       features = create_test_features()
       start = time.time()
       await model.predict(features)
       latency = (time.time() - start) * 1000
       assert latency < 50  # 50ms SLA
   ```

4. **Accuracy Tests**: Validate model accuracy on test set
   ```python
   def test_model_accuracy():
       X_test, y_test = load_test_data()
       predictions = model.predict(X_test)
       accuracy = calculate_accuracy(predictions, y_test)
       assert accuracy > 0.85  # 85% minimum accuracy
   ```

5. **Data Quality Tests**: Validate input data quality
   ```python
   def test_input_data_quality():
       features = create_test_features()
       assert not has_missing_values(features)
       assert features_in_valid_range(features)
       assert no_data_type_mismatch(features)
   ```

### 29. Shadow Testing Pattern

**Pattern:** Test new model without impacting production

**Implementation:**
```python
async def shadow_test_new_model(features: Features):
    # Get production prediction
    prod_prediction = await prod_model.predict(features)

    # Get shadow prediction (not served to user)
    shadow_prediction = await shadow_model.predict(features)

    # Log both for comparison
    await log_shadow_comparison(
        features=features,
        prod_prediction=prod_prediction,
        shadow_prediction=shadow_prediction
    )

    # Serve production prediction
    return prod_prediction
```

**Benefits:**
- Zero risk to production
- Collect real-world performance data
- Compare models on production traffic
- Validate model before full deployment

## Key Takeaways

### For Production Implementations

1. **Start Simple**: Begin with simple models (Linear Regression, Logistic Regression) before complex deep learning
2. **Measure Everything**: Track latency, accuracy, drift, and business metrics
3. **Plan for Failure**: Implement fallback strategies and graceful degradation
4. **Monitor Continuously**: ML models degrade over time, monitor and retrain
5. **Optimize for Production**: Model compression, batch prediction, caching
6. **Test Thoroughly**: Unit, integration, performance, and accuracy tests

### Critical Success Factors

- **Clear Business Value**: Ensure ML solves real business problems
- **Data Quality**: ML is only as good as the data
- **Infrastructure**: Robust ML infrastructure (model registry, serving, monitoring)
- **Team Skills**: ML engineering requires both ML and software engineering expertise
- **Iteration**: Plan for multiple model versions and continuous improvement

### Common Pitfalls to Avoid

- **Premature Optimization**: Start with simple models, add complexity only when needed
- **Overfitting**: Validate on held-out test set, use cross-validation
- **Data Leakage**: Ensure no future data leaks into training
- **Ignoring Latency**: Production models must meet latency SLAs
- **Forgetting Retraining**: Models degrade, schedule regular retraining
- **Lack of Monitoring**: Without monitoring, you won't know when models fail

## References

### Key Technologies (from PR #17)

- **scikit-learn**: ML algorithms (anomaly detection, regression)
- **TensorFlow/PyTorch**: Deep learning (autoencoders, LSTM)
- **Transformers**: NLP (BERT, RoBERTa, DistilBERT)
- **OpenCV**: Computer vision
- **Prophet**: Time series forecasting
- **MLflow**: Experiment tracking and model registry

### Further Reading

- **ML System Design**: "Designing Machine Learning Systems" by Chip Huyen
- **ML in Production**: "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- **MLOps**: "Introducing MLOps" by Mark Treveil
- **Model Serving**: "Serving Machine Learning Models" by Boris Lublinsky

---

**Note**: These patterns were extracted from PR #17 and represent valuable ML integration approaches. They should be adapted and customized for specific production requirements and use cases.
