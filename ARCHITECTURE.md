# System Architecture

This document provides a detailed technical overview of the Object Detection & Image Retrieval System.

## System Overview

The system implements a complete pipeline for angle-invariant, lighting-invariant object detection and image similarity search:

```
┌─────────────┐
│ User Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Object Detection   │ ← YOLOv8
│   (Optional Step)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Feature Extraction  │ ← CLIP
│  (Embeddings 512D)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Similarity Search  │ ← FAISS
│   (Cosine Distance) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Return Results    │
│  (Top-K Matches)    │
└─────────────────────┘
```

## Core Components

### 1. Object Detection (`detector.py`)

**Purpose**: Localize objects in images to focus on relevant regions

**Technology**: YOLOv8 (You Only Look Once version 8)

**Key Features**:
- Real-time object detection
- 80 COCO object classes
- Confidence thresholding
- Bounding box extraction with padding
- Batch processing support

**Model Options**:
- `yolov8n.pt` - Nano (fastest, ~6MB)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (more accurate)
- `yolov8l.pt` - Large (best accuracy)
- `yolov8x.pt` - XLarge (highest accuracy, slowest)

**Flow**:
1. Receive input image
2. Run YOLO inference
3. Filter detections by confidence threshold
4. Extract bounding boxes
5. Add padding for context (10% by default)
6. Crop detected objects
7. Return list of detection dictionaries

**Why YOLOv8?**:
- State-of-the-art accuracy
- Fast inference (real-time capable)
- Pre-trained on COCO dataset (80 common objects)
- Easy to use with ultralytics library
- Supports various model sizes for speed/accuracy tradeoff

### 2. Feature Extraction (`feature_extractor.py`)

**Purpose**: Convert images to semantic embeddings for similarity comparison

**Technology**: CLIP (Contrastive Language-Image Pre-training) by OpenAI

**Key Features**:
- 512-dimensional embeddings (by default)
- Viewpoint-invariant representations
- Lighting-invariant representations
- L2-normalized vectors for cosine similarity
- Batch processing for efficiency

**Model Options**:
- `openai/clip-vit-base-patch32` - Base (512D, faster)
- `openai/clip-vit-large-patch14` - Large (768D, more accurate)

**Flow**:
1. Receive PIL Image(s)
2. Preprocess with CLIP processor (resize, normalize)
3. Extract visual features via CLIP vision encoder
4. L2-normalize embeddings
5. Return numpy array(s)

**Why CLIP?**:
- Pre-trained on 400M image-text pairs
- Robust to viewpoint and lighting changes
- Semantic understanding (not just visual features)
- Normalized embeddings enable cosine similarity
- State-of-the-art on image retrieval tasks
- Multimodal (can also search by text if needed)

**Embedding Properties**:
- Unit vectors (L2 norm = 1)
- Cosine similarity = dot product
- Range: [-1, 1], but typically [0, 1] for similar images
- Higher values = more similar

### 3. Search Engine (`search_engine.py`)

**Purpose**: Fast similarity search over large image databases

**Technology**: FAISS (Facebook AI Similarity Search)

**Key Features**:
- IndexFlatIP (Inner Product for cosine similarity)
- Sub-millisecond search for 10K vectors
- GPU acceleration support
- Persistent storage (save/load indices)
- Incremental updates (add new images)

**Index Type**: `IndexFlatIP`
- Exact search (not approximate)
- Inner product = cosine similarity for normalized vectors
- No quantization or compression
- Optimal for datasets <1M vectors

**Flow**:
1. Receive embeddings and metadata during indexing
2. Normalize embeddings (ensure unit vectors)
3. Build FAISS index
4. Store metadata (image paths)
5. For search queries:
   - Normalize query embedding
   - Compute inner products (= cosine similarity)
   - Return top-K indices and scores
   - Map to image paths

**Scaling Options** (for larger databases):
- `IndexIVFFlat` - Inverted file index (faster, approximate)
- `IndexIVFPQ` - Product quantization (memory efficient)
- `IndexHNSW` - Hierarchical navigable small world (very fast)

**Why FAISS?**:
- Industry-standard for vector search
- Highly optimized (C++ with Python bindings)
- Supports exact and approximate search
- GPU acceleration available
- Handles millions of vectors efficiently
- Developed and maintained by Meta AI

### 4. Flask API (`app.py`)

**Purpose**: Provide REST API and web interface

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check |
| `/api/search` | POST | Search for similar images |
| `/api/add` | POST | Add image to database |
| `/api/index/rebuild` | POST | Rebuild search index |
| `/api/index/status` | GET | Get index statistics |
| `/database/<file>` | GET | Serve database images |

**Request Flow** (`/api/search`):
1. Receive multipart/form-data with image file
2. Validate file type and size
3. Save to temporary upload directory
4. Load and resize image
5. If object detection enabled:
   - Detect objects with YOLO
   - Extract features for each detected object
   - Search for each object
   - Aggregate and rank results
6. If object detection disabled:
   - Extract features for whole image
   - Search directly
7. Format and return JSON response
8. Clean up temporary files

**Response Format**:
```json
{
  "success": true,
  "query_time": 0.234,
  "detected_objects": 2,
  "results": [
    {
      "image_path": "database/product_123.jpg",
      "similarity": 0.95,
      "object_class": "bottle",
      "object_confidence": 0.89,
      "bbox": [120, 80, 300, 450],
      "rank": 0
    }
  ]
}
```

**Lazy Loading**:
- Models are loaded on first use
- Reduces startup time
- Models persist in memory after loading

### 5. Web Interface (`templates/index.html`)

**Features**:
- Drag-and-drop upload
- Live preview
- Configurable search parameters
- Visual results display
- Performance statistics
- Responsive design

**User Flow**:
1. Upload image (click or drag-drop)
2. Preview image
3. Adjust parameters (optional)
4. Click search
5. View results with similarity scores

## Data Flow

### Indexing Phase

```
Database Images
    │
    ├─→ Load & Resize
    │
    ├─→ (Optional) Detect Objects
    │
    ├─→ Extract CLIP Embeddings
    │       │
    │       └─→ 512D vectors (normalized)
    │
    └─→ Build FAISS Index
            │
            └─→ Save to Disk
                 ├─→ faiss.index (index file)
                 ├─→ metadata.pkl (paths)
                 └─→ embeddings.npy (vectors)
```

### Query Phase

```
Query Image
    │
    ├─→ Load & Resize
    │
    ├─→ Detect Objects (if enabled)
    │       │
    │       └─→ Crop to Bounding Boxes
    │
    ├─→ Extract CLIP Embeddings
    │       │
    │       └─→ 512D query vector(s)
    │
    └─→ FAISS Search
            │
            ├─→ Compute Cosine Similarities
            ├─→ Rank by Score
            ├─→ Filter by Threshold
            └─→ Return Top-K Matches
```

## Key Design Decisions

### 1. Why Two-Stage Pipeline (Detection + Embedding)?

**Benefits**:
- Focus on relevant objects, ignore background
- Better matching for cluttered images
- Works with images containing multiple objects
- More accurate for product search

**Trade-offs**:
- Slightly slower (~50ms extra per query)
- Can miss objects if detection fails
- Optional: can be disabled for simple images

### 2. Why CLIP Over Other Embedding Models?

**Alternatives Considered**:
- ResNet/VGG features: Less semantic, not viewpoint-invariant
- SimCLR/MoCo: Good but less robust to transformations
- DINO: Strong but slower
- Custom CNN: Requires training data

**CLIP Advantages**:
- Pre-trained on massive dataset (400M pairs)
- Robust to viewpoint and lighting
- Semantic understanding (not just pixels)
- Can extend to text-based search
- State-of-the-art performance

### 3. Why FAISS Over Simple Numpy?

**For Small Databases (<1K images)**:
- Could use numpy with `np.dot()` and `np.argsort()`
- Simple and sufficient

**For Larger Databases (>1K images)**:
- FAISS is much faster (optimized C++)
- Scalable to millions of vectors
- GPU acceleration available
- Approximate search options for speed

**We chose FAISS for**:
- Future scalability
- Production-ready performance
- Industry-standard tool

### 4. Why Cosine Similarity Over Euclidean Distance?

**Cosine Similarity**:
- Measures angle between vectors
- Invariant to magnitude
- Range: [-1, 1] (or [0, 1] for normalized)
- Better for semantic similarity

**Euclidean Distance**:
- Measures geometric distance
- Sensitive to magnitude
- Less interpretable for embeddings

**For normalized embeddings**: cosine similarity = dot product

## Performance Characteristics

### Latency Breakdown (typical query)

| Component | Time | Notes |
|-----------|------|-------|
| Image Upload | ~10ms | Network dependent |
| Image Loading | ~20ms | Depends on size |
| Object Detection | ~50ms | YOLOv8n on CPU |
| Feature Extraction | ~30ms | CLIP on CPU |
| Vector Search | <1ms | FAISS on 10K vectors |
| **Total** | **~110ms** | Single object, CPU |

### With GPU Acceleration

| Component | Time | Notes |
|-----------|------|-------|
| Object Detection | ~10ms | YOLOv8n on GPU |
| Feature Extraction | ~5ms | CLIP on GPU |
| Vector Search | <0.1ms | FAISS on GPU |
| **Total** | **~25ms** | Single object, GPU |

### Scaling Characteristics

| Database Size | Search Time | Memory Usage |
|---------------|-------------|--------------|
| 1K images | <1ms | ~2MB |
| 10K images | ~1ms | ~20MB |
| 100K images | ~10ms | ~200MB |
| 1M images | ~100ms | ~2GB |

**For larger databases**: Switch to approximate search (IVF, PQ)

## Configuration & Tuning

### Speed vs Accuracy Trade-offs

**For Speed**:
- Use `yolov8n.pt` (nano model)
- Disable object detection
- Lower image resolution
- Use GPU if available

**For Accuracy**:
- Use `yolov8m.pt` or larger
- Enable object detection
- Add more database images (different angles)
- Lower similarity threshold
- Use higher resolution images

### Memory Optimization

**If running out of memory**:
- Use smaller YOLO model
- Reduce `MAX_IMAGE_DIMENSION`
- Reduce `BATCH_SIZE`
- Use CPU instead of GPU (saves VRAM)

### Similarity Threshold Tuning

| Threshold | Use Case |
|-----------|----------|
| 0.9+ | Exact same object/product |
| 0.7-0.9 | Very similar objects |
| 0.5-0.7 | Same category, different style |
| <0.5 | Loosely related |

**Default**: 0.6 (balanced)

## Future Enhancements

### Potential Improvements

1. **Multi-modal Search**:
   - Add text-based search using CLIP text encoder
   - "Find red dresses" without uploading image

2. **Advanced Indexing**:
   - Approximate search (IVF) for 100K+ images
   - Product quantization for memory efficiency
   - Automatic index optimization

3. **Object Tracking**:
   - Track same object across multiple views
   - Build 3D representations

4. **Fine-tuning**:
   - Fine-tune CLIP on domain-specific data
   - Train custom object detector for specific products

5. **Caching**:
   - Redis cache for frequent queries
   - Precompute embeddings for common objects

6. **Analytics**:
   - Track popular searches
   - A/B test different models
   - Measure accuracy metrics

7. **Production Features**:
   - Authentication & rate limiting
   - Async processing for large batches
   - Distributed deployment
   - Monitoring & logging

## References

### Papers

- **YOLO**: [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- **CLIP**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **FAISS**: [Billion-scale similarity search](https://arxiv.org/abs/1702.08734)

### Libraries

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [Transformers](https://github.com/huggingface/transformers) - CLIP
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Flask](https://flask.palletsprojects.com/) - Web framework

### Datasets

- [COCO Dataset](https://cocodataset.org/) - Object detection training
- [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) - CLIP pre-training

