# Object Detection & Image Retrieval System

A production-ready MVP that detects objects in uploaded images and finds the best matching images from a database, regardless of viewing angle or lighting conditions.

## 🎯 Features

- **Object Detection**: Uses YOLOv8 to detect and localize objects in images
- **Robust Feature Extraction**: Leverages CLIP embeddings for viewpoint-invariant matching
- **Fast Similarity Search**: FAISS-powered vector search for efficient retrieval
- **REST API**: Flask-based API for easy integration
- **Web Interface**: Simple UI for uploading and testing
- **Angle-Invariant**: Works with images taken from different angles and lighting

## 🏗️ Architecture

```
Upload Image → YOLOv8 Detection → Crop Object → CLIP Embedding → FAISS Search → Return Matches
```

### Components:

1. **Object Detection** (`detector.py`): YOLOv8-based object detection
2. **Feature Extraction** (`feature_extractor.py`): CLIP-based embeddings
3. **Vector Search** (`search_engine.py`): FAISS similarity search
4. **API** (`app.py`): Flask REST API
5. **Web UI** (`templates/`): Upload interface

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Database

```bash
# Create necessary directories
mkdir -p database static/uploads embeddings

# Add your product images to database/
cp /path/to/your/images/*.jpg database/

# Index the database
python index_database.py
```

### Run the Application

```bash
# Start the Flask server
python app.py

# Open browser to http://localhost:5000
```

## 📡 API Usage

### Upload and Search

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query_image.jpg" \
  -F "top_k=5"
```

Response:
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
      "confidence": 0.89
    }
  ]
}
```

### Add Images to Database

```bash
curl -X POST http://localhost:5000/api/add \
  -F "image=@new_product.jpg"
```

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Model settings
YOLO_MODEL = "yolov8n.pt"  # or yolov8s.pt, yolov8m.pt for better accuracy
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Search settings
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Performance
FAISS_USE_GPU = False
BATCH_SIZE = 32
```

## 📊 Performance

- **Detection Speed**: ~50ms per image (YOLOv8n on CPU)
- **Feature Extraction**: ~30ms per object (CLIP)
- **Search Speed**: <1ms for 10K images (FAISS)
- **Total Pipeline**: ~100-200ms per query

## 🔧 Advanced Usage

### Use Different YOLO Models

```python
# For better accuracy (slower)
detector = ObjectDetector(model_name='yolov8m.pt')

# For faster inference
detector = ObjectDetector(model_name='yolov8n.pt')
```

### Custom Object Classes

```python
# Detect only specific classes
detector.detect(image, classes=[0, 39, 41])  # person, bottle, cup
```

### GPU Acceleration

```python
# Enable GPU for FAISS
search_engine = SearchEngine(use_gpu=True)
```

## 📁 Project Structure

```
mvp/
├── app.py                    # Flask API
├── config.py                 # Configuration
├── detector.py               # YOLOv8 object detection
├── feature_extractor.py      # CLIP embeddings
├── search_engine.py          # FAISS search
├── index_database.py         # Database indexing script
├── utils.py                  # Helper functions
├── requirements.txt          # Dependencies
├── templates/
│   └── index.html           # Web UI
├── static/
│   └── uploads/             # Temporary uploads
├── database/                # Product images
└── embeddings/              # Precomputed embeddings
```

## 🎓 How It Works

### 1. Object Detection
YOLOv8 detects objects and provides bounding boxes. We crop the detected region to focus on the object, removing background noise.

### 2. Feature Extraction
CLIP (Contrastive Language-Image Pre-training) creates 512-dimensional embeddings that are:
- Viewpoint-invariant
- Lighting-invariant
- Semantically meaningful

### 3. Similarity Search
FAISS creates an index of all database embeddings. For each query:
1. Compute cosine similarity between query and all database embeddings
2. Return top-k most similar matches
3. Fast approximate nearest neighbor search for large databases

## 🔍 Troubleshooting

### No objects detected
- Try lowering confidence threshold in `config.py`
- Ensure image quality is good
- Check if object class is in YOLO's 80 classes

### Poor matching results
- Add more database images from different angles
- Use a larger YOLO model (yolov8m)
- Increase `SIMILARITY_THRESHOLD`

### Slow performance
- Use YOLOv8n instead of larger models
- Enable GPU if available
- Reduce image resolution

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use for commercial projects

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **CLIP** by OpenAI
- **FAISS** by Meta Research

