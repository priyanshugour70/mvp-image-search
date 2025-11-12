# 🚀 Quick Start Guide

Get your Object Detection & Image Retrieval System up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- (Optional) CUDA-capable GPU for faster processing

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Note**: The first time you run, YOLO and CLIP models will be downloaded automatically (~500MB total).

## Step 2: Prepare Your Database

Add product images to the `database/` directory:

```bash
# Create the database directory
mkdir -p database

# Copy your product images
cp /path/to/your/images/*.jpg database/

# You can use any image format: jpg, png, jpeg, webp, bmp
```

**Tips for best results:**
- Use clear, well-lit images
- Include multiple angles of each product
- Minimum 10 images recommended
- Each image should be <10MB

## Step 3: Index the Database

Build the search index from your images:

```bash
python index_database.py
```

This will:
1. Load all images from `database/`
2. Extract CLIP embeddings
3. Build a FAISS index
4. Save the index to `embeddings/`

**Expected output:**
```
Found 150 images to index
Loading models...
Extracting features... 100%|████████████| 150/150
Building FAISS index...
Saving index to disk...
✅ Database indexing complete!
```

### Optional: Index with Object Detection

If your database contains images with multiple objects, you can detect and index specific objects:

```bash
python index_database.py --use-objects
```

This will detect the most prominent object in each image and index that instead of the whole image.

## Step 4: Start the Application

```bash
python app.py
```

The server will start at `http://localhost:5000`

**Expected output:**
```
Starting Flask application...
Database directory: /path/to/database
Loading models...
 * Running on http://0.0.0.0:5000
```

## Step 5: Use the Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Click or drag-and-drop to upload an image
3. Adjust search parameters (optional):
   - **Number of Results**: How many matches to return (1-20)
   - **Similarity Threshold**: Minimum match quality (0.0-1.0)
   - **Detect Objects First**: Enable for multi-object images
4. Click "Search for Similar Images"
5. View your results!

## Alternative: Use the API Directly

### Python Example

```python
import requests

# Search for similar images
with open('query_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/search',
        files={'image': f},
        data={'top_k': 5, 'threshold': 0.6}
    )

results = response.json()
print(f"Found {len(results['results'])} matches")
for result in results['results']:
    print(f"  Similarity: {result['similarity']:.2%}")
    print(f"  Image: {result['image_path']}")
```

### cURL Example

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query_image.jpg" \
  -F "top_k=5" \
  -F "threshold=0.6"
```

### Using the Test Script

```bash
# Test the API
python test_api.py

# Test with a specific image
python test_api.py query_image.jpg
```

## Common Issues & Solutions

### Issue: "No images found in database"

**Solution**: Make sure you've added images to the `database/` directory and they have valid extensions (jpg, png, etc.)

### Issue: "Index is empty"

**Solution**: Run `python index_database.py` to build the search index before starting the app.

### Issue: "No objects detected"

**Solutions**:
- Lower the confidence threshold in `config.py` (default is 0.25)
- Try disabling "Detect Objects First" to search with the whole image
- Ensure the image is clear and well-lit

### Issue: Models downloading slowly

**Solution**: The first run downloads ~500MB of models. This only happens once. Be patient!

### Issue: Out of memory

**Solutions**:
- Use `yolov8n.pt` (nano) instead of larger models in `config.py`
- Reduce `MAX_IMAGE_DIMENSION` in `config.py`
- Reduce `BATCH_SIZE` in `config.py`

## Next Steps

### Add More Images

```bash
# Copy new images to database
cp new_images/*.jpg database/

# Rebuild the index
python index_database.py
```

### Add Images via API

```bash
curl -X POST http://localhost:5000/api/add \
  -F "image=@new_product.jpg" \
  -F "name=product_name"
```

### Customize Configuration

Edit `config.py` to:
- Change YOLO model (for speed vs accuracy)
- Adjust detection thresholds
- Change default search parameters
- Enable GPU acceleration

### Production Deployment

For production use:
1. Set `FLASK_DEBUG = False` in `config.py`
2. Use a production WSGI server (gunicorn, uwsgi)
3. Add authentication to API endpoints
4. Set up regular index rebuilds
5. Monitor disk space for uploads

## Performance Tips

### For Speed:
- Use `yolov8n.pt` (fastest)
- Disable object detection for simple queries
- Use GPU if available (`FAISS_USE_GPU = True`)
- Reduce image dimensions

### For Accuracy:
- Use `yolov8m.pt` or `yolov8l.pt`
- Add more database images from different angles
- Lower similarity threshold
- Use object detection mode

## Get Help

Check the full `README.md` for detailed documentation, or review the configuration options in `config.py`.

Happy searching! 🎉

