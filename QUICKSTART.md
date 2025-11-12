# 🚀 Quick Start Guide

Get the Image-Based Product Matching API up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd mvp
pip install -r requirements.txt
```

**Note**: First-time installation may take 5-10 minutes as it downloads the CLIP model (~600MB).

## Step 2: Start the Server

### Option A: Using the start script (Recommended)
```bash
./start.sh
```

### Option B: Using uvicorn directly
```bash
uvicorn app:app --reload
```

### Option C: Using Python
```bash
python app.py
```

The server will start at: `http://127.0.0.1:8000`

## Step 3: Test the API

### Method 1: Using Interactive Swagger UI (Easiest!)

1. Open your browser
2. Go to: `http://127.0.0.1:8000/docs`
3. You'll see a beautiful interactive interface
4. Click on any endpoint to test it

**To upload a product:**
- Click on `POST /upload_product`
- Click "Try it out"
- Upload an image file
- Enter a product name
- Click "Execute"

**To search for similar products:**
- Click on `POST /search_image`
- Click "Try it out"
- Upload a query image
- Click "Execute"
- See the best match with similarity percentage!

### Method 2: Using cURL

**Upload products:**
```bash
# Upload product 1
curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@/path/to/watch1.jpg" \
  -F "product_name=Blue Digital Watch"

# Upload product 2
curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@/path/to/watch2.jpg" \
  -F "product_name=Black Analog Watch"

# Upload product 3
curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@/path/to/shoes.jpg" \
  -F "product_name=Red Sneakers"
```

**Search for similar product:**
```bash
curl -X POST "http://127.0.0.1:8000/search_image" \
  -F "file=@/path/to/query_watch.jpg"
```

**Expected output:**
```json
{
  "query_image": "uploads/query_abc123.jpg",
  "best_match": {
    "product_name": "Blue Digital Watch",
    "product_image": "uploads/product_xyz789.jpg",
    "similarity_percent": 97.83
  }
}
```

### Method 3: Using Python

```python
import requests

# Upload a product
def upload_product(image_path, product_name):
    url = "http://127.0.0.1:8000/upload_product"
    files = {"file": open(image_path, "rb")}
    data = {"product_name": product_name}
    response = requests.post(url, files=files, data=data)
    print(response.json())

# Search for similar product
def search_image(query_image_path):
    url = "http://127.0.0.1:8000/search_image"
    files = {"file": open(query_image_path, "rb")}
    response = requests.post(url, files=files)
    print(response.json())

# Test it
upload_product("watch1.jpg", "Blue Digital Watch")
upload_product("watch2.jpg", "Black Analog Watch")
search_image("query_watch.jpg")
```

### Method 4: Using the Test Script

We've included a test script to verify everything is working:

```bash
python test_api.py
```

## Step 4: Verify It's Working

1. **Check health:**
   ```bash
   curl http://127.0.0.1:8000/
   ```

2. **List all products:**
   ```bash
   curl http://127.0.0.1:8000/products
   ```

3. **Open the docs:**
   - Browser: `http://127.0.0.1:8000/docs`

## 🎯 Example Workflow

Let's say you're building a watch e-commerce site:

1. **Upload 3 different watches:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/upload_product" \
     -F "file=@watch_blue.jpg" \
     -F "product_name=Blue Digital Watch"
   
   curl -X POST "http://127.0.0.1:8000/upload_product" \
     -F "file=@watch_black.jpg" \
     -F "product_name=Black Analog Watch"
   
   curl -X POST "http://127.0.0.1:8000/upload_product" \
     -F "file=@watch_gold.jpg" \
     -F "product_name=Gold Luxury Watch"
   ```

2. **A customer uploads a photo of a watch they like:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/search_image" \
     -F "file=@customer_watch.jpg"
   ```

3. **The API returns the most similar watch:**
   ```json
   {
     "query_image": "uploads/query_123.jpg",
     "best_match": {
       "product_name": "Blue Digital Watch",
       "product_image": "uploads/product_456.jpg",
       "similarity_percent": 94.72
     }
   }
   ```

## 📝 What to Expect

### First Run
- **Time**: 2-3 minutes (downloading CLIP model)
- **Memory**: ~500MB RAM
- **Disk**: ~600MB (model cache)

### Subsequent Runs
- **Startup**: 5-10 seconds (loading model)
- **Upload**: 1-2 seconds per image
- **Search**: <100ms for 5 products

### Performance
- Works great for 3-5 products (MVP scope)
- Can handle up to 100 products smoothly
- For production with thousands of products, you'll need a database

## 🐛 Common Issues

### "Model not found"
**Solution**: The model will auto-download on first run. Make sure you have internet connection.

### "Port already in use"
**Solution**: Another process is using port 8000. Either kill it or use a different port:
```bash
uvicorn app:app --reload --port 8001
```

### "Cannot import name 'CLIPModel'"
**Solution**: Install transformers:
```bash
pip install transformers
```

### "No products available"
**Solution**: You need to upload products first before searching!

## 🎉 You're All Set!

The API is now running and ready to use. Try uploading some product images and searching for similar ones!

### Next Steps:
- Upload 3-5 product images
- Test the search functionality
- Check out the full [README.md](README.md) for more details
- Explore the API docs at `http://127.0.0.1:8000/docs`

---

**Need help?** Check the main [README.md](README.md) for detailed documentation.

