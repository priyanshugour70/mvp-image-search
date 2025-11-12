# 🎯 Image-Based Product Matching API (MVP)

A FastAPI-based MVP that allows uploading product images and searching for visually similar products using **CLIP** (Contrastive Language-Image Pre-training) and **FAISS** (Facebook AI Similarity Search).

## 🚀 Features

- **Upload Products**: Upload product images one by one with names
- **Visual Search**: Upload a query image and find the most similar product
- **High Accuracy**: Uses OpenAI's CLIP model for state-of-the-art image embeddings
- **Fast Search**: FAISS enables lightning-fast similarity search
- **In-Memory Storage**: Simple and fast for MVP (3-5 products)

## 📋 Prerequisites

- Python 3.10+
- pip

## 🛠️ Installation

1. **Navigate to MVP directory**:
```bash
cd mvp
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎬 Running the Application

Start the server:

```bash
uvicorn app:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### Alternative (using Python directly):

```bash
python app.py
```

## 📡 API Endpoints

### 1. **Health Check** - `GET /`

Check if the API is running.

**Response**:
```json
{
  "status": "running",
  "message": "Image-Based Product Matching API",
  "endpoints": {
    "upload": "POST /upload_product",
    "search": "POST /search_image"
  },
  "total_products": 5
}
```

### 2. **Upload Product** - `POST /upload_product`

Upload a product image with its name.

**Parameters**:
- `file` (form-data): Product image file
- `product_name` (form-data): Name of the product

**Example using cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@/path/to/watch1.jpg" \
  -F "product_name=Blue Digital Watch"
```

**Example using Python**:
```python
import requests

url = "http://127.0.0.1:8000/upload_product"
files = {"file": open("watch1.jpg", "rb")}
data = {"product_name": "Blue Digital Watch"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response**:
```json
{
  "success": true,
  "message": "Product uploaded successfully",
  "product": {
    "id": 0,
    "name": "Blue Digital Watch",
    "image_path": "uploads/product_abc123.jpg"
  },
  "total_products": 1
}
```

### 3. **Search Image** - `POST /search_image`

Upload a query image to find the most similar product.

**Parameters**:
- `file` (form-data): Query image file

**Example using cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/search_image" \
  -F "file=@/path/to/query_watch.jpg"
```

**Example using Python**:
```python
import requests

url = "http://127.0.0.1:8000/search_image"
files = {"file": open("query_watch.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

**Response**:
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

### 4. **List Products** - `GET /products`

Get a list of all uploaded products.

**Example using cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/products"
```

**Response**:
```json
{
  "total_products": 3,
  "products": [
    {
      "id": 0,
      "name": "Blue Digital Watch",
      "path": "uploads/product_abc123.jpg"
    },
    {
      "id": 1,
      "name": "Red Sneakers",
      "path": "uploads/product_def456.jpg"
    }
  ]
}
```

### 5. **Delete Product** - `DELETE /products/{product_id}`

Delete a product by its ID.

**Example using cURL**:
```bash
curl -X DELETE "http://127.0.0.1:8000/products/0"
```

**Response**:
```json
{
  "success": true,
  "message": "Product 'Blue Digital Watch' deleted successfully",
  "total_products": 2
}
```

## 🧪 Testing the API

### Using FastAPI Interactive Docs

1. Start the server
2. Open your browser and go to: `http://127.0.0.1:8000/docs`
3. You'll see an interactive Swagger UI where you can test all endpoints

### Step-by-Step Testing

1. **Upload some products**:
```bash
curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@watch1.jpg" \
  -F "product_name=Blue Watch"

curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@watch2.jpg" \
  -F "product_name=Black Watch"

curl -X POST "http://127.0.0.1:8000/upload_product" \
  -F "file=@shoes.jpg" \
  -F "product_name=Red Sneakers"
```

2. **Search for a similar product**:
```bash
curl -X POST "http://127.0.0.1:8000/search_image" \
  -F "file=@query_watch.jpg"
```

3. **View all products**:
```bash
curl -X GET "http://127.0.0.1:8000/products"
```

## 🧠 How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model (`clip-vit-base-patch32`) to generate 512-dimensional embeddings for images
2. **Embedding Generation**: Each uploaded product image is converted to a normalized embedding vector
3. **FAISS Index**: All embeddings are stored in a FAISS IndexFlatIP (Inner Product) index for fast similarity search
4. **Cosine Similarity**: When searching, the query image embedding is compared against all product embeddings using cosine similarity
5. **Best Match**: Returns the product with the highest similarity score (0-100%)

## 📂 Project Structure

```
mvp/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── uploads/           # Uploaded images (created automatically)
│   ├── product_*.jpg  # Product images
│   └── query_*.jpg    # Query images
└── README.md          # This file
```

## 🔧 Technical Details

### Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **CLIP**: OpenAI's vision-language model for image embeddings
- **FAISS**: Facebook's library for efficient similarity search
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing library

### Model Information

- **Model**: `openai/clip-vit-base-patch32`
- **Embedding Size**: 512 dimensions
- **Similarity Metric**: Cosine similarity (via inner product with normalized vectors)

### Performance

- **Upload**: ~1-2 seconds per image (includes model inference)
- **Search**: <100ms for 5 products
- **Memory**: ~500MB (model + embeddings for 5 products)

## 🎨 Example Use Cases

1. **E-commerce Visual Search**: Customers can upload a photo of a product they like and find similar items
2. **Fashion Discovery**: Upload a clothing item and find similar styles
3. **Product Catalog Management**: Identify duplicate or similar products
4. **Reverse Image Search**: Find products in your catalog that match a customer's photo

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" error
**Solution**: The app automatically uses CPU if CUDA is not available. If you have GPU issues, uninstall `torch` and install CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Port already in use"
**Solution**: Change the port:
```bash
uvicorn app:app --reload --port 8001
```

### Issue: Model download is slow
**Solution**: The first time you run the app, it downloads the CLIP model (~600MB). This is a one-time operation. Subsequent runs will use the cached model.

## 📈 Future Enhancements (Beyond MVP)

- [ ] Persistent storage (database for products, S3 for images)
- [ ] Return top-N matches instead of just one
- [ ] Add filters (price, category, etc.)
- [ ] Batch upload support
- [ ] Frontend UI
- [ ] Authentication & user management
- [ ] Docker containerization
- [ ] Cloud deployment

## 📝 License

MIT License - Feel free to use for personal or commercial projects.

## 🤝 Contributing

This is an MVP project. Feel free to fork and enhance!

---

**Built with ❤️ using FastAPI + CLIP + FAISS**

