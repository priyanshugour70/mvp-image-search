"""
Image-Based Product Matching API using FastAPI + CLIP + FAISS
MVP Implementation
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"

# ============================================================================
# GLOBAL STATE
# ============================================================================

# In-memory storage
product_data: List[Dict] = []  # [{id, name, path}]
embeddings: List[np.ndarray] = []  # [embedding arrays]
faiss_index: Optional[faiss.IndexFlatIP] = None  # FAISS index for cosine similarity

# Model and processor (loaded once at startup)
model: Optional[CLIPModel] = None
processor: Optional[CLIPProcessor] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Image-Based Product Matching API",
    description="Upload product images and search for visually similar products using CLIP + FAISS",
    version="1.0.0"
)


# ============================================================================
# STARTUP: LOAD CLIP MODEL
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load CLIP model and processor at startup"""
    global model, processor
    
    print(f"🚀 Loading CLIP model: {MODEL_NAME}")
    print(f"🖥️  Using device: {device}")
    
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(image: Image.Image) -> np.ndarray:
    """
    Generate normalized CLIP embedding for an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Normalized embedding as numpy array (shape: 512,)
    """
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embedding
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Convert to numpy and normalize
    embedding = image_features.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
    
    return embedding


def rebuild_faiss_index():
    """Rebuild FAISS index with all current embeddings"""
    global faiss_index
    
    if len(embeddings) == 0:
        faiss_index = None
        return
    
    # Stack embeddings into matrix
    embedding_matrix = np.vstack(embeddings).astype('float32')
    
    # Create FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
    dimension = embedding_matrix.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    faiss_index.add(embedding_matrix)
    
    print(f"🔄 FAISS index rebuilt with {len(embeddings)} products")


def save_upload_file(upload_file: UploadFile, prefix: str = "product") -> Path:
    """
    Save uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        prefix: Prefix for filename
        
    Returns:
        Path to saved file
    """
    # Generate unique filename
    ext = Path(upload_file.filename).suffix
    filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())
    
    return file_path


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Image-Based Product Matching API",
        "endpoints": {
            "upload": "POST /upload_product",
            "search": "POST /search_image"
        },
        "total_products": len(product_data)
    }


@app.post("/upload_product")
async def upload_product(
    file: UploadFile = File(..., description="Product image file"),
    product_name: str = Form(..., description="Product name")
):
    """
    Upload a product image with its name
    
    Args:
        file: Product image file
        product_name: Name of the product
        
    Returns:
        Success message with product details
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save file
        file_path = save_upload_file(file, prefix="product")
        
        # Load image
        image = Image.open(file_path).convert("RGB")
        
        # Generate embedding
        embedding = get_embedding(image)
        
        # Store product data
        product_id = len(product_data)
        product_info = {
            "id": product_id,
            "name": product_name,
            "path": str(file_path)
        }
        
        product_data.append(product_info)
        embeddings.append(embedding)
        
        # Rebuild FAISS index
        rebuild_faiss_index()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Product uploaded successfully",
                "product": {
                    "id": product_id,
                    "name": product_name,
                    "image_path": str(file_path)
                },
                "total_products": len(product_data)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@app.post("/search_image")
async def search_image(
    file: UploadFile = File(..., description="Query image file")
):
    """
    Search for visually similar products by uploading a query image
    
    Args:
        file: Query image file
        
    Returns:
        Best matching product with similarity percentage
    """
    try:
        # Check if we have any products
        if len(product_data) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No products available. Please upload products first."
            )
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save query file
        query_path = save_upload_file(file, prefix="query")
        
        # Load image
        image = Image.open(query_path).convert("RGB")
        
        # Generate embedding
        query_embedding = get_embedding(image)
        
        # Search in FAISS index
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = faiss_index.search(query_vector, k=1)
        
        # Get best match
        best_match_idx = indices[0][0]
        similarity_score = float(distances[0][0])
        
        # Convert cosine similarity to percentage (range: [-1, 1] -> [0, 100])
        # Since vectors are normalized, cosine similarity is the dot product
        # We convert to percentage: (similarity + 1) / 2 * 100
        # But since we're using IndexFlatIP with normalized vectors, the score is already in [0, 1] range
        similarity_percent = similarity_score * 100
        
        best_product = product_data[best_match_idx]
        
        return JSONResponse(
            status_code=200,
            content={
                "query_image": str(query_path),
                "best_match": {
                    "product_name": best_product["name"],
                    "product_image": best_product["path"],
                    "similarity_percent": round(similarity_percent, 2)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")


@app.get("/products")
async def list_products():
    """
    List all uploaded products
    
    Returns:
        List of all products with their details
    """
    return JSONResponse(
        status_code=200,
        content={
            "total_products": len(product_data),
            "products": product_data
        }
    )


@app.delete("/products/{product_id}")
async def delete_product(product_id: int):
    """
    Delete a product by ID
    
    Args:
        product_id: ID of the product to delete
        
    Returns:
        Success message
    """
    try:
        if product_id < 0 or product_id >= len(product_data):
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Remove from storage
        deleted_product = product_data.pop(product_id)
        embeddings.pop(product_id)
        
        # Update IDs for remaining products
        for i in range(product_id, len(product_data)):
            product_data[i]["id"] = i
        
        # Rebuild FAISS index
        rebuild_faiss_index()
        
        # Optionally delete file
        try:
            os.remove(deleted_product["path"])
        except:
            pass
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Product '{deleted_product['name']}' deleted successfully",
                "total_products": len(product_data)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting product: {str(e)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

