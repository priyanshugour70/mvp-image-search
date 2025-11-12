"""
Configuration file for the Object Detection and Image Retrieval System
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
PRODUCTS_DIR = os.path.join(DATABASE_DIR, 'products')  # Product images folder
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')

# SKU mapping file (CSV with: filename,sku_id,product_name,category,price)
PRODUCTS_CSV = os.path.join(DATABASE_DIR, 'products.csv')

# Create directories if they don't exist
for directory in [DATABASE_DIR, PRODUCTS_DIR, EMBEDDINGS_DIR, UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configurations
YOLO_MODEL = 'yolov8n.pt'  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
YOLO_CONFIDENCE = 0.25  # Detection confidence threshold
YOLO_IOU = 0.45  # NMS IOU threshold

# CLIP model for feature extraction
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'  # Can also use clip-vit-large-patch14
CLIP_EMBEDDING_DIM = 512

# Search engine settings
DEFAULT_TOP_K = 5  # Number of results to return
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score (0-1)
FAISS_USE_GPU = False  # Set to True if GPU available

# API settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Image processing
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_IMAGE_DIMENSION = 1920  # Resize if larger

# Performance settings
BATCH_SIZE = 32  # For batch embedding computation
NUM_WORKERS = 4  # For data loading

# YOLO class names (80 COCO classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Logging
LOG_LEVEL = 'INFO'

