"""
Utility functions for image processing and validation
"""

import os
import logging
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict
import pandas as pd
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        True if extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def load_image(image_path: str, max_size: Optional[int] = None) -> Image.Image:
    """
    Load and optionally resize an image
    
    Args:
        image_path: Path to the image
        max_size: Maximum dimension (width or height)
        
    Returns:
        PIL Image object
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        if max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def save_image(image: Image.Image, save_path: str) -> str:
    """
    Save an image to disk
    
    Args:
        image: PIL Image object
        save_path: Path to save the image
        
    Returns:
        Path where image was saved
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, quality=95)
        return save_path
    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {e}")
        raise


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop an image using bounding box coordinates
    
    Args:
        image: PIL Image object
        bbox: Tuple of (x1, y1, x2, y2)
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def pad_bbox(bbox: Tuple[int, int, int, int], 
             padding: float, 
             img_width: int, 
             img_height: int) -> Tuple[int, int, int, int]:
    """
    Add padding to a bounding box
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        padding: Padding ratio (e.g., 0.1 for 10% padding)
        img_width: Image width
        img_height: Image height
        
    Returns:
        Padded bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_width, x2 + pad_w)
    y2 = min(img_height, y2 + pad_h)
    
    return (x1, y1, x2, y2)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


def get_database_images() -> list:
    """
    Get all image files from the database/products directory
    
    Returns:
        List of image file paths
    """
    image_files = []
    
    # Check products directory first
    products_dir = config.PRODUCTS_DIR
    if os.path.exists(products_dir):
        for filename in os.listdir(products_dir):
            if allowed_file(filename):
                image_path = os.path.join(products_dir, filename)
                image_files.append(image_path)
    
    # Also check root database directory for backward compatibility
    if os.path.exists(config.DATABASE_DIR):
        for filename in os.listdir(config.DATABASE_DIR):
            if allowed_file(filename):
                image_path = os.path.join(config.DATABASE_DIR, filename)
                if image_path not in image_files:  # Avoid duplicates
                    image_files.append(image_path)
    
    logger.info(f"Found {len(image_files)} images in database")
    return sorted(image_files)


def load_sku_mapping() -> Dict[str, Dict]:
    """
    Load SKU mapping from products.csv
    
    Returns:
        Dictionary mapping filename to product metadata
        {
            'product_001.jpg': {
                'sku_id': 'SKU-12345',
                'product_name': 'Blue Shirt',
                'category': 'Clothing',
                'price': '29.99'
            }
        }
    """
    sku_mapping = {}
    
    if not os.path.exists(config.PRODUCTS_CSV):
        logger.warning(f"Products CSV not found: {config.PRODUCTS_CSV}")
        return sku_mapping
    
    try:
        df = pd.read_csv(config.PRODUCTS_CSV)
        
        # Convert to dictionary
        for _, row in df.iterrows():
            filename = row.get('filename', '')
            if filename:
                sku_mapping[filename] = {
                    'sku_id': row.get('sku_id', ''),
                    'product_name': row.get('product_name', ''),
                    'category': row.get('category', ''),
                    'price': row.get('price', '')
                }
        
        logger.info(f"Loaded {len(sku_mapping)} SKU mappings")
    except Exception as e:
        logger.error(f"Error loading SKU mapping: {e}")
    
    return sku_mapping


def get_sku_for_image(image_path: str, sku_mapping: Dict[str, Dict] = None) -> Dict:
    """
    Get SKU information for an image
    
    Args:
        image_path: Full path to image
        sku_mapping: Optional pre-loaded SKU mapping
        
    Returns:
        Dictionary with SKU information
    """
    if sku_mapping is None:
        sku_mapping = load_sku_mapping()
    
    filename = os.path.basename(image_path)
    
    if filename in sku_mapping:
        return sku_mapping[filename]
    
    # If not found, return empty/default values
    return {
        'sku_id': f'SKU-{filename}',
        'product_name': filename,
        'category': 'Unknown',
        'price': ''
    }


def format_similarity_score(score: float) -> str:
    """
    Format similarity score as percentage
    
    Args:
        score: Similarity score (0-1)
        
    Returns:
        Formatted string
    """
    return f"{score * 100:.2f}%"


def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old temporary files
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep
    """
    import time
    
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old temp file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing {filename}: {e}")

