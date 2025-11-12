#!/usr/bin/env python3
"""
Database Indexing Script

This script indexes all images in the database directory by:
1. Extracting CLIP embeddings for each image
2. Building a FAISS search index
3. Saving the index to disk for fast retrieval

Usage:
    python index_database.py [--use-objects]
    
Options:
    --use-objects: Extract embeddings from detected objects instead of whole images
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import config
import utils
from feature_extractor import FeatureExtractor
from search_engine import SearchEngine
from detector import ObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def index_database(use_objects: bool = False):
    """
    Index all images in the database directory
    
    Args:
        use_objects: If True, detect and index objects instead of whole images
    """
    logger.info("Starting database indexing...")
    
    # Get all images from database
    image_paths = utils.get_database_images()
    
    if not image_paths:
        logger.error(f"No images found in {config.DATABASE_DIR}")
        logger.info("Please add images to the database directory first")
        return False
    
    logger.info(f"Found {len(image_paths)} images to index")
    
    # Initialize models
    logger.info("Loading models...")
    feature_extractor = FeatureExtractor()
    
    if use_objects:
        detector = ObjectDetector()
        logger.info("Object detection mode enabled")
    
    # Extract embeddings
    all_embeddings = []
    indexed_paths = []
    
    logger.info("Extracting features...")
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = utils.load_image(image_path, max_size=config.MAX_IMAGE_DIMENSION)
            
            if use_objects:
                # Detect objects and extract embeddings for each
                detections = detector.detect(image)
                
                if detections:
                    # Use the best detection (highest confidence)
                    best_detection = detector.get_best_detection(detections)
                    cropped_image = best_detection['cropped_image']
                    embedding = feature_extractor.extract(cropped_image)
                    
                    all_embeddings.append(embedding)
                    indexed_paths.append(image_path)
                else:
                    logger.warning(f"No objects detected in {image_path}, skipping")
            else:
                # Extract embedding from whole image
                embedding = feature_extractor.extract(image)
                all_embeddings.append(embedding)
                indexed_paths.append(image_path)
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    if not all_embeddings:
        logger.error("No embeddings extracted. Indexing failed.")
        return False
    
    logger.info(f"Successfully extracted {len(all_embeddings)} embeddings")
    
    # Convert to numpy array
    import numpy as np
    embeddings_array = np.vstack(all_embeddings)
    
    # Build search index
    logger.info("Building FAISS index...")
    search_engine = SearchEngine()
    search_engine.build_index(embeddings_array, indexed_paths)
    
    # Save index
    logger.info("Saving index to disk...")
    search_engine.save_index()
    
    logger.info("✅ Database indexing complete!")
    logger.info(f"Indexed {len(indexed_paths)} images")
    logger.info(f"Index saved to: {config.EMBEDDINGS_DIR}")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Index database images for similarity search"
    )
    parser.add_argument(
        '--use-objects',
        action='store_true',
        help='Detect and index objects instead of whole images'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    for directory in [config.DATABASE_DIR, config.EMBEDDINGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Check if database has images
    if not os.path.exists(config.DATABASE_DIR):
        logger.error(f"Database directory does not exist: {config.DATABASE_DIR}")
        logger.info("Please create the directory and add images")
        sys.exit(1)
    
    # Run indexing
    success = index_database(use_objects=args.use_objects)
    
    if success:
        logger.info("\nYou can now run the Flask application:")
        logger.info("  python app.py")
        sys.exit(0)
    else:
        logger.error("\nIndexing failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

