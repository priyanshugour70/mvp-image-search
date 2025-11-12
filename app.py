"""
Flask API for Object Detection and Image Retrieval System

This is the main application that provides REST API endpoints and web interface
for uploading images and searching for similar products.
"""

import os
import time
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image

import config
import utils
from detector import ObjectDetector
from feature_extractor import FeatureExtractor
from search_engine import SearchEngine

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE
CORS(app)

# Global variables for model instances (lazy loading)
detector = None
feature_extractor = None
search_engine = None


def get_detector():
    """Lazy load object detector"""
    global detector
    if detector is None:
        logger.info("Initializing ObjectDetector...")
        detector = ObjectDetector()
    return detector


def get_feature_extractor():
    """Lazy load feature extractor"""
    global feature_extractor
    if feature_extractor is None:
        logger.info("Initializing FeatureExtractor...")
        feature_extractor = FeatureExtractor()
    return feature_extractor


def get_search_engine():
    """Lazy load search engine"""
    global search_engine
    if search_engine is None:
        logger.info("Initializing SearchEngine...")
        search_engine = SearchEngine()
        # Try to load existing index
        if not search_engine.load_index():
            logger.warning("No existing index found. Please run index_database.py first.")
    return search_engine


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'detector': detector is not None,
            'feature_extractor': feature_extractor is not None,
            'search_engine': search_engine is not None
        }
    })


@app.route('/api/search', methods=['POST'])
def search_image():
    """
    Main endpoint for image search
    
    Request:
        - image: uploaded image file
        - top_k: (optional) number of results to return
        - threshold: (optional) similarity threshold
        - detect_objects: (optional) whether to detect objects (default: true)
        
    Response:
        {
            "success": true,
            "query_time": 0.234,
            "detected_objects": 2,
            "results": [
                {
                    "image_path": "database/product_123.jpg",
                    "similarity": 0.95,
                    "object_class": "bottle",
                    "confidence": 0.89,
                    "rank": 0
                }
            ]
        }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not utils.allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Parse parameters
        top_k = int(request.form.get('top_k', config.DEFAULT_TOP_K))
        threshold = float(request.form.get('threshold', config.SIMILARITY_THRESHOLD))
        detect_objects = request.form.get('detect_objects', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_DIR, filename)
        file.save(filepath)
        
        logger.info(f"Processing uploaded image: {filename}")
        
        # Load image
        image = utils.load_image(filepath, max_size=config.MAX_IMAGE_DIMENSION)
        
        # Initialize models
        det = get_detector()
        feat_ext = get_feature_extractor()
        search_eng = get_search_engine()
        
        # Check if search engine has index
        if search_eng.get_index_size() == 0:
            return jsonify({
                'success': False,
                'error': 'Search index is empty. Please run index_database.py first.'
            }), 500
        
        all_results = []
        
        if detect_objects:
            # Detect objects in the image
            detections = det.detect(image)
            
            if not detections:
                return jsonify({
                    'success': False,
                    'error': 'No objects detected in the image. Try adjusting the image or lowering confidence threshold.'
                }), 404
            
            logger.info(f"Found {len(detections)} objects")
            
            # Search for each detected object
            for detection in detections:
                cropped_image = detection['cropped_image']
                
                # Extract features from cropped object
                embedding = feat_ext.extract(cropped_image)
                
                # Search for similar images
                results = search_eng.search(embedding, top_k=top_k, threshold=threshold)
                
                # Add detection metadata to results
                for result in results:
                    result['object_class'] = detection['class_name']
                    result['object_confidence'] = detection['confidence']
                    result['bbox'] = detection['bbox']
                
                all_results.extend(results)
            
            # Sort all results by similarity
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Keep only top_k overall results
            all_results = all_results[:top_k]
            
        else:
            # Search using the whole image (no object detection)
            embedding = feat_ext.extract(image)
            all_results = search_eng.search(embedding, top_k=top_k, threshold=threshold)
        
        # Calculate query time
        query_time = time.time() - start_time
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'query_time': round(query_time, 3),
            'detected_objects': len(detections) if detect_objects else 0,
            'results': all_results
        })
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/add', methods=['POST'])
def add_image():
    """
    Add a new image to the database
    
    Request:
        - image: uploaded image file
        - name: (optional) custom name for the image
    
    Response:
        {
            "success": true,
            "image_path": "database/product_new.jpg",
            "message": "Image added successfully"
        }
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not utils.allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get custom name or use original
        custom_name = request.form.get('name', '')
        if custom_name:
            filename = secure_filename(custom_name)
            # Ensure it has an extension
            if '.' not in filename:
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{filename}.{ext}"
        else:
            filename = secure_filename(file.filename)
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        # Save to database directory
        filepath = os.path.join(config.DATABASE_DIR, filename)
        file.save(filepath)
        
        logger.info(f"Added new image to database: {filename}")
        
        # Extract features and add to search engine
        feat_ext = get_feature_extractor()
        search_eng = get_search_engine()
        
        embedding = feat_ext.extract_from_path(filepath)
        search_eng.add_images(embedding.reshape(1, -1), [filepath])
        
        # Save updated index
        search_eng.save_index()
        
        return jsonify({
            'success': True,
            'image_path': filepath,
            'message': 'Image added successfully'
        })
        
    except Exception as e:
        logger.error(f"Error adding image: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/index/rebuild', methods=['POST'])
def rebuild_index():
    """
    Rebuild the search index from scratch
    
    Response:
        {
            "success": true,
            "num_images": 150,
            "message": "Index rebuilt successfully"
        }
    """
    try:
        logger.info("Rebuilding search index...")
        
        # Get all database images
        image_paths = utils.get_database_images()
        
        if not image_paths:
            return jsonify({
                'success': False,
                'error': 'No images found in database directory'
            }), 404
        
        # Extract features
        feat_ext = get_feature_extractor()
        logger.info(f"Extracting features for {len(image_paths)} images...")
        embeddings = feat_ext.extract_batch(
            [utils.load_image(p) for p in image_paths]
        )
        
        # Build index
        search_eng = get_search_engine()
        search_eng.build_index(embeddings, image_paths)
        
        # Save index
        search_eng.save_index()
        
        logger.info("Index rebuilt successfully")
        
        return jsonify({
            'success': True,
            'num_images': len(image_paths),
            'message': 'Index rebuilt successfully'
        })
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/index/status', methods=['GET'])
def index_status():
    """
    Get status of the search index
    
    Response:
        {
            "indexed": true,
            "num_images": 150,
            "embedding_dim": 512
        }
    """
    try:
        search_eng = get_search_engine()
        
        return jsonify({
            'indexed': search_eng.get_index_size() > 0,
            'num_images': search_eng.get_index_size(),
            'embedding_dim': search_eng.embedding_dim
        })
        
    except Exception as e:
        logger.error(f"Error getting index status: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/database/<path:filename>')
def serve_database_image(filename):
    """Serve images from the database directory"""
    return send_from_directory(config.DATABASE_DIR, filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {config.MAX_UPLOAD_SIZE / (1024*1024)}MB'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database directory: {config.DATABASE_DIR}")
    logger.info(f"Embeddings directory: {config.EMBEDDINGS_DIR}")
    logger.info(f"Upload directory: {config.UPLOAD_DIR}")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

