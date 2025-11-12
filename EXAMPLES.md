# Usage Examples

This document provides practical examples for using the Object Detection & Image Retrieval System.

## Table of Contents

1. [Python API Examples](#python-api-examples)
2. [cURL Examples](#curl-examples)
3. [Web Interface Examples](#web-interface-examples)
4. [Common Use Cases](#common-use-cases)
5. [Troubleshooting Examples](#troubleshooting-examples)

## Python API Examples

### Example 1: Basic Image Search

```python
import requests

# Search for similar images
with open('product_query.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/search',
        files={'image': f},
        data={
            'top_k': 5,
            'threshold': 0.7,
            'detect_objects': 'true'
        }
    )

result = response.json()

if result['success']:
    print(f"Found {len(result['results'])} matches in {result['query_time']}s")
    
    for idx, match in enumerate(result['results'], 1):
        print(f"\n{idx}. Similarity: {match['similarity']:.2%}")
        print(f"   Path: {match['image_path']}")
        if 'object_class' in match:
            print(f"   Object: {match['object_class']}")
else:
    print(f"Error: {result['error']}")
```

### Example 2: Batch Search Multiple Images

```python
import requests
from pathlib import Path

def search_image(image_path):
    """Search for a single image"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/search',
            files={'image': f},
            data={'top_k': 3}
        )
    return response.json()

# Search multiple images
query_images = Path('queries').glob('*.jpg')

for img_path in query_images:
    print(f"\nSearching: {img_path.name}")
    result = search_image(img_path)
    
    if result['success'] and result['results']:
        best_match = result['results'][0]
        print(f"  Best match: {best_match['similarity']:.2%}")
        print(f"  File: {Path(best_match['image_path']).name}")
    else:
        print("  No matches found")
```

### Example 3: Add Images to Database

```python
import requests
from pathlib import Path

def add_to_database(image_path, custom_name=None):
    """Add an image to the database"""
    with open(image_path, 'rb') as f:
        data = {}
        if custom_name:
            data['name'] = custom_name
            
        response = requests.post(
            'http://localhost:5000/api/add',
            files={'image': f},
            data=data
        )
    return response.json()

# Add single image
result = add_to_database('new_product.jpg', 'product_001')
print(result['message'])

# Add multiple images
new_images = Path('new_products').glob('*.jpg')

for img_path in new_images:
    result = add_to_database(img_path)
    if result['success']:
        print(f"✓ Added: {img_path.name}")
    else:
        print(f"✗ Failed: {img_path.name} - {result['error']}")
```

### Example 4: Rebuild Index After Adding Images

```python
import requests

def rebuild_index():
    """Rebuild the search index"""
    response = requests.post('http://localhost:5000/api/index/rebuild')
    return response.json()

# Rebuild index
print("Rebuilding index...")
result = rebuild_index()

if result['success']:
    print(f"✓ Indexed {result['num_images']} images")
else:
    print(f"✗ Error: {result['error']}")
```

### Example 5: Check System Status

```python
import requests

def check_status():
    """Check system health and index status"""
    # Health check
    health = requests.get('http://localhost:5000/api/health').json()
    print("System Status:")
    print(f"  Health: {health['status']}")
    print(f"  Models Loaded: {health['models_loaded']}")
    
    # Index status
    index = requests.get('http://localhost:5000/api/index/status').json()
    print("\nIndex Status:")
    print(f"  Indexed: {index['indexed']}")
    print(f"  Images: {index['num_images']}")
    print(f"  Embedding Dimension: {index['embedding_dim']}")

check_status()
```

### Example 6: Search Without Object Detection

```python
import requests

# For simple images with single objects, disable detection for speed
with open('simple_product.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/search',
        files={'image': f},
        data={
            'top_k': 5,
            'detect_objects': 'false'  # Use whole image
        }
    )

result = response.json()
print(f"Query time: {result['query_time']}s")  # Should be faster
```

### Example 7: Advanced - Using the Models Directly

```python
from PIL import Image
from detector import ObjectDetector
from feature_extractor import FeatureExtractor
from search_engine import SearchEngine

# Initialize models
detector = ObjectDetector()
extractor = FeatureExtractor()
search_engine = SearchEngine()

# Load existing index
search_engine.load_index()

# Process query image
image = Image.open('query.jpg')

# Detect objects
detections = detector.detect(image)
print(f"Detected {len(detections)} objects")

# Extract features for best detection
if detections:
    best = detector.get_best_detection(detections)
    cropped = best['cropped_image']
    embedding = extractor.extract(cropped)
    
    # Search
    results = search_engine.search(embedding, top_k=5)
    
    for result in results:
        print(f"Match: {result['similarity']:.2%} - {result['image_path']}")
```

## cURL Examples

### Example 1: Basic Search

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query_image.jpg" \
  -F "top_k=5" \
  -F "threshold=0.7"
```

### Example 2: Search Without Object Detection

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@simple_product.jpg" \
  -F "detect_objects=false"
```

### Example 3: Add Image to Database

```bash
curl -X POST http://localhost:5000/api/add \
  -F "image=@new_product.jpg" \
  -F "name=product_001"
```

### Example 4: Rebuild Index

```bash
curl -X POST http://localhost:5000/api/index/rebuild
```

### Example 5: Check Health

```bash
curl http://localhost:5000/api/health
```

### Example 6: Check Index Status

```bash
curl http://localhost:5000/api/index/status
```

### Example 7: Pretty Print JSON Response

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query.jpg" \
  | python -m json.tool
```

## Web Interface Examples

### Example 1: Basic Search

1. Open `http://localhost:5000`
2. Click the upload area or drag-and-drop an image
3. Preview appears automatically
4. Click "Search for Similar Images"
5. View results with similarity scores

### Example 2: Adjust Search Parameters

1. Upload an image
2. Change "Number of Results" to 10
3. Set "Similarity Threshold" to 0.8 (more strict)
4. Uncheck "Detect Objects First" for faster search
5. Click search

### Example 3: Interpret Results

The results show:
- **Similarity Badge**: Match percentage (higher = better)
- **Object**: Detected object class (if detection enabled)
- **Confidence**: Detection confidence
- **Rank**: Position in results

## Common Use Cases

### Use Case 1: Product Search (E-commerce)

**Scenario**: Customer takes photo of a product in store, wants to find it online

```python
import requests

def find_product(customer_photo):
    """Find product from customer photo"""
    with open(customer_photo, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/search',
            files={'image': f},
            data={
                'top_k': 10,           # Show multiple options
                'threshold': 0.65,      # Allow similar products
                'detect_objects': 'true' # Focus on product
            }
        )
    
    result = response.json()
    
    if result['success'] and result['results']:
        # Return top 3 matches with high confidence
        return [
            r for r in result['results'][:3]
            if r.get('object_confidence', 1) > 0.7
        ]
    return []

# Customer uploads photo
matches = find_product('customer_photo.jpg')

for match in matches:
    print(f"Product: {match['image_path']}")
    print(f"Match: {match['similarity']:.1%}")
    # Show product page...
```

### Use Case 2: Inventory Management

**Scenario**: Warehouse worker scans item, system identifies it

```python
def identify_inventory_item(scanned_image):
    """Identify item from barcode scan"""
    with open(scanned_image, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/search',
            files={'image': f},
            data={
                'top_k': 1,              # Only need best match
                'threshold': 0.9,        # High threshold for exact match
                'detect_objects': 'true'
            }
        )
    
    result = response.json()
    
    if result['success'] and result['results']:
        best_match = result['results'][0]
        if best_match['similarity'] > 0.95:  # Very high confidence
            return {
                'identified': True,
                'item_id': Path(best_match['image_path']).stem,
                'confidence': best_match['similarity']
            }
    
    return {'identified': False}

# Scan item
item_info = identify_inventory_item('scanned_item.jpg')

if item_info['identified']:
    print(f"Item: {item_info['item_id']}")
    print(f"Confidence: {item_info['confidence']:.1%}")
else:
    print("Item not recognized - manual entry required")
```

### Use Case 3: Quality Control

**Scenario**: Check if product matches reference image

```python
def quality_check(product_image, reference_id):
    """Check if product matches reference"""
    with open(product_image, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/search',
            files={'image': f},
            data={
                'top_k': 5,
                'threshold': 0.85
            }
        )
    
    result = response.json()
    
    # Check if reference_id is in top matches
    for match in result['results']:
        if reference_id in match['image_path']:
            return {
                'passed': match['similarity'] > 0.90,
                'similarity': match['similarity'],
                'rank': match['rank'] + 1
            }
    
    return {'passed': False, 'similarity': 0, 'rank': None}

# Quality check
result = quality_check('product_sample.jpg', 'REF_001')

if result['passed']:
    print(f"✓ Quality check passed ({result['similarity']:.1%})")
else:
    print(f"✗ Quality check failed (similarity: {result['similarity']:.1%})")
```

### Use Case 4: Visual Catalog Building

**Scenario**: Automatically organize and categorize product images

```python
from collections import defaultdict
from pathlib import Path

def build_visual_catalog(images_dir):
    """Group similar products together"""
    all_images = list(Path(images_dir).glob('*.jpg'))
    groups = []
    processed = set()
    
    for img_path in all_images:
        if str(img_path) in processed:
            continue
        
        # Search for similar images
        with open(img_path, 'rb') as f:
            response = requests.post(
                'http://localhost:5000/api/search',
                files={'image': f},
                data={'top_k': 20, 'threshold': 0.75}
            )
        
        result = response.json()
        
        if result['success']:
            # Create group with similar items
            group = [str(img_path)]
            for match in result['results']:
                match_path = match['image_path']
                if match_path not in processed:
                    group.append(match_path)
                    processed.add(match_path)
            
            groups.append(group)
    
    return groups

# Build catalog
catalog = build_visual_catalog('products')

for idx, group in enumerate(catalog, 1):
    print(f"\nGroup {idx}: {len(group)} items")
    for item in group[:3]:  # Show first 3
        print(f"  - {Path(item).name}")
```

## Troubleshooting Examples

### Problem: No Results Found

```python
# Try lowering threshold
response = requests.post(
    'http://localhost:5000/api/search',
    files={'image': open('query.jpg', 'rb')},
    data={'threshold': 0.4}  # Lower threshold
)

# Or disable object detection
response = requests.post(
    'http://localhost:5000/api/search',
    files={'image': open('query.jpg', 'rb')},
    data={'detect_objects': 'false'}
)
```

### Problem: Search Too Slow

```python
# Disable object detection for speed
response = requests.post(
    'http://localhost:5000/api/search',
    files={'image': open('query.jpg', 'rb')},
    data={'detect_objects': 'false'}
)

# Or reduce top_k
response = requests.post(
    'http://localhost:5000/api/search',
    files={'image': open('query.jpg', 'rb')},
    data={'top_k': 3}  # Fewer results = faster
)
```

### Problem: Too Many False Positives

```python
# Increase threshold for stricter matching
response = requests.post(
    'http://localhost:5000/api/search',
    files={'image': open('query.jpg', 'rb')},
    data={'threshold': 0.85}  # Higher threshold
)
```

### Problem: Objects Not Detected

```python
# Check what's being detected
from detector import ObjectDetector
from PIL import Image

detector = ObjectDetector()
image = Image.open('query.jpg')
detections = detector.detect(image, confidence=0.15)  # Lower confidence

print(f"Found {len(detections)} objects:")
for det in detections:
    print(f"  - {det['class_name']}: {det['confidence']:.2%}")
```

## Testing Script

Here's a comprehensive testing script:

```python
#!/usr/bin/env python3
"""Comprehensive API testing"""

import requests
import sys
from pathlib import Path

BASE_URL = 'http://localhost:5000'

def test_all(test_image):
    """Run all tests"""
    print("=" * 60)
    print("COMPREHENSIVE API TEST")
    print("=" * 60)
    
    # 1. Health check
    print("\n1. Health Check")
    r = requests.get(f'{BASE_URL}/api/health')
    print(f"   Status: {r.json()['status']}")
    
    # 2. Index status
    print("\n2. Index Status")
    r = requests.get(f'{BASE_URL}/api/index/status')
    idx = r.json()
    print(f"   Indexed: {idx['indexed']}")
    print(f"   Images: {idx['num_images']}")
    
    # 3. Search with object detection
    print("\n3. Search (with detection)")
    with open(test_image, 'rb') as f:
        r = requests.post(f'{BASE_URL}/api/search',
            files={'image': f},
            data={'top_k': 5, 'detect_objects': 'true'})
    result = r.json()
    print(f"   Time: {result['query_time']}s")
    print(f"   Objects: {result['detected_objects']}")
    print(f"   Results: {len(result['results'])}")
    
    # 4. Search without detection
    print("\n4. Search (without detection)")
    with open(test_image, 'rb') as f:
        r = requests.post(f'{BASE_URL}/api/search',
            files={'image': f},
            data={'detect_objects': 'false'})
    result = r.json()
    print(f"   Time: {result['query_time']}s")
    print(f"   Results: {len(result['results'])}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_comprehensive.py <image_path>")
        sys.exit(1)
    
    test_all(sys.argv[1])
```

Save as `test_comprehensive.py` and run:

```bash
python test_comprehensive.py query_image.jpg
```

