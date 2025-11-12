#!/usr/bin/env python3
"""
API Testing Script

This script provides functions to test the API endpoints.
"""

import requests
import os
import json


def test_health_check(base_url='http://localhost:5000'):
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f'{base_url}/api/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_index_status(base_url='http://localhost:5000'):
    """Test the index status endpoint"""
    print("Testing index status...")
    response = requests.get(f'{base_url}/api/index/status')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_search(image_path, base_url='http://localhost:5000', top_k=5, threshold=0.6, detect_objects=True):
    """Test the search endpoint"""
    print(f"Testing search with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'top_k': top_k,
            'threshold': threshold,
            'detect_objects': str(detect_objects).lower()
        }
        
        response = requests.post(f'{base_url}/api/search', files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Query Time: {result['query_time']}s")
        print(f"Objects Detected: {result['detected_objects']}")
        print(f"Results Found: {len(result['results'])}")
        
        if result['results']:
            print("\nTop Results:")
            for i, res in enumerate(result['results'][:3], 1):
                print(f"  {i}. Similarity: {res['similarity']:.3f}")
                print(f"     Image: {os.path.basename(res['image_path'])}")
                if 'object_class' in res:
                    print(f"     Object: {res['object_class']}")
    else:
        print(f"Error: {response.json()}")
    
    print()


def test_add_image(image_path, base_url='http://localhost:5000', name=None):
    """Test adding an image to the database"""
    print(f"Testing add image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {}
        if name:
            data['name'] = name
        
        response = requests.post(f'{base_url}/api/add', files=files, data=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_rebuild_index(base_url='http://localhost:5000'):
    """Test rebuilding the index"""
    print("Testing index rebuild...")
    response = requests.post(f'{base_url}/api/index/rebuild')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def main():
    """Main testing function"""
    import sys
    
    base_url = 'http://localhost:5000'
    
    print("=" * 60)
    print("API Testing Script")
    print("=" * 60)
    print()
    
    # Test health check
    test_health_check(base_url)
    
    # Test index status
    test_index_status(base_url)
    
    # Test search if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_search(image_path, base_url)
    else:
        print("To test search, provide an image path:")
        print(f"  python {sys.argv[0]} <image_path>")
        print()


if __name__ == '__main__':
    main()

