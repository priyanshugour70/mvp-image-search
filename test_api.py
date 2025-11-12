"""
Simple test script to verify the Image-Based Product Matching API

Usage:
    1. Make sure the API is running (uvicorn app:app --reload)
    2. Run this script: python test_api.py
"""
import requests
import json
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000"

def print_response(response, title="Response"):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print(f"{'='*60}\n")


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{API_BASE_URL}/")
    print_response(response, "Health Check")
    return response.status_code == 200


def test_upload_product(image_path: str, product_name: str):
    """Test uploading a product"""
    print(f"📤 Uploading product: {product_name}")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        print("💡 Please provide a valid image path to test uploading")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"product_name": product_name}
            response = requests.post(f"{API_BASE_URL}/upload_product", files=files, data=data)
        
        print_response(response, f"Upload Product: {product_name}")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"❌ File not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_list_products():
    """Test listing all products"""
    print("📋 Listing all products...")
    response = requests.get(f"{API_BASE_URL}/products")
    print_response(response, "List Products")
    return response.status_code == 200


def test_search_image(image_path: str):
    """Test searching for similar products"""
    print(f"🔎 Searching with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        print("💡 Please provide a valid image path to test searching")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_BASE_URL}/search_image", files=files)
        
        print_response(response, "Search Image")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"❌ File not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  🧪 Image-Based Product Matching API - Test Suite")
    print("="*60)
    
    # Test 1: Health Check
    try:
        health_ok = test_health_check()
        if not health_ok:
            print("\n❌ API is not responding. Please make sure it's running:")
            print("   uvicorn app:app --reload")
            return
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Please make sure it's running:")
        print("   uvicorn app:app --reload")
        return
    
    print("✅ API is running!")
    
    # Test 2: List Products (should be empty initially)
    test_list_products()
    
    # Note about image files
    print("\n" + "="*60)
    print("  📝 Manual Testing Instructions")
    print("="*60)
    print("""
To fully test the API, you need to provide some test images:

1. Upload a few product images:
   
   curl -X POST "http://127.0.0.1:8000/upload_product" \\
     -F "file=@/path/to/your/image1.jpg" \\
     -F "product_name=Product 1"
   
   curl -X POST "http://127.0.0.1:8000/upload_product" \\
     -F "file=@/path/to/your/image2.jpg" \\
     -F "product_name=Product 2"

2. Search for a similar product:
   
   curl -X POST "http://127.0.0.1:8000/search_image" \\
     -F "file=@/path/to/your/query_image.jpg"

3. Or use the interactive docs:
   
   Open: http://127.0.0.1:8000/docs
   
   You can upload and test directly from the browser!

Tip: Use images of similar products (e.g., different watches, shoes, etc.)
     to see the similarity matching in action!
""")
    print("="*60)


if __name__ == "__main__":
    main()

