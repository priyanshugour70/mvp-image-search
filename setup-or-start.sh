#!/bin/bash
# Complete Setup and Startup Script for Product Search System
# This script handles first-time setup AND regular startup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════╗"
echo "║   Product Search System - Setup & Startup             ║"
echo "║   Object Detection + Image Retrieval with SKU IDs     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Detected Python version: $PYTHON_VERSION"
echo ""

# Function to check if venv exists and is valid
check_venv() {
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check if dependencies are installed
check_dependencies() {
    if [ -f "venv/bin/python" ]; then
        venv/bin/python -c "import flask, torch, transformers, ultralytics, faiss" 2>/dev/null
        return $?
    else
        return 1
    fi
}

# Setup Phase
if ! check_venv; then
    echo "🔧 FIRST-TIME SETUP"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "📦 Step 1: Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
    echo ""
    
    echo "📥 Step 2: Upgrading pip..."
    venv/bin/pip install --upgrade pip --quiet
    echo "   ✅ Pip upgraded"
    echo ""
    
    echo "📥 Step 3: Installing dependencies (this may take 5-10 minutes)..."
    echo "   ⏳ Downloading and installing packages..."
    venv/bin/pip install -r requirements.txt --quiet
    echo "   ✅ All dependencies installed"
    echo ""
else
    echo "✅ Virtual environment found"
    
    if ! check_dependencies; then
        echo "⚠️  Some dependencies are missing or outdated"
        echo "📥 Installing/updating dependencies..."
        source venv/bin/activate
        pip install -r requirements.txt --quiet
        echo "   ✅ Dependencies updated"
        echo ""
    else
        echo "✅ All dependencies installed"
        echo ""
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Load environment variables (suppress warnings)
export KMP_DUPLICATE_LIB_OK=TRUE

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p database/products
mkdir -p embeddings
mkdir -p static/uploads
mkdir -p templates
echo "   ✅ Directories created"
echo ""

# Check for dataset JSON (SKU mapping)
if [ ! -f "database/products.json" ]; then
    echo "📋 Creating sample products.json for SKU mapping..."
    cat > database/products.json << 'EOF'
{
  "products": [
    {
      "sku_id": "SKU-001",
      "product_name": "Sample Product 1",
      "category": "Electronics",
      "price": "99.99",
      "description": "Sample product description",
      "images": [
        "sample_product_1.jpg",
        "sample_product_2.jpg",
        "sample_product_3.jpg"
      ]
    }
  ]
}
EOF
    echo "   ✅ Sample products.json created"
    echo "   📝 Edit database/products.json to add your product SKUs"
    echo ""
fi

# Check if database has images
IMAGE_COUNT=$(find database/products -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l | tr -d ' ')

echo "📊 Database Status:"
echo "   Images found: $IMAGE_COUNT"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  WARNING: No product images found!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📝 To add your products:"
    echo "   1. Copy images to: database/products/"
    echo "      Example: cp /path/to/images/*.jpg database/products/"
    echo ""
    echo "   2. Update products.csv with SKU information:"
    echo "      filename,sku_id,product_name,category,price"
    echo "      product_001.jpg,SKU-12345,Blue Shirt,Clothing,29.99"
    echo ""
    echo "   3. Run this script again to index the database"
    echo ""
    read -p "Press Enter to continue anyway (will start in demo mode)..."
    echo ""
else
    echo "   ✅ Found $IMAGE_COUNT product images"
    echo ""
    
    # Check if index exists
    if [ ! -f "embeddings/faiss.index" ]; then
        echo "🔨 Building search index for the first time..."
        echo "   ⏳ This will take a few minutes (extracting features)..."
        echo ""
        python index_database.py
        echo ""
        echo "   ✅ Index built successfully!"
        echo ""
    else
        echo "✅ Search index exists"
        
        # Check if index is older than database
        INDEX_TIME=$(stat -f %m embeddings/faiss.index 2>/dev/null || stat -c %Y embeddings/faiss.index 2>/dev/null)
        NEWEST_IMAGE=$(find database/products -type f -exec stat -f %m {} \; 2>/dev/null | sort -n | tail -1)
        
        if [ ! -z "$NEWEST_IMAGE" ] && [ "$NEWEST_IMAGE" -gt "$INDEX_TIME" ]; then
            echo "⚠️  Database has been updated since last index"
            read -p "   Rebuild index? (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "   🔨 Rebuilding index..."
                python index_database.py
                echo "   ✅ Index rebuilt!"
            fi
        fi
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING APPLICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Server will start at: http://localhost:5000"
echo "📱 API endpoint: http://localhost:5000/api/search"
echo ""
echo "💡 Quick API test:"
echo "   curl -X POST http://localhost:5000/api/search \\"
echo "     -F \"image=@your_image.jpg\""
echo ""
echo "⏹  Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the application
python app.py

