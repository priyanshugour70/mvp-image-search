#!/bin/bash
# Simple startup script for the Image Retrieval System

set -e

echo "🚀 Starting Object Detection & Image Retrieval System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo ""

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p database embeddings static/uploads templates
echo "✅ Directories created"
echo ""

# Check if database has images
IMAGE_COUNT=$(find database -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "⚠️  Warning: No images found in database/"
    echo "   Please add images to the database/ directory"
    echo "   Example: cp /path/to/images/*.jpg database/"
    echo ""
fi

# Check if index exists
if [ ! -f "embeddings/faiss.index" ]; then
    if [ "$IMAGE_COUNT" -gt 0 ]; then
        echo "🔨 Building search index..."
        python index_database.py
        echo "✅ Index built successfully"
        echo ""
    else
        echo "⚠️  Cannot build index: No images in database"
        echo "   Add images first, then run: python index_database.py"
        echo ""
    fi
else
    echo "✅ Search index found"
    echo ""
fi

# Start the Flask application
echo "🌐 Starting Flask application..."
echo "   Open http://localhost:5000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python app.py

