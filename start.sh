#!/bin/bash

# Start script for Image-Based Product Matching API

echo "🚀 Starting Image-Based Product Matching API..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

echo "✅ Starting server..."
echo "📍 API will be available at: http://127.0.0.1:8000"
echo "📚 Interactive docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

