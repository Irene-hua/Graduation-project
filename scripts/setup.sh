#!/bin/bash
# Setup script for privacy-preserving RAG system

set -e

echo "==================================="
echo "Privacy-Preserving RAG System Setup"
echo "==================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/test_datasets
mkdir -p models
mkdir -p logs
mkdir -p qdrant_storage

# Check if Ollama is installed
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
else
    echo "✗ Ollama is not installed"
    echo "Please install Ollama from: https://ollama.com/download"
    echo "After installation, run: ollama pull llama2"
fi

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start Ollama: ollama serve"
echo "3. Pull a model: ollama pull llama2"
echo "4. Add documents to data/raw/"
echo "5. Run ingestion: python scripts/ingest_documents.py --input_dir data/raw/ --generate_key"
echo "6. Run RAG: python scripts/run_rag.py"
echo ""
