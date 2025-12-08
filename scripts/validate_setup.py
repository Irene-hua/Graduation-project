#!/usr/bin/env python3
"""
Validation script to check if the RAG system is properly set up
"""

import sys
import os

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check_module(module_name, package_name=None):
    """Check if a Python module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"{GREEN}✓{RESET} {package_name} is installed")
        return True
    except ImportError:
        print(f"{RED}✗{RESET} {package_name} is NOT installed")
        return False


def check_ollama():
    """Check if Ollama is accessible"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print(f"{GREEN}✓{RESET} Ollama server is running")
            return True
        else:
            print(f"{YELLOW}⚠{RESET} Ollama server is not responding correctly")
            return False
    except Exception:
        print(f"{RED}✗{RESET} Ollama server is not running")
        print("  Start with: ollama serve")
        return False


def check_directories():
    """Check if required directories exist"""
    dirs = ['data/raw', 'logs', 'qdrant_storage']
    all_exist = True
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"{GREEN}✓{RESET} Directory exists: {dir_path}")
        else:
            print(f"{YELLOW}⚠{RESET} Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist


def main():
    print("="*60)
    print("Privacy-Preserving RAG System - Setup Validation")
    print("="*60)
    print()
    
    # Check Python version
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"{GREEN}✓{RESET} Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"{RED}✗{RESET} Python 3.8+ required, found {version.major}.{version.minor}")
    print()
    
    # Check required packages
    print("Checking required packages...")
    required_packages = [
        ('cryptography', 'cryptography'),
        ('yaml', 'pyyaml'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('sentence_transformers', 'sentence-transformers'),
        ('qdrant_client', 'qdrant-client'),
    ]
    
    all_installed = True
    for module, package in required_packages:
        if not check_module(module, package):
            all_installed = False
    print()
    
    # Check Ollama
    print("Checking Ollama...")
    ollama_ok = check_ollama()
    print()
    
    # Check directories
    print("Checking directories...")
    dirs_ok = check_directories()
    print()
    
    # Check custom modules
    print("Checking custom modules...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    custom_modules = [
        'src.encryption',
        'src.document_processing',
        'src.embedding',
        'src.retrieval',
        'src.llm',
        'src.rag_pipeline',
        'src.audit',
        'src.evaluation'
    ]
    
    modules_ok = True
    for module in custom_modules:
        if not check_module(module, module.replace('src.', '')):
            modules_ok = False
    print()
    
    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    
    if all_installed and modules_ok and dirs_ok:
        print(f"{GREEN}✓ System is ready!{RESET}")
        if not ollama_ok:
            print(f"{YELLOW}Note: Ollama is not running. Start it with 'ollama serve'{RESET}")
        return 0
    else:
        print(f"{RED}✗ System is not fully set up{RESET}")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nTo create missing directories:")
        print("  bash scripts/setup.sh")
        return 1


if __name__ == '__main__':
    sys.exit(main())
