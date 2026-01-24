#!/usr/bin/env python3
"""
Pre-flight check script to verify Phase 1 setup.

This script checks all prerequisites before running the RAG inference pipeline.
"""
import sys
import os

def check_imports():
    """Check if all required packages are installed."""
    print("🔍 Checking Python packages...")
    missing = []
    
    packages = [
        ("qdrant_client", "qdrant-client"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
    ]
    
    for module, package in packages:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    return missing


def check_qdrant():
    """Check Qdrant connection."""
    print("\n🔍 Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        from rag.config import QDRANT_URL
        
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        print(f"  ✅ Connected to Qdrant at {QDRANT_URL}")
        print(f"  📚 Available collections: {[c.name for c in collections.collections]}")
        
        # Check if documents collection exists
        from rag.config import COLLECTION_NAME
        collection_names = [c.name for c in collections.collections]
        if COLLECTION_NAME in collection_names:
            print(f"  ✅ Collection '{COLLECTION_NAME}' exists")
            
            # Get collection info
            collection_info = client.get_collection(COLLECTION_NAME)
            print(f"  📊 Collection has {collection_info.points_count} points")
            return True
        else:
            print(f"  ⚠️  Collection '{COLLECTION_NAME}' not found")
            print(f"     Available: {collection_names}")
            return False
            
    except Exception as e:
        print(f"  ❌ Failed to connect to Qdrant: {e}")
        return False


def check_data_files():
    """Check if evaluation data exists."""
    print("\n🔍 Checking data files...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    eval_file = os.path.join(project_root, "evaluate_RAG.jsonl")
    
    if os.path.exists(eval_file):
        print(f"  ✅ Evaluation dataset found: evaluate_RAG.jsonl")
        
        # Count lines
        with open(eval_file, 'r') as f:
            lines = sum(1 for line in f if line.strip())
        print(f"  📊 Contains {lines} cases")
        return True
    else:
        print(f"  ❌ Evaluation dataset not found: {eval_file}")
        return False


def check_rag_system():
    """Check if RAG system can be imported."""
    print("\n🔍 Checking RAG system...")
    
    try:
        from rag.pipeline import RAGPipeline
        from rag.retriever import Retriever
        from rag.generator import Generator
        print("  ✅ RAG pipeline can be imported")
        print("  ✅ Retriever can be imported")
        print("  ✅ Generator can be imported")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import RAG system: {e}")
        return False


def check_directories():
    """Check if output directories exist."""
    print("\n🔍 Checking directory structure...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        "rag_evaluation",
        "rag_evaluation/config",
        "rag_evaluation/results",
        "rag_evaluation/utils",
    ]
    
    all_exist = True
    for dir_path in dirs:
        full_path = os.path.join(project_root, dir_path)
        if os.path.exists(full_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_gpu():
    """Check GPU availability."""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✅ MPS (Apple Silicon) available")
        else:
            print(f"  ⚠️  No GPU detected - will use CPU (slower)")
        
    except Exception as e:
        print(f"  ⚠️  Could not check GPU: {e}")


def main():
    """Run all pre-flight checks."""
    print("="*80)
    print("PHASE 1 PRE-FLIGHT CHECK")
    print("="*80)
    
    results = []
    
    # Check packages
    missing_packages = check_imports()
    results.append(("Packages", len(missing_packages) == 0))
    
    # Check Qdrant
    qdrant_ok = check_qdrant()
    results.append(("Qdrant", qdrant_ok))
    
    # Check data files
    data_ok = check_data_files()
    results.append(("Data files", data_ok))
    
    # Check RAG system
    rag_ok = check_rag_system()
    results.append(("RAG system", rag_ok))
    
    # Check directories
    dirs_ok = check_directories()
    results.append(("Directories", dirs_ok))
    
    # Check GPU (informational only)
    check_gpu()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print(f"   Install with: pip install {' '.join(missing_packages)}")
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("   You're ready to run Phase 1:")
        print("   python run_rag_inference.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("   Please fix the issues above before running Phase 1")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
