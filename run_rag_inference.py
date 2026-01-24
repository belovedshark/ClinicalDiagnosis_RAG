#!/usr/bin/env python3
"""
Runner script for Phase 1: RAG Inference Pipeline

This script provides a simple interface to run the RAG inference pipeline
on the WHO evaluation dataset.
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the main inference pipeline
from rag_evaluation.rag_inference import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user. Checkpoint saved.")
        print("   Run again to resume from the last checkpoint.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
