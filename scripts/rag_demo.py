# scripts/rag_demo.py

# Ensure the repository root is on sys.path so `from rag.pipeline import ...`
# works even when this script is executed with CWD == scripts/
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.pipeline import RAGPipeline

def main():
    rag = RAGPipeline()
    query = "A 26-year-old woman with persistent fever and rash"
    print(f"\n🩺 Query: {query}\n")
    answer = rag.run(query=query, top_k=3)
    print("\n================= 💬 Diagnosis / Response =================")
    print(answer)
    print("===========================================================\n")

if __name__ == "__main__":
    main()