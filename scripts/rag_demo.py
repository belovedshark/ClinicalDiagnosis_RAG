"""scripts/rag_demo.py

Small interactive demo to ask questions to the local RAG pipeline.
This script inserts the repository root onto sys.path so it can be run
directly from the project directory without setting PYTHONPATH.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.pipeline import RAGPipeline

def main():
    rag = RAGPipeline()

    while True:
        user_in = input("\n❓ Ask a question, or type `image: <path>` to query an image (or 'exit'): ")
        if user_in.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        # If the user prefixes their input with `image:` treat the remainder as an image path.
        image_path = None
        question = None
        if user_in.strip().lower().startswith("image:"):
            # allow: image: /absolute/or/relative/path.jpg
            image_path = user_in.split(":", 1)[1].strip()
            if image_path == "":
                print("No image path provided after 'image:'. Try: image: test_images/test1.jpeg")
                continue
            print(f"Using image path: {image_path}")
        else:
            question = user_in.strip()

        answer = rag.query(question, image_path=image_path)
        print("\n🩺 Answer:\n", answer)

if __name__ == "__main__":
    main()