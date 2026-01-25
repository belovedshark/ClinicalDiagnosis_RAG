"""scripts/rag_demo.py

Small interactive demo to ask questions to the local RAG pipeline.
This script inserts the repository root onto sys.path so it can be run
directly from the project directory without setting PYTHONPATH.
"""
from pathlib import Path
import sys
import time

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

        # Track total time
        total_start = time.time()
        
        # Step 1: Retrieval
        print("\n⏱️  Step 1: Retrieving relevant documents...")
        retrieval_start = time.time()
        
        if image_path is not None:
            contexts = rag.retriever.retrieve(query_image=image_path, top_k=3)
        elif question is not None:
            contexts = rag.retriever.retrieve(query_text=question, top_k=3)
        else:
            print("❌ No query provided.")
            continue
            
        retrieval_time = time.time() - retrieval_start
        print(f"   ✅ Retrieved {len(contexts)} documents in {retrieval_time:.2f}s")
        
        if not contexts:
            print("⚠️  No relevant documents found in the database.")
            continue
        
        # Step 2: Generation
        print("\n⏱️  Step 2: Generating answer with LLM...")
        generation_start = time.time()
        
        context_text = "\n\n".join(contexts)
        pre_prompt = """
        You are a clinical reasoning assistant specializing in tropical and infectious diseases.

        The following case describes a patient in a tropical, resource-limited region. 
        Based on the patient's presentation, provide:
        1. The **most likely diagnosis** with reasoning.
        2. The **key differential diagnoses** with brief distinguishing features.
        3. A **recommended management approach** appropriate for a low-resource setting (diagnostic steps + initial treatment priorities).

        Patient case:
        """
        q = pre_prompt + (question or "Describe the image.")
        answer = rag.generator.generate(context_text, q)
        
        generation_time = time.time() - generation_start
        print(f"   ✅ Generated answer in {generation_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # Display results with timing summary
        print("\n" + "="*80)
        print("🩺 Answer:\n")
        print(answer)
        print("\n" + "="*80)
        print("⏱️  Performance Summary:")
        print(f"   • Retrieval:  {retrieval_time:.2f}s ({retrieval_time/total_time*100:.1f}%)")
        print(f"   • Generation: {generation_time:.2f}s ({generation_time/total_time*100:.1f}%)")
        print(f"   • Total:      {total_time:.2f}s")
        print("="*80)

if __name__ == "__main__":
    main()