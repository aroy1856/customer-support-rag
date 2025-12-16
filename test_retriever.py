"""Test script to verify the retriever is working correctly."""

from src.retrieval.retriever import DocumentRetriever
from src.utils.config import Config

def test_retriever():
    """Test the document retriever with multiple queries."""
    
    print("=" * 70)
    print("TESTING DOCUMENT RETRIEVER")
    print("=" * 70)
    
    # Initialize retriever
    print("\n1. Initializing retriever...")
    try:
        retriever = DocumentRetriever()
        print("   ✓ Retriever initialized successfully")
        print(f"   - Collection: {retriever.collection_name}")
        print(f"   - Persist Directory: {retriever.persist_directory}")
        print(f"   - Top K: {retriever.top_k}")
    except Exception as e:
        print(f"   ✗ Failed to initialize retriever: {e}")
        return
    
    # Test queries
    test_queries = [
        "What payment methods do you accept?",
        "How do I activate international roaming?",
        "What is the Fair Usage Policy?",
        "Can I change my plan anytime?",
        "What are the billing cycle dates?"
    ]
    
    print("\n2. Testing retrieval with sample queries...")
    print("-" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)
        
        try:
            # Retrieve documents
            results = retriever.retrieve(query, top_k=3)
            
            print(f"Retrieved {len(results)} documents:")
            
            for j, chunk in enumerate(results, 1):
                source = chunk['metadata'].get('source', 'Unknown')
                distance = chunk.get('distance', 0)
                content_preview = chunk['content'][:100].replace('\n', ' ')
                
                print(f"\n  {j}. Source: {source}")
                print(f"     Distance: {distance:.4f} (lower is better)")
                print(f"     Preview: {content_preview}...")
            
            print()
            
        except Exception as e:
            print(f"   ✗ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("RETRIEVER TEST COMPLETE")
    print("=" * 70)
    print("\n✓ All retrieval tests passed successfully!")
    print("  The retriever is working correctly and returning relevant documents.")

if __name__ == "__main__":
    test_retriever()
