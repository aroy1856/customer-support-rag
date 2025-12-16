"""Functional testing script for the RAG system."""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.answer_generator import AnswerGenerator
from tests.test_queries import TEST_QUERIES


def run_functional_tests():
    """Run functional tests with predefined queries."""
    print("=" * 80)
    print("TELECOM RAG SYSTEM - FUNCTIONAL TESTING")
    print("=" * 80)
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total test queries: {len(TEST_QUERIES)}\n")
    
    # Initialize answer generator
    print("Initializing RAG system...")
    try:
        answer_gen = AnswerGenerator()
        print("✓ System initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        return
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(TEST_QUERIES)}: {test_case['category']}")
        print(f"{'='*80}")
        print(f"\nQuestion: {test_case['question']}")
        
        try:
            # Generate answer
            result = answer_gen.generate_answer(
                query=test_case['question'],
                include_sources=True,
                log_interaction=True
            )
            
            # Display answer
            print(f"\nAnswer:\n{result['answer']}")
            
            # Display retrieved sources
            print(f"\nRetrieved Sources: {', '.join(result['sources'])}")
            print(f"Number of chunks retrieved: {len(result['retrieved_chunks'])}")
            
            # Simple validation: check if answer is not empty
            if result['answer'] and len(result['answer']) > 50:
                print("\n✓ TEST PASSED: Answer generated successfully")
                passed += 1
                test_result = "PASSED"
            else:
                print("\n✗ TEST FAILED: Answer too short or empty")
                failed += 1
                test_result = "FAILED"
            
            # Store result
            results.append({
                'test_number': i,
                'category': test_case['category'],
                'question': test_case['question'],
                'answer': result['answer'],
                'sources': result['sources'],
                'num_chunks': len(result['retrieved_chunks']),
                'result': test_result
            })
            
        except Exception as e:
            print(f"\n✗ TEST FAILED: Error - {e}")
            failed += 1
            results.append({
                'test_number': i,
                'category': test_case['category'],
                'question': test_case['question'],
                'error': str(e),
                'result': "FAILED"
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(TEST_QUERIES)}")
    print(f"Passed: {passed} ({passed/len(TEST_QUERIES)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(TEST_QUERIES)*100:.1f}%)")
    
    # Save results to file
    results_file = Path(__file__).parent / "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(TEST_QUERIES),
            'passed': passed,
            'failed': failed,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    run_functional_tests()
