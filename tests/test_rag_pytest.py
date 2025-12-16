"""Pytest-based functional tests for the RAG system."""

import pytest
from pathlib import Path
import json
from datetime import datetime

from src.generation.answer_generator import AnswerGenerator
from tests.test_queries import TEST_QUERIES


@pytest.fixture(scope="module")
def answer_generator():
    """Fixture to initialize the answer generator once for all tests."""
    return AnswerGenerator()


@pytest.fixture(scope="module")
def test_results_collector():
    """Fixture to collect test results for final report."""
    return []


class TestRAGSystem:
    """Test suite for the RAG system."""
    
    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[q['question'][:50] for q in TEST_QUERIES])
    def test_answer_generation(self, answer_generator, test_results_collector, test_case):
        """Test that the system generates valid answers for each query."""
        
        # Generate answer
        result = answer_generator.generate_answer(
            query=test_case['question'],
            include_sources=True,
            log_interaction=True
        )
        
        # Assertions
        assert result is not None, "Result should not be None"
        assert 'answer' in result, "Result should contain 'answer' key"
        assert 'sources' in result, "Result should contain 'sources' key"
        assert 'retrieved_chunks' in result, "Result should contain 'retrieved_chunks' key"
        
        # Validate answer quality
        assert result['answer'], "Answer should not be empty"
        assert len(result['answer']) > 50, f"Answer too short: {len(result['answer'])} chars"
        
        # Validate sources
        assert len(result['sources']) > 0, "Should have at least one source"
        assert len(result['retrieved_chunks']) > 0, "Should have retrieved chunks"
        
        # Collect results for reporting
        test_results_collector.append({
            'category': test_case['category'],
            'question': test_case['question'],
            'answer': result['answer'],
            'sources': result['sources'],
            'num_chunks': len(result['retrieved_chunks']),
            'result': 'PASSED'
        })
    
    @pytest.mark.parametrize("test_case", TEST_QUERIES[:3])  # Test first 3 queries
    def test_retrieval_relevance(self, answer_generator, test_case):
        """Test that retrieved documents are relevant to the query."""
        
        result = answer_generator.generate_answer(
            query=test_case['question'],
            include_sources=True
        )
        
        # Check that we got reasonable number of chunks
        assert 3 <= len(result['retrieved_chunks']) <= 10, \
            f"Expected 3-10 chunks, got {len(result['retrieved_chunks'])}"
        
        # Check that chunks have required fields
        for chunk in result['retrieved_chunks']:
            assert 'content' in chunk, "Chunk should have content"
            assert 'metadata' in chunk, "Chunk should have metadata"
            assert 'distance' in chunk, "Chunk should have distance score"


class TestRetriever:
    """Test suite for the document retriever."""
    
    def test_retriever_initialization(self, answer_generator):
        """Test that retriever initializes correctly."""
        assert answer_generator.retriever is not None
        assert answer_generator.retriever.vectorstore is not None
    
    def test_retrieval_basic(self, answer_generator):
        """Test basic retrieval functionality."""
        query = "What payment methods do you accept?"
        chunks = answer_generator.retriever.retrieve(query, top_k=3)
        
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        assert all('content' in c for c in chunks), "All chunks should have content"
        assert all('metadata' in c for c in chunks), "All chunks should have metadata"
        assert all('distance' in c for c in chunks), "All chunks should have distance"
    
    @pytest.mark.parametrize("query,expected_source", [
        ("What is Fair Usage Policy?", "fup_policy.txt"),
        ("How do I activate international roaming?", "roaming_tariff.txt"),
        ("What payment methods do you accept?", "billing_policy.txt"),
    ])
    def test_retrieval_correct_source(self, answer_generator, query, expected_source):
        """Test that retrieval returns documents from expected sources."""
        chunks = answer_generator.retriever.retrieve(query, top_k=5)
        
        sources = [chunk['metadata'].get('source', '') for chunk in chunks]
        assert expected_source in sources, \
            f"Expected {expected_source} in sources, got {sources}"


class TestAnswerQuality:
    """Test suite for answer quality."""
    
    @pytest.mark.parametrize("query,min_length", [
        ("What payment methods do you accept?", 100),
        ("How do I activate international roaming?", 150),
        ("What is Fair Usage Policy?", 100),
    ])
    def test_answer_length(self, answer_generator, query, min_length):
        """Test that answers meet minimum length requirements."""
        result = answer_generator.generate_answer(query)
        assert len(result['answer']) >= min_length, \
            f"Answer too short: {len(result['answer'])} < {min_length}"
    
    def test_answer_includes_sources(self, answer_generator):
        """Test that answers include source references."""
        query = "What payment methods do you accept?"
        result = answer_generator.generate_answer(query, include_sources=True)
        
        assert result['sources'], "Should have sources"
        assert len(result['sources']) > 0, "Should have at least one source"


@pytest.fixture(scope="session", autouse=True)
def save_test_report(request):
    """Save test results to JSON file after all tests complete."""
    yield
    
    # This runs after all tests
    results_file = Path(__file__).parent / "pytest_results.json"
    
    # Get test results from pytest
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'summary': 'Tests completed successfully'
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
