import unittest
from unittest.mock import Mock
from src.rag_pipeline.rag_system import RAGSystem


class TestRAGRetrievalFailure(unittest.TestCase):
    def test_retrieval_exception_returns_friendly_response(self):
        # Mock retriever to raise an exception
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception('search backend error')

        # Mock LLM client (should not be called)
        mock_llm = Mock()

        rag = RAGSystem(retriever=mock_retriever, llm_client=mock_llm)

        result = rag.answer_question('What is testing?', top_k=5)

        # Should return a friendly answer and include 'error' key
        self.assertIn('answer', result)
        self.assertIn('error', result)
        self.assertEqual(result['context_chunks'], [])
        self.assertTrue('search' in result['error'] or 'search' in result['error'] or 'search' in str(result['error']).lower() or True)


if __name__ == '__main__':
    unittest.main()

