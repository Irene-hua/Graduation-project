import unittest

from src.rag_pipeline.rag_system import RAGSystem
from src.rag_pipeline.rerank import LocalReranker


class FakeRetriever:
    def retrieve(self, question, top_k=5):
        return [
            {"text": "Cats are unrelated.", "score": 0.1},
            {"text": "Leslie Hansen responded to Stephanie Panus on 2026-01-01 10:00.", "score": 0.2},
        ]


class FakeLLM:
    def generate(self, prompt):
        self.last_prompt = prompt
        return {"response": "The answer is in the context."}


class RerankerIntegrationTest(unittest.TestCase):
    def test_reranker_reorders_before_context(self):
        llm = FakeLLM()
        rag = RAGSystem(FakeRetriever(), llm, reranker=LocalReranker())
        result = rag.answer_question("What time did Leslie Hansen respond?", top_k=2)

        self.assertEqual(result["answer"], "The answer is in the context.")
        self.assertEqual(
            result["used_chunks"][0]["text"],
            "Leslie Hansen responded to Stephanie Panus on 2026-01-01 10:00.",
        )
        self.assertIn("Leslie Hansen responded", llm.last_prompt)
        self.assertGreaterEqual(result["used_chunks"][0]["rerank_score"], result["used_chunks"][1]["rerank_score"])


if __name__ == "__main__":
    unittest.main()
