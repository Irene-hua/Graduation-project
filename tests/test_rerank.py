import unittest

from src.rag_pipeline.rerank import LocalReranker


class LocalRerankerTest(unittest.TestCase):
    def test_rerank_prefers_lexical_overlap(self):
        reranker = LocalReranker()
        chunks = [
            {"text": "A generic paragraph about cats.", "score": 0.1},
            {"text": "Leslie Hansen responded to Stephanie Panus on 2026-01-01 10:00.", "score": 0.2},
        ]
        out = reranker.rerank("What time did Leslie Hansen respond?", chunks, top_k=2)
        self.assertEqual(out[0]["text"], "Leslie Hansen responded to Stephanie Panus on 2026-01-01 10:00.")
        self.assertIn("rerank_score", out[0])
        self.assertIn("rerank_reason", out[0])

    def test_rerank_uses_metadata_prior_for_ties(self):
        reranker = LocalReranker()
        chunks = [
            {"text": "Query-adjacent text.", "score": 0.1, "metadata": {}},
            {"text": "Query-adjacent text.", "score": 0.1, "metadata": {"source_file": "doc.txt", "chunk_id": 1}},
        ]
        out = reranker.rerank("Query adjacent text?", chunks, top_k=2)
        self.assertEqual(out[0]["metadata"]["source_file"], "doc.txt")


if __name__ == "__main__":
    unittest.main()
