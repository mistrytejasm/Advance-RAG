import pytest
from app.utils.score_normalizer import reciprocal_rank_fusion
from app.services.retrieval.reranker import Reranker

def test_reciprocal_rank_fusion_basic():
    vector_results = [
        {"chunk_id": "c1", "score": 0.95, "metadata": {"content_preview": "Vector top match"}},
        {"chunk_id": "c2", "score": 0.85, "metadata": {"content_preview": "Vector second match"}},
    ]
    bm25_results = [
        {"chunk_id": "c2", "bm25_score": 15.0, "content": "BM25 top match"},
        {"chunk_id": "c3", "bm25_score": 5.0, "content": "BM25 second match"},
    ]
    
    fused = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        vector_weight=0.7,
        bm25_weight=0.3,
        k=60
    )
    
    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "c2"
    assert fused[1]["chunk_id"] == "c1"
    assert fused[2]["chunk_id"] == "c3"

def test_reciprocal_rank_fusion_outlier_resilience():
    vector_results = [{"chunk_id": "c1", "score": 0.88}]
    bm25_results = [{"chunk_id": "c2", "bm25_score": 5000.0}]
    
    fused = reciprocal_rank_fusion(vector_results, bm25_results, vector_weight=0.7, bm25_weight=0.3, k=60)
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "c1"

def test_reranker_uses_full_content(monkeypatch):
    class MockModel:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0 if "FULL_CONTENT_MARKER" in p[1] else 0.0 for p in pairs]
            
    reranker = Reranker()
    monkeypatch.setattr(reranker, "_model", MockModel())
    
    candidates = [
        {
            "chunk_id": "c1",
            "content": "This is the full text containing FULL_CONTENT_MARKER past 200 chars...",
            "metadata": {"content_preview": "Short preview"}
        },
        {
            "chunk_id": "c2",
            "content": "Another text without marker",
            "metadata": {"content_preview": "Short preview"}
        }
    ]
    
    reranked = reranker.rerank(query="Find marker", candidates=candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c1"
    assert reranked[0]["rerank_score"] == 1.0
