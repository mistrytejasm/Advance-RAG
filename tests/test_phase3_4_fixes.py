import pytest
from app.generation.context_builder import context_builder
from app.generation.citation_builder import build_citations
from app.generation.response_validator import validate_response
from monitoring.cost_tracker import compute_cost

def test_lost_in_the_middle_context_reordering():
    chunks = [
        {"chunk_id": "c1", "content": "Rank 1 chunk", "rerank_score": 9.0, "page": 1, "section": "A"},
        {"chunk_id": "c2", "content": "Rank 2 chunk", "rerank_score": 8.0, "page": 2, "section": "B"},
        {"chunk_id": "c3", "content": "Rank 3 chunk", "rerank_score": 7.0, "page": 3, "section": "C"},
    ]
    context_str, used = context_builder.build(chunks, max_tokens=1000, max_chunks=5)
    # The alternating boundary distribution should place c1 and c2 at the outer edges
    assert len(used) == 3
    # c2 inserted at 0, c1 at end/start according to left/right alternating strategy
    assert "[Source 1]" in context_str
    assert "[Source 3]" in context_str

def test_citation_builder_preserves_distinct_chunks_same_page():
    used_chunks = [
        {"chunk_id": "c1", "page": 4, "section": "", "source": "pdf", "content_type": "text", "rerank_score": 5.0},
        {"chunk_id": "c2", "page": 4, "section": "", "source": "pdf", "content_type": "text", "rerank_score": 4.0},
    ]
    citations = build_citations(used_chunks)
    assert len(citations) == 2
    assert citations[0]["chunk_id"] == "c1"
    assert citations[1]["chunk_id"] == "c2"

def test_response_validator_pattern_refusals():
    # Subtle variation of refusal
    refusal = "Based on the documents, the provided context does not contain information about salary."
    is_valid, is_grounded, reason = validate_response(refusal)
    assert is_valid is True
    assert is_grounded is False
    
    valid_answer = "The system achieved an accuracy of 94.2% on the test benchmark dataset."
    is_valid, is_grounded, reason = validate_response(valid_answer)
    assert is_valid is True
    assert is_grounded is True

def test_cost_tracker_granular_input_output():
    cost = compute_cost(model="openai/gpt-oss-120b", input_tokens=1000, output_tokens=500)
    # 1000 in ($0.00060) + 500 out ($0.00180 * 0.5 = $0.00090) = $0.00150
    assert cost == 0.0015
