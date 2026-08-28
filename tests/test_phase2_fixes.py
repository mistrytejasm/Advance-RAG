import pytest
import asyncio
from app.generation.answer_service import get_cached_answer, set_cached_answer, _QUERY_CACHE

def test_query_cache_set_and_get():
    _QUERY_CACHE.clear()
    query = "What is Convolutional Neural Network?"
    doc_id = "doc-test-123"
    fake_response = {"answer": "A CNN is a neural network for images.", "is_grounded": True}
    
    set_cached_answer(query, doc_id, fake_response)
    
    cached = get_cached_answer(query, doc_id)
    assert cached is not None
    assert cached["answer"] == fake_response["answer"]
    
    # Check case-insensitivity and trim resilience
    cached_upper = get_cached_answer("  what is CONVOLUTIONAL neural network?  ", doc_id)
    assert cached_upper is not None
    assert cached_upper["answer"] == fake_response["answer"]

@pytest.mark.asyncio
async def test_async_llm_generator_mock(monkeypatch):
    from app.generation.llm_generator import llm_generator
    
    async def mock_create(**kwargs):
        class MockChoice:
            message = type("Message", (), {"content": "Async mock response"})()
        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
        class MockResponse:
            choices = [MockChoice()]
            usage = MockUsage()
        return MockResponse()
        
    monkeypatch.setattr(llm_generator._async_client.chat.completions, "create", mock_create)
    res = await llm_generator.generate_async([{"role": "user", "content": "hello"}])
    assert res["answer"] == "Async mock response"
    assert res["total_tokens"] == 15
