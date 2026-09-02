from rag.hybrid_retriever import HybridRetriever


def test_hybrid_retrieval():
    retriever = HybridRetriever()

    results = retriever.retrieve(
        "Apple total net sales 2025",
        top_k=5,
        candidate_k=10,
    )

    assert len(results) == 5
    assert all(
        "chunk_id" in result
        for result in results
    )
