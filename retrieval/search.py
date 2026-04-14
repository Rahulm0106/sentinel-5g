from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document

from config import config

# ✅ Initialize ONCE (global scope)
client = QdrantClient(
    host=config["qdrant"]["host"],
    port=config["qdrant"]["port"]
)

embedder = SentenceTransformer(config["embedding"]["model_name"])
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def search(query: str, top_k: int = None, rerank_top_k: int = None):
    # Use config defaults if not provided
    top_k = top_k or config["retrieval"]["top_k"]
    rerank_top_k = rerank_top_k or config["retrieval"]["rerank_top_k"]

    # 1. Embed query
    query_vector = embedder.encode(query).tolist()

    # 2. Search Qdrant
    results = client.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=query_vector,
        limit=top_k
    ).points

    # 3. Fetch sibling chunks
    scenario_ids = list(set([res.payload["scenario_id"] for res in results]))

    all_chunks = []
    for sid in scenario_ids:
        hits, _ = client.scroll(
            collection_name=config["qdrant"]["collection_name"],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="scenario_id",
                        match=MatchValue(value=sid)
                    )
                ]
            ),
            limit=10
        )

        for h in hits:
            all_chunks.append({
                "text": h.payload["text"],
                "tag": h.payload.get("tag", ""),
                "scenario_id": sid
            })

    # 4. Deduplicate
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk["text"], chunk["scenario_id"])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    # 5. Re-rank
    pairs = [(query, chunk["text"]) for chunk in unique_chunks]
    scores = cross_encoder.predict(pairs)

    for i, chunk in enumerate(unique_chunks):
        chunk["score"] = float(scores[i])

    reranked = sorted(unique_chunks, key=lambda x: x["score"], reverse=True)

    # 6. Return Documents
    documents = []
    for chunk in reranked[:rerank_top_k]:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "tag": chunk["tag"],
                    "scenario_id": chunk["scenario_id"],
                    "score": chunk["score"]
                }
            )
        )

    return documents

if __name__ == "__main__":
    results = search("handover failure causes")
    for doc in results:
        print(f"Tag: {doc.metadata['tag']}")
        print(f"Score: {doc.metadata['score']:.4f}")
        print(f"Text: {doc.page_content}\n")