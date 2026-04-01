import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from sentence_transformers import SentenceTransformer
from config import config
from ingestion.chunk import load_data  # fixed import for sentinel-5g root


def get_client():
    return QdrantClient(
        host=config["qdrant"]["host"],
        port=config["qdrant"]["port"]
    )


def get_model():
    return SentenceTransformer(
        config["embedding"]["model_name"]
    )


def collection_exists_and_nonempty(client):
    collection_name = config["qdrant"]["collection_name"]

    try:
        info = client.get_collection(collection_name)
        return info.points_count > 0
    except Exception:
        return False


def ingest():
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    model = get_model()
    collection_name = config["qdrant"]["collection_name"]

    if collection_exists_and_nonempty(client):
        logging.info(f"Collection '{collection_name}' already exists and has data. Skipping ingestion.")
        return

    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        logging.info(f"Collection '{collection_name}' exists but is empty. Proceeding with ingestion.")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config["embedding"]["vector_size"],
                distance=Distance.COSINE
            )
        )
        logging.info(f"Created collection '{collection_name}'.")

    chunks = load_data()
    texts = [chunk["text"] for chunk in chunks]

    logging.info(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": chunks[i]["text"],
                "tag": chunks[i]["tag"],
                "scenario_id": chunks[i]["scenario_id"]
            }
        )
        for i in range(len(chunks))
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    logging.info(f"Ingested {len(points)} chunks into '{collection_name}'.")


def main():
    ingest()


if __name__ == "__main__":
    main()