
from qdrant_client import QdrantClient, models

from src.text_chunker import TextChunk
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    QDRANT_BATCH_SIZE
)


class VectorStore:

    def __init__(self):

        self.collection_name = QDRANT_COLLECTION

        self.client = QdrantClient(
            url=QDRANT_URL,
            timeout=30
        )

        self.batch_size = QDRANT_BATCH_SIZE

        self._create_collection()

    def _create_collection(self):

        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )

            print(
                f"Collection created: {self.collection_name}"
            )

        else:

            collection = self.client.get_collection(
                collection_name=self.collection_name
            )

            vector_config = collection.config.params.vectors

            if vector_config.size != EMBEDDING_DIMENSION:
                raise ValueError(
                    "Qdrant collection dimension mismatch. "
                    f"Expected {EMBEDDING_DIMENSION}, "
                    f"found {vector_config.size}."
                )

            print(
                f"Using existing collection: {self.collection_name}"
            )

        # Create indexes for frequently filtered metadata
        for field in [
            "document_id",
            "document_type",
            "section_name"
        ]:

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True
            )

    def store_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]]
    ) -> int:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return 0

        stored_count = 0

        # Store vectors in batches
        for start in range(0, len(chunks), self.batch_size):

            batch_chunks = chunks[
                start:start + self.batch_size
            ]

            batch_embeddings = embeddings[
                start:start + self.batch_size
            ]

            points = []

            for chunk, embedding in zip(
                batch_chunks,
                batch_embeddings
            ):

                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Invalid vector dimension for "
                        f"chunk {chunk.chunk_id}"
                    )

                payload = {
                    "text": chunk.text,
                    **chunk.metadata
                }

                point = models.PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            stored_count += len(points)

        return stored_count

    def delete_document(self, document_id: str):

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id
                            )
                        )
                    ]
                )
            ),
            wait=True
        )

        return result

    def get_collection_count(self) -> int:

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )

        return result.count

    def health_check(self) -> bool:

        try:
            self.client.get_collections()
            return True

        except Exception:
            return False