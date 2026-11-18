
from qdrant_client import models

class QueryProcessor:
    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        document_id: str | None = None,
        document_type: str | None = None,
        limit: int = 5
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        # Generate an embedding for the user's question.
        query_embedding = self.embedding_service.embed_texts([query])[0]

        # Build optional Qdrant filters.
        conditions = []

        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        if document_type:
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type)
                )
            )

        query_filter = (
            models.Filter(must=conditions)
            if conditions else None
        )

        # Retrieve the most similar chunks.
        response = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append({
                "text": payload.get("text", ""),
                "score": point.score,
                "metadata": payload
            })

        return results


    def get_document_chunks(
        self,
        document_id: str,
        document_type: str
    ) -> list[dict]:

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type)
                )
            ]
        )

        results = []
        offset = None

        while True:
            response = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in response[0]:
                payload = point.payload or {}

                results.append({
                    "text": payload.get("text", ""),
                    "metadata": payload
                })

            offset = response[1]

            if offset is None:
                break

        # Preserve the original chunk order.
        results.sort(
            key=lambda item: (
                item["metadata"].get("section_index", 0),
                item["metadata"].get("chunk_index", 0)
            )
        )

        return results
