
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION
)


class EmbeddingService:

    def __init__(self):

        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.model = EMBEDDING_MODEL
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.dimension = EMBEDDING_DIMENSION

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Embedding input cannot contain empty text."
            )

        all_embeddings = []

        # Process texts in batches
        for start in range(0, len(texts), self.batch_size):

            batch = texts[
                start:start + self.batch_size
            ]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"
            )

            # Preserve input order using response index
            batch_embeddings = sorted(
                response.data,
                key=lambda item: item.index
            )

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match input count."
                )

            for item in batch_embeddings:

                embedding = item.embedding

                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Unexpected embedding dimension: "
                        f"{len(embedding)}"
                    )

                all_embeddings.append(embedding)

        return all_embeddings
