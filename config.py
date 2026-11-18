
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

EMBEDDING_DIMENSION = 1536

# Qdrant configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "resume_analysis"
)

# Processing configuration
EMBEDDING_BATCH_SIZE = 64
QDRANT_BATCH_SIZE = 64


def validate_config():

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing in .env"
        )


CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")