
from src.query_processor import QueryProcessor
from src.llm_service import LLMService
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore


class RAGPipeline:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        self.query_processor = QueryProcessor(
            self.embedding_service,
            self.vector_store
        )

        self.llm_service = LLMService()

    def answer(
        self,
        question: str,
        resume_document_id: str,
        job_description_document_id: str,
        top_k: int = 5
    ) -> dict:

        # Retrieve relevant resume chunks.
        resume_results = self.query_processor.search(
            query=question,
            document_id=resume_document_id,
            document_type="resume",
            limit=top_k
        )

        # Retrieve relevant job description chunks.
        jd_results = self.query_processor.search(
            query=question,
            document_id=job_description_document_id,
            document_type="job_description",
            limit=top_k
        )

        if not resume_results and not jd_results:
            return {
                "answer": "No relevant document content was found.",
                "resume_chunks": [],
                "job_description_chunks": []
            }

        # Build context for the LLM.
        context_parts = []

        for result in resume_results:
            metadata = result["metadata"]

            context_parts.append(
                f"""
                SOURCE: RESUME
                SECTION: {metadata.get('section_name', 'Unknown')}
                CONTENT: {result['text']}
                """
            )

        for result in jd_results:
            metadata = result["metadata"]

            context_parts.append(
                f"""
                SOURCE: JOB DESCRIPTION
                SECTION: {metadata.get('section_name', 'Unknown')}
                CONTENT: {result['text']}
                """
            )

        context = "\n".join(context_parts)

        # Generate the final answer.
        answer = self.llm_service.generate_answer(
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "resume_chunks": resume_results,
            "job_description_chunks": jd_results
        }
