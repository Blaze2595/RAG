
from sys import maxsize
import streamlit as st

from src.file_upload_manager import FileUploadManager
from src.document_processor import DocumentProcessor
from src.text_chunker import TextChunker
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.query_processor import QueryProcessor
from src.llm_service import LLMService
from src.resume_analysis_service import ResumeAnalysisService


st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Career Intelligence Assistant")
st.caption("Resume and Job Description Analyzer using OpenAI")


# Initialize

if "resume_document" not in st.session_state:
    st.session_state.resume_document = None

if "jd_documents" not in st.session_state:
    st.session_state.jd_documents = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# Upload documents

st.header("1. Upload Documents")

resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    key="resume_upload",
    max_upload_size = 5
)

jd_files = st.file_uploader(
    "Upload job descriptions",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="jd_upload",
    max_upload_size = 5
)

if st.button("Upload and Process Documents", type="primary"):

    if not resume_file:
        st.error("Please upload a resume.")

    elif not jd_files:
        st.error("Please upload at least one job description.")

    else:
        try:
            with st.spinner("Processing documents..."):

                upload_manager = FileUploadManager()
                document_processor = DocumentProcessor()
                text_chunker = TextChunker()
                embedding_service = EmbeddingService()
                vector_store = VectorStore()

                # Save uploaded files.
                resume_document = upload_manager.save_resume(
                    resume_file
                )

                jd_documents = upload_manager.save_job_descriptions(
                    jd_files
                )

                all_documents = [
                    resume_document,
                    *jd_documents
                ]

                parsed_documents = []

                # Process and chunk documents.
                for document in all_documents:

                    parsed_document = (
                        document_processor.process_document(document)
                    )

                    parsed_documents.append(parsed_document)

                all_chunks = text_chunker.create_chunks(parsed_documents)

                if not all_chunks:
                    raise ValueError(
                        "No text chunks were generated."
                    )

                # Generate embeddings.
                texts = [
                    chunk.text for chunk in all_chunks
                ]

                embeddings = embedding_service.embed_texts(
                    texts
                )

                # Store in Qdrant.
                stored_count = vector_store.store_chunks(
                    all_chunks,
                    embeddings
                )

                if stored_count != len(all_chunks):
                    raise RuntimeError(
                        "Not all chunks were stored successfully."
                    )

                # Save document references.
                st.session_state.resume_document = resume_document
                st.session_state.jd_documents = jd_documents

                st.session_state.analysis_result = None
                st.session_state.processed = True

                st.success(
                    f"Successfully processed and stored "
                    f"{stored_count} chunks!"
                )

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")


# Resume analysis

if st.session_state.processed:

    st.divider()
    st.header("2. Resume Analysis")

    resume_document = st.session_state.resume_document
    jd_documents = st.session_state.jd_documents

    selected_jd = st.selectbox(
        "Select a job description",
        options=jd_documents,
        format_func=lambda doc: doc.original_filename
    )

    if st.button("Analyze Resume", type="primary"):

        try:
            with st.spinner("Analyzing resume against JD..."):

                embedding_service = EmbeddingService()
                vector_store = VectorStore()

                query_processor = QueryProcessor(
                    embedding_service,
                    vector_store
                )

                llm_service = LLMService()

                analysis_service = ResumeAnalysisService(
                    query_processor,
                    llm_service
                )

                result = analysis_service.analyze(
                    resume_document_id=resume_document.document_id,
                    jd_document_id=selected_jd.document_id
                )

                st.session_state.analysis_result = result
                st.session_state.analysis_jd_id = (
                    selected_jd.document_id
                )

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


# Display analysis

result = st.session_state.analysis_result

if result:

    st.divider()
    st.header("3. Analysis Results")

    score = result["overall_match_score"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Overall Match Score",
            f"{score:.0f}%"
        )

    with col2:
        st.progress(score / 100)
        st.caption(result["score_rationale"])

    with st.expander("Matching Skills", expanded=True):

        for item in result["matching_skills"]:
            st.markdown(f"**{item['skill']}**")
            st.caption(
                f"Resume evidence: {item['resume_evidence']}"
            )
            st.caption(
                f"JD requirement: {item['jd_requirement']}"
            )

    with st.expander("Missing Skills", expanded=True):

        for item in result["missing_skills"]:
            priority = item.get("priority", "Not specified")
            st.markdown(
                f"- **{item['skill']}** — {priority} priority"
            )
            st.caption(item.get("jd_requirement", ""))

    with st.expander("Experience Assessment"):
        st.write(result["experience_assessment"])

    with st.expander("Qualification Gaps"):
        for item in result["qualification_gaps"]:
            st.markdown(f"- {item}")

    with st.expander("Strengths"):
        for item in result["strengths"]:
            st.markdown(f"- {item}")

    with st.expander("Recommendations"):
        for item in result["recommendations"]:
            st.markdown(f"- {item}")


# Ask questions

if st.session_state.processed:

    st.divider()
    st.header("4. Ask Questions")

    question = st.text_input(
        "Ask something about your resume and selected JD",
        placeholder="What skills am I missing for this role?"
    )

    if st.button("Ask AI"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Finding an answer..."):

                    rag_pipeline = RAGPipeline()

                    answer = rag_pipeline.answer(
                        question=question,
                        resume_document_id=(
                            st.session_state.resume_document.document_id
                        ),
                        job_description_document_id=(
                            selected_jd.document_id
                        )
                    )

                    st.markdown("### Answer")
                    st.write(answer["answer"])

            except Exception as e:
                st.error(f"Question processing failed: {str(e)}")
