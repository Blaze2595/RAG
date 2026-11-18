# Career Intelligence Assistant – RAG based Resume and Job Description Analyzer

An application that analyzes a candidate's resume against one or more job descriptions using **Retrieval-Augmented Generation (RAG), OpenAI embeddings, and Qdrant vector database**.

The application identifies matching skills, missing skills, experience gaps, strengths, and improvement recommendations. It also provides a conversational interface where users can ask questions about their resume and selected job description.

---

## 1. Technical Architecture

### Technology Stack

| Component | Technology / Library | Purpose |
|---|---|---|
| Programming Language | Python | Core application development |
| Frontend | Streamlit | Interactive web interface |
| Document Processing | PyPDF | Extract text from PDF files |
| Document Processing | python-docx | Extract text from DOCX files |
| Text Chunking | LangChain Text Splitters | Section-based and recursive text chunking |
| Embedding Model | OpenAI `text-embedding-3-small` | Convert text into vector embeddings |
| LLM | OpenAI `gpt-4.1-mini` | Generate answers and resume analysis |
| Vector Database | Qdrant | Store and retrieve document embeddings |
| Vector Database Client | qdrant-client | Python integration with Qdrant |
| Configuration | python-dotenv | Load environment variables from `.env` |
| Database Hosting | Docker | Run Qdrant locally |

### High-Level Architecture

```text
                  USER
                   |
                   v
          +------------------+
          |    Streamlit     |
          |   User Interface |
          +------------------+
                   |
                   v
          +------------------+
          | File Upload      |
          | Manager          |
          +------------------+
                   |
                   v
          +------------------+
          | Document         |
          | Processor        |
          +------------------+
                   |
                   v
          +------------------+
          | Text Chunker     |
          | Section-based    |
          | Recursive Split  |
          +------------------+
                   |
                   v
          +------------------+
          | Embedding Service|
          | OpenAI Embeddings|
          +------------------+
                   |
                   v
          +------------------+
          | Qdrant Vector DB |
          | Embeddings +     |
          | Metadata + Text  |
          +------------------+
                   |
                   v
          +------------------+
          | Query Processor  |
          +------------------+
                   |
                   v
          +------------------+
          | RAG Pipeline /   |
          | Resume Analysis  |
          | Service          |
          +------------------+
                   |
                   v
          +------------------+
          | LLM Service      |
          | OpenAI Chat Model|
          +------------------+
                   |
                   v
          +------------------+
          | Analysis /       |
          | Generated Answer |
          +------------------+
                   |
                   v
                USER
```

### Architecture Overview

The application follows a modular, class-based architecture. Each component has a specific responsibility, making the application easier to maintain, test, and extend.

The processing workflow is divided into three main stages:

1. **Document Ingestion:** Upload, parse, chunk, and generate embeddings.
2. **Information Retrieval:** Search Qdrant for relevant document content.
3. **AI Processing:** Use the retrieved context to generate answers or structured resume analysis.

---

## 2. Basic Framework

The application uses a modular Python architecture with individual classes responsible for document processing, embeddings, vector storage, retrieval, and LLM interactions.

### Project Structure

```text
ResumeRAG/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
│
├── uploads/
│   ├── resumes/
│   └── job_descriptions/
│
└── src/
    ├── __init__.py
    ├── file_upload_manager.py
    ├── document_processor.py
    ├── text_chunker.py
    ├── embedding_service.py
    ├── vector_store.py
    ├── query_processor.py
    ├── llm_service.py
    ├── resume_analysis_service.py
    └── rag_pipeline.py
```

### Component Responsibilities

| File / Class | Responsibility |
|---|---|
| `app.py` | Streamlit frontend and application orchestration |
| `config.py` | Centralized application configuration |
| `file_upload_manager.py` | Upload validation and file management |
| `document_processor.py` | Extract and normalize text from documents |
| `text_chunker.py` | Split documents into meaningful sections and chunks |
| `embedding_service.py` | Generate vector embeddings using OpenAI |
| `vector_store.py` | Manage Qdrant collection, vector storage, and deletion |
| `query_processor.py` | Generate query embeddings and retrieve relevant chunks |
| `llm_service.py` | Interact with the OpenAI chat model |
| `rag_pipeline.py` | Coordinate retrieval and LLM-based question answering |
| `resume_analysis_service.py` | Perform structured resume and JD comparison |

---

## 3. High-Level Project Workflow

### Step 1: Document Upload

**Component:** `FileUploadManager`

Users upload:
- One resume.
- One or more job descriptions.

Supported formats:
- PDF
- DOCX
- TXT

Responsibilities:
- Validate file extensions and file sizes.
- Generate unique document IDs.
- Save files in their respective upload directories.
- Maintain document metadata such as document ID, filename, and document type.

Each document is uniquely identified so that resume and job description content can be retrieved independently.

### Step 2: Document Processing

**Component:** `DocumentProcessor`

Uploaded documents are passed to the document processor for text extraction.

Processing includes:
- Extracting text from PDF files using PyPDF.
- Extracting paragraphs and tables from DOCX files using python-docx.
- Reading text files directly.
- Normalizing whitespace while preserving document structure.

The extracted content is represented using a `ParsedDocument` object containing the document text and associated metadata.

### Step 3: Section-Based Text Chunking

**Component:** `TextChunker`

Instead of splitting documents into arbitrary character-based blocks, the application uses section-aware chunking.

Examples of recognized sections:

**Resume**
- Professional Summary
- Technical Skills
- Work Experience
- Education
- Certifications
- Projects

**Job Description**
- Job Overview
- Responsibilities
- Required Skills
- Preferred Qualifications
- Experience Requirements

The chunking process follows these rules:

1. Identify document sections using heading detection.
2. Group content under the corresponding section.
3. Split oversized sections using `RecursiveCharacterTextSplitter`.
4. Maintain chunk overlap to preserve contextual continuity.
5. Attach metadata to every generated chunk.

**Current chunking configuration:**

| Parameter | Value |
|---|---:|
| Chunk size | 1000 characters |
| Chunk overlap | 150 characters |
| Splitting strategy | Recursive character splitting |
| Section handling | Section-based |

This approach preserves the logical structure of resumes and job descriptions, improving the quality of subsequent retrieval.

### Step 4: Embedding Generation

**Component:** `EmbeddingService`

The application uses OpenAI's `text-embedding-3-small` model to convert each text chunk into a numerical vector representation.

Process:

1. Collect the generated text chunks.
2. Send them to the OpenAI Embeddings API in batches.
3. Receive the vector representation for each chunk.
4. Associate each vector with its original chunk and metadata.

**Embedding configuration:**

| Parameter | Value |
|---|---|
| Model | `text-embedding-3-small` |
| Default dimensions | 1536 |
| Batch size | 64 |
| Encoding | Float |

These embeddings represent the semantic meaning of the text and allow similar content to be retrieved even when the wording differs.

### Step 5: Vector Storage

**Component:** `VectorStore`

Qdrant is used as the vector database and runs locally through Docker.

Each stored point contains:

- Unique point ID.
- Embedding vector.
- Original chunk text.
- Document metadata.

**Qdrant configuration:**

| Parameter | Value |
|---|---|
| Vector database | Qdrant |
| Collection | `CIA` |
| Vector dimensions | 1536 |
| Distance metric | Cosine similarity |
| Storage method | Upsert |

Resume and job description chunks are stored in the same collection. Metadata filters distinguish between document types and individual document IDs.

This allows the application to retrieve content from a specific resume or job description without mixing unrelated documents.

### Step 6: Query Processing and Retrieval

**Component:** `QueryProcessor`

When a user asks a question, the query processor:

1. Converts the question into an embedding using the same embedding model.
2. Searches Qdrant using vector similarity.
3. Applies optional metadata filters.
4. Retrieves the most relevant text chunks.
5. Returns the content, metadata, and similarity scores.

For example:

**Question:** What technical skills am I missing for this role?

The application independently retrieves:
- Relevant chunks from the selected resume.
- Relevant chunks from the selected job description.

The retrieved information is then passed to the RAG pipeline.

The query processor also supports retrieving all chunks belonging to a specific document, which is used for comprehensive resume analysis.

### Step 7: RAG Pipeline

**Component:** `RAGPipeline`

The RAG pipeline connects document retrieval with LLM-based answer generation.

The workflow:

1. Accept the user's question.
2. Retrieve relevant resume chunks.
3. Retrieve relevant job description chunks.
4. Combine the retrieved content into a contextual prompt.
5. Pass the context and question to the LLM.
6. Return the generated answer.

The LLM is instructed to use the provided context and avoid inventing candidate skills, experience, or qualifications.

This ensures that responses are grounded in the uploaded documents rather than relying entirely on the model's general knowledge.

### Step 8: LLM Integration

**Component:** `LLMService`

The application uses OpenAI's `gpt-4.1-mini` model for answer generation and structured analysis.

Responsibilities:
- Generate answers based on retrieved context.
- Compare resume content with job requirements.
- Return structured JSON responses for resume analysis.
- Follow instructions to avoid unsupported claims.

The service supports two types of generation:

**1. Text generation**

Used by the RAG pipeline to answer user questions.

**2. JSON generation**

Used by the resume analysis service to produce predictable structured output.


## 4. Dependencies

The project's `requirements.txt` contains:

```text
streamlit
pypdf
python-docx
langchain-text-splitters
openai
qdrant-client
python-dotenv
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 5. Environment Setup

### Prerequisites

Make sure the following are installed:

- Python 3.10 or later.
- Docker Desktop.
- OpenAI API key.
- Visual Studio Code or another Python IDE.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ResumeRAG
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=resume_analysis
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### Step 5: Start Qdrant Using Docker

Run the following command:

```bash
docker run -d \
  --name resume-rag-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

For Windows Command Prompt, use:

```cmd
docker run -d --name resume-rag-qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant endpoints:

- API: http://localhost:6333
- Dashboard: http://localhost:6333/dashboard

### Step 6: Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 6. Things Needed to Productionize the Application

The solution works great in a small scale but when moved to production environment we will often face issues like latency, llm halucination, etc.,.
We can however resolve them by improvising each module of the application starting with :

- Document Upload : We can move the document upload storage to a central server/cloud as it will be easier and quicker
- Vector Store : Instead of using Qdrant in Docker, we can opt for cloud model. This will make storing and retreiving chunk even faster.
- LLM : Currently I have used OpenAI LLM using API key. But in production we may face issue with token exhaustion, longer response time, halucinated response etc.,. In order to handle this we can assign usage limit for each user so that they donot exhaust the model's token, increase the Request Per Sec time/ select a model with higher RPS, Strict prompting so that the model does not answer outside the provided content

We can also build the whole application using AWS which has their inbuilt vector store, llm models and pipelines. Also it is easier to monitor and maintain in single environment rather than having mulitple dependencies.

---

## 7. AI Tools used in development process.

In this project I have used Claude to develop and improvise modules such as UI, Chunking and Embeddings. 

---

## 7. Things I would do differently in this Project

- I would have an option to tailor make my resume for each Job description I have uploaded
- I will also have an option to delete the uploaded job descriptions and upload new one. In this way I won't end up using more storage.
- Setup a login module and a user profile module where I can upload/enter the contents of the resume. This way I don't have to upload my resume each and everytime I run the application.

---
