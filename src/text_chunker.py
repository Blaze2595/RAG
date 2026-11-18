
from dataclasses import dataclass
from uuid import uuid5, NAMESPACE_URL
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_processor import ParsedDocument


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict


class TextChunker:

    # Common resume and job description section headings
    SECTION_HEADINGS = {
        "professional summary": "Professional Summary",
        "summary": "Professional Summary",
        "profile": "Professional Summary",
        "career objective": "Professional Summary",
        "objective": "Professional Summary",

        "technical skills": "Technical Skills",
        "skills": "Technical Skills",
        "core skills": "Technical Skills",
        "core competencies": "Technical Skills",
        "technical expertise": "Technical Skills",

        "work experience": "Work Experience",
        "professional experience": "Work Experience",
        "employment history": "Work Experience",
        "experience": "Work Experience",

        "projects": "Projects",
        "key projects": "Projects",
        "personal projects": "Projects",
        "academic projects": "Projects",

        "education": "Education",
        "academic qualifications": "Education",
        "educational background": "Education",

        "certifications": "Certifications",
        "certificates": "Certifications",
        "licenses": "Certifications",

        "achievements": "Achievements",
        "awards": "Achievements",
        "accomplishments": "Achievements",

        "job overview": "Job Overview",
        "job summary": "Job Overview",
        "about the role": "Job Overview",
        "about the job": "Job Overview",

        "responsibilities": "Responsibilities",
        "key responsibilities": "Responsibilities",
        "roles and responsibilities": "Responsibilities",
        "what you will do": "Responsibilities",
        "what you'll do": "Responsibilities",

        "requirements": "Requirements",
        "job requirements": "Requirements",
        "qualifications": "Requirements",
        "required qualifications": "Requirements",
        "minimum qualifications": "Requirements",
        "required skills": "Requirements",
        "must have": "Requirements",

        "preferred qualifications": "Preferred Skills",
        "preferred skills": "Preferred Skills",
        "desired skills": "Preferred Skills",
        "nice to have": "Preferred Skills",
        "good to have": "Preferred Skills",

        "benefits": "Benefits",
        "compensation": "Compensation"
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be >= 0 and < chunk size"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ],
            add_start_index=True
        )

    @staticmethod
    def _normalize_heading(line: str) -> str:

        # Remove bullets and numbering
        line = re.sub(
            r"^\s*(?:[-*•]+|\d+[.)])\s*",
            "",
            line
        )

        # Remove trailing colon
        line = line.strip().rstrip(":")

        return re.sub(r"\s+", " ", line).strip().lower()

    def _is_section_heading(self, line: str) -> str | None:

        normalized = self._normalize_heading(line)

        if not normalized:
            return None

        # First check known section headings
        if normalized in self.SECTION_HEADINGS:
            return self.SECTION_HEADINGS[normalized]

        # Recognize additional uppercase headings.
        # Avoid treating ordinary one-word text as headings.
        clean_line = line.strip()

        if (
            len(clean_line) <= 60
            and len(clean_line.split()) >= 2
            and clean_line == clean_line.upper()
            and re.search(r"[A-Z]", clean_line)
            and not clean_line.endswith((".", ",", ";"))
        ):
            return clean_line.title()

        return None

    def _extract_sections(
        self,
        document: ParsedDocument
    ) -> list[dict]:

        lines = document.text.splitlines()

        sections = []

        current_section = "General"
        current_content = []

        def save_section():

            content = "\n".join(current_content).strip()

            if content:
                sections.append({
                    "section_name": current_section,
                    "text": content
                })

        for line in lines:

            heading = self._is_section_heading(line)

            if heading:

                # Save previous section
                save_section()

                current_section = heading
                current_content = []

            else:
                current_content.append(line)

        # Save final section
        save_section()

        return sections

    def create_chunks(
        self,
        documents: list[ParsedDocument]
    ) -> list[TextChunk]:

        all_chunks = []

        for document in documents:

            if not document.text.strip():
                continue

            sections = self._extract_sections(document)

            for section_index, section in enumerate(sections):

                split_documents = self.splitter.create_documents(
                    texts=[section["text"]]
                )

                for chunk_index, split in enumerate(
                    split_documents
                ):

                    chunk_id = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{document.metadata['document_id']}:"
                            f"{section_index}:{chunk_index}"))

                    chunk_metadata = {
                        **document.metadata,
                        "section_name": section["section_name"],
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "start_index": split.metadata.get(
                            "start_index", 0
                        ),
                        "chunk_size": len(split.page_content)
                    }

                    all_chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            text=split.page_content,
                            metadata=chunk_metadata
                        )
                    )

        return all_chunks