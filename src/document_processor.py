
from pathlib import Path
from dataclasses import dataclass
import re

from docx import Document
from pypdf import PdfReader

from src.file_upload_manager import UploadedDocument


@dataclass
class ParsedDocument:
    text: str
    metadata: dict


class DocumentProcessor:

    def process_document(
        self,
        document: UploadedDocument
    ) -> ParsedDocument:

        file_path = Path(document.saved_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            text = self._read_pdf(file_path)

        elif extension == ".docx":
            text = self._read_docx(file_path)

        elif extension == ".txt":
            text = self._read_txt(file_path)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        text = self._clean_text(text)

        if not text.strip():
            raise ValueError(
                f"No readable text found in {file_path.name}"
            )

        metadata = {
            "document_id": document.document_id,
            "document_type": document.document_type,
            "source": document.original_filename,
            "file_extension": extension
        }

        return ParsedDocument(
            text=text,
            metadata=metadata
        )

    def _read_pdf(self, file_path: Path) -> str:

        reader = PdfReader(str(file_path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(
                    f"\n[Page {page_number}]\n{page_text}"
                )

        return "\n".join(pages)

    def _read_docx(self, file_path: Path) -> str:

        doc = Document(str(file_path))

        sections = []

        # Extract paragraphs while preserving heading labels
        for paragraph in doc.paragraphs:

            text = paragraph.text.strip()

            if text:
                sections.append(text)

        # Extract table content (important for resume layouts)
        for table in doc.tables:

            for row in table.rows:

                row_text = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if row_text:
                    sections.append(" | ".join(row_text))

        return "\n".join(sections)

    def _read_txt(self, file_path: Path) -> str:

        return file_path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )

    @staticmethod
    def _clean_text(text: str) -> str:

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces without destroying line structure
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
