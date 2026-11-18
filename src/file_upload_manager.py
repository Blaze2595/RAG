
from pathlib import Path
from dataclasses import dataclass
from uuid import uuid4
import re


@dataclass
class UploadedDocument:
    document_id: str
    original_filename: str
    saved_path: str
    document_type: str
    file_extension: str


class FileUploadManager:

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(
        self,
        upload_directory: str = "uploads",
        max_file_size_mb: int = 5
    ):
        self.upload_directory = Path(upload_directory)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        self.resume_directory = self.upload_directory / "resumes"
        self.jd_directory = self.upload_directory / "job_descriptions"

        self.resume_directory.mkdir(parents=True, exist_ok=True)
        self.jd_directory.mkdir(parents=True, exist_ok=True)

    def _save_file(self, uploaded_file, document_type: str):

        original_filename = Path(uploaded_file.name).name
        extension = Path(original_filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        file_content = uploaded_file.getvalue()

        if not file_content:
            raise ValueError(
                f"File is empty: {original_filename}"
            )

        if len(file_content) > self.max_file_size_bytes:
            raise ValueError(
                f"File exceeds maximum size of "
                f"{self.max_file_size_bytes // (1024 * 1024)} MB: "
                f"{original_filename}"
            )

        # Generate unique ID to prevent filename collisions
        document_id = str(uuid4())

        # Sanitize filename
        safe_filename = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            Path(original_filename).stem
        )

        unique_filename = (
            f"{safe_filename}_{document_id}{extension}"
        )

        target_directory = (
            self.resume_directory
            if document_type == "resume"
            else self.jd_directory
        )

        file_path = target_directory / unique_filename

        file_path.write_bytes(file_content)

        return UploadedDocument(
            document_id=document_id,
            original_filename=original_filename,
            saved_path=str(file_path),
            document_type=document_type,
            file_extension=extension
        )

    def save_resume(self, uploaded_file) -> UploadedDocument:

        if uploaded_file is None:
            raise ValueError("Please upload a resume.")

        return self._save_file(
            uploaded_file,
            document_type="resume"
        )

    def save_job_descriptions(
        self,
        uploaded_files: list
    ) -> list[UploadedDocument]:

        if not uploaded_files:
            raise ValueError(
                "Please upload at least one job description."
            )

        saved_documents = []

        for uploaded_file in uploaded_files:

            document = self._save_file(
                uploaded_file,
                document_type="job_description"
            )

            saved_documents.append(document)

        return saved_documents
