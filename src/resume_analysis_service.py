
class ResumeAnalysisService:

    def __init__(self, query_processor, llm_service):
        self.query_processor = query_processor
        self.llm_service = llm_service

    def analyze(
        self,
        resume_document_id: str,
        jd_document_id: str
    ) -> dict:

        # Retrieve all chunks from both documents.
        resume_chunks = self.query_processor.get_document_chunks(
            document_id=resume_document_id,
            document_type="resume"
        )

        jd_chunks = self.query_processor.get_document_chunks(
            document_id=jd_document_id,
            document_type="job_description"
        )

        if not resume_chunks:
            raise ValueError("No resume content found.")

        if not jd_chunks:
            raise ValueError("No job description content found.")

        # Prepare context.
        resume_context = "\n\n".join(
            f"Section: {chunk['metadata'].get('section_name', 'Unknown')}\n"
            f"{chunk['text']}"
            for chunk in resume_chunks
        )

        jd_context = "\n\n".join(
            f"Section: {chunk['metadata'].get('section_name', 'Unknown')}\n"
            f"{chunk['text']}"
            for chunk in jd_chunks
        )

        system_prompt = """
        You are an experienced technical recruiter and resume analyst.

        Compare the candidate's resume against the job description.

        Use only the information provided. Do not invent skills,
        qualifications, experience, or achievements.

        Scoring rubric:
        - Technical and functional skills: 60%
        - Relevant experience: 20%
        - Qualifications: 10%
        - Role responsibilities: 10%

        Return an overall match score between 0 and 100.
        The score must reflect the evidence in the documents,
        not vector similarity.

        Important rules:
        - A skill not mentioned in the resume is not automatically
          proof that the candidate does not have that skill.
        - Distinguish between explicitly required and preferred skills.
        - Do not treat preferred skills as mandatory.
        - Include evidence from the resume wherever possible.
        - Be objective and avoid inflated scores.

        Return valid JSON with exactly these fields:

        {
            "overall_match_score": 0,
            "score_rationale": "",
            "matching_skills": [
                {
                    "skill": "",
                    "resume_evidence": "",
                    "jd_requirement": ""
                }
            ],
            "missing_skills": [
                {
                    "skill": "",
                    "jd_requirement": "",
                    "priority": "High"
                }
            ],
            "experience_assessment": "",
            "qualification_gaps": [],
            "strengths": [],
            "recommendations": []
        }

        Use only the specified JSON structure.
        """

        user_prompt = f"""
        RESUME:

        {resume_context}

        JOB DESCRIPTION:

        {jd_context}

        Analyse the resume against this job description.
        Return the structured JSON analysis.
        """

        result = self.llm_service.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Basic response validation.
        score = result.get("overall_match_score")

        if not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError("LLM returned an invalid match score.")

        return result
