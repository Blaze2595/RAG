
from openai import OpenAI
import json
from config import OPENAI_API_KEY, CHAT_MODEL


class LLMService:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = CHAT_MODEL

    def generate_answer(self, question: str, context: str) -> str:

        system_prompt = """
        You are an experienced technical recruiter and resume analyst.

        Answer the user's question using only the provided context.

        Instructions:
        1. Do not invent skills, experience, qualifications, or achievements.
        2. Clearly distinguish information found in the resume
           from information found in the job description.
        3. If the context does not contain enough information,
           explicitly say so.
        4. Explain your reasoning breifly.
        5. Use bullet points where appropriate.
        6. Identify the source section when making important claims.
        """

        user_prompt = f"""
        CONTEXT:
        {context}

        QUESTION:
        {question}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content or ""

    
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("The LLM returned an empty response.")

        return json.loads(content)

