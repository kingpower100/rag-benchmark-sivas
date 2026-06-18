from pydantic import BaseModel


class QueryRecord(BaseModel):
    question_id: str
    question: str
    cleaned_question: str | None = None
    detected_category: str | None = None
    category_confidence: float = 0.0
    orchestration_error: str | None = None

    @property
    def retrieval_question(self) -> str:
        return self.cleaned_question or self.question
