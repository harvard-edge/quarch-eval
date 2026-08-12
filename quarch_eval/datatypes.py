import dataclasses
from collections import Counter
from typing import Any, Literal
from pydantic import BaseModel


JudgeLabel = Literal["CORRECT", "INCORRECT", "PARTIALLY-CORRECT"]

@dataclasses.dataclass
class MajorityResult:
    reached: bool
    label: JudgeLabel | None
    counts: Counter[JudgeLabel]   # frequency map of judgments

class Judgment(BaseModel):
    judge_llm_name: str
    judge_instruction_template: str
    judge_response: str
    judge_reasoning_trace: str | None = None
    judge_generation_config: dict[str, Any] | None = None
    judgment: str # "correct" or "incorrect"


class Response(BaseModel):
    student_response: str
    student_apiresponse_raw: str | None = None
    student_reasoning_trace: str | None = None
    # supports pass@k judge evaluation/majority voting
    judgments: list[Judgment] | None = None
    metrics: dict | None = None

class QuestionResponse(BaseModel):
    question_id: str
    # supports pass@k student evaluation
    responses: list[Response]

class ExamResponses(BaseModel):
    exam_id: str
    student_llm_name: str
    student_instruction_template: str
    question_responses: list[QuestionResponse]

# --- MCQ Evaluation Datatypes ---

class MCQResult(BaseModel):
    is_correct: bool | None
    llm_answer: str
    llm_response_text: str

class MCQResultGroup(BaseModel):
    question_id: int
    attempts: list[MCQResult]

class MCQEvaluation(BaseModel):
    llm_name: str
    correct_count: int | None = None
    results: list[MCQResultGroup]
