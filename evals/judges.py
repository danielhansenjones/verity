"""LLM-as-judge dimensions for the RAG eval harness.

Each judge returns a score in [0, 1] alongside structured reasoning. The judge
model defaults to claude-haiku-4-5 (deliberately different from the generator
to reduce self-grading bias); a second judge (claude-opus-4-7) is opt-in for
inter-judge agreement runs.
"""
import logging
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from shared.settings import settings


logger = logging.getLogger(__name__)


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
SECOND_JUDGE_MODEL = "claude-opus-4-7"


class ClaimSupport(BaseModel):
    claim: str
    supported: bool
    reasoning: str


class FaithfulnessJudgment(BaseModel):
    claims: list[ClaimSupport] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


class CitationGrade(BaseModel):
    chunk_id: str
    relevant: bool
    reasoning: str


class CitationAccuracyJudgment(BaseModel):
    grades: list[CitationGrade] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


class ReferencePointCoverage(BaseModel):
    point: str
    present: bool


class CompletenessJudgment(BaseModel):
    coverage: list[ReferencePointCoverage] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


RefusalBucket = Literal[
    "refused_correctly",
    "refused_wrongly",
    "answered_when_should_refuse",
    "answered_correctly",
]


class RefusalCorrectnessJudgment(BaseModel):
    bucket: RefusalBucket
    score: float = Field(ge=0.0, le=1.0)


def _client() -> anthropic.Anthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _call_judge(
    model: str,
    system: str,
    user: str,
    schema_cls: type[BaseModel],
) -> BaseModel:
    msg = _client().messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": "submit_judgment",
                "description": "Submit the structured judgment.",
                "input_schema": schema_cls.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_judgment"},
    )
    tool_use_blocks = [b for b in msg.content if b.type == "tool_use"]
    if not tool_use_blocks:
        raise RuntimeError(
            f"judge {schema_cls.__name__} returned no tool_use block"
        )
    return schema_cls.model_validate(tool_use_blocks[0].input)


_FAITHFULNESS_SYSTEM = (
    "You evaluate whether an answer about a legal contract is faithful to its"
    " cited evidence. Break the answer into discrete factual claims. For"
    " each claim, decide if the cited chunk text supports it. The final"
    " score is the fraction of supported claims; if there are no claims,"
    " the score is 1.0."
)


def judge_faithfulness(
    question: str,
    answer: str,
    citations_with_text: list[dict],
    model: str = DEFAULT_JUDGE_MODEL,
) -> FaithfulnessJudgment:
    if not answer:
        return FaithfulnessJudgment(claims=[], score=1.0)

    cite_blob = "\n\n".join(
        (
            f"[chunk_id={c['chunk_id']}]\nquote: {c['quote']}\n"
            f"chunk text:\n{c['chunk_text']}"
        )
        for c in citations_with_text
    )
    user = (
        f"Question: {question}\n\nAnswer:\n{answer}\n\n"
        f"Cited evidence:\n\n{cite_blob if cite_blob else '(no citations)'}"
    )
    return _call_judge(model, _FAITHFULNESS_SYSTEM, user, FaithfulnessJudgment)


_CITATION_ACCURACY_SYSTEM = (
    "You evaluate whether cited chunks are actually relevant to a question"
    " and its answer. For each citation, mark it relevant if its chunk text"
    " addresses the question; mark it irrelevant otherwise. The final score"
    " is the fraction marked relevant; if there are no citations, the score"
    " is 1.0."
)


def judge_citation_accuracy(
    question: str,
    answer: str,
    citations_with_text: list[dict],
    model: str = DEFAULT_JUDGE_MODEL,
) -> CitationAccuracyJudgment:
    if not citations_with_text:
        return CitationAccuracyJudgment(grades=[], score=1.0)

    cite_blob = "\n\n".join(
        f"[chunk_id={c['chunk_id']}]\nchunk text:\n{c['chunk_text']}"
        for c in citations_with_text
    )
    user = (
        f"Question: {question}\n\nAnswer: {answer}\n\nCitations:\n\n{cite_blob}"
    )
    return _call_judge(
        model, _CITATION_ACCURACY_SYSTEM, user, CitationAccuracyJudgment
    )


_COMPLETENESS_SYSTEM = (
    "You evaluate whether a model answer covers the same factual points as"
    " a reference answer. Use the reference keywords as anchors: for each"
    " keyword, mark it present if it or an unambiguous semantic equivalent"
    " appears in the model answer. The final score is the fraction present;"
    " if there are no reference keywords, the score is 1.0."
)


def judge_completeness(
    question: str,
    answer: str,
    reference_answer: Optional[str],
    reference_keywords: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
) -> CompletenessJudgment:
    if reference_answer is None or not reference_keywords:
        return CompletenessJudgment(coverage=[], score=1.0)

    user = (
        f"Question: {question}\n\nReference answer: {reference_answer}\n\n"
        f"Reference keywords: {reference_keywords}\n\n"
        f"Model answer: {answer if answer else '(empty - model refused)'}"
    )
    return _call_judge(model, _COMPLETENESS_SYSTEM, user, CompletenessJudgment)


def judge_refusal_correctness(
    expected_refusal: bool,
    model_refused: bool,
) -> RefusalCorrectnessJudgment:
    # No LLM call: refusal correctness is a binary comparison. Kept as a
    # function for symmetry with the other dimensions and to centralise
    # the bucket taxonomy.
    if expected_refusal and model_refused:
        return RefusalCorrectnessJudgment(bucket="refused_correctly", score=1.0)
    if expected_refusal and not model_refused:
        return RefusalCorrectnessJudgment(
            bucket="answered_when_should_refuse", score=0.0
        )
    if not expected_refusal and model_refused:
        return RefusalCorrectnessJudgment(bucket="refused_wrongly", score=0.0)
    return RefusalCorrectnessJudgment(bucket="answered_correctly", score=1.0)
