import logging
import re
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from shared.models import Chunk
from shared.settings import settings


logger = logging.getLogger(__name__)


class Citation(BaseModel):
    chunk_id: str = Field(
        description="The id of the chunk that supports this part of the answer"
    )
    chunk_index: int = Field(
        description="The 0-based index of the chunk within its job"
    )
    quote: str = Field(
        description=(
            "A short verbatim excerpt from the chunk's text that supports the"
            " answer. Must be a substring of the chunk."
        )
    )


class AnswerResponse(BaseModel):
    answer: str = Field(
        description=(
            "The answer to the question, grounded in the retrieved chunks."
            " Empty string when refusing."
        )
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting each factual claim in the answer",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description=(
            "Set when retrieved evidence cannot support an answer."
            " When set, answer must be empty and citations must be empty."
        ),
    )


SYSTEM_PROMPT = (
    "You answer questions about a single legal contract using only the"
    " chunks provided in the user message.\n"
    "\n"
    "Rules:\n"
    "1. Cite a specific chunk for every factual claim, using its chunk_id"
    " and chunk_index.\n"
    "2. The quote in each citation must be a verbatim substring of the"
    " named chunk's text.\n"
    "3. If the retrieved chunks do not support an answer, set"
    " refusal_reason, leave answer empty, and use no citations.\n"
    "4. Do not invent or paraphrase quotes. Do not cite chunks that were"
    " not provided.\n"
    "5. Keep answers concise and direct."
)


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    blocks = [
        f"[chunk_id={c.id} chunk_index={c.index}]\n{c.text}" for c in chunks
    ]
    return f"Question: {question}\n\nRetrieved chunks:\n\n" + "\n\n".join(blocks)


_anthropic: Optional[anthropic.Anthropic] = None


def _client() -> anthropic.Anthropic:
    # Module-level singleton: anthropic.Anthropic owns an httpx client with its
    # own connection pool. Re-instantiating per request defeats the pool.
    global _anthropic
    if _anthropic is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        _anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic


def ask(question: str, chunks: list[Chunk]) -> tuple[AnswerResponse, dict]:
    user_prompt = build_user_prompt(question, chunks)

    # System prompt is static across requests, so ephemeral cache cuts the
    # per-call input cost for any RAG burst within the 5-minute cache window.
    msg = _client().messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        tools=[
            {
                "name": "submit_answer",
                "description": (
                    "Submit the final answer to the user's question with"
                    " grounded citations."
                ),
                "input_schema": AnswerResponse.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_answer"},
    )

    tool_use_blocks = [b for b in msg.content if b.type == "tool_use"]
    if not tool_use_blocks:
        raise RuntimeError("model did not return a tool_use block")

    response = AnswerResponse.model_validate(tool_use_blocks[0].input)
    usage = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "cache_read_input_tokens": getattr(
            msg.usage, "cache_read_input_tokens", 0
        ),
        "cache_creation_input_tokens": getattr(
            msg.usage, "cache_creation_input_tokens", 0
        ),
    }
    return response, usage


def _normalize(text: str) -> str:
    # Collapse runs of whitespace so quote-vs-chunk substring matching survives
    # PDF extraction artefacts: line breaks inside sentences, double spaces, etc.
    return re.sub(r"\s+", " ", text).strip().lower()


def grounding_error(
    response: AnswerResponse, chunks: list[Chunk]
) -> Optional[str]:
    # Returns None when the response is correctly grounded, or a human-readable
    # error string when it is not. Callers translate the error into a 502.
    if response.refusal_reason:
        if response.citations:
            return "refusal response must not include citations"
        if response.answer:
            return "refusal response must have an empty answer"
        return None

    chunks_by_id = {c.id: c for c in chunks}

    for cite in response.citations:
        if cite.chunk_id not in chunks_by_id:
            return f"citation chunk_id {cite.chunk_id} was not in the retrieved set"

        cited = chunks_by_id[cite.chunk_id]
        if _normalize(cite.quote) not in _normalize(cited.text):
            return f"citation quote not found in chunk {cite.chunk_id}"

    return None
