"""Faithfulness scoring (LLM-as-judge, 3-point rubric).

Each claim in the answer is scored against its cited context as SUPPORTED (1.0),
PARTIAL (0.5), or UNSUPPORTED (0.0); the result is averaged. Graded runs request
temperature 0; the judge model and date are recorded by the eval harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.base import LLMProvider
from app.llm.types import LLMMessage

JUDGE_SYSTEM = (
    "You are a strict grounding judge. Given CONTEXT and a CLAIM, decide whether the "
    "context supports the claim. Reply with exactly one word: SUPPORTED, PARTIAL, or "
    "UNSUPPORTED. Do not explain."
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[\d+\]")


def split_claims(answer: str, *, limit: int = 8) -> list[str]:
    text = _CITE.sub("", answer)
    claims = [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) > 15]
    return claims[:limit]


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    per_claim: list[tuple[str, float]]


def _label_score(text: str) -> float:
    # Check UNSUPPORTED before SUPPORTED ("SUPPORTED" is a substring of it).
    upper = text.strip().upper()
    if "UNSUPPORTED" in upper:
        return 0.0
    if "PARTIAL" in upper:
        return 0.5
    if "SUPPORTED" in upper:
        return 1.0
    return 0.0


async def score_faithfulness(
    answer: str,
    context_chunks: list[str],
    judge: LLMProvider,
    *,
    temperature: float | None = 0.0,
) -> FaithfulnessResult:
    """Score answer claims against the cited context with an LLM judge."""
    claims = split_claims(answer)
    if not claims:
        return FaithfulnessResult(0.0, [])
    context = "\n\n".join(context_chunks)

    per_claim: list[tuple[str, float]] = []
    for claim in claims:
        prompt = (
            f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\n"
            "Verdict (SUPPORTED / PARTIAL / UNSUPPORTED):"
        )
        response = await judge.complete(
            system=JUDGE_SYSTEM,
            messages=[LLMMessage.user(prompt)],
            max_tokens=8,
            temperature=temperature,
        )
        per_claim.append((claim, _label_score(response.text)))

    score = sum(s for _, s in per_claim) / len(per_claim)
    return FaithfulnessResult(score=score, per_claim=per_claim)
