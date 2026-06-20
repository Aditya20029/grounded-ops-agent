"""Unit tests for the faithfulness LLM-as-judge scoring."""

from __future__ import annotations

import pytest

from app.agent.faithfulness import score_faithfulness, split_claims
from app.llm.types import LLMResponse, LLMUsage
from tests.fakes.providers import FakeLLMProvider


@pytest.mark.unit
def test_split_claims_strips_citations_and_short_fragments() -> None:
    claims = split_claims("The outage lasted three hours [1]. Ok. It was a deploy bug [2].")
    assert any("three hours" in c for c in claims)
    assert all("[" not in c for c in claims)
    assert "Ok." not in claims  # too short to be a claim


@pytest.mark.unit
async def test_all_supported_scores_one() -> None:
    judge = FakeLLMProvider(answer_text="SUPPORTED")
    result = await score_faithfulness(
        "The outage lasted three hours. The cause was a deployment regression.",
        ["context that supports both claims"],
        judge,
    )
    assert result.score == 1.0
    assert len(result.per_claim) == 2


@pytest.mark.unit
async def test_mixed_verdicts_average() -> None:
    judge = FakeLLMProvider(
        responses=[
            LLMResponse(text="SUPPORTED", usage=LLMUsage(1, 1), model="fake"),
            LLMResponse(text="UNSUPPORTED", usage=LLMUsage(1, 1), model="fake"),
        ]
    )
    result = await score_faithfulness(
        "Claim one is here and is long enough. Claim two is also long enough here.",
        ["some context"],
        judge,
    )
    assert result.score == 0.5
