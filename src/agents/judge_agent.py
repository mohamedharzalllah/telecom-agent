"""Judge agent for evaluating agent responses against expected answers."""

from typing import Literal

from pydantic import BaseModel, Field

from src.config.setup import get_llm
from src.prompts.judge_prompts import judge_prompt


class JudgmentResult(BaseModel):
    """Structured output for the judge evaluation."""
    
    score: float = Field(
        description="Score between 0 and 1 (1=correct, 0.5=partial, 0=incorrect)"
    )
    label: Literal["correct", "partial", "incorrect"] = Field(
        description="Categorical label for the answer quality"
    )
    judgment: str = Field(
        description="Brief explanation in French justifying the score and label"
    )


def judge_answer(question: str, expected_answer: str, agent_answer: str) -> dict:
    """
    Evaluate an agent's answer against the expected answer using an LLM judge.
    
    Args:
        question: The original question asked
        expected_answer: The reference/expected answer
        agent_answer: The agent's actual answer to evaluate
        
    Returns:
        Dictionary containing:
        - score: float between 0 and 1
        - label: "correct", "partial", or "incorrect"
        - judgment: explanation string in French
    """
    # Get the LLM and configure it for structured output
    llm = get_llm()
    structured_llm = llm.with_structured_output(JudgmentResult)
    
    # Format the judge prompt with the specific QA pair
    prompt_text = judge_prompt(question, expected_answer, agent_answer)
    
    # Invoke the LLM and get structured response
    result = structured_llm.invoke(prompt_text)
    
    # Convert Pydantic model to dictionary
    return {
        "score": result.score,
        "label": result.label,
        "judgment": result.judgment,
    }
