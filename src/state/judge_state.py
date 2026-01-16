"""Typed state used by the judge agent."""

from typing import Literal, TypedDict


class JudgeState(TypedDict):
    question: str
    expected_answer: str
    agent_answer: str
    judgment: str
    label: Literal["correct", "partial", "incorrect"]
    score: float