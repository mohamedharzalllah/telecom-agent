"""Evaluation script for the TelecomPlus support agent.

This script evaluates the agent's responses against the expected answers
from the evaluation dataset.

Students should modify this script to:
1. Call their actual agent implementation
2. Implement proper evaluation logic (e.g., LLM-as-a-judge)
3. Calculate meaningful metrics (accuracy, relevance, etc.)
"""

import pandas as pd

from src.main import answer  # agent entrypoint
from src.agents.judge_agent import judge_answer


def evaluate_response(question: str, expected_answer: str) -> tuple[float, str, str]:
    """
    Evaluate a single QA pair using the LLM judge.

    Returns:
        score: float in [0, 1]
        explanation: judge rationale
        agent_answer: the agent's answer
    """
    agent_answer = answer(question)
    try:
        judgment = judge_answer(question, expected_answer, agent_answer)
        score = float(judgment.get("score", 0.0))
        explanation = judgment.get("judgment", "")
    except Exception as exc:  # defensive fallback
        score = 0.0
        explanation = f"Judge failed: {exc}"
    return score, explanation, agent_answer


def run_evaluation():
    """Run evaluation on all questions from the evaluation dataset."""

    # Load evaluation questions
    print("Loading evaluation questions...")
    df = pd.read_excel("data/evaluation_questions.xlsx")

    print(f"Loaded {len(df)} questions\n")
    print("=" * 80)

    results = []
    total_score = 0

    # Evaluate each question
    for idx, row in df.iterrows():
        question = row["Question"]
        expected_answer = row["Réponse Attendue"]
        difficulty = row["Difficulté"]

        print(f"\nQuestion {idx + 1}/{len(df)} [{difficulty}]:")
        print(f"Q: {question}")

        score, explanation, agent_answer = evaluate_response(question, expected_answer)
        total_score += score

        print(f"Agent: {agent_answer}")
        print(f"Score: {score:.2f}/1")
        print(f"Judge: {explanation}")

        # Store results
        results.append({
            "Question": question,
            "Expected Answer": expected_answer,
            "Agent Answer": agent_answer,
            "Difficulty": difficulty,
            "Score": score,
            "Judge Explanation": explanation,
        })

        print("-" * 80)

    # Calculate final metrics
    accuracy = total_score / len(df)
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total questions: {len(df)}")
    print(f"Total score: {total_score}/{len(df)}")
    print(f"Accuracy: {accuracy:.2%}")

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel("evaluation_results.xlsx", index=False)
    print("\nResults saved to: evaluation_results.xlsx")


if __name__ == "__main__":
    run_evaluation()
