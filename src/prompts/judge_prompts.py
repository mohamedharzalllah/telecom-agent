from langchain_core.prompts import ChatPromptTemplate


def judge_prompt(question: str, expected: str, agent: str) -> str:
    """Format the judge instruction with the specific QA pair."""
    return f"""Tu es un évaluateur neutre.
Compare la réponse de l'agent à la réponse attendue.
Attribue un score entre 0 et 1:
- 1 si la réponse est globalement correcte et couvre les points essentiels.
- 0.5 si partiellement correcte.
- 0 si incorrecte ou hors sujet.

Fournis aussi un label (correct/partial/incorrect) et une brève justification en français.

Question: {question}
Réponse attendue: {expected}
Réponse de l'agent: {agent}
""".strip()
def build_judge_prompt() -> ChatPromptTemplate:
    """Create the prompt used for judge."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", judge_prompt),
        ]
    )