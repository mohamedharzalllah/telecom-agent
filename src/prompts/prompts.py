from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def router_prompt(question: str) -> str:
    return f"""
Choisis le meilleur mode pour répondre :
- rag : question sur FAQ, procédures, politiques, roaming, résiliation, paiements.
- sql : question sur un client, facture, consommation, ticket, forfait spécifique.
- hybrid : combine infos FAQ + données client.

Question : {question}

Réponds uniquement par : rag, sql, ou hybrid.
""".strip()

def build_prompt() -> ChatPromptTemplate:
    """Create the prompt used for grounded QA."""
    system = SystemMessage(
        content=(
            "Tu es l'assistant support TelecomPlus. "
            "Réponds en français de manière concise et factuelle. "
            "Utilise uniquement le contexte fourni. "
            "Si une information manque, indique-le plutôt que d'inventer."
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            system,
            ("human", "Question: {question}\n\nContexte:\n{context}"),
        ]
    )
    return prompt
