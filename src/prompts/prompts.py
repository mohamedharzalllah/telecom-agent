from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def router_prompt(question: str) -> str:
    return f"""
Choisis le meilleur mode pour répondre :
- rag : question sur FAQ, procédures, politiques, roaming, résiliation, paiements, produits/téléphones.
- sql : question sur un client spécifique (avec nom), facture, consommation, ticket, forfait personnel.
- hybrid : combine infos FAQ + données client personnelles (quand un nom est mentionné + question nécessitant FAQ).

Question : {question}

Réponds uniquement par : rag, sql, ou hybrid.
""".strip()

def build_prompt() -> ChatPromptTemplate:
    """Create the prompt used for grounded QA."""
    system = SystemMessage(
        content=(
            "Tu es l'assistant support TelecomPlus.\n\n"
            "CONTEXTE TEMPOREL:\n"
            "- Aujourd'hui nous sommes le 23 janvier 2026 (23/01/2026).\n"
            "- Quand l'utilisateur demande 'ce mois-ci', utilise les données les plus récentes disponibles.\n"
            "- Si les données datent de novembre 2025 et qu'on est en janvier 2026, présente-les comme les dernières données disponibles.\n\n"
            
            "POLITIQUE ROAMING UE (TRÈS IMPORTANT):\n"
            "- Les pays de l'Union Européenne bénéficient du 'roaming comme à la maison' : AUCUN SURCOÛT.\n"
            "- Pays UE incluent : France, Italie, Espagne, Allemagne, Belgique, Portugal, Grèce, etc.\n"
            "- Si la question concerne un voyage dans un pays de l'UE (ex: Italie, Espagne), TOUJOURS mentionner que le roaming est GRATUIT et inclus.\n"
            "- Hors UE (USA, Suisse, UK, etc.) : des frais s'appliquent, recommander pass international ou espace client.\n\n"
            
            "INSTRUCTIONS DE RÉPONSE:\n"
            "1. Réponds en français de manière concise mais complète.\n"
            "2. Utilise UNIQUEMENT le contexte fourni, ne jamais inventer.\n"
            "3. Pour les questions produits : liste TOUTES les options correspondant aux critères demandés.\n"
            "4. Si tu exclus des options, explique brièvement pourquoi (ex: 'L'iPhone 17 à 1299€ dépasse le budget').\n"
            "5. Fournis toujours des informations actionnables : prix, contacts, étapes suivantes.\n"
            "6. Pour les services (reprise, pass international) : mentionne prix estimé ET comment y accéder (site web, téléphone 3900, espace client).\n"
            "7. Si une information manque dans le contexte, indique-le clairement et suggère de contacter le service client.\n"
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            system,
            ("human", "Question: {question}\n\nContexte:\n{context}"),
        ]
    )
    return prompt
