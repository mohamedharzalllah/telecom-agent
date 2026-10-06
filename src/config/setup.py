"""Configuration settings for the TelecomPlus agent system."""

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
EXCEL_DIR = DATA_DIR / "xlsx"

# Vector database settings
VECTOR_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "telecom_faq" 

# Model settings
EMBEDDING_MODEL = "text-embedding-004"

# API Configuration - supports OpenAI, Google REST API, and Vertex AI
USE_GOOGLE_API = os.getenv("USE_GOOGLE_API", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_llm():
    if not GEMINI_API_KEY:
        load_dotenv()
    if not GEMINI_API_KEY:
        raise EnvironmentError("API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)

# # Langfuse settings
# LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
# LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
# LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Excel data mappings
EXCEL_FILES: Dict[str, str] = {
    "clients": "clients.xlsx",
    "abonnements": "abonnements.xlsx",
    "consommation": "consommation.xlsx",
    "factures": "factures.xlsx",
    "forfaits": "forfaits.xlsx",
    "tickets_support": "tickets_support.xlsx",
}

# PDF FAQ mappings
PDF_FILES: List[str] = [
    "FAQ_Facturation_et_Paiements.pdf",
    "FAQ_Forfaits_et_Abonnements.pdf",
    "FAQ_Support_Technique.pdf",
    "FAQ_Roaming_International.pdf",
    "FAQ_Compte_Client.pdf",
    "FAQ_Résiliation_et_Modifications.pdf",
    "FAQ_Catalogue_Telephones.pdf",
]

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Agent settings
MAX_ITERATIONS = 5
TEMPERATURE = 0.1

def _ensure_api_key() -> str:  # pyright: ignore[reportUndefinedVariable]
    """Return a Gemini API key from environment or raise a clear error."""
    load_dotenv()
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "API key missing. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment."
        )
    return GEMINI_API_KEY

