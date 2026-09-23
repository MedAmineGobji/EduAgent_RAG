# ============================================================
# tools.py — Outils disponibles pour l'agent IA
# ============================================================
#
# ❓ C'EST QUOI UN "TOOL" DANS LANGCHAIN ?
# -----------------------------------------
# Un Tool est une fonction Python que l'agent IA peut appeler
# de manière autonome pour répondre à une question.
#
# L'agent lit la "description" de chaque outil et décide
# AUTOMATIQUEMENT lequel utiliser selon la demande :
#
#   "Explique-moi les pointeurs"          → rag_tool    (cherche dans le cours)
#   "Résume ce chapitre : ..."            → summary_tool (résume le texte)
#   "Génère des questions sur les listes" → quiz_tool    (crée un quiz)
#
# Chaque Tool a :
#   - name        : identifiant court (ex: "RAG_Cours")
#   - func        : la fonction Python à appeler
#   - description : phrase qui explique à l'agent QUAND utiliser cet outil
#
# ============================================================

import os
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

from src.rag_pipeline import answer_with_rag

# Charger les variables d'environnement (.env)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.1-8b-instant"

# Variable de module : stocke les sources LangChain du dernier appel RAG.
# Permet à ask_agent() de récupérer les documents sources pour l'affichage.
# ⚠️ Usage interne uniquement — ne pas modifier depuis l'extérieur.
_last_rag_sources: list = []


def _get_llm() -> ChatGroq:
    """Retourne une instance ChatGroq prête à l'emploi."""
    if not GROQ_API_KEY:
        raise ValueError(
            "❌ Clé API Groq manquante. Ajoutez GROQ_API_KEY dans le fichier .env"
        )
    return ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.3)


# ============================================================
# OUTIL 1 — RAG : Répondre depuis les documents de cours
# ============================================================

def _rag_func(question: str) -> str:
    """
    Cherche dans les documents de cours et retourne une réponse sourcée.

    Appelle answer_with_rag() qui :
      1. Récupère les 4 chunks les plus pertinents dans ChromaDB
      2. Envoie [contexte + question] au LLM
      3. Retourne une réponse basée uniquement sur le cours

    Args:
        question (str): Question de l'étudiant sur le contenu du cours.

    Returns:
        str: Réponse générée par le LLM à partir des documents.
    """
    global _last_rag_sources
    result = answer_with_rag(question)
    # Sauvegarder les sources pour que ask_agent() puisse les récupérer
    _last_rag_sources = result.get("sources", [])
    return result["answer"]


# Création du Tool LangChain
rag_tool = Tool(
    name="RAG_Cours",
    func=_rag_func,
    description=(
        "Utilise cet outil pour répondre à des questions sur le contenu "
        "des documents de cours chargés (PDFs indexés). "
        "Idéal pour : définitions, explications de concepts, exemples du cours, "
        "théorèmes, formules, notions abordées dans le cours. "
        "Input : une question précise sur le cours."
    ),
)


# ============================================================
# OUTIL 2 — SUMMARY : Résumer un texte ou un chapitre
# ============================================================

def _summary_func(text: str) -> str:
    """
    Résume un texte ou un chapitre fourni par l'utilisateur.

    Utilise directement ChatGroq pour produire un résumé structuré,
    avec les idées clés et points importants mis en avant.

    Args:
        text (str): Texte brut ou description du chapitre à résumer.

    Returns:
        str: Résumé structuré avec les points essentiels.
    """
    llm = _get_llm()

    # Construire le prompt de résumé
    prompt = f"""Tu es un assistant pédagogique expert en synthèse de cours.
Résume le texte suivant de manière claire et structurée pour un étudiant.

Texte à résumer :
\"\"\"
{text}
\"\"\"

Format de ta réponse :
- **Idée principale** : (1-2 phrases)
- **Points clés** :
  • Point 1
  • Point 2
  • ...
- **À retenir** : (phrase de conclusion courte)

Réponds en français."""

    # Envoyer le prompt au LLM et récupérer la réponse
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Création du Tool LangChain
summary_tool = Tool(
    name="Résumé_Texte",
    func=_summary_func,
    description=(
        "Utilise cet outil pour résumer un texte, un chapitre ou une notion. "
        "Idéal quand l'étudiant dit : 'résume ce chapitre', 'fais un résumé de...', "
        "'explique brièvement...', 'synthétise ce texte'. "
        "Input : le texte complet à résumer, ou une description du chapitre."
    ),
)


# ============================================================
# OUTIL 3 — QUIZ : Générer des questions de révision
# ============================================================

def _quiz_func(topic: str) -> str:
    """
    Génère 5 questions de révision avec leurs réponses sur un sujet donné.

    Utilise ChatGroq pour créer un quiz pédagogique adapté au niveau étudiant.
    Les questions couvrent différents niveaux : compréhension, application, analyse.

    Args:
        topic (str): Sujet ou thème sur lequel générer le quiz.

    Returns:
        str: 5 questions numérotées avec leurs réponses détaillées.
    """
    llm = _get_llm()

    # Construire le prompt de génération de quiz
    prompt = f"""Tu es un professeur expert qui crée des exercices de révision.
Génère exactement 5 questions de révision sur le sujet suivant : "{topic}"

Les questions doivent varier en difficulté :
- 2 questions de compréhension (définitions, explications)
- 2 questions d'application (exercices pratiques)
- 1 question de synthèse (liens entre concepts)

Format OBLIGATOIRE pour chaque question :

**Question 1 :** [énoncé de la question]
**Réponse :** [réponse complète et pédagogique]

**Question 2 :** [énoncé de la question]
**Réponse :** [réponse complète et pédagogique]

[...jusqu'à la question 5]

Réponds en français. Sois précis et pédagogique."""

    # Envoyer le prompt au LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Création du Tool LangChain
quiz_tool = Tool(
    name="Génération_Quiz",
    func=_quiz_func,
    description=(
        "Utilise cet outil pour générer des questions de révision avec réponses. "
        "Idéal quand l'étudiant dit : 'génère des questions sur...', "
        "'crée un quiz sur...', 'aide-moi à réviser...', 'teste mes connaissances sur...'. "
        "Input : le sujet ou thème sur lequel créer le quiz (ex: 'les pointeurs en C', "
        "'la récursivité', 'les algorithmes de tri')."
    ),
)


# ============================================================
# Fonction utilitaire : retourne tous les outils d'un coup
# ============================================================

def get_all_tools() -> list:
    """
    Retourne la liste de tous les outils disponibles pour l'agent.

    Returns:
        list[Tool]: [rag_tool, summary_tool, quiz_tool]
    """
    tools = [rag_tool, summary_tool, quiz_tool]
    print(f"🛠️  {len(tools)} outil(s) disponible(s) : {[t.name for t in tools]}")
    return tools
