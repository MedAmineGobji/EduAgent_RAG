# ============================================================
# rag_pipeline.py — Pipeline RAG complet
# ============================================================
#
# ❓ C'EST QUOI LE RAG ?
# -----------------------
# RAG = Retrieval-Augmented Generation
# (Génération Augmentée par la Récupération)
#
# Sans RAG : le LLM répond depuis sa mémoire d'entraînement
#   → peut halluciner, ne connaît pas VOS cours
#
# Avec RAG : on donne au LLM les bons passages AVANT qu'il réponde
#   → réponses précises, vérifiables, basées sur VOS documents
#
# Flux de answer_with_rag(question) :
#
#   question
#     │
#     ▼
#   ChromaDB  ──── recherche les 4 chunks les plus proches ───►  [chunk1, chunk2, chunk3, chunk4]
#     │                                                                       │
#     │                                                                       ▼
#     └──────────────────────────────────────────────►  Prompt = Instructions + Contexte + Question
#                                                                             │
#                                                                             ▼
#                                                               ChatGroq (llama-3.1-8b-instant)
#                                                                             │
#                                                                             ▼
#                                                               { "answer": ..., "sources": [...] }
#
# ============================================================

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

from src.vector_store import load_vectorstore, get_retriever

# Charger les variables d'environnement du fichier .env
load_dotenv()

# ---- Paramètres du LLM ----
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.1-8b-instant"   # Modèle Groq rapide et gratuit
TEMPERATURE   = 0.2                       # 0 = précis, 1 = créatif
TOP_K         = 4                         # Nombre de chunks récupérés


# ============================================================
# Prompt personnalisé pour l'assistant pédagogique
# ============================================================
# {context}  → remplacé automatiquement par les chunks récupérés
# {question} → remplacé par la question de l'étudiant

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un assistant pédagogique qui aide les étudiants à comprendre leurs cours.
Réponds UNIQUEMENT en te basant sur le contexte fourni ci-dessous.

--- CONTEXTE EXTRAIT DU COURS ---
{context}
----------------------------------

Question de l'étudiant : {question}

Règles importantes :
- Si la réponse se trouve dans le contexte : réponds clairement et pédagogiquement.
- Si l'information N'EST PAS dans le contexte : réponds exactement :
  "Cette information ne se trouve pas dans les documents chargés."
- Ne jamais inventer une information absente du contexte.
- Réponds toujours en français.

Réponse :"""
)


def _get_llm() -> ChatGroq:
    """
    Initialise le modèle de langage ChatGroq.

    Returns:
        ChatGroq: LLM prêt à générer des réponses.

    Raises:
        ValueError: Si la clé API Groq est absente du fichier .env.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "❌ Clé API Groq manquante !\n"
            "💡 Copiez .env.example en .env et ajoutez : GROQ_API_KEY=votre_clé\n"
            "   Obtenez une clé gratuite sur https://console.groq.com/keys"
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=TEMPERATURE,
    )


def answer_with_rag(question: str) -> dict:
    """
    Répond à une question en cherchant dans la base vectorielle (RAG).

    Étapes internes :
      1. Charge la base Chroma depuis vectorstore/
      2. Récupère les 4 chunks les plus pertinents pour la question
      3. Construit un contexte en assemblant ces chunks
      4. Envoie [contexte + question] au LLM ChatGroq
      5. Retourne la réponse + les sources utilisées

    Args:
        question (str): Question posée par l'étudiant, en langage naturel.

    Returns:
        dict: {
            "answer"  : str   — la réponse générée par le LLM,
            "sources" : list  — les Documents LangChain utilisés comme sources
        }

    Raises:
        FileNotFoundError : si la base vectorielle n'a pas encore été créée.
        ValueError        : si la clé API Groq est absente.
    """
    print(f"\n❓ Question : {question}")

    # --- Étape 1 : Charger la base Chroma depuis le disque ---
    vectorstore = load_vectorstore()

    # --- Étape 2 : Créer le retriever (cherche les 4 chunks les plus proches) ---
    retriever = get_retriever(vectorstore, k=TOP_K)

    # --- Étape 3 : Initialiser le LLM ---
    llm = _get_llm()
    print(f"🤖 LLM : {LLM_MODEL}  |  Température : {TEMPERATURE}  |  Top-K : {TOP_K}")

    # --- Étape 4 : Assembler la chaîne RAG ---
    # RetrievalQA gère automatiquement :
    #   retriever → chunks → prompt → LLM → réponse
    #
    # chain_type="stuff" : tous les chunks sont "stuffés" (insérés) dans un seul prompt.
    # C'est la méthode la plus simple : adaptée quand TOP_K est petit (≤ 6).
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,      # Garder les sources pour les afficher
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    # --- Étape 5 : Poser la question et récupérer la réponse ---
    result = rag_chain.invoke({"query": question})

    answer  = result.get("result", "Aucune réponse générée.")
    sources = result.get("source_documents", [])

    print(f"💬 Réponse générée (basée sur {len(sources)} source(s)).")

    return {
        "answer" : answer,
        "sources": sources,
    }


# ============================================================
# Fonctions de compatibilité pour app.py et agent.py
# ============================================================

def get_llm() -> ChatGroq:
    """Alias public de _get_llm()."""
    return _get_llm()


def create_rag_chain(retriever) -> RetrievalQA:
    """
    Crée une chaîne RAG à partir d'un retriever existant.
    Utilisé par app.py et agent.py.
    """
    llm = _get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )


def ask_question(rag_chain: RetrievalQA, question: str) -> dict:
    """
    Pose une question à une chaîne RAG déjà construite.
    Utilisé par app.py.
    """
    result = rag_chain.invoke({"query": question})
    return {
        "answer" : result.get("result", "Aucune réponse générée."),
        "sources": result.get("source_documents", []),
    }
