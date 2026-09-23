# ============================================================
# agent.py — Agent IA intelligent (Zero-Shot ReAct)
# ============================================================
#
# ❓ COMMENT FONCTIONNE L'AGENT ?
# --------------------------------
# L'agent IA utilise le paradigme ReAct (Reasoning + Acting).
# À chaque question, il suit ce raisonnement automatique :
#
#   Thought  : "La question porte sur le cours → je dois utiliser RAG_Cours"
#   Action   : RAG_Cours
#   Input    : "Qu'est-ce qu'une liste chaînée ?"
#   Observation : [réponse du RAG]
#   Final Answer : "Une liste chaînée est..."
#
# ❓ DIFFÉRENCE AVEC RAG SIMPLE ?
# ---------------------------------
#   RAG Simple → cherche TOUJOURS dans les documents
#   Agent IA   → CHOISIT le bon outil selon la demande :
#     • "Explique-moi X"             → RAG_Cours
#     • "Résume ce texte : ..."      → Résumé_Texte
#     • "Génère un quiz sur X"       → Génération_Quiz
#
# ❓ POURQUOI initialize_agent ?
# --------------------------------
# On utilise initialize_agent avec ZERO_SHOT_REACT_DESCRIPTION :
#   - Aucun téléchargement depuis Hub (pas besoin d'internet au démarrage)
#   - Compatible avec LangChain moderne
#   - L'agent lit les descriptions des tools pour décider
#
# ============================================================

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_groq import ChatGroq

import src.tools as _tools_module
from src.tools import get_all_tools

# Charger les variables d'environnement (.env)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.1-8b-instant"


# ============================================================
# Labels lisibles pour afficher le nom de l'outil choisi
# ============================================================
TOOL_LABELS = {
    "RAG_Cours":        "🔍 Recherche dans le cours (RAG)",
    "Résumé_Texte":     "📝 Résumé du texte",
    "Génération_Quiz":  "🧪 Génération de quiz",
}


def create_agent(rag_chain=None) -> AgentExecutor:
    """
    Crée et configure l'agent IA avec les 3 outils pédagogiques.

    L'agent utilise ZERO_SHOT_REACT_DESCRIPTION : il lit la description
    de chaque outil pour décider lequel appeler sans entraînement préalable.

    Args:
        rag_chain: Ignoré (conservé pour compatibilité avec app.py).
                   Les outils gèrent leur propre LLM/RAG en interne.

    Returns:
        AgentExecutor: Agent prêt à répondre aux questions.

    Raises:
        ValueError: Si GROQ_API_KEY est absent du fichier .env.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "❌ Clé API Groq manquante !\n"
            "💡 Créez un fichier .env et ajoutez : GROQ_API_KEY=votre_clé\n"
            "   Clé gratuite sur : https://console.groq.com"
        )

    # --- LLM de l'agent (prend les décisions) ---
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
    )

    # --- Outils disponibles (rag_tool, summary_tool, quiz_tool) ---
    tools = get_all_tools()

    # --- Préfixe système : empêche l'agent de boucler ---
    # Sans ce préfixe, l'agent répète le même outil plusieurs fois
    # au lieu de s'arrêter après avoir obtenu une observation.
    agent_prefix = (
        "Tu es un assistant pédagogique. "
        "Tu as accès à des outils. "
        "RÈGLE ABSOLUE : utilise UN SEUL outil, lis son résultat, "
        "puis donne IMMÉDIATEMENT ta réponse finale (Final Answer). "
        "Ne rappelle JAMAIS le même outil une seconde fois. "
        "Dès que tu as une Observation, ta prochaine ligne doit être "
        "'Final Answer:' suivi de ta réponse."
    )

    # --- Créer l'agent avec initialize_agent ---
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=2,               # 1 outil + 1 Final Answer = 2 max
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate",
        agent_kwargs={"prefix": agent_prefix},
    )

    print("🤖 Agent IA initialisé avec les outils : "
          f"{[t.name for t in tools]}")
    return agent_executor


def ask_agent(agent_executor: AgentExecutor, question: str) -> dict:
    """
    Pose une question à l'agent et retourne la réponse + l'outil utilisé.

    L'agent va :
      1. Analyser la question (Thought)
      2. Choisir automatiquement le bon outil (Action)
      3. Lire le résultat de l'outil (Observation)
      4. Formuler une réponse finale claire (Final Answer)

    Args:
        agent_executor (AgentExecutor): Agent créé avec create_agent().
        question (str): Question ou demande de l'étudiant.

    Returns:
        dict:
            - "answer"       : réponse finale de l'agent (str)
            - "tool_used"    : nom lisible de l'outil choisi (str)
            - "steps"        : étapes intermédiaires (list)
    """
    print(f"\n🎓 Demande reçue : {question}")

    try:
        result = agent_executor.invoke({"input": question})

        final_answer = result.get("output", "L'agent n'a pas pu générer de réponse.")
        steps        = result.get("intermediate_steps", [])

        # --- Identifier l'outil utilisé depuis les étapes ---
        tool_used = _extract_tool_used(steps)

        # --- Ajouter un préfixe expliquant l'action choisie ---
        if tool_used:
            label  = TOOL_LABELS.get(tool_used, f"Outil : {tool_used}")
            prefix = f"**Action choisie :** {label}\n\n---\n\n"
            final_answer = prefix + final_answer

        # Si l'outil RAG a été utilisé, récupérer les sources sauvegardées
        sources = (
            _tools_module._last_rag_sources
            if tool_used == "RAG_Cours"
            else []
        )

        return {
            "answer":    final_answer,
            "tool_used": tool_used,
            "steps":     steps,
            "sources":   sources,
        }

    except ValueError as e:
        # Clé API manquante ou erreur de configuration
        return {
            "answer":    f"❌ Erreur de configuration : {str(e)}",
            "tool_used": None,
            "steps":     [],
            "sources":   [],
        }
    except Exception as e:
        # Toute autre erreur (réseau, LLM, outil)
        return {
            "answer": (
                f"⚠️ L'agent a rencontré une erreur inattendue.\n\n"
                f"Détail : {str(e)}\n\n"
                "💡 Conseils :\n"
                "  • Vérifiez que des PDFs sont bien indexés (onglet sidebar)\n"
                "  • Reformulez votre question\n"
                "  • Passez en mode RAG Simple si le problème persiste"
            ),
            "tool_used": None,
            "steps":     [],
            "sources":   [],
        }


def _extract_tool_used(steps: list) -> str | None:
    """
    Extrait le nom du premier outil utilisé depuis les étapes intermédiaires.

    Les étapes ont la structure : [(AgentAction, observation), ...]
    AgentAction.tool contient le nom de l'outil appelé.

    Args:
        steps (list): Liste d'étapes intermédiaires de l'agent.

    Returns:
        str | None: Nom de l'outil utilisé, ou None si aucune étape.
    """
    if not steps:
        return None
    # steps[0] = (AgentAction, observation_string)
    first_action = steps[0][0]
    return getattr(first_action, "tool", None)

