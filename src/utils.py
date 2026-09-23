# ============================================================
# utils.py — Fonctions utilitaires réutilisables
# ============================================================
# Ce module contient des fonctions d'aide utilisées dans
# différentes parties du projet.
# ============================================================

import os
import time
from datetime import datetime


def format_sources(source_documents: list) -> str:
    """
    Formate la liste des documents sources pour un affichage lisible.

    Utilisé dans l'interface Streamlit pour montrer à l'étudiant
    d'où vient l'information.

    Args:
        source_documents (list): Liste de Documents LangChain
                                  (retournés par la chaîne RAG).

    Returns:
        str: Texte formaté prêt à être affiché.
    """
    if not source_documents:
        return "Aucune source disponible."

    lines = []
    for i, doc in enumerate(source_documents, 1):
        metadata = doc.metadata

        # Extraire les métadonnées utiles
        page = metadata.get("page", "?")
        source_path = metadata.get("source", "Inconnu")
        source_name = os.path.basename(source_path)  # Juste le nom du fichier

        # Aperçu du contenu (limité à 200 caractères pour ne pas surcharger)
        content = doc.page_content.strip()
        preview = content[:200] + "..." if len(content) > 200 else content

        lines.append(
            f"📌 Source {i} | Fichier : {source_name} | Page : {page + 1 if isinstance(page, int) else page}\n"
            f"   {preview}"
        )

    return "\n\n".join(lines)


def format_agent_steps(steps: list) -> str:
    """
    Formate les étapes de raisonnement de l'agent pour l'affichage.

    Permet à l'étudiant de voir comment l'agent a raisonné
    et quels outils il a utilisés pour répondre.

    Args:
        steps (list): Liste de tuples (AgentAction, observation)
                      retournée par AgentExecutor.

    Returns:
        str: Texte Markdown formaté du raisonnement.
    """
    if not steps:
        return "Aucune étape de raisonnement disponible."

    lines = ["### 🔍 Raisonnement de l'agent\n"]

    for i, (action, observation) in enumerate(steps, 1):
        tool_name = getattr(action, "tool", "Outil inconnu")
        tool_input = getattr(action, "tool_input", "")

        # Limiter l'observation pour ne pas surcharger l'affichage
        obs_str = str(observation)
        obs_preview = obs_str[:400] + "..." if len(obs_str) > 400 else obs_str

        lines.append(
            f"**Étape {i}** — Outil utilisé : `{tool_name}`\n"
            f"- **Requête envoyée** : {tool_input}\n"
            f"- **Résultat obtenu** : {obs_preview}\n"
        )

    return "\n".join(lines)


def measure_time(func):
    """
    Décorateur pour mesurer et afficher le temps d'exécution d'une fonction.

    Usage :
        @measure_time
        def ma_fonction_lente():
            # ... code long ...

    Args:
        func: La fonction à chronométrer.

    Returns:
        wrapper: La fonction encapsulée avec mesure du temps.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"⏱️  '{func.__name__}' exécutée en {duration:.2f} secondes.")
        return result

    return wrapper


def get_timestamp() -> str:
    """
    Retourne l'horodatage actuel sous forme lisible.

    Returns:
        str: Date et heure au format "2024-01-15 14:30:25"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_environment() -> dict:
    """
    Vérifie que tous les prérequis du projet sont en place.

    Returns:
        dict: Dictionnaire {nom_vérification: True/False}
    """
    from src.config import GROQ_API_KEY, DATA_DIR, VECTORSTORE_DIR

    checks = {
        "Clé GROQ_API_KEY": bool(GROQ_API_KEY),
        "Dossier data/": os.path.exists(DATA_DIR),
        "Dossier vectorstore/": os.path.exists(VECTORSTORE_DIR),
    }

    # Vérifier la présence de PDFs dans data/
    if os.path.exists(DATA_DIR):
        pdf_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])
        checks[f"PDF(s) dans data/ ({pdf_count} trouvé(s))"] = pdf_count > 0
    else:
        checks["PDF(s) dans data/"] = False

    return checks


def print_environment_report():
    """
    Affiche un rapport de vérification de l'environnement dans la console.
    Utile pour déboguer lors du premier lancement.
    """
    print("\n" + "=" * 55)
    print("  🔍 VÉRIFICATION DE L'ENVIRONNEMENT")
    print("=" * 55)

    checks = check_environment()
    all_ok = True

    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon}  {name}")
        if not status:
            all_ok = False

    print("=" * 55)
    if all_ok:
        print("  🎉 Tout est configuré ! Vous pouvez démarrer.")
    else:
        print("  ⚠️  Certains éléments sont manquants (voir README.md)")
    print("=" * 55 + "\n")

    return all_ok
