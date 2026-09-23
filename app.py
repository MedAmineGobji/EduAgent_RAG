# ============================================================
# app.py — Interface principale EduAgent RAG
# ============================================================
#
# Point d'entrée de l'application. Lance avec :
#   python -m streamlit run app.py
#
# Structure de cette interface :
#   - Sidebar   : indexation des PDFs + statut du système
#   - Zone principale : historique de conversation + zone de saisie
#
# Séparation des responsabilités :
#   app.py     → interface uniquement (Streamlit)
#   src/agent  → décision intelligente (quel outil ?)
#   src/tools  → exécution des actions (RAG, résumé, quiz)
#   src/rag_pipeline → recherche dans ChromaDB + génération LLM
# ============================================================

import os
import sys
import streamlit as st

# Ajoute le dossier racine du projet au chemin Python
# pour que "from src.xxx import yyy" fonctionne
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_all_pdfs
from src.chunking import split_documents
from src.vector_store import create_vectorstore, vectorstore_exists
from src.agent import create_agent, ask_agent
from src.utils import format_sources, format_agent_steps

# ============================================================
# Configuration globale de la page Streamlit
# ============================================================
st.set_page_config(
    page_title="EduAgent RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Feuille de style CSS embarquée ---
# Styles pour les bulles de conversation et la mise en page générale.
# On évite les frameworks lourds pour garder le code lisible.
st.markdown("""
<style>
/* ---- En-tête principal ---- */
.edu-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a3c5e;
    margin-bottom: 0;
}
.edu-subtitle {
    font-size: 1rem;
    color: #666;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}

/* ---- Bulles de conversation ---- */
.msg-user {
    background: #eef2ff;
    border-left: 4px solid #4f6ef7;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 10px 0;
    font-size: 0.97rem;
}
.msg-bot {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 10px 0;
}

/* ---- Badge outil utilisé ---- */
.tool-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
}

/* ---- Source card ---- */
.source-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cache Streamlit — évite de recharger l'agent à chaque clic
# ============================================================
# @st.cache_resource garde l'objet en mémoire pendant toute
# la session. Sans ce cache, l'agent se rechargerait à chaque
# interaction, ce qui prendrait plusieurs secondes.
@st.cache_resource
def _load_agent():
    """
    Initialise l'agent IA et le met en cache pour la session.

    Returns:
        AgentExecutor: Agent LangChain avec les 3 outils (RAG, résumé, quiz).
    """
    return create_agent()


# ============================================================
# SIDEBAR — Configuration et indexation
# ============================================================
with st.sidebar:
    st.markdown("### 🎓 EduAgent RAG")
    st.markdown(
        "Assistant étudiant alimenté par **LangChain** + **Groq**.\n\n"
        "Il choisit automatiquement le bon outil selon votre demande :\n"
        "- 🔍 **RAG** — répondre depuis vos cours PDF\n"
        "- 📝 **Résumé** — synthétiser un texte\n"
        "- 🧪 **Quiz** — générer des questions de révision"
    )
    st.divider()

    # ---- Section : Indexation des documents ----
    st.subheader("📚 Indexer les documents")
    st.caption(
        "Placez vos fichiers PDF dans le dossier `data/`, "
        "puis cliquez sur le bouton ci-dessous."
    )

    if st.button("🔄 Indexer les documents", type="primary", use_container_width=True):
        # st.status affiche une barre de progression avec les étapes visibles
        with st.status("Indexation en cours…", expanded=True) as status:
            try:
                # Étape 1 : Charger les PDFs depuis data/
                st.write("📄 Chargement des fichiers PDF…")
                documents = load_all_pdfs()
                st.write(f"   → {len(documents)} page(s) chargée(s)")

                # Étape 2 : Découper en chunks (morceaux de 1000 chars)
                st.write("✂️ Découpage en chunks (1000 chars, overlap 200)…")
                chunks = split_documents(documents)
                st.write(f"   → {len(chunks)} chunk(s) créé(s)")

                # Étape 3 : Vectoriser et sauvegarder dans ChromaDB
                st.write("🗄️ Vectorisation et sauvegarde ChromaDB…")
                create_vectorstore(chunks)

                # Vider le cache agent pour forcer le rechargement
                st.cache_resource.clear()

                status.update(
                    label=f"✅ Indexation réussie — {len(chunks)} chunks stockés",
                    state="complete",
                    expanded=False,
                )

            except FileNotFoundError as e:
                status.update(label="❌ Aucun PDF trouvé", state="error")
                st.error(str(e))
                st.info(
                    "**Vérifiez que :**\n"
                    "- Le dossier `data/` existe\n"
                    "- Il contient au moins un fichier `.pdf`"
                )
            except Exception as e:
                status.update(label="❌ Erreur d'indexation", state="error")
                st.error(f"Erreur inattendue : {e}")

    st.divider()

    # ---- Statut de la base vectorielle ----
    if vectorstore_exists():
        st.success("✅ Base vectorielle prête", icon="🗄️")
    else:
        st.warning("⚠️ Aucun cours indexé", icon="📭")

    # ---- Bouton vider l'historique ----
    if st.button("🗑️ Vider l'historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("LangChain · Groq · ChromaDB · HuggingFace · Streamlit")


# ============================================================
# ZONE PRINCIPALE — En-tête
# ============================================================
st.markdown('<p class="edu-title">🎓 EduAgent RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="edu-subtitle">Votre assistant étudiant intelligent — '
    'posez une question sur vos cours, demandez un résumé ou générez un quiz.</p>',
    unsafe_allow_html=True,
)

# ---- Message d'accueil si aucun cours indexé ----
if not vectorstore_exists():
    st.info(
        "**Bienvenue ! Pour commencer :**\n\n"
        "1. 📁 Placez votre fichier PDF dans le dossier `data/`\n"
        "2. 🔄 Cliquez sur **Indexer les documents** (barre latérale)\n"
        "3. 💬 Posez votre première question ci-dessous",
        icon="📌",
    )

# ============================================================
# HISTORIQUE DE CONVERSATION
# ============================================================
# On stocke les échanges dans st.session_state.history
# (persiste tant que l'onglet navigateur reste ouvert)

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Affichage des échanges passés ----
if not st.session_state.history:
    st.markdown(
        "> 💬 *Aucune question posée pour l'instant.*\n\n"
        "> **Exemples :**\n"
        "> - *« Qu'est-ce qu'un arbre binaire ? »*\n"
        "> - *« Résume ce texte : … »*\n"
        "> - *« Génère un quiz sur les algorithmes de tri »*"
    )

for entry in st.session_state.history:
    # ---- Bulle utilisateur ----
    st.markdown(
        f'<div class="msg-user">'
        f'🧑‍🎓 <strong>Vous</strong><br>{entry["question"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Bulle agent ----
    with st.container():
        # Badge indiquant quel outil a été utilisé
        tool = entry.get("tool_used")
        tool_labels = {
            "RAG_Cours":       "🔍 RAG — Recherche dans le cours",
            "Résumé_Texte":    "📝 Résumé du texte",
            "Génération_Quiz": "🧪 Génération de quiz",
        }
        if tool:
            label = tool_labels.get(tool, f"Outil : {tool}")
            st.markdown(
                f'<span class="tool-badge">{label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="msg-bot">🤖 <strong>EduAgent</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(entry["answer"])

        # ---- Sources PDF (uniquement si l'outil RAG a été utilisé) ----
        if entry.get("sources"):
            with st.expander(f"📄 Sources utilisées ({len(entry['sources'])} extrait(s))", expanded=False):
                for i, doc in enumerate(entry["sources"], 1):
                    meta    = doc.metadata
                    page    = meta.get("page", "?")
                    fichier = os.path.basename(meta.get("source", "Inconnu"))
                    page_num = (page + 1) if isinstance(page, int) else page
                    extrait = doc.page_content.strip()[:300]
                    extrait = extrait + "…" if len(doc.page_content.strip()) > 300 else extrait

                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Source {i}</strong> — 📄 {fichier} | Page {page_num}<br>'
                        f'<em>{extrait}</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ---- Raisonnement de l'agent (mode debug pédagogique) ----
        if entry.get("steps"):
            with st.expander("🔍 Raisonnement de l'agent (Thought → Action → Observation)", expanded=False):
                st.markdown(format_agent_steps(entry["steps"]))


# ============================================================
# FORMULAIRE DE SAISIE
# ============================================================
st.divider()

with st.form(key="question_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_question = st.text_area(
            label="Question",
            placeholder=(
                "Posez une question sur vos cours, demandez un résumé ou générez un quiz…\n"
                "Ex : « Explique la récursivité », « Résume : ... », « Quiz sur les arbres »"
            ),
            height=95,
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Envoyer ➤",
            type="primary",
            use_container_width=True,
        )

# ============================================================
# TRAITEMENT DE LA QUESTION
# ============================================================
if submitted and user_question.strip():

    # Vérification préalable : base vectorielle requise
    if not vectorstore_exists():
        st.error(
            "❌ Aucun cours indexé. "
            "Placez un PDF dans `data/` et cliquez sur **Indexer les documents**."
        )
        st.stop()

    with st.spinner("🤔 L'agent analyse votre demande et choisit le bon outil…"):
        try:
            # Récupérer l'agent depuis le cache (instantané si déjà chargé)
            agent_executor = _load_agent()

            # Appel de l'agent — il décide seul quel outil utiliser
            result = ask_agent(agent_executor, user_question.strip())

            # Enregistrer l'échange dans l'historique de session
            st.session_state.history.append({
                "question": user_question.strip(),
                "answer":   result["answer"],
                "tool_used": result.get("tool_used"),
                "steps":    result.get("steps", []),
                "sources":  result.get("sources", []),
            })

        except Exception as e:
            err = str(e)

            # --- Erreur rate limit Groq (429) ---
            if "429" in err or "rate_limit" in err:
                answer = (
                    "⏳ **Limite de débit Groq atteinte (429 Rate Limit).**\n\n"
                    "Le plan gratuit est limité à **6 000 tokens/minute**. "
                    "Attendez quelques secondes puis renvoyez votre question.\n\n"
                    "💡 *Astuce : les questions courtes consomment moins de tokens.*"
                )
            # --- Clé API manquante ou invalide ---
            elif "401" in err or "403" in err or "api_key" in err.lower():
                answer = (
                    "🔑 **Clé API Groq invalide ou manquante.**\n\n"
                    "Vérifiez que votre fichier `.env` contient :\n"
                    "```\nGROQ_API_KEY=gsk_votre_vraie_clé\n```\n"
                    "Obtenez une clé gratuite sur https://console.groq.com/keys"
                )
            # --- Base vectorielle inaccessible ---
            elif "vectorstore" in err.lower() or "FileNotFoundError" in err:
                answer = (
                    "📭 **Base vectorielle introuvable.**\n\n"
                    "Indexez d'abord vos documents depuis la barre latérale."
                )
            # --- Erreur générique ---
            else:
                answer = (
                    f"⚠️ **Erreur inattendue.**\n\n"
                    f"Détail technique : `{err}`\n\n"
                    "Essayez de reformuler votre question."
                )

            st.session_state.history.append({
                "question": user_question.strip(),
                "answer":   answer,
                "tool_used": None,
                "steps":    [],
                "sources":  [],
            })

    # Recharger la page pour afficher le nouvel échange
    st.rerun()

elif submitted and not user_question.strip():
    st.warning("⚠️ Veuillez saisir une question avant d'envoyer.", icon="✏️")
