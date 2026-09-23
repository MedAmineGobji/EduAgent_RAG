# ============================================================
# vector_store.py — Gestion de la base de données vectorielle
# ============================================================
#
# ❓ C'EST QUOI CHROMADB ?
# -------------------------
# ChromaDB est une base de données spécialisée pour stocker
# et rechercher des vecteurs (embeddings).
#
# Flux de données :
#   [Chunks de texte]
#        ↓  (get_embeddings_model)
#   [Vecteurs numériques]
#        ↓  (Chroma.from_documents)
#   [Stockés sur le disque dans vectorstore/]
#
#   Puis, lors d'une question :
#   [Question] → [Vecteur question] → [Recherche dans ChromaDB]
#        → [Top-K chunks les plus proches] → [Réponse LLM]
#
# ❓ PERSISTANCE : c'est quoi ?
# ------------------------------
# En passant persist_directory="vectorstore/", ChromaDB sauvegarde
# tout sur le disque. On n'a donc besoin d'indexer les PDFs
# qu'UNE SEULE FOIS. Les fois suivantes, on charge directement
# la base existante avec load_vectorstore().
#
# ============================================================

import os
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings_model

# Dossier où ChromaDB sauvegarde les vecteurs sur le disque
VECTORSTORE_DIR = "vectorstore"

# Nom de la collection dans ChromaDB (comme une "table" dans une BDD)
COLLECTION_NAME = "cours_collection"


def create_vectorstore(chunks: list) -> Chroma:
    """
    Crée une base vectorielle Chroma persistante à partir de chunks de texte.

    Étapes internes :
      1. Charge le modèle d'embeddings (all-MiniLM-L6-v2)
      2. Transforme chaque chunk en vecteur numérique
      3. Sauvegarde les vecteurs + le texte original dans vectorstore/

    Args:
        chunks (list): Liste de chunks LangChain (sortie de chunking.py).

    Returns:
        Chroma: Instance de la base vectorielle créée et sauvegardée.
    """
    print(f"🗄️  Création de la base vectorielle dans '{VECTORSTORE_DIR}/'...")

    # Récupérer le modèle qui transforme le texte en vecteurs
    embeddings = get_embeddings_model()

    # Chroma.from_documents() fait tout en une seule étape :
    #   - vectorise chaque chunk (appelle embeddings.embed_documents)
    #   - stocke le vecteur + le texte + les métadonnées dans ChromaDB
    #   - sauvegarde automatiquement sur le disque (persist_directory)
    vectorstore = Chroma.from_documents(
        documents=chunks,              # Les morceaux de texte à indexer
        embedding=embeddings,          # Le modèle pour les transformer en vecteurs
        persist_directory=VECTORSTORE_DIR,  # Dossier de sauvegarde sur le disque
        collection_name=COLLECTION_NAME,    # Nom logique de la collection
    )

    print(f"✅ Base vectorielle créée : {len(chunks)} chunks stockés dans '{VECTORSTORE_DIR}/'.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Charge une base vectorielle Chroma déjà existante depuis le disque.

    Utilise quand la base a déjà été créée une fois avec create_vectorstore().
    Beaucoup plus rapide que de re-indexer tous les documents.

    Returns:
        Chroma: Instance de la base vectorielle existante.

    Raises:
        FileNotFoundError: Si la base n'a pas encore été créée.
    """
    # Vérifier que la base existe avant de tenter de la charger
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"❌ Base vectorielle introuvable dans '{VECTORSTORE_DIR}/'.\n"
            "💡 Indexez d'abord vos documents en cliquant sur "
            "'Indexer les documents' dans l'interface Streamlit."
        )

    print(f"📂 Chargement de la base vectorielle depuis '{VECTORSTORE_DIR}/'...")

    # Charger le même modèle d'embeddings que lors de la création
    # (obligatoire : ChromaDB en a besoin pour vectoriser les nouvelles questions)
    embeddings = get_embeddings_model()

    # Ouvrir la base existante sur le disque (sans re-vectoriser quoi que ce soit)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,   # Dossier où la base est sauvegardée
        embedding_function=embeddings,        # Modèle pour vectoriser les questions
        collection_name=COLLECTION_NAME,      # Nom de la collection à ouvrir
    )

    # Afficher le nombre de chunks stockés
    count = vectorstore._collection.count()
    print(f"✅ Base vectorielle chargée : {count} chunk(s) disponible(s).")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 4):
    """
    Crée un retriever à partir d'une base vectorielle Chroma.

    Le retriever est utilisé par la chaîne RAG pour récupérer
    les k chunks les plus pertinents pour une question donnée.

    Args:
        vectorstore (Chroma): Base vectorielle chargée.
        k (int): Nombre de chunks à retourner (défaut : 4).

    Returns:
        VectorStoreRetriever: Retriever prêt à l'emploi.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def vectorstore_exists() -> bool:
    """
    Vérifie si une base vectorielle a déjà été créée sur le disque.

    Returns:
        bool: True si la base existe et est non vide, False sinon.
    """
    return os.path.exists(VECTORSTORE_DIR) and bool(os.listdir(VECTORSTORE_DIR))
