# ============================================================
# embeddings.py — Transformation du texte en vecteurs numériques
# ============================================================
#
# ❓ C'EST QUOI UN EMBEDDING ?
# -----------------------------
# Un embedding est une façon de représenter du texte sous forme
# de tableau de nombres (appelé "vecteur").
#
# Exemple :
#   "Je comprends Python"  →  [0.12, -0.85, 0.34, 0.67, ...]
#   "J'apprends Python"    →  [0.11, -0.83, 0.36, 0.65, ...]  ← très proche !
#   "J'aime le football"   →  [-0.9, 0.20, -0.50, 0.10, ...]  ← très éloigné
#
# Propriété clé : deux textes avec un sens SIMILAIRE produisent
# des vecteurs PROCHES dans l'espace mathématique.
# C'est ce qui permet à ChromaDB de retrouver les passages
# les plus pertinents pour répondre à une question.
#
# ❓ QUEL MODÈLE ON UTILISE ?
# ----------------------------
# Modèle : sentence-transformers/all-MiniLM-L6-v2
#   ✅ Gratuit et open-source (HuggingFace)
#   ✅ Fonctionne localement, sans connexion internet après install
#   ✅ Rapide et léger (ne nécessite pas de GPU)
#   ✅ Produit des vecteurs de 384 dimensions
#   ✅ Très bon pour les langues européennes (français inclus)
#
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings

# Nom du modèle HuggingFace à utiliser
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Crée et retourne le modèle d'embeddings HuggingFace.

    Au premier appel : le modèle est téléchargé depuis HuggingFace
    et mis en cache sur votre machine (~90 Mo).
    Aux appels suivants : le modèle est chargé depuis le cache (rapide).

    Returns:
        HuggingFaceEmbeddings: Modèle prêt à transformer du texte en vecteurs.
    """
    print(f"🔢 Chargement du modèle d'embeddings : '{MODEL_NAME}'")
    print("   (Premier lancement : téléchargement ~90 Mo, puis mis en cache)")

    # Créer le modèle d'embeddings
    embeddings = HuggingFaceEmbeddings(
        # Nom du modèle sur HuggingFace Hub
        model_name=MODEL_NAME,

        # Utiliser le CPU — pas besoin de carte graphique (GPU)
        model_kwargs={"device": "cpu"},

        # Normaliser les vecteurs (longueur = 1)
        # → améliore la précision de la recherche par similarité cosinus
        encode_kwargs={"normalize_embeddings": True},
    )

    print("✅ Modèle d'embeddings prêt.")
    return embeddings


# Alias pour la compatibilité avec le reste du code (vector_store.py)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Alias de get_embeddings_model() pour la compatibilité."""
    return get_embeddings_model()
