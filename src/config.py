# ============================================================
# config.py — Configuration centralisée du projet EduAgent RAG
# ============================================================
# Ce fichier regroupe TOUS les paramètres du projet.
# Modifier ces valeurs pour personnaliser le comportement
# sans avoir à toucher au reste du code.
# ============================================================

import os
from dotenv import load_dotenv

# Charger les variables du fichier .env (ex: GROQ_API_KEY)
load_dotenv()

# ============================================================
# Clé API
# ============================================================

# Clé Groq lue depuis le fichier .env
# Si la clé est manquante, une erreur claire sera affichée plus tard
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ============================================================
# Modèle de langage (LLM)
# ============================================================

# Modèle Groq utilisé pour générer les réponses
# Autres options : "llama3-70b-8192" (plus puissant), "mixtral-8x7b-32768"
LLM_MODEL: str = "llama-3.1-8b-instant"

# Température : contrôle la créativité du modèle
# 0.0 = réponses précises et déterministes
# 1.0 = réponses créatives et variées
# Pour un assistant de cours, on préfère la précision → 0.2
LLM_TEMPERATURE: float = 0.2

# ============================================================
# Modèle d'embeddings (transformation texte → vecteur)
# ============================================================

# Modèle HuggingFace local (téléchargé automatiquement, gratuit)
# "all-MiniLM-L6-v2" : petit, rapide, efficace pour français et anglais
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# Base de données vectorielle (ChromaDB)
# ============================================================

# Dossier où ChromaDB sauvegarde les vecteurs sur le disque
VECTORSTORE_DIR: str = "vectorstore"

# Nom de la collection dans ChromaDB (comme une "table" dans une BDD)
COLLECTION_NAME: str = "cours_collection"

# ============================================================
# Découpage du texte (Chunking)
# ============================================================

# Nombre maximum de caractères par morceau de texte
# Plus c'est petit → recherche plus précise, mais moins de contexte
# Plus c'est grand → plus de contexte, mais recherche moins précise
CHUNK_SIZE: int = 500

# Chevauchement entre deux morceaux consécutifs (en caractères)
# Permet de ne pas perdre le contexte à la jonction de deux chunks
CHUNK_OVERLAP: int = 100

# ============================================================
# Données (PDFs)
# ============================================================

# Dossier contenant les fichiers PDF à indexer
DATA_DIR: str = "data"

# Nom du fichier PDF principal (utilisé par défaut)
PDF_FILENAME: str = "cours.pdf"

# ============================================================
# Récupération de documents (RAG)
# ============================================================

# Nombre de passages à récupérer depuis ChromaDB pour chaque question
# Plus k est grand → plus de contexte, mais prompt plus long (et plus lent)
TOP_K_RESULTS: int = 4
