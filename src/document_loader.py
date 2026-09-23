# ============================================================
# document_loader.py — Chargement des fichiers PDF
# ============================================================
# Ce module se charge de lire les PDFs et de les transformer
# en objets "Document" que LangChain peut manipuler.
#
# Chaque Document contient :
#   - page_content : le texte brut de la page
#   - metadata     : infos comme le numéro de page, le fichier source
# ============================================================

import os
from langchain_community.document_loaders import PyPDFLoader
from src.config import DATA_DIR, PDF_FILENAME


def load_documents(folder_path: str = None) -> list:
    """
    Lit tous les fichiers PDF dans un dossier et retourne une liste de Documents LangChain.

    Chaque page de chaque PDF devient un Document séparé avec :
      - page_content : le texte de la page
      - metadata     : {"source": "chemin/du/fichier.pdf", "page": 0, ...}

    Args:
        folder_path (str): Chemin du dossier contenant les PDFs.
                           Si non spécifié, utilise le dossier 'data/' par défaut.

    Returns:
        list: Liste de tous les Documents LangChain chargés depuis les PDFs.

    Raises:
        FileNotFoundError: Si le dossier n'existe pas ou si aucun PDF n'est trouvé.
    """
    # Utiliser le dossier par défaut si aucun chemin n'est fourni
    if folder_path is None:
        folder_path = DATA_DIR

    # --- Vérification 1 : le dossier existe-t-il ? ---
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"❌ Dossier introuvable : '{folder_path}'\n"
            f"💡 Créez le dossier et placez-y vos fichiers PDF."
        )

    # --- Récupérer la liste de tous les fichiers .pdf dans le dossier ---
    # os.listdir() liste tous les fichiers ; on filtre ceux qui finissent par ".pdf"
    pdf_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    # --- Vérification 2 : y a-t-il au moins un PDF ? ---
    if not pdf_files:
        raise FileNotFoundError(
            f"❌ Aucun fichier PDF trouvé dans '{folder_path}/'.\n"
            f"💡 Placez au moins un fichier .pdf dans ce dossier."
        )

    print(f"📂 {len(pdf_files)} fichier(s) PDF trouvé(s) dans '{folder_path}/'")

    all_documents = []  # Contiendra tous les Documents de tous les PDFs

    # --- Charger chaque PDF un par un ---
    for pdf_file in pdf_files:
        # Construire le chemin complet vers le fichier
        filepath = os.path.join(folder_path, pdf_file)

        print(f"  📄 Chargement : {pdf_file} ...")

        # PyPDFLoader lit le PDF et retourne une liste de Documents
        # (un Document par page du PDF)
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        print(f"     ✅ {len(documents)} page(s) chargée(s)")

        # Ajouter les pages de ce PDF à la liste globale
        all_documents.extend(documents)

    # --- Afficher le résumé final ---
    print(
        f"\n📚 Chargement terminé :\n"
        f"   - Fichiers PDF lus  : {len(pdf_files)}\n"
        f"   - Documents totaux  : {len(all_documents)}"
    )

    return all_documents


def load_pdf(filepath: str = None) -> list:
    """
    Charge un seul fichier PDF et retourne ses pages sous forme de Documents.

    Args:
        filepath (str): Chemin vers le fichier PDF.
                        Si non spécifié, utilise 'data/cours.pdf' par défaut.

    Returns:
        list: Liste d'objets Document LangChain (un par page du PDF).

    Raises:
        FileNotFoundError: Si le fichier PDF n'existe pas.
    """
    # Utiliser le chemin par défaut si aucun chemin n'est donné
    if filepath is None:
        filepath = os.path.join(DATA_DIR, PDF_FILENAME)

    # Vérifier que le fichier existe avant d'essayer de le lire
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Fichier PDF introuvable : '{filepath}'\n"
            f"💡 Placez votre fichier PDF dans le dossier '{DATA_DIR}/'"
        )

    print(f"📄 Chargement du fichier : {filepath}")

    # PyPDFLoader extrait le texte de chaque page du PDF
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    print(f"✅ {len(documents)} page(s) chargée(s) depuis : {os.path.basename(filepath)}")
    return documents


def load_all_pdfs() -> list:
    """Alias de load_documents() pour la compatibilité avec app.py."""
    return load_documents()


def _load_all_pdfs_legacy() -> list:
    """
    Charge TOUS les fichiers PDF présents dans le dossier 'data/'.

    Utile si vous avez plusieurs cours à indexer en même temps.

    Returns:
        list: Liste de tous les Documents chargés depuis tous les PDFs.

    Raises:
        FileNotFoundError: Si le dossier 'data/' n'existe pas.
        ValueError: Si aucun fichier PDF n'est trouvé dans 'data/'.
    """
    all_documents = []

    # Vérifier que le dossier data/ existe
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"❌ Dossier '{DATA_DIR}/' introuvable.\n"
            f"💡 Créez le dossier '{DATA_DIR}/' à la racine du projet."
        )

    # Lister uniquement les fichiers .pdf dans le dossier
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(
            f"❌ Aucun fichier PDF trouvé dans '{DATA_DIR}/'.\n"
            f"💡 Placez au moins un fichier .pdf dans ce dossier."
        )

    # Charger chaque PDF et ajouter ses pages à la liste globale
    for pdf_file in pdf_files:
        filepath = os.path.join(DATA_DIR, pdf_file)
        docs = load_pdf(filepath)
        all_documents.extend(docs)

    print(
        f"\n📚 Résumé : {len(all_documents)} page(s) chargée(s) "
        f"depuis {len(pdf_files)} fichier(s) PDF."
    )
    return all_documents
