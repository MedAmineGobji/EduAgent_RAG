# ============================================================
# chunking.py — Découpage du texte en morceaux (chunks)
# ============================================================
#
# ❓ POURQUOI DÉCOUPER LE TEXTE EN CHUNKS ?
# -----------------------------------------
# Les LLMs (modèles de langage) ont une limite de tokens :
# ils ne peuvent pas lire un cours entier d'un seul coup.
#
# La solution : découper le texte en petits morceaux (chunks),
# vectoriser chaque morceau, puis ne récupérer que ceux
# qui sont pertinents pour répondre à la question.
#
# Exemple concret :
#   Cours de 50 pages → 200 chunks de ~1000 caractères chacun
#   Question : "Qu'est-ce qu'une liste chaînée ?"
#   → On récupère uniquement les 4 chunks qui parlent de listes
#   → Le LLM lit 4 000 caractères au lieu de 50 pages ✅
#
#
# ❓ C'EST QUOI LE CHUNK_OVERLAP (chevauchement) ?
# -------------------------------------------------
# Problème : si on coupe le texte sans chevauchement, une idée
# peut se retrouver coupée en deux entre deux chunks différents.
#
# Solution : chaque chunk "récupère" les 200 derniers caractères
# du chunk précédent → aucune idée n'est perdue à la jonction.
#
# Visualisation avec chunk_size=1000, chunk_overlap=200 :
#
#   |<-------- Chunk 1 : 1000 car. -------->|
#                                  |<---200--->|<--- Chunk 2 : 1000 car. --->|
#                                                              |<---200--->|<--- Chunk 3 ...
#
# Le chevauchement de 200 caractères garantit la continuité du sens.
#
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paramètres de découpage
CHUNK_SIZE    = 1000   # Nombre max de caractères par chunk
CHUNK_OVERLAP = 200    # Nombre de caractères partagés entre deux chunks consécutifs


def split_documents(documents: list) -> list:
    """
    Découpe une liste de Documents LangChain en petits morceaux de texte.

    Utilise RecursiveCharacterTextSplitter, qui coupe intelligemment :
    il essaie d'abord de couper aux paragraphes (\n\n),
    puis aux sauts de ligne (\n), puis aux espaces,
    pour ne jamais couper au milieu d'un mot ou d'une phrase.

    Args:
        documents (list): Liste de Documents LangChain
                          (sortie de document_loader.load_documents()).

    Returns:
        list: Liste de chunks — chaque chunk est un Document LangChain
              avec son propre texte et ses métadonnées (page source, fichier...).
    """
    # --- Créer le découpeur de texte ---
    text_splitter = RecursiveCharacterTextSplitter(
        # Taille maximale de chaque chunk (en nombre de caractères)
        # 1000 caractères ≈ 150-200 mots ≈ un paragraphe dense
        chunk_size=CHUNK_SIZE,

        # Chevauchement entre deux chunks consécutifs (en caractères)
        # 200 caractères partagés = on ne perd pas le contexte à la jonction
        chunk_overlap=CHUNK_OVERLAP,

        # Séparateurs essayés dans l'ordre (du plus au moins préféré) :
        #   "\n\n" → saut de paragraphe (idéal, sens bien délimité)
        #   "\n"   → saut de ligne
        #   " "    → espace entre mots (dernier recours)
        separators=["\n\n", "\n", " "],

        # Mesurer la taille en nombre de caractères (simple et rapide)
        length_function=len,
    )

    # --- Découper tous les documents d'un coup ---
    # split_documents() conserve les métadonnées (page, source) dans chaque chunk
    chunks = text_splitter.split_documents(documents)

    print(
        f"✂️  Découpage terminé :\n"
        f"   - Documents originaux : {len(documents)}\n"
        f"   - Chunks créés        : {len(chunks)}\n"
        f"   - Taille par chunk    : ~{CHUNK_SIZE} caractères\n"
        f"   - Chevauchement       : {CHUNK_OVERLAP} caractères"
    )

    return chunks
