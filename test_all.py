# ============================================================
# test_all.py — Tests complets de toutes les fonctionnalités
# ============================================================
# Lance avec :  python test_all.py
#
# Ce script teste, dans l'ordre :
#   1. Chargement de l'environnement (.env)
#   2. Création d'un PDF de test
#   3. Document Loader (lecture PDF)
#   4. Chunking (découpage en morceaux)
#   5. Embeddings (vectorisation du texte)
#   6. VectorStore (création + chargement ChromaDB)
#   7. RAG Pipeline (question → réponse avec sources)
#   8. Tools (outil résumé + quiz — pas besoin de PDF)
#   9. Agent (question complète via l'agent ReAct)
#  10. Utils (format_sources, format_agent_steps)
# ============================================================

import os
import sys
import struct
import zlib
import time

# Ajouter le dossier racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Helpers d'affichage
# ============================================================

PASS  = "  ✅ PASS"
FAIL  = "  ❌ FAIL"
SKIP  = "  ⏭️  SKIP"
SEP   = "─" * 58

def section(title: str):
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print(f"{'=' * 58}")

def ok(msg: str):
    print(f"{PASS}  {msg}")

def fail(msg: str, err=None):
    print(f"{FAIL}  {msg}")
    if err:
        print(f"       Erreur : {err}")

def warn(msg: str):
    print(f"  ⚠️  {msg}")


# ============================================================
# ÉTAPE 0 — Vérification de l'environnement
# ============================================================
section("0. ENVIRONNEMENT & .env")

try:
    from dotenv import load_dotenv
    load_dotenv()
    ok("dotenv chargé")
except Exception as e:
    fail("dotenv non disponible", e)

groq_key = os.getenv("GROQ_API_KEY", "")
if groq_key.startswith("gsk_") and len(groq_key) > 20:
    ok(f"GROQ_API_KEY présente ({groq_key[:8]}...{groq_key[-4:]})")
else:
    fail("GROQ_API_KEY manquante ou invalide — les tests LLM échoueront")

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
ok(f"Dossier data/ prêt")


# ============================================================
# ÉTAPE 1 — Création d'un PDF de test minimal
# ============================================================
section("1. CRÉATION DU PDF DE TEST")

TEST_PDF = os.path.join(data_dir, "cours_test.pdf")

COURS_CONTENT = """Cours d'Informatique : Algorithmes et Structures de Données

Chapitre 1 : Les Algorithmes de Tri

Un algorithme de tri est une procédure qui réorganise une séquence d'éléments
dans un ordre défini (croissant ou décroissant).

Le tri à bulles (Bubble Sort) est l'un des algorithmes les plus simples.
Il compare deux éléments adjacents et les échange s'ils sont dans le mauvais ordre.
Sa complexité temporelle est O(n²) dans le pire cas.

Le tri rapide (Quick Sort) utilise la stratégie "diviser pour régner".
Il choisit un élément pivot, puis place les éléments plus petits à gauche
et les éléments plus grands à droite. Complexité moyenne : O(n log n).

Le tri fusion (Merge Sort) divise récursivement le tableau en deux moitiés,
les trie séparément, puis les fusionne. Complexité garantie : O(n log n).

Chapitre 2 : Les Structures de Données

Une liste chaînée est une structure de données linéaire.
Chaque noeud contient une valeur et un pointeur vers le noeud suivant.
Les opérations d'insertion et de suppression sont en O(1) en tête de liste.

Un arbre binaire est une structure hiérarchique où chaque noeud
possède au plus deux enfants : un fils gauche et un fils droit.
Les arbres binaires de recherche (ABR) permettent des recherches en O(log n).

Une table de hachage (hash table) utilise une fonction de hachage pour
associer des clés à des valeurs. L'accès, l'insertion et la suppression
sont en O(1) en moyenne.

Chapitre 3 : La Récursivité

La récursivité est une technique où une fonction s'appelle elle-même.
Elle nécessite toujours un cas de base qui arrête la récursion.

Exemple : calcul du factoriel de n
  factoriel(0) = 1  (cas de base)
  factoriel(n) = n * factoriel(n-1)  (cas récursif)

La suite de Fibonacci est un exemple classique de récursivité :
  fib(0) = 0, fib(1) = 1
  fib(n) = fib(n-1) + fib(n-2)
"""

def _make_minimal_pdf(filepath: str, text: str):
    """Crée un PDF valide minimal en pur Python (sans librairie externe)."""
    # Encoder le texte en lignes de 80 chars max (ASCII safe)
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    lines = []
    for raw_line in safe.split("\n"):
        while len(raw_line) > 80:
            lines.append(raw_line[:80])
            raw_line = raw_line[80:]
        lines.append(raw_line)

    # Construire les commandes de page PDF
    font_size = 9
    page_h = 841.89  # A4
    margin = 40
    y = page_h - margin
    line_h = font_size + 3
    pdf_lines = [
        "BT",
        "/F1 9 Tf",
    ]
    for l in lines:
        escaped = l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        pdf_lines.append(f"  {margin} {y:.2f} Td")
        pdf_lines.append(f"  ({escaped}) Tj")
        pdf_lines.append("  0 0 Td")  # reset x
        y -= line_h
        if y < margin + 40:
            break  # une seule page pour ce test
    pdf_lines.append("ET")
    stream_content = "\n".join(pdf_lines)
    stream_bytes = stream_content.encode("latin-1")
    stream_len = len(stream_bytes)

    body = []
    offsets = []

    body.append(b"%PDF-1.4\n")

    offsets.append(len(b"".join(body)))
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    body.append(obj1)

    offsets.append(len(b"".join(body)))
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    body.append(obj2)

    offsets.append(len(b"".join(body)))
    obj3 = (
        f"3 0 obj\n"
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 595.28 841.89] "
        f"/Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R >> >> >>\n"
        f"endobj\n"
    ).encode("latin-1")
    body.append(obj3)

    offsets.append(len(b"".join(body)))
    obj4 = (
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n"
    ).encode("latin-1") + stream_bytes + b"\nendstream\nendobj\n"
    body.append(obj4)

    offsets.append(len(b"".join(body)))
    obj5 = (
        "5 0 obj\n"
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        "endobj\n"
    ).encode("latin-1")
    body.append(obj5)

    raw = b"".join(body)
    xref_offset = len(raw)
    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    full = raw + xref.encode() + trailer.encode()
    with open(filepath, "wb") as f:
        f.write(full)


try:
    _make_minimal_pdf(TEST_PDF, COURS_CONTENT)
    size = os.path.getsize(TEST_PDF)
    ok(f"PDF créé : {TEST_PDF} ({size} octets)")
except Exception as e:
    fail("Impossible de créer le PDF de test", e)
    sys.exit(1)


# ============================================================
# ÉTAPE 2 — Document Loader
# ============================================================
section("2. DOCUMENT LOADER")

try:
    from src.document_loader import load_all_pdfs, load_documents
    docs = load_all_pdfs()
    if docs:
        ok(f"{len(docs)} document(s) chargé(s) depuis data/")
        ok(f"Premier doc — source : {docs[0].metadata.get('source', '?')} | page : {docs[0].metadata.get('page', '?')}")
        preview = docs[0].page_content[:80].replace("\n", " ")
        ok(f"Aperçu contenu : «{preview}…»")
    else:
        fail("Aucun document chargé")
except Exception as e:
    fail("Erreur dans document_loader", e)
    docs = []


# ============================================================
# ÉTAPE 3 — Chunking
# ============================================================
section("3. CHUNKING")

chunks = []
try:
    from src.chunking import split_documents
    if docs:
        chunks = split_documents(docs)
        ok(f"{len(chunks)} chunk(s) créé(s) depuis {len(docs)} document(s)")
        ok(f"Taille premier chunk : {len(chunks[0].page_content)} caractères")
        ok(f"Chevauchement observé — source préservée : {chunks[0].metadata.get('source', '?')}")
    else:
        warn("Pas de documents à découper — test ignoré")
except Exception as e:
    fail("Erreur dans chunking", e)


# ============================================================
# ÉTAPE 4 — Embeddings
# ============================================================
section("4. EMBEDDINGS (HuggingFace)")

embeddings_model = None
try:
    from src.embeddings import get_embeddings_model
    t0 = time.time()
    embeddings_model = get_embeddings_model()
    elapsed = time.time() - t0
    ok(f"Modèle chargé en {elapsed:.1f}s")

    # Test de vectorisation d'une phrase simple
    test_phrase = "Qu'est-ce que la récursivité ?"
    vec = embeddings_model.embed_query(test_phrase)
    ok(f"Vecteur produit : {len(vec)} dimensions")
    ok(f"Extrait : [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, ...]")

    # Test de similarité : deux phrases proches doivent avoir des vecteurs proches
    import numpy as np
    v1 = np.array(embeddings_model.embed_query("algorithme de tri"))
    v2 = np.array(embeddings_model.embed_query("méthode de classement"))
    v3 = np.array(embeddings_model.embed_query("recette de cuisine"))
    cos12 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cos13 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))
    ok(f"Similarité 'tri' / 'classement' = {cos12:.3f}  (attendu : élevé)")
    ok(f"Similarité 'tri' / 'cuisine'    = {cos13:.3f}  (attendu : faible)")
    if cos12 > cos13:
        ok("Similarité sémantique correcte ✓")
    else:
        warn("Similarité sémantique inattendue")

except Exception as e:
    fail("Erreur dans embeddings", e)


# ============================================================
# ÉTAPE 5 — VectorStore (ChromaDB)
# ============================================================
section("5. VECTORSTORE (ChromaDB)")

vectorstore = None
try:
    from src.vector_store import create_vectorstore, load_vectorstore, get_retriever, vectorstore_exists

    if chunks:
        # Supprimer l'ancienne base pour repartir propre
        import shutil
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            os.makedirs("vectorstore", exist_ok=True)
            ok("Ancienne base supprimée")

        t0 = time.time()
        vectorstore = create_vectorstore(chunks)
        elapsed = time.time() - t0
        ok(f"Base créée en {elapsed:.1f}s — {len(chunks)} chunks stockés")

        # Test chargement depuis disque
        vectorstore2 = load_vectorstore()
        count = vectorstore2._collection.count()
        ok(f"Base rechargée depuis disque : {count} chunk(s)")

        # Test retriever
        retriever = get_retriever(vectorstore2, k=3)
        results = retriever.invoke("algorithme de tri")
        ok(f"Retriever : {len(results)} résultat(s) pour 'algorithme de tri'")
        for i, r in enumerate(results, 1):
            preview = r.page_content[:60].replace("\n", " ")
            ok(f"  Résultat {i} : «{preview}…»")

        # Test vectorstore_exists
        assert vectorstore_exists(), "vectorstore_exists() doit retourner True"
        ok("vectorstore_exists() = True ✓")
    else:
        warn("Pas de chunks — test vectorstore ignoré")

except Exception as e:
    fail("Erreur dans vector_store", e)
    import traceback; traceback.print_exc()


# ============================================================
# ÉTAPE 6 — RAG Pipeline
# ============================================================
section("6. RAG PIPELINE")

try:
    from src.rag_pipeline import answer_with_rag

    questions_rag = [
        "Qu'est-ce que le tri à bulles ?",
        "Explique la récursivité.",
        "Comment fonctionne une table de hachage ?",
    ]

    for q in questions_rag:
        print(f"\n  Question : {q}")
        t0 = time.time()
        result = answer_with_rag(q)
        elapsed = time.time() - t0
        answer  = result["answer"]
        sources = result["sources"]
        print(f"  Réponse  : {answer[:150].replace(chr(10), ' ')}…")
        ok(f"Répondu en {elapsed:.1f}s — {len(sources)} source(s) PDF trouvée(s)")
        time.sleep(1)  # Éviter rate-limit

except Exception as e:
    fail("Erreur dans rag_pipeline", e)
    import traceback; traceback.print_exc()


# ============================================================
# ÉTAPE 7 — Tools (Résumé + Quiz)
# ============================================================
section("7. TOOLS")

# --- Outil Résumé ---
print(f"\n  {SEP}")
print("  Outil : Résumé_Texte")
print(f"  {SEP}")
try:
    from src.tools import _summary_func
    texte = (
        "La récursivité est une technique de programmation où une fonction "
        "s'appelle elle-même pour résoudre un problème. Elle repose sur "
        "un cas de base (qui arrête la récursion) et un cas récursif. "
        "Exemple : factoriel(n) = n * factoriel(n-1), avec factoriel(0) = 1."
    )
    t0 = time.time()
    resume = _summary_func(texte)
    elapsed = time.time() - t0
    ok(f"Résumé généré en {elapsed:.1f}s")
    print(f"\n--- RÉSUMÉ ---\n{resume}\n--- FIN ---\n")
except Exception as e:
    fail("Erreur dans _summary_func", e)

time.sleep(2)  # Éviter rate-limit

# --- Outil Quiz ---
print(f"\n  {SEP}")
print("  Outil : Génération_Quiz")
print(f"  {SEP}")
try:
    from src.tools import _quiz_func
    t0 = time.time()
    quiz = _quiz_func("les algorithmes de tri (bubble sort, quick sort, merge sort)")
    elapsed = time.time() - t0
    ok(f"Quiz généré en {elapsed:.1f}s")
    print(f"\n--- QUIZ ---\n{quiz[:600]}…\n--- FIN ---\n")
except Exception as e:
    fail("Erreur dans _quiz_func", e)

time.sleep(2)


# ============================================================
# ÉTAPE 8 — Agent complet (3 outils)
# ============================================================
section("8. AGENT ReAct (3 outils)")

try:
    from src.agent import create_agent, ask_agent

    agent = create_agent()
    ok("Agent créé avec succès")

    tests_agent = [
        ("RAG",   "Qu'est-ce qu'un arbre binaire selon le cours ?"),
        ("Résumé", "Résume ce texte : La table de hachage utilise une fonction pour associer clés et valeurs avec une complexité O(1) en moyenne."),
        ("Quiz",  "Génère 3 questions de révision sur la récursivité"),
    ]

    for label, question in tests_agent:
        print(f"\n  {SEP}")
        print(f"  Test {label} : {question[:60]}…")
        print(f"  {SEP}")
        t0 = time.time()
        result = ask_agent(agent, question)
        elapsed = time.time() - t0
        answer   = result["answer"]
        tool     = result.get("tool_used", "?")
        steps    = result.get("steps", [])
        sources  = result.get("sources", [])

        ok(f"Répondu en {elapsed:.1f}s")
        ok(f"Outil choisi : {tool}")
        ok(f"Étapes intermédiaires : {len(steps)}")
        ok(f"Sources PDF : {len(sources)}")
        print(f"\n  Réponse :\n  {answer[:250].replace(chr(10), chr(10) + '  ')}\n")
        time.sleep(3)  # Éviter rate-limit entre appels agent

except Exception as e:
    fail("Erreur dans agent", e)
    import traceback; traceback.print_exc()


# ============================================================
# ÉTAPE 9 — Utils
# ============================================================
section("9. UTILS")

try:
    from src.utils import format_sources, format_agent_steps, check_environment
    from langchain.schema import Document

    # Test format_sources
    fake_docs = [
        Document(page_content="Le tri rapide est un algorithme efficace.", metadata={"source": "data/cours_test.pdf", "page": 0}),
        Document(page_content="La récursivité est une technique fondamentale.", metadata={"source": "data/cours_test.pdf", "page": 2}),
    ]
    formatted = format_sources(fake_docs)
    ok("format_sources() fonctionne")
    print(f"\n{formatted}\n")

    # Test format_agent_steps
    from langchain.schema import AgentAction
    fake_steps = [
        (AgentAction(tool="RAG_Cours", tool_input="qu'est-ce que la récursivité", log=""), "La récursivité est..."),
    ]
    formatted_steps = format_agent_steps(fake_steps)
    ok("format_agent_steps() fonctionne")
    print(f"\n{formatted_steps}\n")

    # Test check_environment
    env_checks = check_environment()
    ok("check_environment() fonctionne")
    for k, v in env_checks.items():
        status = "✅" if v else "❌"
        print(f"  {status}  {k}")

except Exception as e:
    fail("Erreur dans utils", e)
    import traceback; traceback.print_exc()


# ============================================================
# RÉSUMÉ FINAL
# ============================================================
section("RÉSUMÉ DES TESTS")
print("""
  Module                État
  ─────────────────────────────────────────────────────
  0. Environnement      voir ci-dessus
  1. Création PDF       voir ci-dessus
  2. Document Loader    voir ci-dessus
  3. Chunking           voir ci-dessus
  4. Embeddings         voir ci-dessus
  5. VectorStore        voir ci-dessus
  6. RAG Pipeline       voir ci-dessus
  7. Tools              voir ci-dessus
  8. Agent ReAct        voir ci-dessus
  9. Utils              voir ci-dessus

  Pour lancer l'interface Streamlit :
  > python -m streamlit run app.py
""")
