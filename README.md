# 🎓 EduAgent RAG — Assistant Étudiant Intelligent

> Assistant pédagogique alimenté par **LangChain**, **Groq** et **ChromaDB**.  
> Il analyse vos cours PDF et répond à vos questions, résume des textes, ou génère des quiz de révision.

---

## 📌 Présentation du projet

**EduAgent RAG** est une application Python qui combine deux approches complémentaires :

- **RAG (Retrieval-Augmented Generation)** : le modèle cherche dans vos propres documents avant de répondre, garantissant des réponses fidèles à votre cours.
- **Agent IA (ReAct)** : un agent intelligent qui analyse chaque demande et choisit automatiquement le bon outil parmi ses 3 capacités.

L'interface Streamlit permet à n'importe quel étudiant d'utiliser le système sans coder.

---

## 🎯 Objectif

Permettre à un étudiant de :
1. Charger ses cours PDF en quelques clics
2. Poser des questions sur le contenu et obtenir des réponses précises avec sources
3. Demander des résumés de chapitres ou de textes longs
4. Générer des quiz de révision personnalisés sur n'importe quel sujet

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INTERFACE STREAMLIT                       │
│                            app.py                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │ question utilisateur
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT IA (ReAct)                          │
│                         src/agent.py                              │
│                                                                    │
│   Thought → Action → Observation → Final Answer                   │
│   "Quelle action est la plus adaptée à cette demande ?"           │
└───────────┬───────────────┬───────────────────┬───────────────────┘
            │               │                   │
            ▼               ▼                   ▼
     ┌────────────┐  ┌─────────────┐  ┌──────────────────┐
     │  RAG_Cours  │  │ Résumé_Texte│  │ Génération_Quiz  │
     │ rag_tool   │  │summary_tool │  │   quiz_tool      │
     └─────┬──────┘  └──────┬──────┘  └────────┬─────────┘
           │                │                   │
           ▼                ▼                   ▼
     ┌──────────┐     ┌──────────┐        ┌──────────┐
     │ ChromaDB │     │  ChatGroq │        │ ChatGroq │
     │vectorstore│   │  (LLM)   │        │  (LLM)   │
     └──────────┘     └──────────┘        └──────────┘
```

### Pipeline d'indexation

```
Fichiers PDF
    │
    ▼  PyPDFLoader                ← src/document_loader.py
    │  Documents LangChain
    ▼  RecursiveCharacterTextSplitter  ← src/chunking.py
    │  Chunks (1000 chars, overlap 200)
    ▼  HuggingFaceEmbeddings      ← src/embeddings.py
    │  Vecteurs 384 dimensions
    ▼  ChromaDB (persist)         ← src/vector_store.py
       Base vectorielle locale
```

---

## 🗂️ Structure du projet

```
EduAgent_RAG/
│
├── app.py                  # Interface Streamlit principale
├── requirements.txt        # Dépendances Python
├── .env                    # Cle API (a creer, non versionne)
├── .env.example            # Modele de configuration
│
├── data/                   # Placez vos PDF ici
│   └── cours.pdf
│
├── vectorstore/            # Base ChromaDB (generee automatiquement)
│
└── src/
    ├── __init__.py
    ├── config.py           # Constantes globales
    ├── document_loader.py  # Chargement des PDF
    ├── chunking.py         # Decoupage en chunks
    ├── embeddings.py       # Modele d'embeddings
    ├── vector_store.py     # Gestion ChromaDB
    ├── rag_pipeline.py     # Pipeline RAG complet
    ├── tools.py            # Outils de l'agent (3 tools)
    ├── agent.py            # Agent IA ReAct
    └── utils.py            # Fonctions utilitaires
```

---

## 🛠️ Technologies utilisées

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| LLM | Groq · `llama-3.1-8b-instant` | Génération de texte rapide |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Vectorisation des chunks |
| Vectorstore | ChromaDB | Base de données vectorielle locale |
| Framework IA | LangChain | Orchestration RAG + Agent |
| Agent | `ZERO_SHOT_REACT_DESCRIPTION` | Sélection automatique des outils |
| Chargement PDF | PyPDFLoader | Extraction du texte |
| Interface | Streamlit | Application web |
| Config | python-dotenv | Gestion des variables d'environnement |

---

## ⚙️ Installation

### Prérequis

- Python 3.10 ou supérieur
- Une clé API Groq gratuite → https://console.groq.com

### Étapes

```bash
# 1. Se placer dans le dossier du projet
cd EduAgent_RAG

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🔑 Configuration du fichier `.env`

```bash
# Copier le modèle
copy .env.example .env       # Windows
# cp .env.example .env        # macOS / Linux
```

Ouvrez `.env` et renseignez votre clé :

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Clé gratuite sur https://console.groq.com → onglet *API Keys* → *Create API Key*.

---

## 🚀 Lancement

```bash
# Depuis le dossier EduAgent_RAG/
python -m streamlit run app.py
```

> Sur Windows, utilisez `python -m streamlit run app.py` si la commande `streamlit` n'est pas reconnue.

L'application s'ouvre sur http://localhost:8501

### Démarrage rapide

1. Placez votre PDF dans le dossier `data/`
2. Cliquez sur **"Indexer les documents"** dans la barre latérale
3. Attendez la fin de l'indexation (quelques secondes)
4. Posez votre première question dans la zone de texte

---

## 🔍 Explication du pipeline RAG

Le RAG (Retrieval-Augmented Generation) résout le problème principal des LLMs : ils ne connaissent pas votre cours spécifique.

### Phase 1 — Indexation (une seule fois)

```
PDF → Texte brut → Chunks → Embeddings → ChromaDB
```

1. **Chargement** (`document_loader.py`) : `PyPDFLoader` extrait le texte de chaque page du PDF.
2. **Chunking** (`chunking.py`) : le texte est découpé en morceaux de 1000 caractères avec un chevauchement de 200 caractères pour ne pas perdre le contexte aux jointures.
3. **Embeddings** (`embeddings.py`) : chaque chunk est transformé en vecteur numérique (384 dimensions) grâce au modèle `all-MiniLM-L6-v2` qui capture le sens du texte.
4. **Stockage** (`vector_store.py`) : les vecteurs sont persistés localement dans ChromaDB (dossier `vectorstore/`).

### Phase 2 — Requête (à chaque question)

```
Question → Embedding → Similarité cosinus → Top-4 chunks → LLM → Réponse
```

1. La question est transformée en vecteur avec le même modèle.
2. ChromaDB calcule la similarité cosinus entre la question et tous les chunks.
3. Les 4 chunks les plus similaires sont récupérés comme contexte.
4. Le LLM (Groq) reçoit `[contexte + question]` et génère une réponse ancrée dans le cours.

> Si l'information n'est pas dans le cours, le système répond : "Cette information ne se trouve pas dans les documents chargés."

---

## 🤖 Explication des outils de l'agent

L'agent utilise le paradigme **ReAct** (Reasoning + Acting) :

```
Thought  →  Action  →  Observation  →  Final Answer
```

Il lit la **description** de chaque outil et décide lequel appeler.

### Outil 1 — `RAG_Cours`

| Attribut | Valeur |
|----------|--------|
| **Déclencheur** | Question sur le contenu du cours |
| **Exemples** | "Qu'est-ce qu'un arbre binaire ?", "Explique la récursivité" |
| **Fonctionnement** | Appelle `answer_with_rag()` → cherche dans ChromaDB → réponse sourcée |

### Outil 2 — `Résumé_Texte`

| Attribut | Valeur |
|----------|--------|
| **Déclencheur** | Demande de résumé ou de synthèse |
| **Exemples** | "Résume ce chapitre : …", "Synthétise ce texte : …" |
| **Fonctionnement** | Envoie le texte à ChatGroq avec un prompt de résumé structuré |
| **Format de sortie** | Idée principale · Points clés · À retenir |

### Outil 3 — `Génération_Quiz`

| Attribut | Valeur |
|----------|--------|
| **Déclencheur** | Demande de quiz ou de questions de révision |
| **Exemples** | "Génère un quiz sur les listes", "Teste mes connaissances sur les arbres" |
| **Fonctionnement** | Envoie le sujet à ChatGroq → génère 5 questions avec réponses détaillées |

---

## 💬 Exemples de questions

### Questions de cours (RAG_Cours)

```
Qu'est-ce qu'une liste chaînée ?
Explique le principe de la récursivité avec un exemple.
Quelle est la complexité de l'algorithme de tri rapide ?
Comment fonctionne un arbre AVL ?
```

### Résumés (Résumé_Texte)

```
Résume ce texte : "Un arbre binaire est une structure de données hiérarchique
dans laquelle chaque nœud possède au plus deux fils..."
Fais une synthèse du chapitre sur les graphes.
Explique brièvement ce qu'est la programmation dynamique.
```

### Quiz (Génération_Quiz)

```
Génère 5 questions de révision sur les algorithmes de tri.
Crée un quiz sur les structures de données.
Teste mes connaissances sur la complexité algorithmique.
```

---

## ⚠️ Limites du projet

| Limite | Description |
|--------|-------------|
| **Langue des PDF** | Fonctionne mieux avec des PDFs en français ou en anglais bien formatés |
| **PDFs scannés** | Les PDF image (scan) ne sont pas supportés — le texte doit être extractible |
| **Contexte limité** | Seuls les 4 chunks les plus proches sont utilisés ; des nuances peuvent être manquées |
| **Pas de mémoire** | L'agent ne retient pas les échanges précédents dans la même session |
| **Un seul sujet** | L'architecture est optimisée pour un ensemble de cours d'un même domaine |
| **Clé API requise** | Nécessite une connexion internet et une clé Groq valide |
| **Modèle léger** | `llama-3.1-8b-instant` est moins précis sur des raisonnements très complexes |

---

## 🔮 Améliorations futures

- [ ] **Mémoire conversationnelle** — mémoriser les échanges pour des dialogues cohérents
- [ ] **Multi-PDF** — indexer plusieurs cours et choisir lequel interroger
- [ ] **Support OCR** — intégrer `pytesseract` pour les PDFs scannés
- [ ] **Mode examen** — générer des examens complets avec barème et correction automatique
- [ ] **Export** — télécharger les résumés et quiz en PDF ou Word
- [ ] **Streaming** — afficher la réponse token par token comme ChatGPT
- [ ] **Évaluation RAG** — mesurer la pertinence des réponses (RAGAS)
- [ ] **Support DOCX / TXT** — étendre le chargement à d'autres formats
- [ ] **Déploiement cloud** — Docker + Streamlit Cloud ou Azure

---

## 📄 Licence

Projet open-source à des fins éducatives.

---

*Propulsé par LangChain · Groq · ChromaDB · HuggingFace · Streamlit*