# 🎓 EduAgent RAG — AI Study Assistant

An intelligent study assistant that combines **Retrieval-Augmented Generation (RAG)** with a **ReAct agent** to help students learn from their own course material.

## ✨ What it does

- 📄 Ingests course PDFs
- 🔎 Retrieves relevant passages with semantic search
- 🤖 Uses an AI agent to select the right tool
- 💬 Answers questions grounded in the uploaded documents
- 📝 Summarizes long content
- 🧠 Generates personalized revision quizzes
- 🖥️ Provides a simple Streamlit web interface

## 🏗️ Architecture

```text
PDFs
 │
 ▼
PyPDFLoader → Chunking → Embeddings → ChromaDB
                                      │
                                      ▼
User → Streamlit → ReAct Agent → RAG / Summary / Quiz
                                      │
                                      ▼
                                    LLM
```

### RAG pipeline

```text
Question
   ↓
Embedding
   ↓
Similarity search
   ↓
Top relevant chunks
   ↓
LLM + retrieved context
   ↓
Grounded answer
```

The project uses local vector storage with **ChromaDB** and semantic embeddings from **Sentence Transformers**.

## 🛠️ Tech stack

| Area | Technologies |
|---|---|
| Language | Python 3.10+ |
| AI orchestration | LangChain |
| LLM | Groq / Llama |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | ChromaDB |
| PDF processing | PyPDFLoader |
| UI | Streamlit |
| Configuration | python-dotenv |

## 🚀 Getting started

### 1. Clone

```bash
git clone https://github.com/MedAmineGobji/EduAgent_RAG.git
cd EduAgent_RAG
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and add your Groq API key.

> Never commit real API keys or secrets.

### 5. Run

```bash
python -m streamlit run app.py
```

The application runs locally on port 8501 by default.

## 📁 Project structure

```text
EduAgent_RAG/
├── app.py
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py
│   ├── rag_pipeline.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── tools.py
│   └── utils.py
└── test_all.py
```

## 🔬 Engineering focus

This project demonstrates practical experience with:

- RAG system design
- Vector search and embeddings
- LLM application architecture
- Agent/tool orchestration
- Prompt-driven workflows
- Document processing
- Python modularization
- Automated testing

## 🔮 Roadmap

- [ ] Conversation memory
- [ ] Multi-PDF knowledge bases
- [ ] OCR support for scanned documents
- [ ] RAG evaluation with RAGAS
- [ ] Streaming responses
- [ ] Docker deployment
- [ ] Cloud deployment

## 📄 License

Open-source educational project.

---

**Author:** [Med Amine Gobji](https://github.com/MedAmineGobji)
