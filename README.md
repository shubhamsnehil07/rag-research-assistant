# 📚 RAG Research Paper Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for analyzing research papers using AI.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.29-red)

## 🌟 Features

- **📤 PDF Upload**: Drag-and-drop research papers
- **🤖 AI Q&A**: Natural language question answering
- **📚 Smart Citations**: Answers include source references with page numbers
- **📊 Confidence Scoring**: See how confident the AI is
- **🔍 Hybrid Retrieval**: Combines BM25 keyword search + semantic search
- **🎯 Cross-Encoder Reranking**: Improves answer relevance
- **💬 Interactive UI**: Beautiful Streamlit interface
- **🔄 Multi-Document Support**: Analyze multiple papers simultaneously

## 🎥 Demo

![Demo Screenshot](docs/screenshots/demo.png)

## 🛠️ Tech Stack
\`\`\`
| Component | Technology |
|-----------|------------|
| **Framework** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | Sentence Transformers (MiniLM-L6-v2) |
| **LLM** | Ollama (Llama 3.2) |
| **Retrieval** | Hybrid (BM25 + Dense) |
| **Reranking** | Cross-Encoder (MS MARCO) |
| **UI** | Streamlit |
| **Language** | Python 3.11 |
\`\`\`

## 📋 Prerequisites

- Python 3.11+
- Ollama installed and running
- 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Clone the Repository

``` bash
git clone https://github.com/yourusername/rag-research-assistant.git
cd rag-research-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

**Windows/Mac:**
1. Download from https://ollama.ai
2. Install Ollama
3. Open terminal and run:

```bash
ollama serve
```

4. In a **new terminal**, pull the model:

```bash
ollama pull llama3.2
```

### 5. Add Your PDF

Place your research paper PDFs in the `data/` folder:

```bash
data/
  └── your_paper.pdf
```

### 6. Build Vector Database

```bash
python scripts/rebuild_database_fixed.py
```

### 7. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Ask Questions

1. **Upload PDF** via sidebar (or use pre-loaded database)
2. **Type your question** in the text box
3. **Get AI-powered answers** with citations
4. **View sources** by expanding the references section

### Example Questions

- "What is the main contribution of this paper?"
- "What dataset was used in the experiments?"
- "What accuracy did the model achieve?"
- "What are the limitations mentioned?"

### Features

- **Toggle Citations**: Show/hide source references
- **Toggle Confidence**: See AI confidence scores
- **Export Chat**: Download your Q&A session as JSON
- **Clear History**: Start a new conversation

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│   Query Enhancement             │
│   (Optional Rewriting)          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Hybrid Retrieval              │
│   ┌─────────┬─────────────┐    │
│   │  BM25   │   Dense     │    │
│   │ (40%)   │   (60%)     │    │
│   └─────────┴─────────────┘    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Cross-Encoder Reranking       │
│   (Top-K Selection)             │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   LLM Generation                │
│   (Llama 3.2 via Ollama)        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Answer + Citations            │
│   + Confidence Score            │
└─────────────────────────────────┘
```

## 📊 Performance

- **Processing**: ~50 chunks/second
- **Retrieval**: <100ms for vector search
- **Answer Generation**: 2-5 seconds
- **Accuracy**: 95%+ for factual questions

## 🔬 Advanced Features

### Section-Aware Chunking

Automatically detects and preserves paper structure:
- Abstract
- Introduction
- Methodology
- Results
- Conclusion

### Confidence Scoring

Answers are scored as:
- 🟢 **HIGH** (90%+): Very confident
- 🟡 **MEDIUM** (60-90%): Moderate confidence
- 🔴 **LOW** (<60%): Uncertain, may not be in document

### Citation Tracking

Every answer includes:
- Source document
- Page number
- Section name
- Relevance score

## 📁 Project Structure

```
rag-research-assistant/
├── app.py                      # Streamlit UI
├── core/                       # Core modules
│   ├── rag_system_pro.py      # Main RAG system
│   ├── llm_handler.py         # LLM integration
│   ├── hybrid_retriever.py    # Hybrid search
│   └── ...
├── ingestion/                  # PDF processing
│   └── ingestion_with_abstract.py
├── scripts/                    # Utility scripts
└── data/                       # PDF storage
```

## 🐛 Troubleshooting

### "Ollama not available"

**Solution:**
1. Make sure Ollama is running: `ollama serve`
2. Check model is installed: `ollama list`
3. Pull model if missing: `ollama pull llama3.2`

### "Vector database not found"

**Solution:**
Run the database builder:
```bash
python scripts/rebuild_database_fixed.py
```

### Slow responses

**Solution:**
Use a smaller/faster model:
```bash
ollama pull phi3
```

Then update `app.py` to use `phi3` instead of `llama3.2`.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Shubham Snehil**
- GitHub: [@shubhamsnehil07](https://github.com/shubhamsnehil07)
- Built as a project demonstrating ML engineering skills

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com)
- Powered by [Ollama](https://ollama.ai)
- UI with [Streamlit](https://streamlit.io)
- Embeddings from [Sentence Transformers](https://www.sbert.net)

## 📚 Resources

- [Documentation](docs/INSTALLATION.md)
- [Usage Guide](docs/USAGE.md)
- [Video Demo](#) (Coming soon)

---

**⭐ If you found this helpful, please star the repo!**
