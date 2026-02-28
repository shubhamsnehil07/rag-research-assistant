"""
Streamlit Web UI for RAG Research Paper Assistant
Beautiful, interactive interface for asking questions about research papers
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import json


from ingestion import PDFProcessor
from core.rag_system_pro import ProductionRAGSystem
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    .confidence-medium {
        background-color: #FFC107;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    .confidence-low {
        background-color: #F44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Citation boxes */
    .citation-box {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_db_loaded' not in st.session_state:
    st.session_state.vector_db_loaded = False

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_rag_system():
    """Load the RAG system"""
    try:
        with st.spinner("��� Loading RAG system..."):
            rag = EnhancedRAGSystem(
                vector_db_path="vector_db",
                model_name="llama3.2",
                top_k=3,
                temperature=0.1
            )
        st.session_state.rag_system = rag
        st.session_state.vector_db_loaded = True
        return True
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        return False

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and rebuild vector database"""
    try:
        # Save uploaded file
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        pdf_path = data_dir / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.processing_status = "📄 Processing PDF..."
        
        # Process PDF
        processor = PDFProcessor(chunk_size=800, chunk_overlap=150)
        chunks = processor.process_pdf(str(pdf_path))
        
        st.session_state.processing_status = "🔢 Generating embeddings..."
        
        # Generate embeddings
        embedder = EmbeddingGenerator()
        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        
        st.session_state.processing_status = "💾 Building vector database..."
        
        # Build vector store
        vector_store = VectorStore(embedding_dimension=384)
        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save("vector_db")
        
        st.session_state.processing_status = "✅ Complete!"
        
        return True, len(chunks)
        
    except Exception as e:
        st.session_state.processing_status = f"❌ Error: {str(e)}"
        return False, 0

def format_confidence_badge(label, score):
    """Format confidence as HTML badge"""
    if label == "HIGH":
        css_class = "confidence-high"
        emoji = "🟢"
    elif label == "MEDIUM":
        css_class = "confidence-medium"
        emoji = "🟡"
    else:
        css_class = "confidence-low"
        emoji = "🔴"
    
    return f'<span class="{css_class}">{emoji} {label} ({score:.0%})</span>'

def export_chat_history():
    """Export chat history as JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    export_data = {
        "timestamp": timestamp,
        "conversation": st.session_state.chat_history
    }
    
    return json.dumps(export_data, indent=2), filename

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("📚 RAG Research Assistant")
    st.markdown("---")
    
    # System status
    st.subheader("🔧 System Status")
    
    if st.session_state.vector_db_loaded:
        st.success("✅ RAG System Ready")
        
        # Get stats
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.vector_store.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", stats['total_chunks'])
            with col2:
                st.metric("Sources", stats['unique_sources'])
    else:
        st.warning("⚠️ No database loaded")
        if st.button("🔄 Load Existing Database"):
            if os.path.exists("vector_db/faiss.index"):
                load_rag_system()
                st.rerun()
            else:
                st.error("No vector database found. Please upload a PDF first.")
    
    st.markdown("---")
    
    # PDF Upload
    st.subheader("📤 Upload Research Paper")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a research paper to analyze"
    )
    
    if uploaded_file:
        if st.button("🚀 Process PDF", type="primary"):
            with st.spinner("Processing..."):
                success, num_chunks = process_uploaded_pdf(uploaded_file)
                
                if success:
                    st.success(f"✅ Processed {num_chunks} chunks!")
                    # Reload RAG system
                    load_rag_system()
                    st.rerun()
                else:
                    st.error("Failed to process PDF")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    
    show_sources = st.checkbox("Show Sources", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    
    st.markdown("---")
    
    # Chat actions
    st.subheader("💬 Chat Actions")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("💾 Export Conversation"):
            export_data, filename = export_chat_history()
            st.download_button(
                label="📥 Download JSON",
                data=export_data,
                file_name=filename,
                mime="application/json"
            )
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **RAG Research Assistant**
        
        A Retrieval-Augmented Generation system for analyzing research papers.
        
        **Features:**
        - PDF upload and processing
        - Semantic search
        - AI-powered Q&A
        - Citation tracking
        - Confidence scoring
        
        **Tech Stack:**
        - LangChain
        - FAISS
        - Ollama (Llama 3.2)
        - Streamlit
        """)

# ==============================================================================
# MAIN CONTENT
# ==============================================================================

# Header
st.title("🤖 Research Paper Q&A Assistant")
st.markdown("Ask questions about your research papers and get AI-powered answers with citations.")

# Check if system is ready
if not st.session_state.vector_db_loaded:
    st.info("👈 Please load an existing database or upload a PDF to get started!")
    
    # Show quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Option 1: Load Existing Database**
        1. Click "Load Existing Database" in sidebar
        2. Start asking questions!
        """)
    
    with col2:
        st.markdown("""
        **Option 2: Upload New Paper**
        1. Upload a PDF in the sidebar
        2. Click "Process PDF"
        3. Wait for processing
        4. Start asking questions!
        """)
    
    st.stop()

# Display chat history
st.markdown("### 💬 Conversation")

for entry in st.session_state.chat_history:
    # User question
    with st.container():
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong><br>
            {entry['question']}
        </div>
        """, unsafe_allow_html=True)
    
    # Assistant answer
    with st.container():
        answer_html = f"""
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {entry['answer']}
        """
        
        # Add confidence badge
        if show_confidence and 'confidence_label' in entry:
            confidence_badge = format_confidence_badge(
                entry['confidence_label'],
                entry['confidence_score']
            )
            answer_html += f"<br><br>{confidence_badge}"
        
        answer_html += "</div>"
        st.markdown(answer_html, unsafe_allow_html=True)
    
    # Show sources
    if show_sources and 'sources' in entry:
        with st.expander(f"📚 View {len(entry['sources'])} Source(s)"):
            for i, source in enumerate(entry['sources'], 1):
                st.markdown(f"""
                <div class="citation-box">
                    <strong>[{i}] {source['source']} - Page {source['page']}</strong><br>
                    <em>Relevance: {source['relevance_score']:.1f}%</em><br>
                    <small>{source['text'][:200]}...</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")

# Question input
st.markdown("### ❓ Ask a Question")

col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What dataset was used in this study?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# Process question
if ask_button and question:
    if st.session_state.rag_system:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response
                response = st.session_state.rag_system.ask(
                    question,
                    include_citations=show_sources,
                    include_confidence=show_confidence
                )
                
                # Add to history
                st.session_state.chat_history.append(response)
                
                # Rerun to show new message
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.error("❌ RAG system not loaded")

# Sample questions
if not st.session_state.chat_history:
    st.markdown("### 💡 Sample Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 What datasets were used?"):
            st.session_state.chat_history.append({
                'question': "What datasets were used?",
                'answer': "Processing...",
            })
            st.rerun()
    
    with col2:
        if st.button("🎯 What is the main contribution?"):
            st.session_state.chat_history.append({
                'question': "What is the main contribution?",
                'answer': "Processing...",
            })
            st.rerun()
    
    with col3:
        if st.button("📈 What results were achieved?"):
            st.session_state.chat_history.append({
                'question': "What results were achieved?",
                'answer': "Processing...",
            })
            st.rerun()

# Paper Summary Section
if st.session_state.vector_db_loaded and not st.session_state.chat_history:
    st.markdown("### 📄 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Summarize Paper", type="secondary", use_container_width=True):
            with st.spinner("📄 Generating summary..."):
                try:
                    response = st.session_state.rag_system.summarize_paper()
                    st.session_state.chat_history.append(response)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("📊 What datasets?", use_container_width=True):
            question = "What datasets were used in this study?"
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.ask(question)
                st.session_state.chat_history.append(response)
                st.rerun()
    
    with col3:
        if st.button("🎯 Main contribution?", use_container_width=True):
            question = "What is the main contribution of this research?"
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.ask(question)
                st.session_state.chat_history.append(response)
                st.rerun()
    
    with col4:
        if st.button("📈 Key results?", use_container_width=True):
            question = "What are the key results and findings?"
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.ask(question)
                st.session_state.chat_history.append(response)
                st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Built with ❤️ using Streamlit, LangChain, FAISS, and Ollama</small>
</div>
""", unsafe_allow_html=True)