"""
RAG Document Summarizer - Streamlit Application

Ứng dụng tóm tắt văn bản thông minh sử dụng RAG (Retrieval-Augmented Generation).

Tính năng:
- Upload file (TXT, DOCX, PDF)
- 2 chế độ tóm tắt:
  1. Tóm tắt toàn bộ văn bản
  2. Tóm tắt theo truy vấn cụ thể
- Giao diện ChatGPT-style
- Quản lý files trong sidebar
- Tự động xóa files khi thoát
"""

import os
import shutil
import streamlit as st
import os
from pathlib import Path
import logging
from typing import List

# Import các modules đã implement
from core.loaders.txt_loader import TXTLoader
from core.loaders.docx_loader import DOCXLoader
from core.loaders.pdf_loader import PDFLoader
from core.chunking.splitter import chunk_documents
from core.embeddings.embedder import Embedder
from core.vectorstore.faiss_store import FAISSVectorStore
from core.retriever.retriever import Retriever
from core.llm.summarizer import GeminiSummarizer

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình trang
st.set_page_config(
    page_title="RAG Document Summarizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Sidebar 30%
st.markdown("""
<style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        width: 30% !important;
        min-width: 350px;
        max-width: 500px;
    }
    
    /* Main content */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Hide detailed processing logs */
    .stStatus {
        display: none;
    }
    
    /* Main container */
    .main {
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202123;
    }
    
    /* File item styling */
    .file-item {
        background-color: white;
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Summary box */
    .summary-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Chat message */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .user-message {
        background-color: #f0f0f0;
    }
    
    .assistant-message {
        background-color: #e8f4f8;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Directories
UPLOAD_DIR = Path("storage/uploads")
VECTOR_DB_DIR = Path("storage/vectordb")
CACHE_DIR = Path("storage/cache")

# Tạo directories nếu chưa có
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def initialize_session_state():
    """Khởi tạo session state."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None
    
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_document(file_path: str) -> List:
    """Load document dựa vào extension."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.txt':
        loader = TXTLoader()
        return loader.load_txt(file_path)
    
    elif ext == '.docx':
        loader = DOCXLoader()
        return loader.load_docx(file_path)
    
    elif ext == '.pdf':
        # TODO: Implement PDF loader
        st.error("PDF loader chưa được implement")
        return []
    
    else:
        st.error(f"File type không được hỗ trợ: {ext}")
        return []


def process_document(file_path: str):
    """Xử lý document: Load → Chunk → Embed → Store."""
    
    try:
        with st.spinner("🔄 Đang xử lý document..."):
            # 1. Load document
            with st.status("📖 Loading document...", expanded=True) as status:
                st.write("Đọc file...")
                documents = load_document(file_path)
                
                if not documents:
                    st.error("Không thể load document")
                    return False
                
                st.write(f"✓ Đã load {len(documents)} document(s)")
                
                # 2. Chunking
                st.write("✂️ Chia thành chunks...")
                chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
                st.write(f"✓ Đã tạo {len(chunks)} chunks")
                
                # 3. Initialize embedder nếu chưa có
                if st.session_state.embedder is None:
                    st.write("🔧 Khởi tạo embedding model...")
                    st.session_state.embedder = Embedder(
                        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        batch_size=16,
                        normalize_embeddings=True,
                        enable_cache=True,
                        cache_dir=str(CACHE_DIR)
                    )
                    st.write("✓ Embedding model ready")
                
                # 4. Create embeddings
                st.write("🧮 Tạo embeddings...")
                embedded_chunks = st.session_state.embedder.embed_documents(chunks)
                st.write(f"✓ Đã tạo {len(embedded_chunks)} embeddings")
                
                # 5. Create/update vector store
                st.write("💾 Lưu vào vector database...")
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = FAISSVectorStore(
                        dimension=st.session_state.embedder.get_embedding_dimension(),
                        index_type="Flat",
                        metric="cosine"
                    )
                
                st.session_state.vector_store.add_documents(embedded_chunks)
                st.write(f"✓ Đã lưu vào vector store (total: {st.session_state.vector_store.index.ntotal} docs)")
                
                # 6. Initialize retriever
                st.write("🔍 Khởi tạo retriever...")
                st.session_state.retriever = Retriever(
                    vector_store=st.session_state.vector_store,
                    embedder=st.session_state.embedder,
                    add_neighboring_chunks=False
                )
                st.write("✓ Retriever ready")
                
                status.update(label="✅ Document đã được xử lý thành công!", state="complete")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý document: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def process_document_simple(file_path: str) -> bool:
    """Xử lý document with simple UI (no detailed logs)."""
    try:
        # 1. Load
        documents = load_document(file_path)
        if not documents:
            return False
        
        # 2. Chunk
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        
        # 3. Embedder
        if st.session_state.embedder is None:
            st.session_state.embedder = Embedder(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                batch_size=16,
                normalize_embeddings=True,
                enable_cache=True,
                cache_dir=str(CACHE_DIR)
            )
        
        # 4. Embed
        embedded_chunks = st.session_state.embedder.embed_documents(chunks)
        
        # 5. Vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISSVectorStore(
                dimension=st.session_state.embedder.get_embedding_dimension(),
                index_type="Flat",
                metric="cosine"
            )
        
        st.session_state.vector_store.add_documents(embedded_chunks)
        
        # 6. Retriever
        st.session_state.retriever = Retriever(
            vector_store=st.session_state.vector_store,
            embedder=st.session_state.embedder,
            add_neighboring_chunks=False
        )
        
        # Delete uploaded file to save space
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return False


def sidebar():
    """Render sidebar với file management."""
    with st.sidebar:
        st.title("📁 Quản lý tài liệu")
        
        # Check if already has file or document is ready
        has_file = len(st.session_state.uploaded_files) > 0
        is_processed = st.session_state.retriever is not None
        
        if is_processed and not has_file:
            # Document is ready (file was deleted after processing)
            st.success("✅ **Tài liệu đã sẵn sàng**")
            st.caption("File đã được xử lý và xóa")
            
            st.divider()
            
            # Delete button to clear everything
            if st.button("🗑️ Xóa và tải file mới", type="secondary", use_container_width=True):
                # Reset state
                st.session_state.uploaded_files = []
                st.session_state.current_file = None
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.chat_history = []
                
                st.rerun()
        
        elif has_file:
            # Show current file
            file_info = st.session_state.uploaded_files[0]
            st.success(f"📄 **{file_info['name']}**")
            st.caption(f"Kích thước: {file_info['size'] / 1024:.1f} KB")
            
            # Show status
            is_processed = st.session_state.retriever is not None
            if is_processed:
                st.success("✅ Đã xử lý xong - sẵn sàng sử dụng")
            else:
                st.info("⏳ Đang xử lý...")
            
            st.divider()
            
            # Delete button
            if st.button("🗑️ Xóa và tải file khác", type="secondary", use_container_width=True):
                # Delete file
                if os.path.exists(file_info['path']):
                    os.remove(file_info['path'])
                
                # Reset state
                st.session_state.uploaded_files = []
                st.session_state.current_file = None
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.chat_history = []
                
                st.rerun()
        
        else:
            # Upload new file
            st.subheader("Tải lên tài liệu")
            uploaded_file = st.file_uploader(
                "Chọn file (TXT, DOCX, PDF)",
                type=['txt', 'docx', 'pdf'],
                help="Chỉ được tải 1 file tại một thời điểm",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Save file
                file_path = UPLOAD_DIR / uploaded_file.name
                
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to session state
                st.session_state.uploaded_files.append({
                    'name': uploaded_file.name,
                    'path': str(file_path),
                    'size': uploaded_file.size
                })
                
                st.session_state.current_file = str(file_path)
                
                # Auto process with simple status
                with st.spinner("⏳ Đang xử lý tài liệu..."):
                    success = process_document_simple(str(file_path))
                
                if success:
                    # Clear the uploaded file info since file is deleted
                    st.session_state.uploaded_files = []
                    st.rerun()
                else:
                    st.error("❌ Lỗi khi xử lý tài liệu")
        
        st.divider()


def main_area():
    """Render main content area."""
    st.title("🤖 RAG Document Summarizer")
    st.caption("Tóm tắt văn bản thông minh với AI - Hỏi bất cứ điều gì!")
    
    # Check if document is ready
    if st.session_state.current_file is None or st.session_state.retriever is None:
        st.info("👈 Vui lòng upload và xử lý document ở sidebar để bắt đầu")
        
        # Hướng dẫn sử dụng
        with st.expander("📖 Hướng dẫn sử dụng", expanded=True):
            st.markdown("""
            **Bạn có thể hỏi:**
            - 🌍 "Tóm tắt toàn bộ tài liệu"
            - 📑 "Tóm tắt phần phương pháp nghiên cứu"
            - ❓ "Độ chính xác của mô hình là bao nhiêu?"
            
            **Hệ thống tự động:**
            - Phát hiện intent (tóm tắt toàn bộ/một phần/trả lời câu hỏi)
            - Xử lý document theo chunks (không quá tải)
            - Hiển thị nguồn để bạn verify
            """)
        return
    
    # Initialize summarizer nếu chưa có
    if st.session_state.summarizer is None:
        try:
            with st.spinner("🔧 Khởi tạo Gemini AI..."):
                st.session_state.summarizer = GeminiSummarizer(
                    model_name="gemini-2.5-flash",
                    temperature=0.3
                )
        except Exception as e:
            st.error(f"❌ Lỗi khi khởi tạo Gemini: {e}")
            st.info("💡 Kiểm tra GEMINI_API_KEY trong file .env")
            return
    
    st.success(f"✅ Document đã sẵn sàng: **{Path(st.session_state.current_file).name}**")
    
    # Chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant"):
                # Show intent badge
                if 'intent' in msg:
                    intent_badges = {
                        'GLOBAL_SUMMARY': '🌍 Tóm tắt toàn bộ',
                        'PARTIAL_SUMMARY': '📑 Tóm tắt một phần',
                        'QUESTION_ANSWERING': '❓ Trả lời câu hỏi'
                    }
                    badge = intent_badges.get(msg['intent'], msg['intent'])
                    st.caption(f"*Intent: {badge}*")
                
                # Show summary/answer
                st.markdown(msg['content'])
                
                 # Show metadata if available
                if 'metadata' in msg and msg['metadata']:
                    with st.expander(f"📍 Nguồn tham khảo ({len(msg['metadata'])} đoạn)", expanded=False):
                        for idx, meta in enumerate(msg['metadata'][:10]):  # Limit to 10
                            st.markdown(f"""
                            **Đoạn {meta.get('chunk_id', idx+1)}** - *{meta.get('source', 'Unknown')}*
                            
                            > {meta.get('text_preview', 'N/A')}
                            """)
                            if idx < len(msg['metadata']) - 1:
                                st.divider()
    
    # Query input
    query = st.chat_input("Nhập câu hỏi hoặc yêu cầu của bạn...")
    
    if query:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query
        })
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang phân tích..."):
                try:
                    # Step 1: Classify intent
                    intent = st.session_state.summarizer.classify_intent(query)
                    
                    intent_badges = {
                        'GLOBAL_SUMMARY': '🌍 Tóm tắt toàn bộ',
                        'PARTIAL_SUMMARY': '📑 Tóm tắt một phần',
                        'QUESTION_ANSWERING': '❓ Trả lời câu hỏi'
                    }
                    st.caption(f"*Intent: {intent_badges.get(intent, intent)}*")
                    
                    # Step 2: Route based on intent
                    if intent == "GLOBAL_SUMMARY":
                        with st.spinner("📚 Đang tóm tắt toàn bộ document (xử lý từng chunk)..."):
                            # Get all chunks from vector store
                            all_chunks = st.session_state.retriever.vector_store.documents
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Progress callback
                            def update_progress(current, total, msg):
                                progress_bar.progress(int(current / total * 100))
                                status_text.text(msg)
                            
                            # Summarize chunk by chunk to avoid overload
                            result = st.session_state.summarizer.summarize_chunk_by_chunk(
                                chunks=all_chunks,
                                language="Vietnamese",
                                delay_seconds=0.5,  # 0.5s delay giữa các requests
                                progress_callback=update_progress
                            )
                            
                            progress_bar.progress(100)
                            status_text.text(f"✅ Đã xử lý {result['num_chunks']} chunks")
                            
                            answer = result['summary']
                            metadata = result['metadata']
                            
                            st.markdown(answer)
                            st.caption(f"*Đã xử lý {result['num_chunks']} chunks*")
                    
                    elif intent == "PARTIAL_SUMMARY":
                        with st.spinner("🔍 Đang tìm và tóm tắt phần liên quan..."):
                            # Retrieve top-k relevant chunks
                            relevant_results = st.session_state.retriever.retrieve_top_k(
                                query=query,
                                top_k=10,
                                format_for_llm=False  # Get raw chunks with metadata
                            )
                            
                            # Summarize relevant chunks
                            result = st.session_state.summarizer.summarize_in_chunks(
                                chunks=relevant_results,
                                language="Vietnamese",
                                chunks_per_batch=5
                            )
                            
                            answer = result['summary']
                            metadata = result['metadata']
                            
                            st.markdown(answer)
                    
                    else:  # QUESTION_ANSWERING
                        with st.spinner("💭 Đang tìm câu trả lời..."):
                            # Retrieve relevant chunks
                            relevant_text = st.session_state.retriever.retrieve_top_k(
                                query=query,
                                top_k=5,
                                format_for_llm=True
                            )
                            
                            # Get raw chunks for metadata
                            relevant_chunks = st.session_state.retriever.retrieve_top_k(
                                query=query,
                                top_k=5,
                                format_for_llm=False
                            )
                            
                            # Generate answer
                            answer = st.session_state.summarizer.summarize_by_query(
                                query=query,
                                relevant_text=relevant_text,
                                language="Vietnamese"
                            )
                            
                            metadata = [{
                                'chunk_id': c.get('metadata', {}).get('chunk', 'N/A'),
                                'source': c.get('metadata', {}).get('source', 'Unknown'),
                                'text_preview': c['text'][:200] + '...' if len(c['text']) > 200 else c['text']
                            } for c in relevant_chunks]
                            
                            st.markdown(answer)
                    
                    # Show metadata
                    if metadata:
                        with st.expander(f"📍 Nguồn ({len(metadata)} chunks)", expanded=False):
                            for idx, meta in enumerate(metadata[:10]):
                                st.markdown(f"""
                                **Chunk {meta.get('chunk_id', idx+1)}** - *{meta.get('source', 'Unknown')}*
                                
                                > {meta.get('text_preview', 'N/A')}
                                """)
                                if idx < len(metadata) - 1:
                                    st.divider()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'intent': intent,
                        'metadata': metadata
                    })
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    import traceback
                    st.error(traceback.format_exc())


def cleanup_on_exit():
    """Cleanup files khi app đóng (nếu cần)."""
    # Streamlit không có built-in cleanup hook
    # User có thể dùng button "Xóa tất cả files" trong sidebar
    pass


def main():
    """Main application."""
    initialize_session_state()
    sidebar()
    main_area()


if __name__ == "__main__":
    main()
