"""
Retriever module cho RAG application.

Module này cung cấp các tính năng:
- Retrieve all chunks từ vector store (cho tóm tắt toàn bộ)
- Retrieve top-k relevant chunks dựa trên query (cho tóm tắt theo truy vấn)
- Format results cho LLM consumption
- Merge chunks liền kề để tăng context
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

from ..vectorstore.faiss_store import FAISSVectorStore
from ..embeddings.embedder import Embedder

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever để lấy documents từ vector store cho LLM.
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Embedder,
        add_neighboring_chunks: bool = False,
        neighbors_before: int = 1,
        neighbors_after: int = 1
    ):
        """
        Khởi tạo Retriever.
        
        Args:
            vector_store: FAISS vector store instance
            embedder: Embedder instance (để embed queries)
            add_neighboring_chunks: Có thêm chunks lân cận vào kết quả không
            neighbors_before: Số chunks trước đó để thêm vào
            neighbors_after: Số chunks sau đó để thêm vào
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.add_neighboring_chunks = add_neighboring_chunks
        self.neighbors_before = neighbors_before
        self.neighbors_after = neighbors_after
        
        logger.info(f"Khởi tạo Retriever với vector store ({vector_store.get_stats()['num_documents']} docs)")
    
    def retrieve_all(
        self,
        format_for_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Lấy TẤT CẢ chunks từ vector store.
        Dùng cho Flow 1: Tóm tắt toàn bộ tài liệu.
        
        Args:
            format_for_llm: Format kết quả cho LLM (concatenate text)
        
        Returns:
            List tất cả documents hoặc formatted text
        """
        logger.info("Retrieving all chunks từ vector store...")
        
        # Lấy tất cả documents từ doc_store
        all_docs = []
        for doc_id, doc_data in self.vector_store.doc_store.items():
            all_docs.append({
                "doc_id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"]
            })
        
        logger.info(f"Retrieved {len(all_docs)} chunks")
        
        if format_for_llm:
            return self._format_for_llm(all_docs)
        
        return all_docs
    
    def retrieve_top_k(
        self,
        query: str,
        top_k: int = 5,
        format_for_llm: bool = True
    ) -> List[Dict[str, Any]] | str:
        """
        Retrieve top-k most relevant chunks based on query.
        Dùng cho Flow 2: Tóm tắt/trả lời theo truy vấn.
        
        Args:
            query: Query từ user
            top_k: Số chunks muốn lấy
            format_for_llm: Nếu True, return formatted text; nếu False, return raw chunks với metadata
        
        Returns:
            Nếu format_for_llm=True: str (formatted text)
            Nếu format_for_llm=False: List[Dict] (raw chunks với metadata)
        """
        logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:50]}...'")
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search trong vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        logger.info(f"Retrieved {len(results)} chunks")
        
        # Add neighboring chunks nếu cần
        if self.add_neighboring_chunks and results:
            results = self._add_neighboring_chunks(results) # Removed num_neighbors as it's not in original signature
            logger.info(f"After adding neighbors: {len(results)} chunks")
        
        # Sort by original order
        results = sorted(results, key=lambda x: x.get('metadata', {}).get('chunk', 0))
        
        if format_for_llm:
            # Format for LLM
            return self._format_for_llm(results)
        else:
            # Return raw chunks
            return results
    
    def _add_neighboring_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Thêm các chunks lân cận vào kết quả để tăng context.
        
        Args:
            chunks: List chunks đã retrieve
        
        Returns:
            List chunks với neighbors đã thêm
        """
        # Group chunks by source document
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk["metadata"].get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        expanded_chunks = []
        seen_doc_ids = set()
        
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            
            if doc_id in seen_doc_ids:
                continue
            
            # Thêm chunk hiện tại
            expanded_chunks.append(chunk)
            seen_doc_ids.add(doc_id)
            
            # Lấy chunk index từ metadata
            chunk_idx = chunk["metadata"].get("chunk", None)
            total_chunks = chunk["metadata"].get("total_chunks", None)
            
            if chunk_idx is None or total_chunks is None:
                continue
            
            # Thêm neighbors (nếu có trong doc_store)
            for offset in range(-self.neighbors_before, self.neighbors_after + 1):
                if offset == 0:  # Skip chunk hiện tại
                    continue
                
                neighbor_idx = chunk_idx + offset
                if neighbor_idx < 1 or neighbor_idx > total_chunks:
                    continue
                
                # Tìm neighbor chunk trong doc_store
                # (Giả định doc_id tăng dần theo chunk order)
                neighbor_doc_id = doc_id + offset
                
                if neighbor_doc_id in self.vector_store.doc_store:
                    neighbor_data = self.vector_store.doc_store[neighbor_doc_id]
                    
                    if neighbor_doc_id not in seen_doc_ids:
                        expanded_chunks.append({
                            "doc_id": neighbor_doc_id,
                            "text": neighbor_data["text"],
                            "metadata": neighbor_data["metadata"],
                            "is_neighbor": True
                        })
                        seen_doc_ids.add(neighbor_doc_id)
        
        logger.info(f"Expanded from {len(chunks)} to {len(expanded_chunks)} chunks (with neighbors)")
        return expanded_chunks
    
    def _format_for_llm(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Format chunks thành text duy nhất để đưa cho LLM.
        
        Args:
            chunks: List chunks
        
        Returns:
            Formatted text string
        """
        # Sắp xếp chunks theo thứ tự (nếu có metadata chunk index)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: (
                x["metadata"].get("source", ""),
                x["metadata"].get("chunk", 0)
            )
        )
        
        # Concatenate text
        formatted_parts = []
        current_source = None
        
        for i, chunk in enumerate(sorted_chunks):
            source = chunk["metadata"].get("source", "unknown")
            
            # Thêm separator giữa các sources
            if source != current_source:
                if current_source is not None:
                    formatted_parts.append("\n" + "=" * 80 + "\n")
                current_source = source
            
            # Thêm chunk text
            chunk_text = chunk["text"].strip()
            if chunk_text:
                formatted_parts.append(chunk_text)
        
        formatted_text = "\n\n".join(formatted_parts)
        
        logger.info(f"Formatted {len(chunks)} chunks → {len(formatted_text)} chars for LLM")
        return formatted_text
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 3000,
        top_k: int = 10
    ) -> str:
        """
        Lấy context tối ưu cho query trong giới hạn max_tokens.
        
        Args:
            query: Query string
            max_tokens: Số tokens tối đa (ước lượng: ~4 chars = 1 token)
            top_k: Số chunks ban đầu để retrieve
        
        Returns:
            Context text đã format
        """
        # Retrieve top-k chunks
        chunks = self.retrieve_top_k(
            query=query,
            top_k=top_k,
            format_for_llm=False
        )
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        
        # Thêm chunks cho đến khi đạt max_chars
        selected_chunks = []
        total_chars = 0
        
        for chunk in chunks:
            chunk_chars = len(chunk["text"])
            
            if total_chars + chunk_chars > max_chars:
                break
            
            selected_chunks.append(chunk)
            total_chars += chunk_chars
        
        logger.info(f"Selected {len(selected_chunks)}/{len(chunks)} chunks (≈{total_chars//4} tokens)")
        
        return self._format_for_llm(selected_chunks)


# Hàm tiện ích
def create_retriever(
    vector_store: FAISSVectorStore,
    embedder: Embedder,
    add_neighboring_chunks: bool = False
) -> Retriever:
    """
    Tạo Retriever instance.
    
    Args:
        vector_store: FAISS vector store
        embedder: Embedder instance
        add_neighboring_chunks: Thêm chunks lân cận
    
    Returns:
        Retriever instance
    """
    return Retriever(
        vector_store=vector_store,
        embedder=embedder,
        add_neighboring_chunks=add_neighboring_chunks
    )
