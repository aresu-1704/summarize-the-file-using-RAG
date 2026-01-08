"""
FAISS Vector Store cho RAG application.

Module này cung cấp các tính năng:
- Lưu trữ embeddings trong FAISS index (hiệu quả cho similarity search)
- Tìm kiếm similar documents bằng cosine similarity hoặc L2 distance
- Lưu/load index từ disk
- Quản lý metadata của documents (vì FAISS chỉ lưu vectors)
- Hỗ trợ nhiều loại FAISS index (Flat, IVF, HNSW)
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

# Import FAISS
import faiss

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store để lưu và tìm kiếm embeddings.
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100
    ):
        """
        Khởi tạo FAISS vector store.
        
        Args:
            dimension: Dimension của embedding vectors
            index_type: Loại FAISS index:
                - "Flat": Brute-force search, chính xác nhất (tốt cho < 1M vectors)
                - "IVF": Inverted File Index, nhanh hơn Flat (tốt cho > 1M vectors)
                - "HNSW": Hierarchical NSW, rất nhanh, độ chính xác cao
            metric: Metric để tính similarity:
                - "cosine": Cosine similarity (phổ biến cho text)
                - "l2": Euclidean distance (L2)
                - "ip": Inner product
            nlist: Số clusters cho IVF index (chỉ dùng khi index_type="IVF")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        
        # Tạo FAISS index
        self.index = self._create_index()
        
        # Lưu metadata của documents (FAISS chỉ lưu vectors)
        # Format: {doc_id: {"text": "...", "metadata": {...}}}
        self.doc_store: Dict[int, Dict[str, Any]] = {}
        
        # Mapping từ internal ID (FAISS index position) sang doc_id
        self.id_mapping: List[int] = []
        
        # Counter cho doc IDs
        self.next_doc_id = 0
        
        logger.info(f"Khởi tạo FAISS vector store: {index_type} index, metric={metric}, dimension={dimension}")
    
    def _create_index(self) -> faiss.Index:
        """
        Tạo FAISS index dựa trên cấu hình.
        
        Returns:
            FAISS index object
        """
        if self.index_type == "Flat":
            # Flat index: brute-force search
            if self.metric == "cosine":
                # Cosine similarity = IP với normalized vectors
                index = faiss.IndexFlatIP(self.dimension)
                logger.info("Sử dụng IndexFlatIP (cosine similarity)")
            elif self.metric == "l2":
                index = faiss.IndexFlatL2(self.dimension)
                logger.info("Sử dụng IndexFlatL2 (L2 distance)")
            elif self.metric == "ip":
                index = faiss.IndexFlatIP(self.dimension)
                logger.info("Sử dụng IndexFlatIP (inner product)")
            else:
                raise ValueError(f"Metric không hợp lệ: {self.metric}")
        
        elif self.index_type == "IVF":
            # IVF index: faster search với trade-off về độ chính xác
            # Tạo quantizer (index để cluster)
            if self.metric == "cosine" or self.metric == "ip":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                logger.info(f"Sử dụng IndexIVFFlat (IP), nlist={self.nlist}")
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
                logger.info(f"Sử dụng IndexIVFFlat (L2), nlist={self.nlist}")
        
        elif self.index_type == "HNSW":
            # HNSW index: very fast, high accuracy
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(self.dimension, M)
            
            if self.metric == "cosine" or self.metric == "ip":
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                index.metric_type = faiss.METRIC_L2
            
            logger.info(f"Sử dụng IndexHNSWFlat, M={M}")
        
        else:
            raise ValueError(f"Index type không hợp lệ: {self.index_type}")
        
        return index
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embedding_field: str = "embedding"
    ) -> List[int]:
        """
        Thêm documents với embeddings vào vector store.
        
        Args:
            documents: List documents với format:
                {
                    "text": "nội dung...",
                    "metadata": {...},
                    "embedding": [0.1, 0.2, ...]  # numpy array
                }
            embedding_field: Tên field chứa embedding vector
        
        Returns:
            List doc IDs đã được thêm
        """
        if not documents:
            logger.warning("Không có documents để thêm")
            return []
        
        # Trích xuất embeddings và metadata
        embeddings = []
        doc_ids = []
        
        for doc in documents:
            if embedding_field not in doc:
                raise ValueError(f"Document không có field '{embedding_field}': {doc}")
            
            embedding = doc[embedding_field]
            
            # Convert sang numpy array nếu chưa phải
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Check dimension
            if embedding.shape[0] != self.dimension:
                raise ValueError(f"Embedding dimension không khớp: {embedding.shape[0]} != {self.dimension}")
            
            embeddings.append(embedding)
            
            # Lưu metadata
            doc_id = self.next_doc_id
            self.doc_store[doc_id] = {
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {})
            }
            doc_ids.append(doc_id)
            self.id_mapping.append(doc_id)
            self.next_doc_id += 1
        
        # Convert sang numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize nếu dùng cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings_array)
        
        # Train index nếu cần (IVF index cần train)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info(f"Training IVF index với {len(embeddings_array)} vectors...")
            self.index.train(embeddings_array)
            logger.info("Training hoàn thành")
        
        # Thêm vào FAISS index
        self.index.add(embeddings_array)
        
        logger.info(f"Đã thêm {len(documents)} documents vào vector store (total: {self.index.ntotal})")
        return doc_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        nprobe: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm top-k documents tương tự nhất với query.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Số lượng kết quả trả về
            nprobe: Số clusters để search (chỉ cho IVF index)
        
        Returns:
            List documents với scores:
                {
                    "doc_id": 123,
                    "text": "nội dung...",
                    "metadata": {...},
                    "score": 0.95  # similarity score
                }
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store rỗng, không có kết quả")
            return []
        
        # Convert sang numpy array nếu chưa phải
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Reshape thành (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize nếu dùng cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Set nprobe cho IVF index
        if self.index_type == "IVF":
            self.index.nprobe = nprobe
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Tạo kết quả
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Bỏ qua kết quả không hợp lệ (-1 là không tìm thấy)
            if idx == -1:
                continue
            
            # Lấy doc_id từ mapping
            doc_id = self.id_mapping[idx]
            
            # Lấy document từ store
            if doc_id not in self.doc_store:
                logger.warning(f"Không tìm thấy doc_id {doc_id} trong doc_store")
                continue
            
            doc = self.doc_store[doc_id]
            
            # Convert distance thành similarity score
            # - Cosine/IP: distance đã là similarity (cao hơn = tốt hơn)
            # - L2: cần convert (thấp hơn = tốt hơn)
            if self.metric == "l2":
                score = 1.0 / (1.0 + float(distance))  # Convert distance sang similarity
            else:
                score = float(distance)
            
            results.append({
                "doc_id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": score,
                "rank": i + 1
            })
        
        logger.info(f"Tìm thấy {len(results)} kết quả")
        return results
    
    def save(self, save_dir: str):
        """
        Lưu vector store vào disk.
        
        Args:
            save_dir: Thư mục lưu index và metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Lưu FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        logger.info(f"Đã lưu FAISS index: {index_path}")
        
        # Lưu metadata
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        metadata = {
            "doc_store": self.doc_store,
            "id_mapping": self.id_mapping,
            "next_doc_id": self.next_doc_id,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Đã lưu metadata: {metadata_path}")
        
        # Lưu config dễ đọc
        config_path = os.path.join(save_dir, "config.json")
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "num_documents": self.index.ntotal
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu config: {config_path}")
    
    @classmethod
    def load(cls, load_dir: str) -> 'FAISSVectorStore':
        """
        Load vector store từ disk.
        
        Args:
            load_dir: Thư mục chứa index và metadata
        
        Returns:
            FAISSVectorStore object
        """
        # Load metadata
        metadata_path = os.path.join(load_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Tạo instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            nlist=metadata["nlist"]
        )
        
        # Load FAISS index
        index_path = os.path.join(load_dir, "faiss_index.bin")
        store.index = faiss.read_index(index_path)
        
        # Restore metadata
        store.doc_store = metadata["doc_store"]
        store.id_mapping = metadata["id_mapping"]
        store.next_doc_id = metadata["next_doc_id"]
        
        logger.info(f"Đã load vector store từ {load_dir} ({store.index.ntotal} documents)")
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về vector store.
        
        Returns:
            Dictionary chứa stats
        """
        return {
            "num_documents": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "doc_store_size": len(self.doc_store)
        }


# Hàm tiện ích
def create_vector_store(
    documents: List[Dict[str, Any]],
    dimension: Optional[int] = None,
    index_type: str = "Flat",
    metric: str = "cosine",
    embedding_field: str = "embedding"
) -> FAISSVectorStore:
    """
    Tạo vector store và thêm documents vào.
    
    Args:
        documents: List documents với embeddings
        dimension: Dimension của embeddings (None = auto-detect từ document đầu tiên)
        index_type: Loại FAISS index
        metric: Similarity metric
        embedding_field: Tên field chứa embedding
    
    Returns:
        FAISSVectorStore instance
    """
    if not documents:
        raise ValueError("Không có documents để tạo vector store")
    
    # Auto-detect dimension
    if dimension is None:
        first_embedding = documents[0][embedding_field]
        if isinstance(first_embedding, np.ndarray):
            dimension = first_embedding.shape[0]
        else:
            dimension = len(first_embedding)
        logger.info(f"Auto-detected dimension: {dimension}")
    
    # Tạo store
    store = FAISSVectorStore(
        dimension=dimension,
        index_type=index_type,
        metric=metric
    )
    
    # Thêm documents
    store.add_documents(documents, embedding_field=embedding_field)
    
    return store
