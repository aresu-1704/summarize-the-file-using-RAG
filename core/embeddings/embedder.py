"""
Embedder module cho RAG application.

Module này cung cấp các tính năng:
- Tạo embeddings từ text chunks sử dụng sentence-transformers
- Hỗ trợ nhiều models (multilingual, Vietnamese-optimized, English)
- Batch processing để tối ưu performance
- Cache embeddings để tránh tính toán lặp lại
- Normalize embeddings cho cosine similarity
"""

import os
import hashlib
import pickle
from typing import List, Dict, Any, Optional
import logging
import numpy as np

# Import sentence-transformers
from sentence_transformers import SentenceTransformer

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder class để tạo vector embeddings từ text chunks.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Khởi tạo Embedder.
        
        Args:
            model_name: Tên model từ sentence-transformers
                - paraphrase-multilingual-mpnet-base-v2: Multilingual, tốt cho nhiều ngôn ngữ
                - keepitreal/vietnamese-sbert: Tối ưu cho tiếng Việt
                - all-MiniLM-L6-v2: Nhẹ, nhanh, tốt cho tiếng Anh
            device: Device để chạy model ('cuda', 'cpu', None = auto-detect)
            batch_size: Số lượng texts xử lý cùng lúc
            normalize_embeddings: Normalize vectors về unit length (tốt cho cosine similarity)
            enable_cache: Bật/tắt cache embeddings
            cache_dir: Thư mục lưu cache (None = không cache)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
        # Load model
        logger.info(f"Đang load embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded thành công trên device: {self.model.device}")
        
        # Tạo cache directory nếu cần
        if self.enable_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")
    
    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Tạo embeddings cho danh sách documents/chunks.
        
        Args:
            documents: List các documents với format:
                {
                    "text": "nội dung...",
                    "metadata": {...}
                }
            text_field: Tên field chứa text cần embed
        
        Returns:
            List documents với embeddings đã thêm vào:
                {
                    "text": "nội dung...",
                    "metadata": {...},
                    "embedding": [0.1, 0.2, ...]  # numpy array
                }
        """
        if not documents:
            logger.warning("Không có documents để embed")
            return []
        
        # Trích xuất texts
        texts = []
        for doc in documents:
            if text_field not in doc:
                raise ValueError(f"Document không có field '{text_field}': {doc}")
            texts.append(doc[text_field])
        
        logger.info(f"Đang tạo embeddings cho {len(texts)} documents...")
        
        # Tạo embeddings (batch processing)
        embeddings = self._embed_texts(texts)
        
        # Thêm embeddings vào documents
        embedded_documents = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc["embedding"] = embedding
            embedded_documents.append(embedded_doc)
        
        logger.info(f"Hoàn thành tạo {len(embedded_documents)} embeddings")
        return embedded_documents
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Tạo embedding cho query text.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector (numpy array)
        """
        if not query or not query.strip():
            raise ValueError("Query không được rỗng")
        
        logger.debug(f"Tạo embedding cho query: '{query[:50]}...'")
        embeddings = self._embed_texts([query])
        return embeddings[0]
    
    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Tạo embeddings cho danh sách texts với cache support.
        
        Args:
            texts: List các text strings
        
        Returns:
            List embeddings (numpy arrays)
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache nếu bật
        if self.enable_cache and self.cache_dir:
            for i, text in enumerate(texts):
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            if len(texts) - len(texts_to_embed) > 0:
                logger.info(f"Tìm thấy {len(texts) - len(texts_to_embed)} embeddings trong cache")
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
        
        # Tạo embeddings cho texts chưa có trong cache
        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                show_progress_bar=len(texts_to_embed) > 100,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Cache các embeddings mới
            if self.enable_cache and self.cache_dir:
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self._cache_embedding(text, embedding)
            
            # Merge embeddings từ cache và embeddings mới
            if self.enable_cache and self.cache_dir:
                # Tạo list embeddings đầy đủ theo đúng thứ tự
                final_embeddings = [None] * len(texts)
                
                # Đặt cached embeddings vào đúng vị trí
                cached_idx = 0
                for i in range(len(texts)):
                    if i not in text_indices:
                        final_embeddings[i] = embeddings[cached_idx]
                        cached_idx += 1
                
                # Đặt new embeddings vào đúng vị trí
                for idx, embedding in zip(text_indices, new_embeddings):
                    final_embeddings[idx] = embedding
                
                embeddings = final_embeddings
            else:
                embeddings = list(new_embeddings)
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """
        Tạo cache key từ text content.
        
        Args:
            text: Text content
        
        Returns:
            Cache key (hash)
        """
        # Sử dụng hash của text + model name làm cache key
        content = f"{self.model_name}|{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Lấy embedding từ cache nếu có.
        
        Args:
            text: Text content
        
        Returns:
            Cached embedding hoặc None nếu không có trong cache
        """
        if not self.cache_dir:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                return embedding
            except Exception as e:
                logger.warning(f"Không thể đọc cache {cache_key}: {e}")
                return None
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """
        Lưu embedding vào cache.
        
        Args:
            text: Text content
            embedding: Embedding vector
        """
        if not self.cache_dir:
            return
        
        cache_key = self._get_cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Không thể lưu cache {cache_key}: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Lấy dimension của embedding vectors.
        
        Returns:
            Dimension (int)
        """
        return self.model.get_sentence_embedding_dimension()


# Hàm tiện ích để tương thích ngược
def embed_documents(
    documents: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    enable_cache: bool = False,
    cache_dir: Optional[str] = None,
    text_field: str = "text"
) -> List[Dict[str, Any]]:
    """
    Tạo embeddings cho danh sách documents/chunks.
    
    Args:
        documents: List documents với format {"text": "...", "metadata": {...}}
        model_name: Tên embedding model
        device: Device để chạy model ('cuda', 'cpu', None)
        batch_size: Batch size cho encoding
        normalize_embeddings: Normalize vectors
        enable_cache: Bật cache
        cache_dir: Thư mục cache
        text_field: Tên field chứa text
    
    Returns:
        List documents với embeddings đã thêm vào
    """
    embedder = Embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        enable_cache=enable_cache,
        cache_dir=cache_dir
    )
    return embedder.embed_documents(documents, text_field=text_field)


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device: Optional[str] = None,
    normalize_embeddings: bool = True
) -> np.ndarray:
    """
    Tạo embedding cho query text.
    
    Args:
        query: Query string
        model_name: Tên embedding model
        device: Device để chạy model
        normalize_embeddings: Normalize vector
    
    Returns:
        Embedding vector (numpy array)
    """
    embedder = Embedder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        enable_cache=False  # Không cần cache cho query
    )
    return embedder.embed_query(query)
