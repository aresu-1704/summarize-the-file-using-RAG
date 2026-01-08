# Text splitter implementation
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Chia 1 text thành nhiều chunks với overlap.
    
    Args:
        text: Text cần chia thành chunks
        chunk_size: Kích thước của mỗi chunk
        chunk_overlap: Overlap giữa các chunks
    
    Returns:
        List các chunks
    """
    if chunk_size < chunk_overlap:
        raise ValueError("Chunk size must be greater than chunk overlap")
    
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - chunk_overlap
    
    return chunks

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chia các document thành nhiều chunks với overlap.
    
    Args:
        documents: List các document cần chia thành chunks
        chunk_size: Kích thước của mỗi chunk
        chunk_overlap: Overlap giữa các chunks
    
    Returns:
        List các chunks
    """
    chunked_documents = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "text": chunk,
                "metadata": {
                    "page": doc["metadata"]["page"],
                    "source": doc["metadata"]["source"],
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "processing_method": doc["metadata"]["processing_method"],
                    "total_pages": doc["metadata"]["total_pages"]
                }
            })
    
    return chunked_documents

# Hàm tiện ích tương thích ngược
def splitter(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chia các document thành nhiều chunks với overlap.
    
    Args:
        documents: List các document cần chia thành chunks
        chunk_size: Kích thước của mỗi chunk
        chunk_overlap: Overlap giữa các chunks
    
    Returns:
        List các chunks
    """
    return chunk_documents(documents, chunk_size, chunk_overlap)