"""
Summarizer module sử dụng Google Gemini API.

Module này cung cấp các tính năng:
- Tóm tắt toàn bộ document (từ all chunks)
- Tóm tắt theo query cụ thể (từ top-k relevant chunks)
- Load API key từ .env file
- Hỗ trợ nhiều Gemini models (gemini-pro, gemini-1.5-flash, etc.)
- Configurable prompt templates
"""

import os
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Import Gemini SDK
import google.generativeai as genai

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables từ .env
load_dotenv()


class GeminiSummarizer:
    """
    Summarizer sử dụng Google Gemini API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 2048
    ):
        """
        Khởi tạo Gemini Summarizer.
        
        Args:
            api_key: Gemini API key (None = load từ GEMINI_API_KEY env var)
            model_name: Tên model Gemini:
                - gemini-1.5-flash: Nhanh, rẻ, tốt cho tóm tắt
                - gemini-1.5-pro: Chậm hơn nhưng chất lượng cao hơn
                - gemini-pro: Model cũ hơn
            temperature: Độ "sáng tạo" (0.0-1.0, thấp = deterministic hơn)
            max_output_tokens: Số tokens tối đa cho output
        """
        # Load API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY không được tìm thấy. "
                "Vui lòng set trong .env file hoặc truyền vào api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Khởi tạo model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        
        logger.info(f"Khởi tạo GeminiSummarizer với model: {model_name}")
    
    def summarize_full_document(
        self,
        document_text: str,
        custom_prompt: Optional[str] = None,
        language: str = "Vietnamese"
    ) -> str:
        """
        Tóm tắt TOÀN BỘ document.
        Dùng cho Flow 1: Tóm tắt toàn bộ văn bản.
        
        Args:
            document_text: Text của toàn bộ document (đã format từ retriever)
            custom_prompt: Custom prompt (None = dùng default)
            language: Ngôn ngữ output (Vietnamese/English)
        
        Returns:
            Tóm tắt của document
        """
        logger.info(f"Tóm tắt toàn bộ document ({len(document_text)} chars)...")
        
        # Tạo prompt
        if custom_prompt:
            prompt = custom_prompt.format(document=document_text)
        else:
            prompt = self._get_full_summary_prompt(document_text, language)
        
        # Generate summary
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            logger.info(f"Tạo tóm tắt thành công ({len(summary)} chars)")
            return summary
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo tóm tắt: {e}")
            raise
    
    def summarize_by_query(
        self,
        query: str,
        relevant_text: str,
        custom_prompt: Optional[str] = None,
        language: str = "Vietnamese"
    ) -> str:
        """
        Tóm tắt theo QUERY cụ thể.
        Dùng cho Flow 2: Tóm tắt theo truy vấn.
        
        Args:
            query: Câu hỏi/query từ user
            relevant_text: Text của top-k relevant chunks (đã format từ retriever)
            custom_prompt: Custom prompt (None = dùng default)
            language: Ngôn ngữ output
        
        Returns:
            Tóm tắt liên quan đến query
        """
        logger.info(f"Tóm tắt theo query: '{query[:50]}...' ({len(relevant_text)} chars)")
        
        # Tạo prompt
        if custom_prompt:
            prompt = custom_prompt.format(query=query, context=relevant_text)
        else:
            prompt = self._get_query_summary_prompt(query, relevant_text, language)
        
        # Generate summary
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            logger.info(f"Tạo tóm tắt thành công ({len(summary)} chars)")
            return summary
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo tóm tắt: {e}")
            raise
    
    def _get_full_summary_prompt(self, document_text: str, language: str) -> str:
        """
        Tạo prompt cho tóm tắt toàn bộ document.
        
        Args:
            document_text: Text của document
            language: Ngôn ngữ output
        
        Returns:
            Prompt string
        """
        if language.lower() == "vietnamese":
            prompt = f"""Bạn là một trợ lý AI chuyên nghiệp về tóm tắt văn bản.

Nhiệm vụ: Đọc kỹ văn bản dưới đây và tạo một bản tóm tắt toàn diện, chi tiết.

Yêu cầu:
- Tóm tắt phải bao quát TẤT CẢ các ý chính trong văn bản
- Giữ nguyên thông tin quan trọng (số liệu, tên riêng, thuật ngữ kỹ thuật)
- Cấu trúc rõ ràng với các mục/phần logic
- Độ dài: Khoảng 20-30% của văn bản gốc
- Ngôn ngữ: Tiếng Việt, dễ hiểu, mạch lạc

Văn bản cần tóm tắt:
{document_text}

Hãy tạo bản tóm tắt toàn diện:"""
        
        else:  # English
            prompt = f"""You are a professional AI assistant specialized in document summarization.

Task: Read the following document carefully and create a comprehensive, detailed summary.

Requirements:
- The summary must cover ALL main points in the document
- Preserve important information (numbers, proper nouns, technical terms)
- Clear structure with logical sections
- Length: Approximately 20-30% of original text
- Language: English, clear and coherent

Document to summarize:
{document_text}

Create a comprehensive summary:"""
        
        return prompt
    
    def _get_query_summary_prompt(self, query: str, relevant_text: str, language: str) -> str:
        """
        Tạo prompt cho tóm tắt theo query.
        
        Args:
            query: Query từ user
            relevant_text: Relevant chunks
            language: Ngôn ngữ output
        
        Returns:
            Prompt string
        """
        if language.lower() == "vietnamese":
            prompt = f"""Bạn là một trợ lý AI chuyên nghiệp về tóm tắt và trả lời câu hỏi dựa trên tài liệu.

Nhiệm vụ: Dựa vào context được cung cấp, hãy trả lời câu hỏi/truy vấn của người dùng một cách chi tiết và chính xác.

Câu hỏi/Truy vấn:
{query}

Context (từ tài liệu):
{relevant_text}

Yêu cầu:
- Trả lời dựa HOÀN TOÀN trên context được cung cấp
- Nếu context không có thông tin để trả lời, hãy nói rõ
- Trích dẫn thông tin cụ thể từ context
- Ngôn ngữ: Tiếng Việt, rõ ràng, dễ hiểu
- Cấu trúc câu trả lời logic, có tổ chức

Câu trả lời chi tiết:"""
        
        else:  # English
            prompt = f"""You are a professional AI assistant specialized in answering questions based on documents.

Task: Based on the provided context, answer the user's question/query in detail and accurately.

Question/Query:
{query}

Context (from document):
{relevant_text}

Requirements:
- Answer ENTIRELY based on the provided context
- If context doesn't contain information to answer, state it clearly
- Cite specific information from context
- Language: English, clear and understandable
- Logical, organized answer structure

Detailed answer:"""
        
        return prompt
    
    def chat(
        self,
        message: str,
        context: Optional[str] = None
    ) -> str:
        """
        Chat tự do với Gemini (có hoặc không có context).
        
        Args:
            message: Message từ user
            context: Context bổ sung (optional)
        
        Returns:
            Response từ Gemini
        """
        if context:
            prompt = f"""Context:\n{context}\n\nUser: {message}\n\nAssistant:"""
        else:
            prompt = message
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi chat: {e}")
            raise


# Hàm tiện ích
def create_summarizer(
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.3
) -> GeminiSummarizer:
    """
    Tạo GeminiSummarizer instance.
    
    Args:
        api_key: Gemini API key (None = load từ env)
        model_name: Tên model Gemini
        temperature: Temperature cho generation
    
    Returns:
        GeminiSummarizer instance
    """
    return GeminiSummarizer(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )
