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
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv

# Import Gemini SDK (new package)
from google import genai
from google.genai import types

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
        model_name: str = "gemini-2.5-flash-exp",
        temperature: float = 0.3,
        max_output_tokens: int = 4096
    ):
        """
        Khởi tạo Gemini Summarizer.
        
        Args:
            api_key: Gemini API key (None = load từ GEMINI_API_KEY env var)
            model_name: Tên model Gemini:
                - gemini-2.0-flash-exp: Model mới nhất, nhanh (recommended)
                - gemini-1.5-flash-latest: Ổn định hơn
                - gemini-1.5-pro-latest: Chất lượng cao nhất
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
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Khởi tạo Gemini client (new API)
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"Khởi tạo GeminiSummarizer với model: {model_name}")
    
    def classify_intent(self, query: str) -> str:
        """
        Phân loại intent của user query.
        
        Args:
            query: Câu hỏi/truy vấn từ user
        
        Returns:
            Intent label: GLOBAL_SUMMARY, PARTIAL_SUMMARY, hoặc QUESTION_ANSWERING
        """
        prompt = f"""Classify the user's intent into one of the following:
                1. GLOBAL_SUMMARY - User wants a complete summary of the entire document
                2. PARTIAL_SUMMARY - User wants a summary of a specific topic/section
                3. QUESTION_ANSWERING - User is asking a specific question that needs a direct answer

                User query: "{query}"

                Return only the label (GLOBAL_SUMMARY, PARTIAL_SUMMARY, or QUESTION_ANSWERING).
                """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic cho classification
                    max_output_tokens=20
                )
            )
            
            # Handle None response
            if not response or not response.text:
                logger.warning("Received empty response from intent classification")
                return "QUESTION_ANSWERING"
            
            intent = response.text.strip().upper()
            
            # Validate intent
            valid_intents = ["GLOBAL_SUMMARY", "PARTIAL_SUMMARY", "QUESTION_ANSWERING"]
            if any(valid in intent for valid in valid_intents):
                for valid in valid_intents:
                    if valid in intent:
                        logger.info(f"Classified intent: {valid} for query: '{query[:50]}...'")
                        return valid
            
            # Default to QUESTION_ANSWERING if unclear
            logger.warning(f"Could not classify intent, defaulting to QUESTION_ANSWERING. Response: {intent}")
            return "QUESTION_ANSWERING"
        
        except Exception as e:
            logger.error(f"Lỗi khi classify intent: {e}")
            # Default fallback
            return "QUESTION_ANSWERING"

    
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )
            )
            summary = response.text.strip()
            
            logger.info(f"Tạo tóm tắt thành công ({len(summary)} chars)")
            return summary
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo tóm tắt: {e}")
            raise
    
    def summarize_in_chunks(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "Vietnamese",
        chunks_per_batch: int = 7
    ) -> Dict[str, Any]:
        """
        Tóm tắt document theo từng batch chunks thay vì toàn bộ.
        Dùng cho GLOBAL_SUMMARY với documents lớn.
        
        Args:
            chunks: List các chunks với metadata
            language: Ngôn ngữ output
            chunks_per_batch: Số chunks xử lý mỗi lần
        
        Returns:
            Dict với 'summary' và 'metadata' (source chunks)
        """
        logger.info(f"Tóm tắt {len(chunks)} chunks theo batch ({chunks_per_batch} chunks/batch)...")
        
        batch_summaries = []
        source_metadata = []
        
        # Process từng batch
        for i in range(0, len(chunks), chunks_per_batch):
            batch = chunks[i:i + chunks_per_batch]
            batch_num = i // chunks_per_batch + 1
            total_batches = (len(chunks) + chunks_per_batch - 1) // chunks_per_batch
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            # Combine batch text
            batch_text = "\n\n".join([
                f"--- Chunk {j+1} ---\n{chunk['text']}"
                for j, chunk in enumerate(batch)
            ])
            
            # Tạo prompt cho batch summary
            if language.lower() == "vietnamese":
                prompt = f"""Tóm tắt chi tiết phần văn bản sau (Batch {batch_num}/{total_batches}):{batch_text}
                            Tạo tóm tắt TOÀN DIỆN, bao quát TẤT CẢ nội dung quan trọng trong phần này:"""
            else:
                prompt = f"""Summarize the following text section in detail (Batch {batch_num}/{total_batches}):{batch_text}
                            Create a COMPREHENSIVE summary covering ALL important content in this section:"""
            
            # Generate batch summary
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=2048
                    )
                )
                batch_summary = response.text.strip()
                batch_summaries.append(batch_summary)
                
                # Track metadata
                for chunk in batch:
                    source_metadata.append({
                        'chunk_id': chunk.get('metadata', {}).get('chunk', 'N/A'),
                        'source': chunk.get('metadata', {}).get('source', 'Unknown'),
                        'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                    })
                
            except Exception as e:
                logger.error(f"Lỗi khi tóm tắt batch {batch_num}: {e}")
                batch_summaries.append(f"[Lỗi xử lý batch {batch_num}]")
        
        # Combine all batch summaries into final summary
        logger.info(f"Combining {len(batch_summaries)} batch summaries...")
        
        combined_text = "\n\n".join([
            f"Phần {i+1}:\n{summary}"
            for i, summary in enumerate(batch_summaries)
        ])
        
        if language.lower() == "vietnamese":
            final_prompt = f"""Dưới đây là các tóm tắt từng phần của một tài liệu:{combined_text}
                            Hãy kết hợp các phần trên thành một bản tóm tắt TỔNG HỢP, MẠCH LẠC, bao quát toàn bộ tài liệu:"""
        else:
            final_prompt = f"""Below are section summaries of a document:{combined_text}
                            Combine the above sections into a COMPREHENSIVE, COHERENT summary covering the entire document:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )
            )
            final_summary = response.text.strip()
            
            logger.info(f"Tóm tắt hoàn thành ({len(final_summary)} chars)")
            
            return {
                'summary': final_summary,
                'metadata': source_metadata,
                'num_chunks': len(chunks),
                'num_batches': len(batch_summaries)
            }
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo final summary: {e}")
            raise

    def summarize_chunk_by_chunk(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "Vietnamese",
        delay_seconds: float = 1.0,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Tóm tắt document bằng cách xử lý TỪNG CHUNK MỘT.
        Dùng khi API bị overload với batch processing.
        
        Args:
            chunks: List các chunks với metadata
            language: Ngôn ngữ output
            delay_seconds: Delay giữa mỗi request (tránh rate limit)
            progress_callback: Optional callback function(current, total, status_msg)
        
        Returns:
            Dict với 'summary' và 'metadata'
        """
        import time
        
        logger.info(f"Tóm tắt {len(chunks)} chunks một-by-một với delay {delay_seconds}s...")
        
        chunk_summaries = []
        source_metadata = []
        
        # Process từng chunk riêng lẻ
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Update progress
            if progress_callback:
                progress_callback(i+1, len(chunks), f"Đang xử lý chunk {i+1}/{len(chunks)}...")
            
            # Tạo prompt cho chunk
            if language.lower() == "vietnamese":
                prompt = f"""Tóm tắt ngắn gọn phần văn bản sau (Chunk {i+1}/{len(chunks)}):{chunk['text']}
                            Tạo tóm tắt NGẮN GỌN (2-3 câu) nắm bắt ý chính:"""
            else:
                prompt = f"""Briefly summarize the following text (Chunk {i+1}/{len(chunks)}):{chunk['text']}
                            Create a BRIEF summary (2-3 sentences) capturing the main idea:"""
            
            # Generate summary cho chunk này
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=500  # Ngắn gọn cho mỗi chunk
                    )
                )
                chunk_summary = response.text.strip()
                chunk_summaries.append(f"Phần {i+1}: {chunk_summary}")
                
                # Track metadata
                source_metadata.append({
                    'chunk_id': chunk.get('metadata', {}).get('chunk', i+1),
                    'source': chunk.get('metadata', {}).get('source', 'Unknown'),
                    'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                })
                
                logger.info(f"✓ Chunk {i+1} summarized ({len(chunk_summary)} chars)")
                
            except Exception as e:
                logger.error(f"Lỗi khi tóm tắt chunk {i+1}: {e}")
                chunk_summaries.append(f"Phần {i+1}: [Lỗi xử lý]")
            
            # Delay giữa các requests
            if i < len(chunks) - 1:  # Không delay sau chunk cuối
                time.sleep(delay_seconds)
        
        # Combine tất cả chunk summaries
        logger.info(f"Combining {len(chunk_summaries)} chunk summaries...")
        
        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Đang kết hợp các phần tóm tắt...")
        
        combined_text = "\n\n".join(chunk_summaries)
        
        if language.lower() == "vietnamese":
            final_prompt = f"""Dưới đây là các tóm tắt từng phần của tài liệu:{combined_text}
                            Hãy kết hợp thành một bản tóm tắt TỔNG HỢP, MẠCH LẠC cho toàn bộ tài liệu:"""
        else:
            final_prompt = f"""Below are summaries of each section:{combined_text}
                            Combine into a COMPREHENSIVE, COHERENT summary of the entire document:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )
            )
            final_summary = response.text.strip()
            
            logger.info(f"Tóm tắt hoàn thành ({len(final_summary)} chars)")
            
            return {
                'summary': final_summary,
                'metadata': source_metadata,
                'num_chunks': len(chunks)
            }
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo final summary: {e}")
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
        max_retries = 3
        retry_delay = 2.0  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                )
                
                if not response or not response.text:
                    raise ValueError("Received empty response from API")
                
                summary = response.text.strip()
                
                logger.info(f"Tạo tóm tắt thành công ({len(summary)} chars)")
                return summary
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Lỗi khi tạo tóm tắt (attempt {attempt+1}/{max_retries}): {error_msg}")
                
                # Check if it's a 503 or rate limit error
                if "503" in error_msg or "overloaded" in error_msg.lower() or "UNAVAILABLE" in error_msg:
                    if attempt < max_retries - 1:
                        import time
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                
                # If not retryable or last attempt, raise
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
            prompt = f"""Bạn là một chuyên gia phân tích và tóm tắt văn bản học thuật/kỹ thuật.
                    Nhiệm vụ: Đọc KỸ LƯỠNG toàn bộ văn bản dưới đây và tạo một bản tóm tắt TOÀN DIỆN, CHI TIẾT.

                    YÊU CẦU QUAN TRỌNG:
                    1. **Độ dài**: Tóm tắt phải dài ít nhất 30-40% độ dài văn bản gốc
                    2. **Bao quát toàn diện**: Đề cập ĐẦY ĐỦ TẤT CẢ các phần, chương, mục trong văn bản
                    3. **Cấu trúc rõ ràng**: 
                    - Sử dụng tiêu đề và phân mục
                    - Đánh số thứ tự các phần chính
                    - Bullet points cho các điểm chi tiết
                    4. **Giữ nguyên thông tin quan trọng**:
                    - Số liệu, con số, phần trăm
                    - Tên riêng (người, địa điểm, mô hình, thuật ngữ)
                    - Kết quả thí nghiệm, so sánh
                    - Phương pháp nghiên cứu
                    5. **Ngôn ngữ**: Tiếng Việt chuyên nghiệp, mạch lạc, dễ hiểu

                    CẤU TRÚC TÓM TẮT NÊN BAO GỒM:
                    - Giới thiệu tổng quan
                    - Mục tiêu/vấn đề nghiên cứu
                    - Phương pháp/cách tiếp cận
                    - Kết quả chính (với số liệu cụ thể)
                    - Các phát hiện quan trọng
                    - Kết luận và đề xuất

                    Văn bản cần tóm tắt:
                    {document_text}

                    Hãy tạo bản tóm tắt TOÀN DIỆN, CHI TIẾT:"""
                            
        else:  # English
            prompt = f"""You are an expert in analyzing and summarizing academic/technical documents.

                    Task: Read the following document THOROUGHLY and create a COMPREHENSIVE, DETAILED summary.

                    CRITICAL REQUIREMENTS:
                    1. **Length**: Summary must be at least 30-40% of original document length
                    2. **Comprehensive coverage**: Mention ALL sections, chapters, and topics in the document
                    3. **Clear structure**: 
                    - Use headings and sections
                    - Number main sections
                    - Use bullet points for details
                    4. **Preserve important information**:
                    - Numbers, statistics, percentages
                    - Proper nouns (people, places, models, terms)
                    - Experimental results, comparisons
                    - Research methods
                    5. **Language**: Professional English, clear and coherent

                    SUMMARY STRUCTURE SHOULD INCLUDE:
                    - Overall introduction
                    - Research objectives/problems
                    - Methods/approach
                    - Main results (with specific numbers)
                    - Key findings
                    - Conclusions and recommendations

                    Document to summarize:
                    {document_text}

                    Create a COMPREHENSIVE, DETAILED summary:"""

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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi chat: {e}")
            raise


# Hàm tiện ích
def create_summarizer(
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-exp",
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
