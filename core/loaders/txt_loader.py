"""
TXT Loader nâng cao cho RAG application.

Module này cung cấp các tính năng:
- Tự động phát hiện encoding (UTF-8, Latin-1, CP1252, etc.)
- Xử lý nhiều định dạng text: plain text, markdown, log files
- Phát hiện cấu trúc văn bản (sections, headers, lists)
- Làm sạch và chuẩn hóa text cho embedding tối ưu
- Xử lý văn bản đa ngôn ngữ
- Tách văn bản thành semantic chunks
"""

import os
import re
import unicodedata
import chardet
from typing import List, Dict, Any, Optional
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TXTLoader:
    """TXT loader với khả năng tự động phát hiện encoding và xử lý cấu trúc."""
    
    def __init__(
        self,
        encoding: Optional[str] = None,
        auto_detect_encoding: bool = True,
        enable_structure_detection: bool = True,
        enable_text_cleaning: bool = True,
        preserve_formatting: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        min_line_length: int = 1
    ):
        """
        Khởi tạo TXT loader.
        
        Args:
            encoding: Encoding cố định (None = auto-detect)
            auto_detect_encoding: Tự động phát hiện encoding
            enable_structure_detection: Phát hiện cấu trúc văn bản (headers, sections)
            enable_text_cleaning: Làm sạch text (whitespace, unicode)
            preserve_formatting: Giữ nguyên format (spaces, indentation)
            remove_urls: Xóa URLs khỏi text
            remove_emails: Xóa email addresses khỏi text
            min_line_length: Độ dài tối thiểu của line (bỏ qua lines ngắn hơn)
        """
        self.encoding = encoding
        self.auto_detect_encoding = auto_detect_encoding
        self.enable_structure_detection = enable_structure_detection
        self.enable_text_cleaning = enable_text_cleaning
        self.preserve_formatting = preserve_formatting
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.min_line_length = min_line_length
    
    def load_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load file TXT và trích xuất nội dung.
        
        Args:
            file_path: Đường dẫn file TXT
            
        Returns:
            List documents với text và metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file TXT: {file_path}")
        
        # Đọc file với encoding detection
        text, detected_encoding = self._read_file_with_encoding(file_path)
        
        if not text:
            logger.warning(f"File rỗng hoặc không đọc được: {file_path}")
            return []
        
        # Phát hiện loại file (plain text, markdown, log, etc.)
        file_type = self._detect_file_type(file_path, text)
        
        # Làm sạch text
        if self.enable_text_cleaning:
            text = self._clean_text(text)
        
        # Phát hiện cấu trúc
        structure_info = {}
        if self.enable_structure_detection:
            structure_info = self._detect_structure(text)
        
        # Tính toán metadata
        metadata = {
            "source": file_path,
            "encoding": detected_encoding,
            "file_type": file_type,
            "num_lines": len(text.split('\n')),
            "num_chars": len(text),
            "num_words": len(text.split()),
            **structure_info
        }
        
        documents = []
        if text.strip():
            documents.append({
                "text": text,
                "metadata": metadata
            })
        
        logger.info(f"Đã load file TXT: {file_path} ({detected_encoding}, {metadata['num_lines']} lines)")
        return documents
    
    def _read_file_with_encoding(self, file_path: str) -> tuple[str, str]:
        """
        Đọc file với tự động phát hiện encoding.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            Tuple (text content, detected encoding)
        """
        # Nếu có encoding cố định, dùng luôn
        if self.encoding:
            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    text = f.read()
                return text, self.encoding
            except UnicodeDecodeError as e:
                logger.warning(f"Không đọc được file với encoding {self.encoding}: {e}")
        
        # Auto-detect encoding
        if self.auto_detect_encoding:
            # Đọc một phần file để detect
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Sử dụng chardet để detect encoding
            detection = chardet.detect(raw_data)
            detected_encoding = detection['encoding']
            confidence = detection['confidence']
            
            logger.debug(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")
            
            # Thử các encoding phổ biến theo thứ tự ưu tiên
            encodings_to_try = [
                detected_encoding,
                'utf-8',
                'utf-8-sig',  # UTF-8 with BOM
                'latin-1',
                'cp1252',  # Windows Western European
                'iso-8859-1',
                'ascii'
            ]
            
            # Loại bỏ None và duplicates
            encodings_to_try = list(dict.fromkeys([e for e in encodings_to_try if e]))
            
            for enc in encodings_to_try:
                try:
                    text = raw_data.decode(enc)
                    logger.info(f"Đọc thành công với encoding: {enc}")
                    return text, enc
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # Fallback: ignore errors
            logger.warning("Không thể decode hoàn toàn, sử dụng UTF-8 với errors='ignore'")
            text = raw_data.decode('utf-8', errors='ignore')
            return text, 'utf-8 (with errors ignored)'
        
        # Default: UTF-8
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text, 'utf-8'
        except UnicodeDecodeError:
            # Fallback to latin-1 (luôn thành công)
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return text, 'latin-1'
    
    def _detect_file_type(self, file_path: str, text: str) -> str:
        """
        Phát hiện loại file text.
        
        Args:
            file_path: Đường dẫn file
            text: Nội dung text
            
        Returns:
            Loại file: 'markdown', 'log', 'code', 'plain_text'
        """
        filename = os.path.basename(file_path).lower()
        
        # Check extension
        if filename.endswith('.md') or filename.endswith('.markdown'):
            return 'markdown'
        elif filename.endswith('.log'):
            return 'log'
        elif filename.endswith(('.py', '.java', '.cpp', '.js', '.ts', '.c', '.h')):
            return 'code'
        
        # Check content patterns
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        # Markdown indicators
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers: # ## ###
            r'^\*\*.*\*\*',  # Bold: **text**
            r'^\[.*\]\(.*\)',  # Links: [text](url)
            r'^[-*+]\s',  # Lists: - * +
            r'^```',  # Code blocks
        ]
        
        markdown_count = 0
        for line in lines:
            if any(re.match(pattern, line) for pattern in markdown_patterns):
                markdown_count += 1
        
        if markdown_count >= 3:
            return 'markdown'
        
        # Log file indicators
        log_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # Date: 2024-01-01
            r'^\d{2}:\d{2}:\d{2}',  # Time: 12:30:45
            r'\b(ERROR|WARNING|INFO|DEBUG)\b',  # Log levels
            r'\[\d+\]',  # Process IDs
        ]
        
        log_count = 0
        for line in lines:
            if any(re.search(pattern, line) for pattern in log_patterns):
                log_count += 1
        
        if log_count >= 5:
            return 'log'
        
        return 'plain_text'
    
    def _detect_structure(self, text: str) -> Dict[str, Any]:
        """
        Phát hiện cấu trúc văn bản (headers, sections, lists).
        
        Args:
            text: Nội dung text
            
        Returns:
            Dictionary chứa thông tin cấu trúc
        """
        structure = {
            "num_sections": 0,
            "num_headers": 0,
            "num_lists": 0,
            "num_code_blocks": 0,
            "has_toc": False
        }
        
        lines = text.split('\n')
        
        # Đếm markdown headers
        for line in lines:
            # Headers: # ## ### ####
            if re.match(r'^#{1,6}\s', line):
                structure["num_headers"] += 1
                structure["num_sections"] += 1
            
            # Headers: Underline style
            # Title
            # =====
            elif re.match(r'^[=]{3,}$', line) or re.match(r'^[-]{3,}$', line):
                structure["num_headers"] += 1
                structure["num_sections"] += 1
            
            # Lists
            elif re.match(r'^[\s]*[-*+]\s', line) or re.match(r'^[\s]*\d+\.\s', line):
                structure["num_lists"] += 1
            
            # Code blocks
            elif re.match(r'^```', line):
                structure["num_code_blocks"] += 1
            
            # Table of Contents detection
            if re.search(r'(table of contents|mục lục|目次)', line.lower()):
                structure["has_toc"] = True
        
        # Nếu không có markdown headers, thử phát hiện sections bằng cách khác
        if structure["num_headers"] == 0:
            # Sections được phân cách bằng nhiều newlines
            sections = re.split(r'\n{3,}', text)
            structure["num_sections"] = len(sections) - 1
        
        return structure
    
    def _clean_text(self, text: str) -> str:
        """
        Làm sạch và chuẩn hóa text.
        
        Args:
            text: Text thô
            
        Returns:
            Text đã làm sạch
        """
        if not text:
            return text
        
        # 1. Chuẩn hóa Unicode (NFKC: compatibility decomposition + composition)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Xóa URLs (nếu bật)
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Xóa emails (nếu bật)
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 4. Xóa control characters (trừ newline, tab)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        
        # 5. Chuẩn hóa whitespace (nếu không preserve formatting)
        if not self.preserve_formatting:
            text = self._normalize_whitespace(text)
        
        # 6. Lọc lines quá ngắn
        if self.min_line_length > 1:
            lines = text.split('\n')
            lines = [line for line in lines if len(line.strip()) >= self.min_line_length or line.strip() == '']
            text = '\n'.join(lines)
        
        return text.strip()
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Chuẩn hóa khoảng trắng.
        
        Args:
            text: Text input
            
        Returns:
            Text với whitespace chuẩn hóa
        """
        # Nhiều spaces thành 1 space
        text = re.sub(r' {2,}', ' ', text)
        
        # Nhiều tabs thành 1 space
        text = re.sub(r'\t+', ' ', text)
        
        # Nhiều newlines (>2) thành 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Xóa spaces đầu/cuối mỗi dòng
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Xóa trailing whitespace sau dấu câu
        text = re.sub(r'([.!?,;:])\s{2,}', r'\1 ', text)
        
        return text


# Hàm tiện ích để tương thích ngược
def load_txt(
    file_path: str,
    encoding: Optional[str] = None,
    auto_detect_encoding: bool = True,
    enable_structure_detection: bool = True,
    enable_text_cleaning: bool = True,
    preserve_formatting: bool = False,
    remove_urls: bool = False,
    remove_emails: bool = False,
    min_line_length: int = 1
) -> List[Dict[str, Any]]:
    """
    Load file TXT và trả về documents với text đã làm sạch.
    
    Args:
        file_path: Đường dẫn file TXT
        encoding: Encoding cố định (None = auto-detect)
        auto_detect_encoding: Tự động phát hiện encoding
        enable_structure_detection: Phát hiện cấu trúc văn bản
        enable_text_cleaning: Làm sạch text
        preserve_formatting: Giữ nguyên format
        remove_urls: Xóa URLs
        remove_emails: Xóa emails
        min_line_length: Độ dài tối thiểu của line
        
    Returns:
        List documents với text đã làm sạch và metadata
    """
    loader = TXTLoader(
        encoding=encoding,
        auto_detect_encoding=auto_detect_encoding,
        enable_structure_detection=enable_structure_detection,
        enable_text_cleaning=enable_text_cleaning,
        preserve_formatting=preserve_formatting,
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        min_line_length=min_line_length
    )
    return loader.load_txt(file_path)
