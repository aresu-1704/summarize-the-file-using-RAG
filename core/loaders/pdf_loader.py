"""
PDF Loader nâng cao với khả năng xử lý 2 cột và OCR.

Module này cung cấp các tính năng:
- Trích xuất text từ PDF thông thường
- Tự động phát hiện và xử lý bố cục 2 cột
- OCR cho PDF scan
- OCR thông minh cho ảnh embedded (bỏ qua diagrams/charts)
- Làm sạch text cho embedding tối ưu
"""

import os
import re
import unicodedata
from typing import List, Dict, Any
import logging

# Import thư viện PDF
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import fitz  # PyMuPDF fallback

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """PDF loader với khả năng phát hiện 2 cột và OCR thông minh."""
    
    def __init__(
        self,
        ocr_languages: str = "vie+eng+kor+jpn+chi+rus+ara+fra+deu+ita+spa+pol+por+fin+heb+ind",
        column_threshold: float = 0.4,
        min_text_threshold: int = 50,
        enable_ocr: bool = True,
        image_dpi: int = 300,
        enable_image_extraction: bool = True,
        enable_text_cleaning: bool = True,
        min_image_confidence: float = 60.0,
        min_image_words: int = 5
    ):
        """
        Khởi tạo PDF loader.
        
        Args:
            ocr_languages: Ngôn ngữ cho OCR (format Tesseract)
            column_threshold: Ngưỡng phát hiện 2 cột (0-1)
            min_text_threshold: Độ dài text tối thiểu để bỏ qua OCR
            enable_ocr: Bật/tắt OCR cho trang scan
            image_dpi: DPI khi convert PDF sang ảnh
            enable_image_extraction: Bật/tắt OCR cho ảnh embedded
            enable_text_cleaning: Bật/tắt làm sạch text
            min_image_confidence: Ngưỡng confidence tối thiểu cho OCR ảnh (0-100)
            min_image_words: Số từ tối thiểu để coi ảnh có text hữu ích
        """
        self.ocr_languages = ocr_languages
        self.column_threshold = column_threshold
        self.min_text_threshold = min_text_threshold
        self.enable_ocr = enable_ocr
        self.image_dpi = image_dpi
        self.enable_image_extraction = enable_image_extraction
        self.enable_text_cleaning = enable_text_cleaning
        self.min_image_confidence = min_image_confidence
        self.min_image_words = min_image_words
    
    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load PDF và trích xuất text.
        
        Args:
            file_path: Đường dẫn file PDF
            
        Returns:
            List documents với text và metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF: {file_path}")
        
        documents = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Trích xuất text từ trang
                text = self._extract_page_text(page)
                
                # Kiểm tra xem trang có phải scan không (ít text)
                if len(text.strip()) < self.min_text_threshold and self.enable_ocr:
                    logger.info(f"Trang {page_num} là scan. Áp dụng OCR...")
                    text = self._extract_with_ocr(file_path, page_num)
                    processing_method = "OCR"
                else:
                    processing_method = "Text extraction"
                    
                    # Trích xuất text từ ảnh embedded
                    if self.enable_image_extraction and self.enable_ocr:
                        image_text = self._extract_images_from_page(page)
                        if image_text:
                            text += "\n\n[Text từ ảnh]\n" + image_text
                            processing_method = "Text extraction + Image OCR"
                
                # Làm sạch text
                if self.enable_text_cleaning:
                    text = self._clean_text(text)
                
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "page": page_num,
                            "source": file_path,
                            "processing_method": processing_method,
                            "total_pages": len(pdf.pages)
                        }
                    })
        
        return documents
    
    def _extract_page_text(self, page) -> str:
        """
        Trích xuất text từ trang với phát hiện cột.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Text đã trích xuất với thứ tự cột đúng
        """
        # Lấy từng từ với vị trí
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        if not words:
            return ""
        
        # Phát hiện bố cục 2 cột
        if self._is_two_column_layout(words, page):
            return self._extract_two_column_text(words, page)
        else:
            # Cột đơn - trích xuất bình thường
            return page.extract_text(x_tolerance=3, y_tolerance=3)
    
    def _is_two_column_layout(self, words: List[Dict], page) -> bool:
        """
        Phát hiện trang có bố cục 2 cột.
        
        Args:
            words: List từ điển từ với vị trí
            page: pdfplumber page object
            
        Returns:
            True nếu phát hiện 2 cột
        """
        if not words:
            return False
        
        page_width = page.width
        middle_x = page_width / 2
        
        # Đếm từ ở nửa trái và phải
        left_count = sum(1 for w in words if w['x0'] < middle_x)
        right_count = sum(1 for w in words if w['x0'] >= middle_x)
        
        total_words = len(words)
        left_ratio = left_count / total_words if total_words > 0 else 0
        right_ratio = right_count / total_words if total_words > 0 else 0
        
        # Cả 2 bên phải có ít nhất threshold % từ
        return (left_ratio >= self.column_threshold and 
                right_ratio >= self.column_threshold)
    
    def _extract_two_column_text(self, words: List[Dict], page) -> str:
        """
        Trích xuất text từ bố cục 2 cột theo thứ tự đọc đúng.
        
        Args:
            words: List từ điển từ với vị trí
            page: pdfplumber page object
            
        Returns:
            Text với thứ tự cột đúng
        """
        page_width = page.width
        middle_x = page_width / 2
        
        # Tách từ thành cột trái và phải
        left_words = [w for w in words if w['x0'] < middle_x]
        right_words = [w for w in words if w['x0'] >= middle_x]
        
        # Sắp xếp theo vị trí dọc (trên xuống dưới)
        left_words.sort(key=lambda w: (w['top'], w['x0']))
        right_words.sort(key=lambda w: (w['top'], w['x0']))
        
        # Ghép text cho mỗi cột
        left_text = self._build_text_from_words(left_words)
        right_text = self._build_text_from_words(right_words)
        
        # Kết hợp: cột trái trước, sau đó cột phải
        return f"{left_text}\n\n{'='*50}\n[Cột phải]\n{'='*50}\n\n{right_text}"
    
    def _build_text_from_words(self, words: List[Dict]) -> str:
        """
        Ghép text từ list từ đã sắp xếp.
        
        Args:
            words: List từ đã sắp xếp
            
        Returns:
            Text đã ghép
        """
        if not words:
            return ""
        
        lines = []
        current_line = []
        current_top = words[0]['top']
        line_threshold = 5  # pixels
        
        for word in words:
            # Kiểm tra từ này ở dòng mới
            if abs(word['top'] - current_top) > line_threshold:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word['text']]
                current_top = word['top']
            else:
                current_line.append(word['text'])
        
        # Thêm dòng cuối
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _extract_with_ocr(self, file_path: str, page_num: int) -> str:
        """
        Trích xuất text bằng OCR cho trang scan.
        
        Args:
            file_path: Đường dẫn file PDF
            page_num: Số trang (1-indexed)
            
        Returns:
            Text từ OCR
        """
        # Convert trang PDF sang ảnh
        images = convert_from_path(
            file_path,
            first_page=page_num,
            last_page=page_num,
            dpi=self.image_dpi
        )
        
        if not images:
            return ""
        
        image = images[0]
        
        # Tiền xử lý ảnh
        image = self._preprocess_image(image)
        
        # OCR
        text = pytesseract.image_to_string(
            image,
            lang=self.ocr_languages,
            config='--psm 1'  # Phân trang tự động
        )
        
        return text
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Tiền xử lý ảnh để tăng độ chính xác OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Ảnh đã xử lý
        """
        # Convert sang OpenCV format
        img_array = np.array(image)
        
        # Convert sang grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Deskew (sửa nghiêng)
        gray = self._deskew_image(gray)
        
        # Khử nhiễu
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Binarization (Otsu's method)
        _, binary = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Convert về PIL
        return Image.fromarray(binary)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sửa nghiêng của ảnh scan.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Ảnh đã sửa nghiêng
        """
        # Phát hiện cạnh
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Phát hiện đường thẳng bằng Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return image
        
        # Tính góc trung bình
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Chỉ xét các đường gần ngang và gần dọc
            if (angle < 45 or angle > 135):
                angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        
        # Điều chỉnh góc để lấy góc xoay cần thiết
        if median_angle > 45:
            rotation_angle = median_angle - 90
        else:
            rotation_angle = median_angle
        
        # Chỉ xoay nếu góc đáng kể (> 0.5 độ)
        if abs(rotation_angle) > 0.5:
            # Xoay ảnh
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def _extract_images_from_page(self, page) -> str:
        """
        Trích xuất và OCR text từ ảnh embedded trong trang PDF.
        Sử dụng bộ lọc thông minh để tránh OCR các diagram/chart.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Text từ những ảnh có nội dung text hữu ích
        """
        images = page.images
        if not images:
            return ""
        
        extracted_texts = []
        
        for img_idx, img_info in enumerate(images):
            # Lấy tọa độ ảnh
            x0, y0, x1, y1 = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
            
            # Bỏ qua ảnh quá nhỏ (icon/logo)
            img_width = x1 - x0
            img_height = y1 - y0
            if img_width < 50 or img_height < 50:
                logger.debug(f"Bỏ qua ảnh nhỏ {img_idx} ({img_width}x{img_height})")
                continue
            
            # Crop vùng ảnh
            img_bbox = (x0, y0, x1, y1)
            cropped_page = page.crop(img_bbox)
            
            # Convert sang PIL Image
            img = cropped_page.to_image(resolution=150)
            pil_img = img.original
            
            # Tiền xử lý và OCR với dữ liệu chi tiết
            pil_img = self._preprocess_image(pil_img)
            
            # Lấy kết quả OCR với confidence data
            ocr_data = pytesseract.image_to_data(
                pil_img,
                lang=self.ocr_languages,
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Lọc và kiểm tra kết quả OCR
            if not self._is_image_text_meaningful(ocr_data):
                logger.debug(f"Bỏ qua ảnh {img_idx}: chất lượng thấp hoặc là diagram")
                continue
            
            # Trích xuất text thực tế
            text = pytesseract.image_to_string(
                pil_img,
                lang=self.ocr_languages,
                config='--psm 6'
            )
            
            text = text.strip()
            if text:
                extracted_texts.append(text)
                logger.info(f"Trích xuất text từ ảnh {img_idx} (phát hiện nội dung hữu ích)")
        
        return "\n\n".join(extracted_texts)
    
    def _is_image_text_meaningful(self, ocr_data: Dict) -> bool:
        """
        Xác định ảnh có chứa text hữu ích hay chỉ là diagram/chart.
        
        Args:
            ocr_data: Dữ liệu OCR từ pytesseract.image_to_data
            
        Returns:
            True nếu ảnh có text hữu ích, False nếu là diagram/chart
        """
        # Trích xuất confidence và text
        confidences = []
        words = []
        
        for i, conf in enumerate(ocr_data['conf']):
            # Bỏ qua confidence không hợp lệ
            if conf == -1:
                continue
            
            text = ocr_data['text'][i].strip()
            if text:
                confidences.append(float(conf))
                words.append(text)
        
        # Không tìm thấy text
        if not words:
            return False
        
        # Check 1: Số từ tối thiểu
        if len(words) < self.min_image_words:
            logger.debug(f"Chỉ có {len(words)} từ, dưới ngưỡng {self.min_image_words}")
            return False
        
        # Check 2: Ngưỡng confidence trung bình
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < self.min_image_confidence:
            logger.debug(f"OCR confidence thấp: {avg_confidence:.1f}% < {self.min_image_confidence}%")
            return False
        
        # Check 3: Mật độ text - diagram thường có label ngắn rải rác
        # Tính % "từ thật" (độ dài > 2)
        real_words = [w for w in words if len(w) > 2]
        word_quality_ratio = len(real_words) / len(words) if words else 0
        
        if word_quality_ratio < 0.5:  # Dưới 50% từ thật
            logger.debug(f"Tỉ lệ từ chất lượng thấp: {word_quality_ratio:.2f}")
            return False
        
        # Check 4: Phát hiện chart/diagram
        # Chart thường có nhiều số, ít từ
        number_count = sum(1 for w in words if w.isdigit())
        number_ratio = number_count / len(words) if words else 0
        
        if number_ratio > 0.7:  # Hơn 70% là số → khả năng cao là chart
            logger.debug(f"Tỉ lệ số cao: {number_ratio:.2f} - có thể là chart/diagram")
            return False
        
        logger.debug(f"Ảnh hợp lệ: {len(words)} từ, {avg_confidence:.1f}% conf, {word_quality_ratio:.2f} chất lượng")
        return True
    
    def _clean_text(self, text: str) -> str:
        """
        Làm sạch và chuẩn hóa text để tối ưu cho embedding.
        
        Args:
            text: Text thô
            
        Returns:
            Text đã làm sạch
        """
        if not text:
            return text
        
        # 1. Chuẩn hóa Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Sửa lỗi OCR thường gặp
        text = self._fix_ocr_artifacts(text)
        
        # 3. Sửa từ bị ngắt dòng
        text = self._fix_hyphenation(text)
        
        # 4. Chuẩn hóa khoảng trắng
        text = self._normalize_whitespace(text)
        
        # 5. Xóa số trang và header/footer
        text = self._remove_page_artifacts(text)
        
        return text.strip()
    
    def _fix_ocr_artifacts(self, text: str) -> str:
        """Sửa lỗi OCR thường gặp."""
        # Xóa khoảng trắng thừa giữa các ký tự
        text = re.sub(r'(\w)\s+(\w)', lambda m: m.group(1) + m.group(2) if len(m.group(0)) > 3 else m.group(0), text)
        
        # Sửa lỗi thay thế ký tự OCR
        ocr_fixes = {
            r'\bl\b': 'I',  # l thường nhầm thành I
            r'\bO(?=\d)': '0',  # O trước số là 0
            r'(?<=\d)O\b': '0',  # O sau số là 0
            r'\brn\b': 'm',  # rn thường nhầm thành m
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Xóa ký tự unicode lạ (nhiễu OCR)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """Sửa từ bị ngắt dòng với dấu gạch ngang."""
        # Ghép từ-\n thành từ hoàn chỉnh
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Chuẩn hóa khoảng trắng thừa nhưng giữ cấu trúc."""
        # Nhiều space thành 1 space
        text = re.sub(r' {2,}', ' ', text)
        
        # Nhiều newline (>2) thành 2 newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Xóa space đầu/cuối mỗi dòng
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _remove_page_artifacts(self, text: str) -> str:
        """Xóa số trang, header, footer thường gặp."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Bỏ qua dòng chỉ là số trang
            if re.match(r'^\d+$', line_stripped):
                continue
            
            # Bỏ qua header/footer pattern thường gặp
            if re.match(r'^Page \d+', line_stripped, re.IGNORECASE):
                continue
            
            # Bỏ qua dòng quá ngắn ở đầu/cuối
            if len(line_stripped) < 3 and (not cleaned_lines or line == lines[-1]):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


# Hàm tiện ích để tương thích ngược
def load_pdf(
    file_path: str,
    ocr_languages: str = "vie+eng+kor+jpn+chi+rus+ara+fra+deu+ita+spa+pol+por+fin+heb+ind",
    enable_ocr: bool = True,
    enable_image_extraction: bool = True,
    enable_text_cleaning: bool = True,
    min_image_confidence: float = 60.0,
    min_image_words: int = 5
) -> List[Dict[str, Any]]:
    """
    Load file PDF và trả về documents với text đã làm sạch.
    
    Args:
        file_path: Đường dẫn file PDF
        ocr_languages: Ngôn ngữ cho OCR (mặc định: đa ngôn ngữ)
        enable_ocr: Bật OCR cho trang scan
        enable_image_extraction: Bật trích xuất và OCR ảnh embedded
        enable_text_cleaning: Bật làm sạch text
        min_image_confidence: Ngưỡng confidence tối thiểu cho OCR ảnh (0-100)
        min_image_words: Số từ tối thiểu để coi ảnh có text hữu ích
        
    Returns:
        List documents với text đã làm sạch và metadata
    """
    loader = PDFLoader(
        ocr_languages=ocr_languages,
        enable_ocr=enable_ocr,
        enable_image_extraction=enable_image_extraction,
        enable_text_cleaning=enable_text_cleaning,
        min_image_confidence=min_image_confidence,
        min_image_words=min_image_words
    )
    return loader.load_pdf(file_path)