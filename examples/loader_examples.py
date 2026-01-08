"""
Ví dụ minh họa dữ liệu mà các module loader return về.

Module này cho thấy:
- Cấu trúc dữ liệu của từng loader (TXT, PDF, DOCX)
- Các trường metadata khác nhau
- Cách sử dụng các loader
"""

import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.loaders.txt_loader import TXTLoader
from core.loaders.pdf_loader import PDFLoader
from core.loaders.docx_loader import DOCXLoader


def example_txt_loader():
    """
    Ví dụ dữ liệu return từ TXTLoader.
    
    Cấu trúc return:
    [
        {
            "text": "Nội dung văn bản đã được làm sạch...",
            "metadata": {
                "source": "đường/dẫn/file.txt",
                "encoding": "utf-8",
                "file_type": "plain_text" | "markdown" | "log" | "code",
                "num_lines": 100,
                "num_chars": 5000,
                "num_words": 800,
                # Nếu enable_structure_detection=True:
                "has_headers": True/False,
                "has_lists": True/False,
                "num_sections": 5
            }
        }
    ]
    """
    print("=" * 80)
    print("VÍ DỤ TXT LOADER")
    print("=" * 80)
    
    loader = TXTLoader(
        auto_detect_encoding=True,
        enable_text_cleaning=True,
        enable_structure_detection=True
    )
    
    # Giả sử load một file TXT
    # documents = loader.load_txt("example.txt")
    
    # Ví dụ kết quả:
    example_result = [
        {
            "text": "Đây là nội dung văn bản đã được làm sạch.\n\nNội dung có thể nhiều đoạn văn.\n\nCác ký tự đặc biệt và khoảng trắng thừa đã được xử lý.",
            "metadata": {
                "source": "c:/example/document.txt",
                "encoding": "utf-8",
                "file_type": "plain_text",
                "num_lines": 15,
                "num_chars": 850,
                "num_words": 125,
                "has_headers": False,
                "has_lists": False,
                "num_sections": 1
            }
        }
    ]
    
    print("\n📄 Cấu trúc dữ liệu TXT:")
    print(f"  - Kiểu: List[Dict[str, Any]]")
    print(f"  - Số documents: {len(example_result)}")
    print(f"\n📝 Document đầu tiên:")
    print(f"  - Text preview: {example_result[0]['text'][:100]}...")
    print(f"  - Metadata:")
    for key, value in example_result[0]['metadata'].items():
        print(f"      • {key}: {value}")
    
    return example_result


def example_pdf_loader():
    """
    Ví dụ dữ liệu return từ PDFLoader.
    
    Cấu trúc return (mỗi trang là 1 document):
    [
        {
            "text": "Nội dung trang 1 với xử lý 2 cột nếu có...\n\n[Bảng 1]\n...",
            "metadata": {
                "page": 1,
                "source": "đường/dẫn/file.pdf",
                "processing_method": "Text extraction" | "OCR" | "Text extraction + Image OCR",
                "total_pages": 10
            }
        },
        {
            "text": "Nội dung trang 2...",
            "metadata": {
                "page": 2,
                "source": "đường/dẫn/file.pdf",
                "processing_method": "OCR",
                "total_pages": 10
            }
        }
    ]
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ PDF LOADER")
    print("=" * 80)
    
    loader = PDFLoader(
        column_threshold=0.3,
        enable_ocr=True,
        enable_image_extraction=True,
        enable_table_extraction=True,
        enable_text_cleaning=True
    )
    
    # Giả sử load một file PDF 3 trang
    # documents = loader.load_pdf("example.pdf")
    
    # Ví dụ kết quả:
    example_result = [
        {
            "text": "TIÊU ĐỀ CHƯƠNG 1\n\nĐây là nội dung trang 1. PDF này có bố cục 2 cột nên text được sắp xếp đúng thứ tự từ trái sang phải.\n\n[Bảng 1]\n==================================================\nTên sản phẩm | Giá      | Số lượng\n-------------+----------+---------\nSản phẩm A   | 100,000đ | 50\nSản phẩm B   | 200,000đ | 30\n==================================================",
            "metadata": {
                "page": 1,
                "source": "c:/example/report.pdf",
                "processing_method": "Text extraction",
                "total_pages": 3
            }
        },
        {
            "text": "CHƯƠNG 2\n\nTrang này có ảnh scan nên được xử lý bằng OCR.\n\nNội dung được trích xuất từ ảnh scan với độ chính xác cao.",
            "metadata": {
                "page": 2,
                "source": "c:/example/report.pdf",
                "processing_method": "OCR",
                "total_pages": 3
            }
        },
        {
            "text": "KẾT LUẬN\n\nTrang cuối có cả text thông thường và ảnh embedded.\n\n[Text từ ảnh]\nĐây là text được OCR từ ảnh trong trang PDF.",
            "metadata": {
                "page": 3,
                "source": "c:/example/report.pdf",
                "processing_method": "Text extraction + Image OCR",
                "total_pages": 3
            }
        }
    ]
    
    print("\n📄 Cấu trúc dữ liệu PDF:")
    print(f"  - Kiểu: List[Dict[str, Any]]")
    print(f"  - Số documents: {len(example_result)} (mỗi trang = 1 document)")
    
    for i, doc in enumerate(example_result, 1):
        print(f"\n📝 Document {i} (Trang {doc['metadata']['page']}):")
        print(f"  - Text preview: {doc['text'][:80]}...")
        print(f"  - Processing method: {doc['metadata']['processing_method']}")
        print(f"  - Metadata:")
        for key, value in doc['metadata'].items():
            print(f"      • {key}: {value}")
    
    return example_result


def example_docx_loader():
    """
    Ví dụ dữ liệu return từ DOCXLoader.
    
    Cấu trúc return:
    [
        {
            "text": "Paragraphs...\n\n[Bảng 1]\n...\n\n[Đồ thị 1]\n...\n\n[Text từ ảnh]\n...",
            "metadata": {
                "source": "đường/dẫn/file.docx",
                "num_paragraphs": 50,
                "num_tables": 3
            }
        }
    ]
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ DOCX LOADER")
    print("=" * 80)
    
    loader = DOCXLoader(
        enable_image_extraction=True,
        enable_table_extraction=True,
        enable_chart_extraction=True,
        enable_text_cleaning=True
    )
    
    # Giả sử load một file DOCX
    # documents = loader.load_docx("example.docx")
    
    # Ví dụ kết quả:
    example_result = [
        {
            "text": """TIÊU ĐỀ TÀI LIỆU

Đây là đoạn văn giới thiệu trong tài liệu DOCX.

Nội dung được trích xuất theo thứ tự: paragraphs, tables, charts, và images.

[Bảng 1]
==================================================
Tháng | Doanh thu | Lợi nhuận
------+-----------+-----------
Jan   | 1,000,000 | 200,000
Feb   | 1,200,000 | 250,000
Mar   | 1,500,000 | 300,000
==================================================

Phần giải thích về bảng số liệu trên.

[Đồ thị 1]
==================================================
Caption: Biểu đồ tăng trưởng doanh thu theo tháng
Bảng số liệu gốc:
Tháng | Giá trị
------+--------
Jan   | 100
Feb   | 120
Mar   | 150
Nội dung đồ thị:
Text từ OCR: Q1 2024, +50% growth, Target achieved
==================================================

[Text từ ảnh]
Đây là text được OCR từ ảnh screenshot hoặc diagram có chứa text trong tài liệu.""",
            "metadata": {
                "source": "c:/example/report.docx",
                "num_paragraphs": 25,
                "num_tables": 5
            }
        }
    ]
    
    print("\n📄 Cấu trúc dữ liệu DOCX:")
    print(f"  - Kiểu: List[Dict[str, Any]]")
    print(f"  - Số documents: {len(example_result)}")
    print(f"\n📝 Document đầu tiên:")
    print(f"  - Text preview: {example_result[0]['text'][:150]}...")
    print(f"  - Text length: {len(example_result[0]['text'])} chars")
    print(f"  - Metadata:")
    for key, value in example_result[0]['metadata'].items():
        print(f"      • {key}: {value}")
    
    print("\n💡 Đặc điểm DOCX Loader:")
    print("  - Trích xuất theo thứ tự: paragraphs → tables → charts → images")
    print("  - Tables được format dạng markdown-style")
    print("  - Charts được phân tích 3 cách: caption + bảng gốc + OCR")
    print("  - Images được lọc thông minh (bỏ qua diagrams không có text)")
    
    return example_result


def summary_comparison():
    """So sánh tổng quan các loader."""
    print("\n" + "=" * 80)
    print("TỔNG QUAN SO SÁNH CÁC LOADER")
    print("=" * 80)
    
    comparison = """
┌──────────────┬─────────────────────────┬──────────────────────────────────────┐
│   Loader     │   Số documents return   │   Metadata chính                     │
├──────────────┼─────────────────────────┼──────────────────────────────────────┤
│ TXTLoader    │ 1 document              │ • source                             │
│              │ (toàn bộ file)          │ • encoding (utf-8, cp1252, ...)      │
│              │                         │ • file_type (plain/markdown/log)     │
│              │                         │ • num_lines, num_chars, num_words    │
│              │                         │ • structure info (sections, headers) │
├──────────────┼─────────────────────────┼──────────────────────────────────────┤
│ PDFLoader    │ N documents             │ • source                             │
│              │ (mỗi trang = 1 doc)     │ • page (số trang)                    │
│              │                         │ • total_pages                        │
│              │                         │ • processing_method                  │
│              │                         │   (Text/OCR/Text+Image OCR)          │
├──────────────┼─────────────────────────┼──────────────────────────────────────┤
│ DOCXLoader   │ 1 document              │ • source                             │
│              │ (toàn bộ file)          │ • num_paragraphs                     │
│              │                         │ • num_tables                         │
└──────────────┴─────────────────────────┴──────────────────────────────────────┘

📌 ĐIỂM CHUNG:
  • Tất cả đều return: List[Dict[str, Any]]
  • Mỗi dict có 2 keys: "text" và "metadata"
  • "text" là string chứa nội dung đã xử lý
  • "metadata" là dict chứa thông tin về nguồn và cách xử lý

📌 KHÁC BIỆT:
  • TXT/DOCX: 1 file = 1 document (toàn bộ nội dung)
  • PDF: 1 file = N documents (mỗi trang riêng biệt)
  
📌 ỨNG DỤNG TRONG RAG:
  1. Load documents từ các loader
  2. Chunking: chia nhỏ text thành các chunks (đoạn ngắn)
  3. Embedding: chuyển chunks thành vectors
  4. VectorStore: lưu vectors để retrieval
  5. Retrieval: tìm chunks liên quan khi user query
"""
    print(comparison)


def main():
    """Chạy tất cả ví dụ."""
    print("\n🚀 DEMO: CẤU TRÚC DỮ LIỆU CỦA CÁC MODULE LOADER\n")
    
    # Chạy các ví dụ
    example_txt_loader()
    example_pdf_loader()
    example_docx_loader()
    summary_comparison()
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT - Bạn đã hiểu cấu trúc dữ liệu của các loader!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
