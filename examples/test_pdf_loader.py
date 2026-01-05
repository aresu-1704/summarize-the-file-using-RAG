"""
Script demo để test PDF Loader với nhiều loại file PDF.

Test cases:
1. PDF text thuần (native text)
2. PDF scan (OCR)
3. PDF 2 cột layout
4. PDF có ảnh embedded
5. PDF có bảng
6. PDF đa ngôn ngữ
"""

import sys
import os
from pathlib import Path

# Thêm đường dẫn đến thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.loaders.pdf_loader import PDFLoader


def test_loader_initialization():
    """Test khởi tạo PDF Loader với các config khác nhau."""
    print("\n" + "=" * 70)
    print("TEST 1: KHỞI TẠO PDF LOADER")
    print("=" * 70)
    
    # Config 1: Default (full features)
    loader1 = PDFLoader()
    print("\n✅ Config 1 - Default (Full Features):")
    print(f"   OCR enabled: {loader1.enable_ocr}")
    print(f"   Image extraction: {loader1.enable_image_extraction}")
    print(f"   Layout analysis: {loader1.enable_layout_analysis}")
    print(f"   Table extraction: {loader1.enable_table_extraction}")
    print(f"   Text cleaning: {loader1.enable_text_cleaning}")
    
    # Config 2: Text-only (fast mode)
    loader2 = PDFLoader(
        enable_ocr=False,
        enable_image_extraction=False,
        enable_layout_analysis=False,
        enable_table_extraction=False
    )
    print("\n✅ Config 2 - Text-Only Mode (Fast):")
    print(f"   OCR enabled: {loader2.enable_ocr}")
    print(f"   Image extraction: {loader2.enable_image_extraction}")
    print(f"   Layout analysis: {loader2.enable_layout_analysis}")
    
    # Config 3: OCR-focused (for scanned PDFs)
    loader3 = PDFLoader(
        enable_ocr=True,
        ocr_language="vie+eng",
        min_ocr_confidence=70.0
    )
    print("\n✅ Config 3 - OCR-Focused (Scanned Documents):")
    print(f"   OCR language: {loader3.ocr_language}")
    print(f"   Min confidence: {loader3.min_ocr_confidence}%")


def test_basic_pdf_loading(pdf_path):
    """Test load PDF cơ bản."""
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  File không tồn tại: {pdf_path}")
        print("   Tạo một file PDF mẫu và chạy lại test này.")
        return None
    
    print("\n" + "=" * 70)
    print("TEST 2: LOAD PDF CƠ BẢN")
    print("=" * 70)
    print(f"\n📂 File: {pdf_path}")
    
    loader = PDFLoader()
    docs = loader.load_pdf(pdf_path)
    
    if docs:
        print(f"\n✅ Load thành công!")
        print(f"   Số trang: {len(docs)}")
        
        # Thông tin trang đầu tiên
        first_doc = docs[0]
        metadata = first_doc['metadata']
        
        print(f"\n📄 Thông tin trang 1:")
        print(f"   Page: {metadata.get('page', 'N/A')}")
        print(f"   Total pages: {metadata.get('total_pages', 'N/A')}")
        print(f"   Encoding: {metadata.get('encoding', 'N/A')}")
        print(f"   Words: {metadata.get('num_words', 'N/A')}")
        print(f"   Has images: {metadata.get('has_images', False)}")
        print(f"   Has tables: {metadata.get('has_tables', False)}")
        
        print(f"\n📝 Preview nội dung (200 ký tự đầu):")
        print("─" * 70)
        print(first_doc['text'][:200] + "...")
        print("─" * 70)
        
        return docs
    else:
        print("\n❌ Không load được file!")
        return None


def test_two_column_layout(pdf_path):
    """Test xử lý PDF 2 cột."""
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  File không tồn tại: {pdf_path}")
        return
    
    print("\n" + "=" * 70)
    print("TEST 3: XỬ LÝ PDF 2 CỘT")
    print("=" * 70)
    print(f"\n📂 File: {pdf_path}")
    
    # Load với layout analysis
    loader = PDFLoader(enable_layout_analysis=True)
    docs = loader.load_pdf(pdf_path)
    
    if docs:
        print(f"\n✅ Phát hiện và xử lý layout:")
        for i, doc in enumerate(docs[:3], 1):  # Chỉ show 3 trang đầu
            metadata = doc['metadata']
            print(f"\n   Trang {i}:")
            print(f"      Layout type: {metadata.get('layout_type', 'unknown')}")
            print(f"      Columns detected: {metadata.get('num_columns', 'N/A')}")
            

def test_ocr_extraction(pdf_path):
    """Test OCR cho PDF scan."""
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  File không tồn tại: {pdf_path}")
        return
    
    print("\n" + "=" * 70)
    print("TEST 4: OCR CHO PDF SCAN")
    print("=" * 70)
    print(f"\n📂 File: {pdf_path}")
    
    loader = PDFLoader(
        enable_ocr=True,
        ocr_language="vie+eng",
        min_ocr_confidence=60.0
    )
    docs = loader.load_pdf(pdf_path)
    
    if docs:
        print(f"\n✅ OCR hoàn thành!")
        for i, doc in enumerate(docs[:2], 1):  # Show 2 trang đầu
            metadata = doc['metadata']
            print(f"\n   Trang {i}:")
            print(f"      OCR applied: {metadata.get('ocr_applied', False)}")
            print(f"      Confidence: {metadata.get('ocr_confidence', 'N/A')}")
            print(f"      Words extracted: {metadata.get('num_words', 'N/A')}")


def test_image_extraction(pdf_path):
    """Test trích xuất và OCR ảnh trong PDF."""
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  File không tồn tại: {pdf_path}")
        return
    
    print("\n" + "=" * 70)
    print("TEST 5: TRÍCH XUẤT ẢNH VÀ OCR")
    print("=" * 70)
    print(f"\n📂 File: {pdf_path}")
    
    loader = PDFLoader(
        enable_image_extraction=True,
        enable_ocr=True,
        min_image_confidence=60.0
    )
    docs = loader.load_pdf(pdf_path)
    
    if docs:
        total_images = sum(doc['metadata'].get('num_images', 0) for doc in docs)
        print(f"\n✅ Tổng số ảnh tìm thấy: {total_images}")
        
        for i, doc in enumerate(docs, 1):
            img_count = doc['metadata'].get('num_images', 0)
            if img_count > 0:
                print(f"\n   Trang {i}: {img_count} ảnh")
                img_text = doc['metadata'].get('image_text', [])
                if img_text:
                    for j, text in enumerate(img_text[:2], 1):  # Show 2 ảnh đầu
                        print(f"      Ảnh {j}: {text[:100]}...")


def test_table_extraction(pdf_path):
    """Test trích xuất bảng từ PDF."""
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  File không tồn tại: {pdf_path}")
        return
    
    print("\n" + "=" * 70)
    print("TEST 6: TRÍCH XUẤT BẢNG")
    print("=" * 70)
    print(f"\n📂 File: {pdf_path}")
    
    loader = PDFLoader(enable_table_extraction=True)
    docs = loader.load_pdf(pdf_path)
    
    if docs:
        total_tables = sum(doc['metadata'].get('num_tables', 0) for doc in docs)
        print(f"\n✅ Tổng số bảng tìm thấy: {total_tables}")
        
        for i, doc in enumerate(docs, 1):
            table_count = doc['metadata'].get('num_tables', 0)
            if table_count > 0:
                print(f"\n   Trang {i}: {table_count} bảng")


def test_text_cleaning():
    """Test text cleaning pipeline."""
    print("\n" + "=" * 70)
    print("TEST 7: TEXT CLEANING")
    print("=" * 70)
    
    # Tạo sample text có nhiễu
    sample_text = """
    Đây    là   văn bản    có   nhiều   khoảng   trắng.
    
    
    
    Và   nhiều   dòng   trống.
    
    URL: https://example.com   Email: test@example.com
    
    Số điện thoại: 0123-456-789
    """
    
    # Load với cleaning
    loader1 = PDFLoader(
        enable_text_cleaning=True,
        remove_urls=True,
        remove_emails=True
    )
    
    # Load không cleaning
    loader2 = PDFLoader(enable_text_cleaning=False)
    
    print("\n📝 Text gốc:")
    print("─" * 70)
    print(sample_text)
    print("─" * 70)
    
    # Note: Thực tế cần test với PDF file, đây chỉ là demo
    print("\n✅ Text cleaning sẽ:")
    print("   - Gộp nhiều khoảng trắng thành 1")
    print("   - Xóa dòng trống dư thừa")
    print("   - Xóa URLs và emails (nếu enable)")
    print("   - Chuẩn hóa encoding")


def demo_with_sample_files():
    """Demo với các file PDF mẫu."""
    print("\n" + "=" * 70)
    print("DEMO VỚI FILE PDF MẪU")
    print("=" * 70)
    
    # Định nghĩa các file test
    test_files = {
        "native_text": "test_native.pdf",
        "scanned": "test_scanned.pdf",
        "two_column": "test_two_column.pdf",
        "with_images": "test_with_images.pdf",
        "with_tables": "test_with_tables.pdf",
        "multilang": "test_multilang.pdf"
    }
    
    print("\n📋 Danh sách file test cần tạo:")
    for test_type, filename in test_files.items():
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"   {exists} {filename} ({test_type})")
    
    # Test với file có sẵn
    print("\n🔍 Tìm file PDF trong thư mục hiện tại:")
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if pdf_files:
        print(f"   Tìm thấy {len(pdf_files)} file PDF:")
        for pdf_file in pdf_files:
            print(f"      - {pdf_file}")
        
        # Test với file đầu tiên
        test_file = str(pdf_files[0])
        print(f"\n🧪 Chạy test với file: {test_file}")
        test_basic_pdf_loading(test_file)
    else:
        print("   Không tìm thấy file PDF nào.")
        print("\n💡 Hướng dẫn:")
        print("   1. Tạo hoặc copy file PDF vào thư mục hiện tại")
        print("   2. Chạy lại script này")
        print("   3. Hoặc sử dụng đường dẫn cụ thể:")
        print("      docs = loader.load_pdf('path/to/your/file.pdf')")


def main():
    """Chạy tất cả tests."""
    print("=" * 70)
    print("DEMO PDF LOADER - RAG APPLICATION")
    print("=" * 70)
    
    try:
        # Test 1: Khởi tạo loader
        test_loader_initialization()
        
        # Test 2-7: Cần file PDF thực tế
        print("\n" + "=" * 70)
        print("📝 LƯU Ý:")
        print("=" * 70)
        print("Các test sau cần file PDF thực tế để chạy:")
        print("  - Test 2: Load PDF cơ bản")
        print("  - Test 3: Xử lý 2 cột")
        print("  - Test 4: OCR")
        print("  - Test 5: Trích xuất ảnh")
        print("  - Test 6: Trích xuất bảng")
        print("  - Test 7: Text cleaning")
        
        # Test text cleaning (không cần file)
        test_text_cleaning()
        
        # Demo với file có sẵn
        demo_with_sample_files()
        
        print("\n" + "=" * 70)
        print("✅ TESTS HOÀN THÀNH")
        print("=" * 70)
        
        print("\n💡 Ví dụ sử dụng:")
        print("─" * 70)
        print("""
from core.loaders.pdf_loader import PDFLoader

# Khởi tạo loader
loader = PDFLoader(
    enable_ocr=True,
    enable_layout_analysis=True,
    ocr_language="vie+eng"
)

# Load PDF
docs = loader.load_pdf("your_file.pdf")

# Xem kết quả
for doc in docs:
    print(f"Page {doc['metadata']['page']}:")
    print(doc['text'][:200])
    print("-" * 50)
        """)
        print("─" * 70)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
