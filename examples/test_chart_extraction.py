"""
Script demo để test tính năng trích xuất đồ thị từ DOCX.

Xử lý 3 trường hợp:
1. Đồ thị có caption
2. Đồ thị có bảng số liệu gốc
3. Đồ thị được OCR hoặc mô tả lại
"""

import sys
import os

# Thêm đường dẫn đến thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.loaders.docx_loader import DOCXLoader


def test_chart_extraction():
    """Test trích xuất đồ thị với 3 trường hợp."""
    
    # Khởi tạo loader với chart extraction enabled
    loader = DOCXLoader(
        ocr_languages="vie+eng",
        enable_image_extraction=True,
        enable_table_extraction=True,
        enable_chart_extraction=True,  # Bật xử lý đồ thị
        enable_text_cleaning=True,
        min_image_confidence=60.0,
        min_image_words=5
    )
    
    # Đường dẫn file DOCX test (bạn cần tạo file này)
    test_file = "test_document_with_charts.docx"
    
    if not os.path.exists(test_file):
        print(f"❌ Không tìm thấy file test: {test_file}")
        print("\nVui lòng tạo file DOCX với:")
        print("1. Ít nhất 1 đồ thị có caption (ví dụ: 'Hình 1: Biểu đồ doanh thu')")
        print("2. Ít nhất 1 đồ thị có bảng số liệu gốc ở gần")
        print("3. Ít nhất 1 đồ thị có text trên trục/labels (để OCR)")
        return
    
    print("=" * 70)
    print("TEST TRÍCH XUẤT ĐỒ THỊ TỪ DOCX")
    print("=" * 70)
    
    # Load document
    try:
        documents = loader.load_docx(test_file)
        
        if not documents:
            print("❌ Không trích xuất được nội dung từ file")
            return
        
        # Hiển thị kết quả
        for doc_idx, doc in enumerate(documents):
            print(f"\n📄 Document {doc_idx + 1}:")
            print(f"   Số paragraphs: {doc['metadata'].get('num_paragraphs', 'N/A')}")
            print(f"   Số tables: {doc['metadata'].get('num_tables', 'N/A')}")
            print("\n" + "─" * 70)
            print("NỘI DUNG:")
            print("─" * 70)
            print(doc['text'])
            print("─" * 70)
        
        # Phân tích kết quả
        text = documents[0]['text']
        
        print("\n" + "=" * 70)
        print("PHÂN TÍCH KẾT QUẢ")
        print("=" * 70)
        
        # Đếm số đồ thị được trích xuất
        chart_count = text.count("[Đồ thị")
        print(f"\n✅ Số đồ thị phát hiện: {chart_count}")
        
        # Kiểm tra các trường hợp
        has_caption = "Caption:" in text
        has_source_table = "Bảng số liệu gốc:" in text
        has_ocr = "Text từ OCR:" in text or "Nội dung đồ thị:" in text
        
        print("\n📊 CÁC TRƯỜNG HỢP ĐÃ XỬ LÝ:")
        print(f"   1. Có caption:           {'✅ Có' if has_caption else '❌ Không'}")
        print(f"   2. Có bảng số liệu gốc:  {'✅ Có' if has_source_table else '❌ Không'}")
        print(f"   3. OCR/Mô tả:            {'✅ Có' if has_ocr else '❌ Không'}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {e}")
        import traceback
        traceback.print_exc()


def create_sample_instructions():
    """In hướng dẫn tạo file DOCX mẫu."""
    print("\n" + "=" * 70)
    print("HƯỚNG DẪN TẠO FILE DOCX MẪU")
    print("=" * 70)
    print("""
Để test đầy đủ 3 trường hợp, tạo file 'test_document_with_charts.docx' với:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 TRƯỜNG HỢP 1: Đồ thị có caption
   
   1. Insert một biểu đồ (Chart) bất kỳ
   2. Thêm paragraph ngay sau chart với text:
      "Hình 1: Biểu đồ doanh thu theo quý"
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 TRƯỜNG HỢP 2: Đồ thị có bảng số liệu gốc
   
   1. Tạo bảng với dữ liệu số (ví dụ):
      ┌────────┬────────┬────────┐
      │ Quý    │ Q1     │ Q2     │
      ├────────┼────────┼────────┤
      │ Doanh  │ 100    │ 150    │
      └────────┴────────┴────────┘
   
   2. Insert chart ngay sau bảng (data table sẽ được tự động detect)
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 TRƯỜNG HỢP 3: Đồ thị được OCR
   
   1. Insert chart có labels/text rõ ràng trên:
      - Trục X, Y
      - Legend
      - Data labels
   
   2. Chart sẽ được OCR để trích xuất text từ các labels này
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 LƯU Ý:
   - Có thể combine cả 3 trường hợp trong 1 chart
   - File cần được save ở cùng thư mục với script này
   - Tên file: test_document_with_charts.docx
    
""")


if __name__ == "__main__":
    # Kiểm tra có file test không
    if not os.path.exists("test_document_with_charts.docx"):
        print("⚠️  Chưa có file test!")
        create_sample_instructions()
    else:
        test_chart_extraction()
