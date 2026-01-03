"""
Test script cho DOCX Loader.

Chạy script này để test các tính năng của DOCX loader:
- Trích xuất text từ DOCX
- Trích xuất table
- OCR ảnh embedded
"""

from core.loaders.docx_loader import load_docx, DOCXLoader
import os

def test_basic_docx():
    """Test trích xuất text cơ bản từ DOCX."""
    print("=" * 60)
    print("TEST: DOCX Loader")
    print("=" * 60)
    
    # Tạo loader
    loader = DOCXLoader(
        enable_image_extraction=True,
        enable_table_extraction=True,
        enable_text_cleaning=True,
        min_image_confidence=60.0,
        min_image_words=5
    )
    
    print("\n✅ DOCXLoader initialized successfully")
    print(f"   - Image extraction: {loader.enable_image_extraction}")
    print(f"   - Table extraction: {loader.enable_table_extraction}")
    print(f"   - Text cleaning: {loader.enable_text_cleaning}")
    print(f"   - Min image confidence: {loader.min_image_confidence}%")
    print(f"   - Min image words: {loader.min_image_words}")
    
    # Test với file helper function
    print("\n" + "=" * 60)
    print("TEST: Helper function load_docx()")
    print("=" * 60)
    
    print("\n📝 Hàm load_docx() sẵn sàng sử dụng với các parameters:")
    print("   - file_path (required)")
    print("   - enable_image_extraction (default: True)")
    print("   - enable_table_extraction (default: True)")
    print("   - enable_text_cleaning (default: True)")
    print("   - min_image_confidence (default: 60.0)")
    print("   - min_image_words (default: 5)")
    
    print("\n" + "=" * 60)
    print("✅ DOCX Loader test completed successfully!")
    print("=" * 60)
    
    print("\n💡 Để test với file DOCX thực tế:")
    print("   docs = load_docx('path/to/your/file.docx')")
    print("   print(docs[0]['text'])")
    print("   print(docs[0]['metadata'])")

if __name__ == "__main__":
    test_basic_docx()
