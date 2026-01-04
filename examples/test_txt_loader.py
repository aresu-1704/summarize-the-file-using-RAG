"""
Script demo để test TXT Loader với nhiều loại file và encoding.

Test cases:
1. Plain text với encoding khác nhau
2. Markdown file
3. Log file
4. File với URLs và emails
5. File với nhiều ngôn ngữ
"""

import sys
import os

# Thêm đường dẫn đến thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.loaders.txt_loader import TXTLoader


def create_test_files():
    """Tạo các file test với các trường hợp khác nhau."""
    
    # 1. Plain text UTF-8
    with open("test_plain_utf8.txt", "w", encoding="utf-8") as f:
        f.write("""Đây là file text tiếng Việt có dấu.

Paragraph 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.

Paragraph 2: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Paragraph 3: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
""")
    
    # 2. Markdown file
    with open("test_markdown.md", "w", encoding="utf-8") as f:
        f.write("""# Tiêu đề chính

## Section 1

Đây là nội dung section 1 với:

- Item 1
- Item 2
- Item 3

## Section 2

**Bold text** và *italic text*.

### Subsection 2.1

```python
def hello():
    print("Hello, World!")
```

## Section 3

Link: [Google](https://www.google.com)

""")
    
    # 3. Log file
    with open("test_app.log", "w", encoding="utf-8") as f:
        f.write("""2024-01-04 10:30:45 INFO Application started
2024-01-04 10:30:46 INFO Loading configuration
2024-01-04 10:30:47 WARNING Configuration file not found, using defaults
2024-01-04 10:30:48 INFO Server listening on port 8080
2024-01-04 10:35:12 ERROR Database connection failed: timeout
2024-01-04 10:35:13 INFO Retrying connection...
2024-01-04 10:35:15 INFO Connected to database successfully
2024-01-04 11:00:00 DEBUG Processing request [ID: 12345]
2024-01-04 11:00:01 DEBUG Request completed [ID: 12345, duration: 245ms]
""")
    
    # 4. File với URLs và emails
    with open("test_urls_emails.txt", "w", encoding="utf-8") as f:
        f.write("""Liên hệ: support@example.com

Website: https://www.example.com

Tài liệu: https://docs.example.com/api

Email admin: admin@company.com

More info at www.info.com
""")
    
    # 5. File đa ngôn ngữ
    with open("test_multilang.txt", "w", encoding="utf-8") as f:
        f.write("""English: Hello, World!
Tiếng Việt: Xin chào thế giới!
日本語: こんにちは世界！
한국어: 안녕하세요 세계!
中文: 你好世界！
Русский: Привет, мир!
العربية: مرحبا بالعالم!
""")
    
    print("✅ Đã tạo các file test:")
    print("   - test_plain_utf8.txt")
    print("   - test_markdown.md")
    print("   - test_app.log")
    print("   - test_urls_emails.txt")
    print("   - test_multilang.txt")


def test_basic_loading():
    """Test load file cơ bản."""
    print("\n" + "=" * 70)
    print("TEST 1: LOAD FILE CƠ BẢN")
    print("=" * 70)
    
    loader = TXTLoader()
    docs = loader.load_txt("test_plain_utf8.txt")
    
    if docs:
        print(f"\n✅ Đã load file:")
        print(f"   Encoding: {docs[0]['metadata']['encoding']}")
        print(f"   File type: {docs[0]['metadata']['file_type']}")
        print(f"   Lines: {docs[0]['metadata']['num_lines']}")
        print(f"   Words: {docs[0]['metadata']['num_words']}")
        print(f"\n📝 Nội dung:")
        print("─" * 70)
        print(docs[0]['text'][:200] + "...")
        print("─" * 70)


def test_markdown_detection():
    """Test phát hiện markdown structure."""
    print("\n" + "=" * 70)
    print("TEST 2: PHÁT HIỆN CẤU TRÚC MARKDOWN")
    print("=" * 70)
    
    loader = TXTLoader(enable_structure_detection=True)
    docs = loader.load_txt("test_markdown.md")
    
    if docs:
        metadata = docs[0]['metadata']
        print(f"\n✅ Phát hiện được:")
        print(f"   File type: {metadata['file_type']}")
        print(f"   Số headers: {metadata['num_headers']}")
        print(f"   Số sections: {metadata['num_sections']}")
        print(f"   Số lists: {metadata['num_lists']}")
        print(f"   Số code blocks: {metadata['num_code_blocks']}")


def test_log_file():
    """Test load log file."""
    print("\n" + "=" * 70)
    print("TEST 3: PHÁT HIỆN LOG FILE")
    print("=" * 70)
    
    loader = TXTLoader()
    docs = loader.load_txt("test_app.log")
    
    if docs:
        metadata = docs[0]['metadata']
        print(f"\n✅ Phát hiện log file:")
        print(f"   File type: {metadata['file_type']}")
        print(f"   Lines: {metadata['num_lines']}")


def test_url_email_removal():
    """Test xóa URLs và emails."""
    print("\n" + "=" * 70)
    print("TEST 4: XÓA URLs VÀ EMAILS")
    print("=" * 70)
    
    # Load không xóa
    loader1 = TXTLoader(remove_urls=False, remove_emails=False)
    docs1 = loader1.load_txt("test_urls_emails.txt")
    
    # Load có xóa
    loader2 = TXTLoader(remove_urls=True, remove_emails=True)
    docs2 = loader2.load_txt("test_urls_emails.txt")
    
    print("\n📝 KHÔNG XÓA:")
    print("─" * 70)
    print(docs1[0]['text'])
    print("─" * 70)
    
    print("\n🧹 CÓ XÓA:")
    print("─" * 70)
    print(docs2[0]['text'])
    print("─" * 70)


def test_multilingual():
    """Test file đa ngôn ngữ."""
    print("\n" + "=" * 70)
    print("TEST 5: FILE ĐA NGÔN NGỮ")
    print("=" * 70)
    
    loader = TXTLoader()
    docs = loader.load_txt("test_multilang.txt")
    
    if docs:
        metadata = docs[0]['metadata']
        print(f"\n✅ Load thành công:")
        print(f"   Encoding: {metadata['encoding']}")
        print(f"   Lines: {metadata['num_lines']}")
        print(f"\n📝 Nội dung:")
        print("─" * 70)
        print(docs[0]['text'])
        print("─" * 70)


def test_encoding_detection():
    """Test auto-detect encoding."""
    print("\n" + "=" * 70)
    print("TEST 6: AUTO-DETECT ENCODING")
    print("=" * 70)
    
    # Tạo file với các encoding khác nhau
    test_files = []
    
    # UTF-8
    with open("test_utf8.txt", "w", encoding="utf-8") as f:
        f.write("UTF-8: Tiếng Việt có dấu ăâêôơư")
    test_files.append(("test_utf8.txt", "utf-8"))
    
    # UTF-8 with BOM
    with open("test_utf8_bom.txt", "w", encoding="utf-8-sig") as f:
        f.write("UTF-8 BOM: Tiếng Việt có dấu")
    test_files.append(("test_utf8_bom.txt", "utf-8-sig"))
    
    # Latin-1
    with open("test_latin1.txt", "w", encoding="latin-1") as f:
        f.write("Latin-1: Hello World")
    test_files.append(("test_latin1.txt", "latin-1"))
    
    # Test auto-detection
    loader = TXTLoader(auto_detect_encoding=True)
    
    for filename, expected_enc in test_files:
        docs = loader.load_txt(filename)
        detected_enc = docs[0]['metadata']['encoding']
        print(f"\n   {filename}:")
        print(f"      Expected: {expected_enc}")
        print(f"      Detected: {detected_enc}")
        print(f"      Status: {'✅' if expected_enc in detected_enc else '⚠️'}")


def cleanup_test_files():
    """Xóa các file test."""
    test_files = [
        "test_plain_utf8.txt",
        "test_markdown.md",
        "test_app.log",
        "test_urls_emails.txt",
        "test_multilang.txt",
        "test_utf8.txt",
        "test_utf8_bom.txt",
        "test_latin1.txt"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\n🧹 Đã xóa các file test")


def main():
    """Chạy tất cả tests."""
    print("=" * 70)
    print("DEMO TXT LOADER - RAG APPLICATION")
    print("=" * 70)
    
    try:
        # Tạo test files
        create_test_files()
        
        # Run tests
        test_basic_loading()
        test_markdown_detection()
        test_log_file()
        test_url_email_removal()
        test_multilingual()
        test_encoding_detection()
        
        print("\n" + "=" * 70)
        print("✅ TẤT CẢ TESTS HOÀN THÀNH")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_test_files()


if __name__ == "__main__":
    main()
