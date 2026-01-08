# RAG Document Summarizer

Ứng dụng tóm tắt tài liệu thông minh sử dụng RAG (Retrieval-Augmented Generation) và AI.

## Tính năng

- **Hỗ trợ nhiều định dạng**: TXT, DOCX, PDF
- **3 chế độ tự động**:
  - Tóm tắt toàn bộ tài liệu
  - Tóm tắt theo chủ đề cụ thể
  - Hỏi đáp về nội dung
- **Giao diện ChatGPT-style**: Trò chuyện tự nhiên với tài liệu
- **Trích dẫn nguồn**: Hiển thị nguồn tham khảo cho mỗi câu trả lời
- **Xử lý tự động**: Tải lên là tự động xử lý, không cần bấm nút

## Công nghệ

**AI & LLM:**

- Google Gemini 2.5 Flash - LLM generation
- Sentence Transformers - Text embeddings (multilingual)

**Vector Database:**

- FAISS - Similarity search & retrieval

**Frontend:**

- Streamlit - Web UI

**Document Processing:**

- python-docx - DOCX parsing
- PyMuPDF - PDF parsing
- Tesseract OCR - Image text extraction (optional)

## Cài đặt

1. **Clone repository**

```bash
git clone <repo-url>
cd summarize-the-file-using-RAG
```

2. **Tạo virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. **Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

4. **Cấu hình API key**

Tạo file `.env` trong thư mục gốc:

```
GEMINI_API_KEY=your_api_key_here
```

5. **Chạy ứng dụng**

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

## Cách sử dụng

1. **Upload tài liệu** - Kéo thả file vào sidebar
2. **Đợi xử lý** - Hệ thống tự động phân tích
3. **Hỏi câu hỏi** - Gõ trực tiếp vào chat
4. **Nhận kết quả** - Tóm tắt/câu trả lời với nguồn trích dẫn

## Cấu trúc dự án

```
├── app.py                 # Main Streamlit app
├── core/
│   ├── loaders/          # Document loaders (TXT, DOCX, PDF)
│   ├── chunking/         # Text splitting
│   ├── embeddings/       # Sentence embeddings
│   ├── vectorstore/      # FAISS vector database
│   ├── retriever/        # RAG retrieval logic
│   └── llm/              # Gemini API integration
├── storage/              # Temporary file storage
└── requirements.txt      # Python dependencies
```

## Tính năng nâng cao

- **Chunked processing** - Xử lý tài liệu lớn theo từng phần
- **Auto retry** - Tự động retry khi API quá tải
- **Cache embeddings** - Lưu cache để tăng tốc
- **Intent classification** - Tự động nhận diện loại câu hỏi
- **Auto file cleanup** - Tự động xóa file sau khi xử lý

## Lưu ý

- File upload tự động xóa sau khi embedding để tiết kiệm dung lượng
- Hệ thống tự động phát hiện intent từ câu hỏi
- Hỗ trợ retry với exponential backoff khi API quá tải (503 errors)
- OCR cho ảnh trong DOCX yêu cầu Tesseract (optional)

## Yêu cầu hệ thống

- Python 3.8+
- 2GB RAM (tối thiểu)
- Internet connection (cho Gemini API)
