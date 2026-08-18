# Individual Reflection — Lab 18

**Tên:** Duong Kien
**Module phụ trách:** M1–M5 (toàn bộ 5 modules, bài cá nhân)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 Chunking (semantic, hierarchical, structure-aware), M2 Hybrid Search (BM25 + Dense + RRF), M3 Reranking (CrossEncoder), M4 RAGAS Eval + Failure Analysis, M5 Enrichment (combined single-call mode).
- Các hàm/class chính đã viết: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()`, `CrossEncoderReranker`, `evaluate_ragas()`, `failure_analysis()`, `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `_enrich_single_call()`.
- Số tests pass: 37/37 (`pytest tests/ -v`).

## 2. Kiến thức học được

- Khái niệm mới nhất: RRF (Reciprocal Rank Fusion) để hợp nhất kết quả BM25 và Dense mà không cần chuẩn hoá score trực tiếp — dùng rank thay vì score tuyệt đối để tránh lệch thang đo giữa 2 hệ thống.
- Điều bất ngờ nhất: sau khi chạy `python -m src.m1_chunking`, semantic chunking tạo ra **208 chunks** (avg 99 ký tự) từ cùng bộ tài liệu mà basic chunking chỉ tạo **51 chunks** (avg 410 ký tự) — chênh lệch gấp 4 lần vì threshold cosine similarity 0.85 khá chặt, mỗi khi câu tiếp theo lệch topic nhẹ là tách chunk mới. Hierarchical (110 chunks, avg 189, 11 parents) và structure-aware (106 chunks, avg 196, max 788) cho kích thước đồng đều hơn, phù hợp production hơn semantic thuần.
- Kết nối với bài giảng: đúng như lecture nói CrossEncoder rerank rất nhạy với độ liên quan ngữ nghĩa — test thực tế cho thấy câu đúng chủ đề ("nghỉ phép") được điểm 0.99, còn câu lệch chủ đề ("thử việc") chỉ 0.02 và câu hoàn toàn không liên quan ("mật khẩu") gần như 0 — reranker phân biệt rất rõ ràng dù retrieval trước đó (BM25/Dense) có thể trả cả 3 docs với score gần nhau.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: chạy `pipeline.py` bị `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3` khi print tiếng Việt có emoji trên console Windows (cp1252) — lỗi này chỉ xảy ra khi redirect output ra file, không xảy ra khi chạy trực tiếp trong terminal.
- Cách giải quyết: set biến môi trường `PYTHONIOENCODING=utf-8` trước khi chạy script để buộc stdout dùng UTF-8 thay vì codepage mặc định của Windows.
- Khó khăn thứ hai (nghiêm trọng hơn): Groq free-tier có giới hạn **200,000 token/ngày (TPD)** cho mỗi model. Model mặc định `openai/gpt-oss-120b` cạn quota ngay giữa quá trình chạy M5 enrichment (117 chunks × 1 call/chunk) và RAGAS eval (80 job = 4 metric × 20 câu), gây ra hàng loạt lỗi `RateLimitError 429`.
- Cách giải quyết: đổi tạm sang model `openai/gpt-oss-20b` (quota TPD độc lập) để tiếp tục chạy — nhưng model yếu hơn, JSON structured output cho enrichment thường xuyên fail validation (`json_validate_failed`), phải dựa vào fallback graceful đã code sẵn (`enrich_chunks()` trả về context rỗng khi API fail thay vì crash). Ngay cả `gpt-oss-20b` cũng cạn quota giữa RAGAS eval (~35/80 job lỗi 429) — RAGAS tự loại các job lỗi khỏi trung bình thay vì tính là 0, nên vẫn ra được điểm cuối nhưng độ tin cậy thấp hơn.
- Thời gian debug: ~45 phút (bao gồm restart Docker Desktop vì container Qdrant bị treo giữa chừng, và 2 lần thử model khác nhau).

## 4. Nếu làm lại

- Sẽ làm khác: thêm exponential backoff + retry cho các lệnh gọi Groq API thay vì fail nhanh và fallback ngay — với daily quota, retry có thể giúp hoàn thành nhiều job hơn thay vì bỏ cuộc ở request đầu tiên gặp 429.
- Sẽ chạy M5 enrichment và RAGAS eval vào đầu ngày (quota mới) thay vì sau khi đã dùng nhiều quota cho debug/test lặp lại nhiều lần trong session.
- Module muốn thử tiếp: M2 Hybrid Search — muốn thử nghiệm nhiều giá trị RRF k khác nhau và so sánh context_recall, vì đây là điểm mạnh nhất trong kết quả hiện tại (Production RAG có context_recall 0.80 > Naive baseline 0.75).

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | - (bài cá nhân) |
| Problem solving | 5 |
