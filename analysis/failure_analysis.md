# Failure Analysis — Lab 18: Production RAG

**Thành viên:** Duong Kien

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6906 | 0.5600 | -0.1306 |
| Answer Relevancy | 0.8486 | 0.5662 | -0.2824 |
| Context Precision | 0.9643 | 0.7917 | -0.1726 |
| Context Recall | 0.7546 | 0.8000 | +0.0454 |

> ⚠️ **Lưu ý về độ tin cậy của phép so sánh:** Trong lần chạy này, Groq free-tier daily token
> quota (200k TPD) bị cạn ngay trong quá trình chạy `naive_baseline.py` với model
> `openai/gpt-oss-120b`. Khi chạy `pipeline.py` (production), model đã phải đổi tạm sang
> `openai/gpt-oss-20b` (quota riêng) để tránh chờ reset — model judge yếu hơn, và trong quá
> trình RAGAS evaluate (80 job = 4 metric × 20 câu) có ~35 job bị `RateLimitError` khi quota
> `gpt-oss-20b` cũng cạn giữa chừng; các job lỗi bị RAGAS loại khỏi trung bình (không tính là 0).
> Vì vậy điểm Production ở trên **thấp hơn thực tế** và không hoàn toàn so sánh ngang hàng với
> Naive (đánh giá bởi 2 model LLM-judge khác nhau, với số job hợp lệ khác nhau). Xu hướng đáng
> tin cậy nhất là **Context Recall tăng** (hybrid search + reranking tìm được nhiều chunk liên
> quan hơn) — đúng như kỳ vọng lý thuyết. Để so sánh công bằng, cần chạy lại cả 2 pipeline cùng
> một model judge, trong cùng ngày quota mới.

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Got:** "3 ngày làm việc."
- **Worst metric:** faithfulness (NaN — job bị rate-limit khi RAGAS chấm câu này)
- **Error Tree:** Output sai? → Không rõ (job lỗi) → Context đúng? → Không xác định được → Query OK? → Có (câu hỏi rõ ràng)
- **Root cause:** RAGAS không đánh giá được câu này do quota cạn giữa chừng, không phải lỗi hệ thống RAG.
- **Suggested fix:** Chạy lại eval với quota còn đủ, hoặc tăng `RunConfig.max_retries`/giảm `max_workers` để tránh burst request.

### #2
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Got:** "Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm."
- **Worst metric:** faithfulness (NaN — cùng lý do rate-limit)
- **Error Tree:** Output sai? → Không rõ (job lỗi) → Context đúng? → Không xác định
- **Root cause:** Job bị rate-limit, không đánh giá được.
- **Suggested fix:** Chạy lại eval.

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai (trả lời "không tìm thấy") → Context đúng? Cần kiểm tra thủ công — nhiều khả năng chunk chứa ngưỡng phê duyệt theo mức tiền không được retrieve đúng, hoặc chunk bị enrichment/hierarchical chunking cắt mất bảng ngưỡng phê duyệt → Query OK, câu hỏi rõ ràng.
- **Root cause:** Context recall thất bại cho câu hỏi có điều kiện số học (ngưỡng tiền) — loại câu hỏi này cần chunk chứa nguyên bảng/threshold, dễ bị cắt rời khi chunking theo kích thước cố định (hierarchical child 256 token).
- **Suggested fix:** Với các bảng ngưỡng phê duyệt/số liệu, ưu tiên `chunk_structure_aware()` (giữ nguyên section) thay vì hierarchical, hoặc tăng child size cho các section dạng bảng.

### #4
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0, avg_score = 0.33
- **Error Tree:** Output sai (không tìm thấy dù test set kỳ vọng có câu trả lời) → Context đúng? Nhiều khả năng chunk IT Policy về đổi mật khẩu không nằm trong top-K sau rerank → Query OK.
- **Root cause:** Context recall/precision — câu hỏi ngắn, ít từ khóa đặc trưng ("mật khẩu", "đổi") dễ bị BM25/Dense đánh giá thấp hơn các chunk khác có nhiều từ trùng hơn.
- **Suggested fix:** Tăng `BM25_TOP_K`/`DENSE_TOP_K` trước rerank, hoặc thêm HyQA (M5) cho chunk IT policy để bridge vocabulary gap.

### #5
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0, avg_score = 0.50
- **Error Tree:** Output sai (không tìm thấy) → Context đúng? Câu hỏi liên quan trực tiếp tới chunk "12 ngày phép + thâm niên" (đã thấy trong test M5) — có khả năng bị hierarchical chunking tách câu điều kiện thâm niên ra khỏi câu chính về số ngày phép → Query OK.
- **Root cause:** Chunking tách rời 2 câu liên quan ngữ nghĩa (số ngày phép cơ bản + điều kiện tăng theo thâm niên) thành 2 chunk khác nhau, retrieval chỉ lấy 1 trong 2.
- **Suggested fix:** Giảm `HIERARCHICAL_CHILD_SIZE` hoặc dùng `chunk_semantic()` với threshold thấp hơn để nhóm các câu liên quan ngữ nghĩa lại cùng 1 chunk.

## Case Study (cho presentation)

**Question chọn phân tích:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"

**Error Tree walkthrough:**
1. Output đúng? → Sai, trả lời "Không tìm thấy" trong khi test set có ground truth.
2. Context đúng? → Nhiều khả năng sai/thiếu — chunk chứa điều kiện thâm niên bị tách khỏi chunk chính do child size 256 token quá nhỏ cho đoạn văn 2 câu liên quan.
3. Query rewrite OK? → Có, câu hỏi rõ ràng, không cần rewrite.
4. Fix ở bước: **Chunking (M1)** — đây là lỗi retrieval do chiến lược chunk, không phải lỗi generation/LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. Chạy lại toàn bộ RAGAS eval trong khung giờ quota mới (không bị rate-limit giữa chừng) để có số liệu tin cậy, so sánh naive vs production công bằng với cùng 1 model judge.
2. Thử `chunk_semantic()` thay `chunk_hierarchical()` làm chunking chính trong `pipeline.py` cho các document dạng policy có câu điều kiện liên tiếp (nghỉ phép theo thâm niên, ngưỡng phê duyệt theo số tiền), so sánh context_recall.
3. Thêm retry với backoff (thay vì để RAGAS/enrichment fail rồi bỏ qua) để tận dụng hết quota còn lại thay vì mất job giữa chừng.
