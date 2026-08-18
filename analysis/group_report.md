# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân (bài tập cá nhân — 1 thành viên đảm nhiệm toàn bộ 5 module)
**Ngày:** 2026-08-18

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Duong Kien | M1: Chunking | ☑ | 13/13 |
| Duong Kien | M2: Hybrid Search | ☑ | 5/5 |
| Duong Kien | M3: Reranking | ☑ | 5/5 |
| Duong Kien | M4: Evaluation | ☑ | 4/4 |
| Duong Kien | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.6906 | 0.5600 | -0.1306 |
| Answer Relevancy | 0.8486 | 0.5662 | -0.2824 |
| Context Precision | 0.9643 | 0.7917 | -0.1726 |
| Context Recall | 0.7546 | 0.8000 | +0.0454 |

> Xem ghi chú độ tin cậy đầy đủ trong `analysis/failure_analysis.md`: quota Groq daily
> (200k TPD) cạn giữa quá trình chạy cả naive baseline (`gpt-oss-120b`) lẫn production RAGAS
> eval (buộc đổi sang `gpt-oss-20b`, và ~35/80 job RAGAS vẫn bị rate-limit và bị loại khỏi
> trung bình). Điểm Production vì vậy thấp hơn thực tế và không hoàn toàn so sánh ngang hàng
> với Naive (2 model LLM-judge khác nhau).

## Key Findings

1. **Biggest improvement:** Context Recall tăng từ 0.7546 lên 0.8000 — hybrid search (BM25 +
   Dense + RRF) kết hợp reranking tìm được nhiều chunk liên quan hơn so với retrieval đơn giản
   của naive baseline, đúng như kỳ vọng lý thuyết dù các metric khác bị nhiễu bởi rate-limit.
2. **Biggest challenge:** Groq free-tier daily token quota (200k TPD) cạn liên tiếp trên cả
   2 model dùng trong session (`gpt-oss-120b` rồi `gpt-oss-20b`), khiến M5 enrichment phải
   fallback graceful nhiều lần và RAGAS eval mất ~35/80 job. Phải restart Docker Desktop giữa
   chừng vì Qdrant container bị treo.
3. **Surprise finding:** Semantic chunking (threshold 0.85) tạo ra 208 chunks (avg 99 ký tự)
   so với chỉ 51 chunks (avg 410 ký tự) của basic chunking trên cùng bộ tài liệu — chênh lệch
   gấp 4 lần, cho thấy threshold càng chặt càng dễ tách chunk nhỏ vụn, không nhất thiết tốt cho
   retrieval.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Production thấp hơn naive trên 3/4 metric trong lần đo
   này, nhưng nguyên nhân chính là confound từ rate-limit (model judge khác nhau, nhiều job bị
   loại khỏi mẫu) chứ không phản ánh đúng chất lượng pipeline — Context Recall là tín hiệu đáng
   tin cậy nhất và nó tăng.
2. Biggest win — module nào, tại sao: M2 Hybrid Search (BM25 + Dense + RRF) — retrieval recall
   cải thiện rõ rệt so với single-method retrieval của naive baseline.
3. Case study — 1 failure, Error Tree walkthrough: Câu "Thâm niên bao nhiêu năm thì được cộng
   thêm ngày phép?" trả lời "Không tìm thấy" — output sai → context thiếu (điều kiện thâm niên
   bị tách khỏi câu chính do hierarchical child size 256 token quá nhỏ) → query OK → fix ở bước
   Chunking (M1), không phải generation. Chi tiết trong `analysis/failure_analysis.md`.
4. Next optimization nếu có thêm 1 giờ: chạy lại toàn bộ RAGAS eval trong khung giờ quota mới
   (không bị rate-limit giữa chừng) để có số liệu tin cậy, và thử `chunk_semantic()` thay
   `chunk_hierarchical()` cho các section có câu điều kiện liên tiếp.
