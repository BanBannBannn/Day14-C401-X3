# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50 (V1: 50, V2: 50)
- **Tỉ lệ Pass/Fail (V2):** 40 pass / 10 fail (80% / 20%)
- **Điểm RAGAS trung bình:**
    - Faithfulness: V1=0.673 → V2=0.600 (Δ -0.073)
    - Relevancy (hit_rate): 1.0
- **Điểm LLM-Judge trung bình:** V1=4.750 → V2=3.020 (Δ -1.730)
- **Hallucination rate:** V1=0.327 → V2=0.400 (Δ +0.073)
- **Error rate:** V1=0.000 → V2=0.200 (Δ +0.200)
- **Latency (p95):** V1=6262ms → V2=5453ms (Δ -809ms, cải thiện)
- **Latency (p99):** V1=8165ms → V2=5910ms (Δ -2254ms, cải thiện)
- **Auto-gate decision:** ❌ ROLLBACK

> ⚠️ **Lưu ý quan trọng:** avg_score V2 bị ô nhiễm bởi lỗi hạ tầng Judge — toàn bộ ~50 lần gọi Claude Judge trong V2 đều thất bại với lỗi 429 (rate limit OpenRouter free tier). Điểm giảm từ 4.75 → 3.02 phản ánh **lỗi scoring system**, không hoàn toàn là suy giảm chất lượng agent.

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng bị ảnh hưởng | Nguyên nhân dự kiến |
|----------|----------------------|---------------------|
| Claude Judge Rate Limit (429) | 50/50 V2 cases | OpenRouter free tier bị cạn (50 req/day), toàn bộ claude judge call fail → fallback score=3, kéo avg_score xuống |
| Over-cautious refusal (agent) | 8 cases | System prompt V2 kích hoạt fallback "Vinhomes hotline" sai ngữ cảnh cho câu hỏi kỹ thuật hợp lệ |
| Out-of-domain (agent) | 2 cases | Câu hỏi ngoài phạm vi knowledge base hoàn toàn (phim hài lãng mạn, điểm danh học sinh) |

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: avg_score toàn bộ V2 bị kéo xuống 3.02 (scoring system failure)
1. **Symptom:** avg_score giảm -1.730, nhưng các response có nội dung tốt vẫn nhận điểm thấp.
2. **Why 1:** Claude Judge trả về fallback score=3 cho tất cả 50 cases trong V2 do 429 rate limit.
3. **Why 2:** `_handle_disagreement()` trong `llm_judge.py` resolve xung đột bằng cách so sánh độ dài chuỗi reasoning — chuỗi error 429 dài hơn reasoning thật của GPT nên claude fallback luôn được chọn.
4. **Why 3:** OpenRouter free tier giới hạn 50 request/ngày — V2 chạy 50 cases × 1 claude call = vượt hết quota.
5. **Why 4:** Không có retry mechanism hoặc fallback sang model thứ ba khi claude judge fail.
6. **Root Cause:** Bug trong `_handle_disagreement`: dùng string length làm tie-breaker vô tình ưu tiên error message dài thay vì real reasoning. Kết hợp với rate limit → tất cả V2 scores bị kéo về 3.0.

### Case #2: "Ưu điểm chính của việc sử dụng vector store là gì?" — Agent trả lời "Xin lỗi, không có thông tin" (error_rate case)
1. **Symptom:** Agent từ chối trả lời câu hỏi kỹ thuật hợp lệ trong phạm vi RAG domain, redirect "Vinhomes hotline".
2. **Why 1:** System prompt V2 có fallback message cứng nhắc với nhãn "Vinhomes" cho mọi câu hỏi không retrieve được.
3. **Why 2:** Retriever không tìm được document về vector store vì query embedding không khớp.
4. **Why 3:** Knowledge base thiếu document về vector store advantages — gap trong corpus.
5. **Why 4:** Không có coverage audit giữa test set và knowledge base trước khi chạy benchmark V2.
6. **Root Cause:** System prompt V2 hardcode "Vinhomes" vào generic fallback, biến mọi retrieval miss thành "xin lỗi Vinhomes" dù câu hỏi là kỹ thuật — V1 không có pattern này.

### Case #3: "Trí tuệ nhân tạo AI có thể giúp tìm kiếm các bộ phim hài lãng mạn không?" — Out-of-domain, faithfulness=0.4146 (thấp nhất)
1. **Symptom:** Agent không nhận diện câu hỏi off-topic, chạy toàn bộ retrieval pipeline rồi trả fallback.
2. **Why 1:** Không có bước intent classification/domain guard trước retrieval.
3. **Why 2:** System prompt V2 không định nghĩa rõ phạm vi domain được phép trả lời.
4. **Why 3:** Khi V2 tối ưu latency, bước pre-check domain bị loại bỏ.
5. **Why 4:** Test set chứa câu hỏi out-of-domain nhưng không có guard để xử lý sớm, tiêu tốn quota judge.
6. **Root Cause:** Thiếu domain guard ở đầu pipeline — câu hỏi off-topic tiêu tốn judge API quota và vẫn nhận điểm thấp, kép với rate limit làm đội chi phí lỗi.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **[Bug fix - ưu tiên cao]** Sửa `_handle_disagreement()` trong `llm_judge.py`: không dùng string length làm tie-breaker — thay bằng chọn điểm trung bình hoặc điểm của judge không phải fallback.
- [ ] **[Infra]** Thêm retry + fallback model khi Claude Judge 429: thử lại 1 lần, nếu vẫn fail thì chỉ dùng GPT score thay vì fallback=3.
- [ ] **[Agent]** Sửa system prompt V2: bỏ hardcode "Vinhomes" trong generic fallback message, tách fallback theo domain (kỹ thuật RAG vs sản phẩm).
- [ ] **[Pipeline]** Thêm domain guard đầu pipeline: detect câu hỏi hoàn toàn off-domain (phim, học sinh...) trước khi gọi retrieval.
- [ ] **[Test]** Audit coverage test set vs knowledge base: loại câu hỏi không có ground truth trong corpus hoặc bổ sung document tương ứng.
- [ ] **[Decision]** Rollback V2 system prompt, giữ lại cải tiến latency (p95 -809ms) vì không gây regression chất lượng.
