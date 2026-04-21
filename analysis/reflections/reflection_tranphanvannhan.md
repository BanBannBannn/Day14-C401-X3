# 📋 Báo cáo Cá nhân - Lab Day 14: AI Evaluation Factory

**Họ tên:** Trần Phan Văn Nhân

**Mã HV:** 2A202600301

**Vai trò:** Data Collection và System Prompt

**Ngày hoàn thành:** 21 tháng 4, 2026

---

## 1️⃣ Đóng góp Kỹ thuật (Engineering Contribution) - 15 điểm

### 1.1 Triển khai synthetic_gen.py
**Mô tả:** Tôi chịu trách nhiệm thiết kế System Prompt và xây dựng pipeline sinh dữ liệu tổng hợp (synthetic QA) từ tài liệu thô phục vụ RAG evaluation.

#### Các thành phần chính:
```python
✅ def _build_prompt()          # Thiết kế system prompt cho Gemini
✅ def _generate_with_gemini()  # Gọi Gemini API với fallback
✅ def _extract_json_array()    # Parse JSON từ LLM response
✅ async def generate_qa_from_text()  # Orchestrator async
✅ async def main()             # Pipeline đọc file → sinh QA → lưu JSONL
```

### 1.2 Công nghệ được áp dụng

- **Model:** Google Gemini (`gemini-2.5-flash-lite` theo env `GEMINI_MODEL`)
- **Output format:** JSONL — mỗi QA pair là một JSON object riêng biệt, lưu tại `data/golden_set.jsonl`
- **Input:** Các file `.md` trong `data/raw/`, đọc tuần tự theo thứ tự sorted

- **Async Pattern:** Dùng `asyncio.to_thread` để chạy hàm Gemini đồng bộ trong thread pool mà không block event loop:
  ```python
  model_text = await asyncio.to_thread(_generate_with_gemini, client, prompt)
  ```

- **Service Tier Fallback:** Ưu tiên gọi với `service_tier: "flex"` (chi phí thấp hơn), tự động fallback về config mặc định nếu lỗi:
  ```python
  try:
      response = client.models.generate_content(..., config={**base_config, "service_tier": "flex"})
  except Exception:
      response = client.models.generate_content(..., config=base_config)
  ```

- **Cấu hình linh hoạt qua Environment Variables:**
  ```
  GEMINI_MODEL       → tên model (default: gemini-2.5-flash-lite)
  QA_PAIRS_PER_DOC  → số cặp QA mỗi doc (default: 12)
  GEMINI_API_KEY / GOOGLE_API_KEY → xác thực
  ```

### 1.3 Schema QA Pair được thiết kế:
```json
{
  "question": "string (tiếng Việt)",
  "expected_answer": "string (tiếng Việt)",
  "context": "trích dẫn hoặc paraphrase từ tài liệu",
  "metadata": {
    "difficulty": "easy|medium|hard",
    "type": "fact-check|reasoning|adversarial|out-of-context",
    "source_doc": "tên file .md",
    "doc_id": 0
  }
}
```

### 1.4 Code Quality & Best Practices:
- ✅ Type hints đầy đủ: `List[Dict[str, Any]]`, `Tuple` → IDE autocomplete tốt
- ✅ Graceful import handling: `try/except ImportError` cho `google-genai`
- ✅ JSON parsing robust: thử `json.loads` trực tiếp, fallback sang regex nếu fail
- ✅ Normalization: lọc bỏ pair thiếu field bắt buộc, tự động thêm `source_doc` và `doc_id`
- ✅ Modular design: mỗi hàm có trách nhiệm rõ ràng, dễ test độc lập

---

## 2️⃣ Technical Depth - 15 điểm

### 2.1 Hiểu về Synthetic Data cho RAG Evaluation

**Tại sao cần Synthetic QA?**
- Dữ liệu labeled thủ công tốn kém và chậm
- Synthetic QA từ tài liệu thực → golden set đáng tin cậy để đánh giá retrieval + generation
- Có thể scale lên hàng trăm tài liệu với chi phí thấp

**Đa dạng loại câu hỏi:**
| Loại | Mô tả | Mục đích |
|------|-------|----------|
| `fact-check` | Câu hỏi dữ kiện cụ thể | Test retrieval chính xác |
| `reasoning` | Câu hỏi suy luận từ nhiều phần | Test khả năng tổng hợp |
| `adversarial` | Câu hỏi đánh lừa nhưng vẫn trả lời được | Test độ robust |
| `out-of-context` | Câu hỏi ngoài phạm vi tài liệu | Test hallucination avoidance |

### 2.2 Prompt Engineering cho LLM Judge

**Nguyên tắc thiết kế `_build_prompt()`:**
- **Explicit schema:** Định nghĩa chính xác JSON object cần trả về, kể cả kiểu dữ liệu
- **Ràng buộc ngôn ngữ:** Rule 1 — `question`, `expected_answer`, `context` phải bằng tiếng Việt
- **Edge case specification:** Rule 3 — `out-of-context` thì `context` là chuỗi rỗng
- **Adversarial requirement:** Rule 4 — bắt buộc có ít nhất 1 câu hỏi adversarial
- **Output constraint:** Rule 6 — "Return JSON only. No markdown fences, no extra keys."

**Cài đặt Generation:**
```python
base_config = {
    "temperature": 0.9,           # Đủ đa dạng, không quá ngẫu nhiên
    "response_mime_type": "application/json",  # Force JSON output
    "max_output_tokens": 4096,
}
```

### 2.3 asyncio.to_thread — Tích hợp Sync API vào Async Pipeline

**Vấn đề:** `google-genai` client là synchronous, nhưng pipeline cần async để scale.

**Giải pháp:** `asyncio.to_thread` chạy hàm sync trong thread pool executor mà không block event loop chính:
```python
model_text = await asyncio.to_thread(_generate_with_gemini, client, prompt)
```

**Lợi ích:** Có thể mở rộng sau thành `asyncio.gather` để xử lý nhiều tài liệu song song nếu cần.

### 2.4 JSONL Format — Tại Sao Không Dùng JSON Thông Thường?

- **Streaming-friendly:** Đọc từng dòng mà không cần load toàn bộ file vào RAM
- **Append-safe:** Dễ thêm record mới mà không cần parse toàn bộ file
- **Compatible:** Tương thích với các ML frameworks (HuggingFace datasets, LangChain, etc.)

---

## 3️⃣ Problem Solving - 10 điểm

### 3.1 Vấn đề 1: JSON Parsing từ LLM Response (CRITICAL)

**Vấn đề:** Dù đã yêu cầu "Return JSON only", Gemini đôi khi trả về markdown code block:
```
```json
[{"question": "...", ...}]
```
```

**Giải pháp — 2 lớp phòng thủ:**
```python
def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\\n", "", stripped)
        stripped = re.sub(r"\\n```$", "", stripped)

    try:
        data = json.loads(stripped)          # Lớp 1: parse trực tiếp
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))    # Lớp 2: regex tìm array

    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array")

    return [item for item in data if isinstance(item, dict)]
```

---

### 3.2 Vấn đề 2: Gemini API Service Tier Failure

**Vấn đề:** Gọi với `service_tier: "flex"` (chi phí thấp, latency cao) có thể bị lỗi trong một số điều kiện:
```python
# ❌ Không có fallback → crash toàn bộ pipeline
response = client.models.generate_content(..., config={**base_config, "service_tier": "flex"})
```

**Giải pháp:** Try/except với fallback sang default tier:
```python
try:
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt,
        config={**base_config, "service_tier": "flex"},
    )
except Exception:
    response = client.models.generate_content(   # Fallback: default tier
        model=MODEL_NAME, contents=prompt,
        config=base_config,
    )
```

**Ý tưởng:** Ưu tiên flex để tối ưu chi phí, đảm bảo pipeline không bị gián đoạn dù API có lỗi.

---

### 3.3 Vấn đề 3: Response Normalization — Pair Thiếu Field

**Vấn đề:** Gemini đôi khi sinh ra object thiếu key bắt buộc, gây lỗi khi sử dụng downstream:
```python
# Pair thiếu "context" → downstream code crash khi access pair["context"]
{"question": "...", "expected_answer": "..."}
```

**Giải pháp:** Lọc bỏ pair không hợp lệ và inject metadata tự động:
```python
normalized: List[Dict[str, Any]] = []
for pair in pairs:
    if not all(key in pair for key in ["question", "expected_answer", "context"]):
        continue                            # Bỏ qua pair thiếu field
    pair.setdefault("metadata", {})
    pair["metadata"]["source_doc"] = doc_name   # Inject provenance
    pair["metadata"]["doc_id"] = doc_id
    normalized.append(pair)
```

---

### 3.4 Vấn đề 4: Xác Thực API Key Linh Hoạt

**Vấn đề:** Google cung cấp 2 tên env var khác nhau cho cùng API key (`GEMINI_API_KEY` vs `GOOGLE_API_KEY`), gây nhầm lẫn khi setup.

**Giải pháp:** Hỗ trợ cả hai với fallback rõ ràng:
```python
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running this script.")
```

---

## 4️⃣ Kết quả Định lượng

| Chỉ số | Giá trị | Status |
|-------|--------|--------|
| **Model sử dụng** | gemini-2.5-flash-lite | ✅ Cấu hình qua env |
| **QA pairs mỗi doc** | 12 (default) | ✅ Cấu hình qua env |
| **Loại câu hỏi** | 4 types | ✅ fact-check / reasoning / adversarial / out-of-context |
| **JSON Parsing** | 2-layer fallback | ✅ Regex backup nếu parse trực tiếp fail |
| **API Failure Recovery** | Service tier fallback | ✅ flex → default tự động |
| **Output format** | JSONL | ✅ `data/golden_set.jsonl` |
| **Ngôn ngữ QA** | Tiếng Việt | ✅ Ràng buộc trong system prompt |

---

## 5️⃣ Git Contributions

```bash
# Main commits:
1. Add _build_prompt(): Vietnamese QA schema với 4 loại câu hỏi
2. Add _generate_with_gemini(): Gemini API call với flex tier + fallback
3. Add _extract_json_array(): 2-layer JSON parsing (direct + regex)
4. Add generate_qa_from_text(): async orchestrator với asyncio.to_thread
5. Add main(): pipeline đọc .md → sinh QA → lưu JSONL
6. Add env-based config: GEMINI_MODEL, QA_PAIRS_PER_DOC, API keys
```

---

## 6️⃣ Reflection & Learning

### Những gì học được:
1. **Prompt engineering** cho data generation — schema rõ ràng + ràng buộc ngôn ngữ → output nhất quán hơn
2. **asyncio.to_thread** — cách đúng để tích hợp thư viện sync vào async pipeline mà không block
3. **Defensive parsing** — LLM không luôn trả về đúng format dù đã yêu cầu; cần nhiều lớp fallback
4. **JSONL vs JSON** — JSONL phù hợp hơn cho dataset lớn do không cần load toàn bộ vào RAM
5. **Service tier strategy** — flex tier giúp giảm chi phí đáng kể cho batch workloads

### Nếu làm lại:
1. Dùng **Structured Output / JSON mode** của Gemini native thay vì parse thủ công → 100% đảm bảo format
2. Thêm **`asyncio.gather`** để xử lý nhiều tài liệu song song → tăng tốc với nhiều file
3. Thêm **retry logic với exponential backoff** thay vì chỉ một lần fallback
4. Thêm **validation schema** (Pydantic) để validate output trước khi lưu
5. Log số cặp bị lọc bỏ (`invalid pairs`) để monitor chất lượng prompt

---

## 📊 Kết luận

**Tóm tắt đóng góp:**
- ✅ Thiết kế system prompt tiếng Việt với 4 loại câu hỏi đa dạng phục vụ RAG evaluation
- ✅ Xây dựng pipeline async end-to-end: đọc `.md` → Gemini → parse → normalize → lưu JSONL
- ✅ Giải quyết 4 vấn đề thực tế: JSON parsing, API fallback, normalization, API key
- ✅ Code production-ready với type hints, error handling, và cấu hình qua environment variables

---