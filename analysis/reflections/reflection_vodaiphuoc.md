# 📋 Báo cáo Cá nhân - Lab Day 14: AI Evaluation Factory

**Họ tên:** Võ Đại Phước 
**Mã HV:** 2A202600334
**Vai trò:** SDG Nhóm Data
**Ngày hoàn thành:** 21 tháng 4, 2026

---

## 1️⃣ Đóng góp Kỹ thuật (Engineering Contribution) - 15 điểm

### 1.1 Tạo synthetic data với LLM
**Mô tả:** Xây dựng inference pipeline với gemini api để tạo data


#### Các thành phần chính:
```python
function generate_qa_from_text
function _generate_with_gemini
function _build_prompt
```

### 1.2 Code Quality & Best Practices:
- Sử dụng enumerate để đánh dấu doc_id, cùng với đó doc_source là file name để làm metadata
- Tinh chỉnh prompt template trong hàm _build_prompt để hướng gemini model trả về đúng độ khó, kiểu các edge case
như out-of-context, fact-check
- Tinh chỉnh temperature bằng 0.9, sử dụng flex inference để tiết cost với config {"service_tier": "flex"}
---

## 2️⃣ Technical Depth - 15 điểm

### 2.1 MRR Retrieval Metric

```
MRR = (1/N) * Σ(1 / rank_of_first_relevant_document)
```

Nếu document đúng ở vị trí thứ 2:
- rank = 2 → MRR contribution = 1/2 = 0.5 (50%)

Đánh giá chất lượng retrieval - document đúng có xếp hạng cao không.

### 2.2 Cohen's Kappa - Agreement Metric

**Công thức đơn giản:**
```
kappa = (Pa - Pe) / (1 - Pe)

Pa = Actual agreement (tỉ lệ 2 judge đồng ý)
Pe = Expected agreement by chance (tỉ lệ ngẫu nhiên)
```

- Kappa = 0.8-1.0: Excellent agreement
- Kappa = 0.6-0.8: Good
- Kappa < 0.6: Poor 

---

## 3️⃣ Problem Solving - 10 điểm

### 3.1 Vấn đề 1: câu hỏi và expect answer bằng tiếng anh

**Vấn đề:** Gemini nhìn tài liệu doc tiếng Việt/Anh nhưng ghi ra câu hỏi và expect answer bằng tiếng anh


**Giải pháp:**
thêm rules vào system prompt để ép model trả lời Tiếng Việt
---

### 3.2 Vấn đề 2: Kiểm tra định dạng response từ model

**Vấn đề:** Gemini thường trả về JSON không đúng yêu cầu, bị nhiễu và không cho ra JSON format theo ý mặc dù
đã có format response trong prompt

**Giải pháp:**
- Dùng regex để loại bỏ các nhiễu (trong hàm _extract_json_array), dùng try except để bắt các case không đúng format chuẩn
- Thêm rule `6) Return JSON only. No markdown fences, no extra keys.` để ép model không output ra sai format khi temperature
được set giá trị cao
---


## 5️⃣ Git Contributions

```bash
# commit hash:
5b8b4ae8ad25c8a22a9794e5144f4f9229344c94
2309c1b6cefba28954b94b5aa7fa6741bba4aa43
```

---

## 6️⃣ Reflection & Learning

### Những gì học được:
1. Bài lab yêu cầu phải inference nhiều lần, phải canh chỉnh để đáp ứng rate limit
2. Đánh giá theo phương pháp multi judge, tuy hơn tốn tài nguyên nhưng đánh giá chính xác, không
bị bias

### Nếu làm lại:
1. Tính toán tổng số lần inference để chạy benchmark tốt hơn
2. Thêm rule để LLM tạo ra các case prompt injection

---

## 📊 Kết luận

**Tóm tắt đóng góp:**
- ✅ Căn chỉnh code synthetic_gen để tạo data đúng yêu cầu
- ✅ Tối ưu code api inference tránh hit rate limit
- ✅ Tư vấn code tích hợp ở main, không dùng mock class evaluator của code mẫu

---
