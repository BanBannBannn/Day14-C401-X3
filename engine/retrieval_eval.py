from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:

        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        total_hit_rate = 0.0
        total_mrr = 0.0
        count = len(dataset)

        if count == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        for item in dataset:
            # Giả định dataset có định dạng từ kết quả chạy Agent thực tế 
            expected = item.get('expected_retrieval_ids', [])
            actual = item.get('retrieved_ids', [])
            
            total_hit_rate += self.calculate_hit_rate(expected, actual)
            total_mrr += self.calculate_mrr(expected, actual)

        return {
            "avg_hit_rate": total_hit_rate / count,
            "avg_mrr": total_mrr / count,
            "sample_size": count
        }

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Tỉ lệ token của text_a xuất hiện trong text_b (proxy không cần LLM)."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a)

    async def score(self, _case: dict, _resp: dict) -> dict:
        # ── Retrieval metrics ─────────────────────────────────────────────────
        # expected_ids: source doc được ground truth khai báo trong golden_set
        # retrieved_ids: danh sách sources agent thực sự trả về
        expected_source = _case.get("metadata", {}).get("source_doc", "")
        retrieved_sources = _resp.get("metadata", {}).get("sources", [])

        expected_ids = [expected_source] if expected_source else []
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_sources)
        mrr = self.calculate_mrr(expected_ids, retrieved_sources)

        # ── Faithfulness ──────────────────────────────────────────────────────
        # Đo mức độ câu trả lời bám vào contexts đã retrieve được
        # (proxy: % token của answer xuất hiện trong contexts)
        answer = _resp.get("answer", "")
        contexts_text = " ".join(_resp.get("contexts", []))
        faithfulness = self._token_overlap(answer, contexts_text)

        # ── Relevancy ─────────────────────────────────────────────────────────
        # Đo contexts retrieve được có khớp với ground truth context không
        # (proxy: % token của expected context xuất hiện trong retrieved contexts)
        expected_context = _case.get("context", "")
        relevancy = self._token_overlap(expected_context, contexts_text)

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
            },
        }