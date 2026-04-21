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
