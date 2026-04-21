from agent.main_agent import MainAgent
from typing import List

class AgentImproveV2(MainAgent):
    def __init__(self):
       super().__init__()
    
    @staticmethod
    def _build_prompt(question: str, contexts: List[str]) -> str:
        ctx_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        return (
    "--- ROLE ---\n"
    "You are a helpful assistant. Your goal is to provide accurate, "
    "fact-based information from the provided context.\n\n"
    
    "--- STRICT RULES ---\n"
    "1. ONLY use the information from the 'Reference Context' below.\n"
    "2. If the answer is not explicitly mentioned in the context, strictly respond: "
    "'Xin lỗi, dữ liệu hệ thống hiện tại không có thông tin chi tiết về [vấn đề này] của Vinhomes. "
    "Vui lòng liên hệ hotline hoặc kiểm tra lại câu hỏi.'\n"
    "3. NEVER use your internal knowledge or guess prices/policies if not in context.\n"
    "4. If the question is vague, ask for clarification based on available projects in the context.\n"
    "5. Format the answer clearly using bullet points for technical specs (area, price, location).\n\n"
    
    "--- RESPONSE GUIDELINE ---\n"
    "- Cite specific details like building codes or policy dates if available.\n\n"
    
    f"### Reference Context:\n{ctx_block}\n\n"
    f"### Question: {question}\n\n"
    "### Answer (in Vietnamese):"
)