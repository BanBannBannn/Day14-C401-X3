import asyncio
import os
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Corpus từ data/raw/*.md ───────────────────────────────────────────────────
_CORPUS = [
    {
        "id": "1",
        "text": """# Chunking Experiment Report

## Purpose

This report summarizes a small experiment comparing fixed-size chunking, sentence-based chunking, and recursive chunking on internal documentation. The objective was to understand how chunk boundaries affect retrieval quality, context preservation, and the usefulness of returned passages.

## Fixed-Size Chunking

Fixed-size chunking was simple to implement and produced predictable chunk counts. It worked reasonably well for long technical documents because every chunk stayed below a target size. However, some chunks split explanations in awkward places, especially when a procedure spanned multiple sentences. In those cases, search results sometimes returned a fragment that mentioned the right keyword but omitted the actual instruction.

## Sentence-Based Chunking

Sentence-based chunking improved readability because each chunk aligned with natural language boundaries. This made manual inspection easier and often produced more coherent retrieval results for short policy documents and FAQs. The downside was that chunk sizes became less consistent, and some dense sections still exceeded ideal embedding length when too many long sentences were grouped together.

## Recursive Chunking

Recursive chunking offered the best balance in the experiment. It first tried to split on larger structural boundaries such as paragraphs, then fell back to smaller separators only when needed. As a result, most chunks preserved context while still staying within the target size range. For the tested data, recursive chunking produced the most consistently useful passages for downstream question answering.

## Conclusion

The experiment suggests that there is no universal best strategy, but recursive chunking is a strong default for mixed technical documentation. Teams should still validate this assumption with their own queries, because retrieval quality depends on both the document style and the kinds of questions users actually ask.""",
        "source": "chunking_experiment_report.md",
    },
    {
        "id": "2",
        "text": """# RAG System Design for an Internal Knowledge Assistant

## Background

A product team wants an assistant that can answer questions about onboarding, deployment workflows, service ownership, and troubleshooting steps. The company already has documentation spread across markdown handbooks, engineering runbooks, and support notes, but employees waste time searching across disconnected folders.

## Goal

Build a retrieval-augmented generation system that finds relevant internal documents before producing an answer. The assistant should reduce hallucinations by grounding its responses in retrieved text and should clearly separate retrieved context from generated synthesis.

## Proposed Architecture

The ingestion pipeline reads markdown and text files from trusted directories, chunks them into semantically coherent segments, and stores those segments with metadata. Each stored record includes the source path, document identifier, document type, and department. The retrieval layer embeds user questions, performs similarity search, and optionally applies metadata filters when the question is scoped to a specific team.

The application layer takes the top retrieved chunks and constructs a prompt that instructs the language model to answer only from the supplied evidence. If the retrieval results are weak or contradictory, the assistant should say so explicitly instead of pretending the answer is complete.

## Evaluation Plan

The team should measure retrieval quality with realistic employee questions such as "How do I deploy the billing API?" or "Who owns the checkout service?" Success is not only whether the answer sounds fluent, but whether the retrieved evidence is relevant, traceable, and up to date.

A useful test plan includes comparing chunking strategies, checking whether metadata filters improve relevance, and recording failure cases. Example failure cases might include outdated documents outranking current runbooks, small chunks losing critical caveats, or multilingual content confusing the embedding model.

## Operational Considerations

As the document set grows, the team should track re-indexing behavior, document deletion, and source freshness. The system should also log which chunks were retrieved so reviewers can inspect why a given answer was produced. That feedback loop is essential for improving both the data and the prompting strategy.""",
        "source": "rag_system_design.md",
    },
    {
        "id": "3",
        "text": """# Vector Store Notes

A vector store is a database or storage layer designed to keep embeddings and retrieve the most similar items to a query vector. In practical AI systems, a vector store is often used to support semantic search, recommendation, clustering workflows, and retrieval-augmented generation.

## Typical Workflow

A common vector search pipeline has four stages:

1. **Chunk documents** into smaller units that preserve meaning.
2. **Embed each chunk** into a dense numerical vector.
3. **Store the vector and metadata** so records can be searched and filtered.
4. **Embed the query** and rank stored vectors by similarity.

The quality of the retrieval system depends heavily on the quality of the chunks. If chunks are too small, they may lose context and produce incomplete matches. If chunks are too large, they may contain too many unrelated ideas and dilute semantic relevance.

## Metadata Matters

Metadata is often as important as the vector itself. Teams frequently store fields such as document source, language, author, product area, publication date, and access control level. When a user asks a question about a specific domain, metadata filters can narrow the search space and improve precision.

For example, a support assistant might restrict retrieval to only public troubleshooting guides, while an internal analyst tool might search engineering postmortems and incident documentation. This filtering step reduces noise and can prevent the application from surfacing text from the wrong department or outdated material.

## Common Risks

Vector stores are powerful, but retrieval is not magically correct. Poor chunking, low-quality embeddings, missing metadata, and weak evaluation practices can all cause misleading results. A system may retrieve passages that are semantically adjacent but not actually useful for the user's task.

That is why teams should test retrieval quality with realistic queries, compare filtered versus unfiltered search, and inspect the actual chunks returned by the system. Good retrieval is a product of careful data preparation, not just a database choice.""",
        "source": "vector_store_notes.md",
    },
    {
        "id": "4",
        "text": """# Ghi chú về Retrieval cho Trợ lý Tri thức Nội bộ

Trong một hệ thống trợ lý tri thức nội bộ, retrieval đóng vai trò tìm ra những đoạn tài liệu phù hợp nhất trước khi mô hình ngôn ngữ tạo câu trả lời. Mục tiêu không chỉ là trả lời trôi chảy mà còn phải bám sát nguồn dữ liệu đã được lưu trữ.

Một pipeline retrieval điển hình bắt đầu từ việc thu thập tài liệu, làm sạch nội dung, chia nhỏ thành các chunk có ý nghĩa, sau đó tạo embedding cho từng chunk. Khi người dùng đặt câu hỏi, hệ thống sẽ tạo embedding cho câu hỏi đó và so sánh với các vector đã lưu để tìm các đoạn gần nhất về mặt ngữ nghĩa.

Chất lượng chunking ảnh hưởng trực tiếp đến chất lượng retrieval. Nếu chunk quá ngắn, hệ thống có thể trả về các câu rời rạc thiếu ngữ cảnh. Nếu chunk quá dài, nhiều ý không liên quan sẽ bị gộp lại, làm giảm độ chính xác của kết quả. Vì vậy, nhiều nhóm chọn chiến lược recursive chunking để ưu tiên tách theo đoạn, rồi mới tách nhỏ hơn khi cần.

Metadata cũng rất quan trọng. Ví dụ, một công ty có thể gắn nhãn tài liệu theo phòng ban, ngôn ngữ, độ nhạy cảm, hoặc ngày cập nhật. Khi người dùng hỏi về tài liệu kỹ thuật bằng tiếng Việt, bộ lọc metadata có thể giúp hệ thống tránh lấy nhầm các tài liệu marketing hoặc tài liệu tiếng Anh không liên quan.

Trong thực tế, retrieval không phải lúc nào cũng đúng. Một số lỗi thường gặp là tài liệu cũ vẫn xếp hạng cao, từ khóa trong câu hỏi không khớp với cách diễn đạt trong tài liệu, hoặc embedding model chưa xử lý tốt nội dung song ngữ. Vì vậy, đội ngũ phát triển nên kiểm thử bằng các truy vấn thực tế, xem trực tiếp các chunk được trả về, và ghi nhận failure cases để cải thiện dữ liệu cũng như chiến lược truy xuất.""",
        "source": "vi_retrieval_notes.md",
    },
    {
        "id": "5",
        "text": """CHƯƠNG III
KIỂM TRA, THI VÀ ĐÁNH GIÁ KẾT QUẢ HỌC TẬP
Điều 21. Đánh giá học phần
1. Giảng viên dạy học phần có trách nhiệm thiết kế đánh giá học phần liên quan
phù hợp với chuẩn đầu ra và tuân thủ các nguyên tắc và tiêu chuẩn của Nhà trường.
2. Kế hoạch đánh giá (phương pháp, hình thức, nhiệm vụ và tỷ trọng các cấu
phần đánh giá) cho một học phần phải được quy định trong đề cương chi tiết học phần
và cung cấp cho sinh viên trước khi đến lớp. Ngoại trừ trường hợp cần thiết bất khả
kháng, khung đánh giá không nên thay đổi trong suốt học kỳ. Đánh giá trực tuyến có
thể được thực hiện nếu đảm bảo trung thực, công bằng và khách quan như đánh giá trực
tiếp và sẽ đóng góp không quá 50% vào tổng điểm trọng số của học phần đó.
3. Chuyên cần
Mỗi học sinh có trách nhiệm tham dự tất cả các lớp học theo lịch trình trừ khi có
sự đồng ý của người hướng dẫn khóa học. Chính sách tham gia lớp học cụ thể sẽ được
nêu trong giáo trình của khóa học.
Học sinh nên đăng ký tham dự bằng các phương pháp (điện tử hoặc thủ công) do
người hướng dẫn chỉ định. Nếu một học sinh không đăng ký bằng các phương pháp đã
được phê duyệt, thì học sinh đó có trách nhiệm cung cấp bằng chứng về việc họ đi học
để được tín chỉ.
Việc không tham dự các buổi học bắt buộc sẽ dẫn đến việc tự động vắng mặt
không lý do trừ khi người hướng dẫn khóa học cho phép ngoại lệ. Học sinh chịu trách
nhiệm cho tất cả các nội dung được đề cập; người hướng dẫn thường sẽ không thực hiện
các khoản phụ cấp để đáp ứng sự vắng mặt không lý do.
Sự chậm trễ hoặc vắng mặt trong các hoạt động giáo dục theo lịch trình thường
sẽ dẫn đến việc bị giảm điểm hoặc các hình phạt khác do người hướng dẫn áp dụng.
Học sinh phải đến đúng giờ cho tất cả các hoạt động giáo dục, ở lại trong suốt
thời gian của hoạt động và không gây ra hoặc tham gia gây rối cho bất kỳ môi trường
giáo dục nào trong suốt hoạt động giáo dục. Kỳ vọng này được mã hóa trong quy tắc
ứng xử của học sinh và là một thành phần thiết yếu của tính chuyên nghiệp.""",
        "source": "vin_policy_trimm.md",
    },
]


class MainAgent:
    """RAG Agent chuyên về thông tin bất động sản Vinhomes.

    Pipeline: ChromaDB retrieval → prompt build → LLM (Anthropic/OpenAI) → JSON response.
    """

    _CHROMA_PATH = "./chroma_db"
    _COLLECTION = "raw_docs_knowledge"
    _EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    _TOP_K = 3

    def __init__(self):
        self.name = "VinhomesRAGAgent-v1"

        self._embedder = SentenceTransformer(self._EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=self._CHROMA_PATH)
        self._col = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_if_empty()

        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._openai_key = os.getenv("OPENAI_API_KEY")

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_if_empty(self) -> None:
        if self._col.count() > 0:
            return
        texts = [d["text"] for d in _CORPUS]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True).tolist()
        self._col.add(
            ids=[d["id"] for d in _CORPUS],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": d["source"]} for d in _CORPUS],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> Tuple[List[str], List[dict], List[float]]:
        q_emb = self._embedder.encode([question], normalize_embeddings=True).tolist()
        res = self._col.query(
            query_embeddings=q_emb,
            n_results=self._TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        # cosine distance ∈ [0,2] → similarity ∈ [-1,1]; normalised vecs → [0,1]
        scores = [round(1.0 - d, 4) for d in res["distances"][0]]
        return docs, metas, scores

    # ── Prompt ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str, contexts: List[str]) -> str:
        ctx_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        return (
             "You are a helpful assistant. "
            "Answer only based on the provided context. "
            "If context is insufficient, say clearly what is missing."
            f"### Reference context:\n{ctx_block}\n\n"
            f"### Question: {question}\n\n"
            "### Answer:"
        )

    # ── LLM calls ─────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        if self._anthropic_key:
            return await self._call_anthropic(prompt)
        if self._openai_key:
            return await self._call_openai(prompt)
        return (
            "[Thiếu API key – vui lòng đặt ANTHROPIC_API_KEY hoặc OPENAI_API_KEY "
            "trong file .env]"
        )

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    async def _call_openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._openai_key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    # ── Public interface ──────────────────────────────────────────────────────

    async def query(self, question: str) -> Dict:
        """Thực hiện RAG pipeline và trả về JSON chuẩn."""
        docs, metas, scores = self._retrieve(question)
        prompt = self._build_prompt(question, docs)
        answer = await self._call_llm(prompt)
        return {
            "answer": answer,
            "contexts": docs,
            "metadata": {
                "sources": [m["source"] for m in metas],
                "score": scores,
            },
        }


# if __name__ == "__main__":
#     agent = MainAgent()

#     async def _demo():
#         questions = [
#             "Giá căn hộ Vinhomes Ocean Park bao nhiêu?",
#             "Tiện ích của Vinhomes Smart City gồm những gì?",
#             "Chính sách vay mua nhà Vinhomes Grand Park như thế nào?",
#         ]
#         for q in questions:
#             print(f"\n❓ {q}")
#             resp = await agent.query(q)
#             print(f"💬 {resp['answer']}")
#             print(f"📚 Nguồn: {resp['metadata']['sources']}")
#             print(f"🎯 Điểm: {resp['metadata']['score']}")

#     asyncio.run(_demo())
