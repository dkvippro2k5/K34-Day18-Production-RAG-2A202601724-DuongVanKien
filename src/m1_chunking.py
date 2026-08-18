from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    if not text or not text.strip():
        return []

    metadata = metadata or {}
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if part.strip()
    ]
    if not sentences:
        return []

    # Load lazily and use only a cached model so chunking never depends on
    # network availability.
    embeddings = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2", local_files_only=True
        )
        embeddings = model.encode(sentences)
    except Exception:
        # Token-count vectors are a deterministic cosine-similarity fallback
        # when sentence-transformers or its model is unavailable.
        vocabulary = sorted({
            token
            for sentence in sentences
            for token in re.findall(r"\w+", sentence.lower(), re.UNICODE)
        })
        positions = {token: index for index, token in enumerate(vocabulary)}
        embeddings = []
        for sentence in sentences:
            vector = [0.0] * len(vocabulary)
            tokens = re.findall(r"\w+", sentence.lower(), re.UNICODE)
            for token, count in Counter(tokens).items():
                vector[positions[token]] = float(count)
            embeddings.append(vector)

    def cosine_similarity(left, right) -> float:
        numerator = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = sum(float(a) * float(a) for a in left) ** 0.5
        right_norm = sum(float(b) * float(b) for b in right) ** 0.5
        return numerator / (left_norm * right_norm + 1e-9)

    groups = [[sentences[0]]]
    for index in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[index - 1], embeddings[index])
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentences[index])

    return [
        Chunk(
            text="\n\n".join(group).strip(),
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": index,
            },
        )
        for index, group in enumerate(groups)
        if group
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if not text or not text.strip():
        return ([], [])
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = metadata or {}

    def split_to_limit(value: str, limit: int) -> list[str]:
        """Split on paragraphs/words while guaranteeing a character limit."""
        value = value.strip()
        if not value:
            return []
        paragraphs = [
            part.strip()
            for part in re.split(r"\n{2,}", value)
            if part.strip()
        ]
        pieces: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                pieces.append(current.strip())
                current = ""
            if len(paragraph) <= limit:
                current = paragraph
                continue

            for word in paragraph.split():
                if len(word) > limit:
                    if current:
                        pieces.append(current.strip())
                        current = ""
                    pieces.extend(
                        word[offset:offset + limit]
                        for offset in range(0, len(word), limit)
                    )
                    continue
                candidate = f"{current} {word}" if current else word
                if len(candidate) <= limit:
                    current = candidate
                else:
                    pieces.append(current.strip())
                    current = word
        if current:
            pieces.append(current.strip())
        return pieces

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_text in split_to_limit(text, parent_size):
        parent_id = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "parent",
                "parent_id": parent_id,
                "chunk_index": len(parents),
            },
            parent_id=parent_id,
        ))
        for child_index, child_text in enumerate(
            split_to_limit(parent_text, child_size)
        ):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "chunk_index": child_index,
                },
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    if not text or not text.strip():
        return []

    metadata = metadata or {}
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []
    fence_marker: str | None = None
    header_pattern = re.compile(r"^#{1,3}\s+.+$")

    for line in text.splitlines():
        fence_match = re.match(r"^\s*(?:\x60{3,}|~{3,})", line)
        if fence_match:
            marker = "~" if "~" in fence_match.group(0) else "\x60"
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None

        is_header = (
            fence_marker is None
            and bool(header_pattern.match(line.strip()))
        )
        if is_header:
            if current_lines and "\n".join(current_lines).strip():
                sections.append((current_header, current_lines))
            current_header = line.strip()
            current_lines = [line.rstrip()]
        else:
            current_lines.append(line.rstrip())

    if current_lines and "\n".join(current_lines).strip():
        sections.append((current_header, current_lines))

    chunks: list[Chunk] = []
    for index, (header, lines) in enumerate(sections):
        section_text = "\n".join(lines).strip()
        if not section_text:
            continue
        chunks.append(Chunk(
            text=section_text,
            metadata={
                **metadata,
                "strategy": "structure",
                "section": header or "preamble",
                "chunk_index": index,
            },
        ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
