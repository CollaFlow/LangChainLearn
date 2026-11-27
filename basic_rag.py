"""用一篇 Markdown 文档演示最小 RAG：切分、向量化、检索和回答。"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCUMENT = (
    ROOT_DIR
    / "RAG-Multi-Corpus-main"
    / "datasets"
    / "ZX Bank"
    / "md"
    / "Apply for a Cheque Book.md"
)
DEFAULT_QUESTION = "如何申请支票簿？通常需要多久送达？"
DEFAULT_TOKENHUB_BASE_URL = "https://tokenhub.tencentmaas.com/v1"
DEFAULT_TOKENHUB_CHAT_MODEL = "hy4"
DEFAULT_TOKENHUB_EMBEDDING_MODEL = "kinfra-text-embedding-0.6b"
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库助手。只能根据提供的资料回答；资料不足时，明确说不知道。"
            "不要编造事实，并使用与用户问题相同的语言回答。",
        ),
        ("human", "资料：\n{context}\n\n问题：{question}"),
    ]
)


def load_markdown(path: Path) -> Document:
    """读取 UTF-8 Markdown，并把文件路径保存到 metadata。"""
    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")
    return Document(
        page_content=path.read_text(encoding="utf-8-sig"),
        metadata={"source": str(path.resolve())},
    )


def markdown_sections(text: str) -> list[tuple[str, int, list[str]]]:
    """按 Markdown 标题分段，返回正文、原文偏移和标题路径。"""
    headings = list(MARKDOWN_HEADING_PATTERN.finditer(text))
    if not headings:
        return [(text, 0, [])]

    sections = []
    if text[: headings[0].start()].strip():
        sections.append((text[: headings[0].start()], 0, []))

    heading_stack: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        level = len(heading.group(1))
        title = heading.group(2)
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        section_end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(text)
        )
        section_text = text[heading.end() : section_end]
        content_start = heading.end() + (
            len(section_text) - len(section_text.lstrip())
        )
        section_text = section_text.strip()
        section_text = re.sub(
            r"(?:\r?\n)?[ \t]*-{3,}[ \t]*$", "", section_text
        ).rstrip()
        if section_text:
            sections.append(
                (
                    section_text,
                    content_start,
                    [item[1] for item in heading_stack],
                )
            )
    return sections


def split_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    strategy: str = "recursive",
) -> list[Document]:
    """切分文档；Markdown 策略会保留标题路径并避免跨章节切分。"""
    if strategy not in {"recursive", "markdown"}:
        raise ValueError(f"未知切分策略：{strategy}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    if strategy == "recursive":
        return splitter.split_documents([document])

    chunks = []
    for section_text, section_start, headings in markdown_sections(
        document.page_content
    ):
        section_document = Document(
            page_content=section_text, metadata=document.metadata.copy()
        )
        for chunk in splitter.split_documents([section_document]):
            chunk.metadata["start_index"] += section_start
            if headings:
                section = " > ".join(headings)
                chunk.metadata["section"] = section
                chunk.page_content = f"[{section}]\n\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """为 Chunk 生成向量，并建立内存中的 FAISS 索引。"""
    return FAISS.from_documents(chunks, embeddings)


def retrieve_documents(
    vector_store: FAISS, question: str, k: int = 3
) -> list[Document]:
    """检索与问题最相似的 k 个 Chunk。"""
    return vector_store.similarity_search(question, k=k)


def format_context(documents: list[Document]) -> str:
    """把检索结果整理成 LLM 可读取的上下文。"""
    sections = []
    for index, document in enumerate(documents, start=1):
        source = Path(document.metadata["source"]).name
        sections.append(f"[片段 {index}，来源：{source}]\n{document.page_content}")
    return "\n\n".join(sections)


def generate_answer(
    question: str, documents: list[Document], llm: BaseChatModel
) -> str:
    """将问题和检索上下文交给 LLM，生成有依据的回答。"""
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {"context": format_context(documents), "question": question}
    )


def print_retrieval(documents: list[Document]) -> None:
    """在控制台展示检索结果，便于检查回答依据。"""
    print("\n=== 检索到的片段 ===")
    for index, document in enumerate(documents, start=1):
        source = Path(document.metadata["source"]).name
        start_index = document.metadata.get("start_index", "未知")
        section = document.metadata.get("section")
        section_label = f" | 章节：{section}" if section else ""
        print(
            f"\n--- 片段 {index} | {source} | 起始字符：{start_index}"
            f"{section_label} ---"
        )
        print(document.page_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=3, help="返回的片段数量")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument(
        "--chunk-strategy",
        choices=("recursive", "markdown"),
        default="recursive",
        help="recursive：通用递归切分；markdown：保留标题路径且不跨章节切分",
    )
    parser.add_argument(
        "--retrieve-only", action="store_true", help="只检索，不调用聊天模型"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise SystemExit("k 必须大于等于 1")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        raise SystemExit("需要满足 chunk_size > chunk_overlap >= 0")

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("TOKENHUB_API_KEY")
    if not api_key:
        raise SystemExit("缺少 TOKENHUB_API_KEY，请复制 .env.example 为 .env 后填写。")
    base_url = os.getenv("TOKENHUB_BASE_URL", DEFAULT_TOKENHUB_BASE_URL)

    print(f"1. 读取文档：{args.document}")
    document = load_markdown(args.document)

    chunks = split_document(
        document, args.chunk_size, args.chunk_overlap, args.chunk_strategy
    )
    print(
        f"2. 文档切分完成：{len(chunks)} 个 Chunk"
        f"（策略：{args.chunk_strategy}）"
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv(
            "TOKENHUB_EMBEDDING_MODEL", DEFAULT_TOKENHUB_EMBEDDING_MODEL
        ),
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    vector_store = build_vector_store(chunks, embeddings)
    print("3. Embedding 与 FAISS 索引构建完成")

    documents = retrieve_documents(vector_store, args.question, args.k)
    print_retrieval(documents)
    if args.retrieve_only:
        return

    llm = ChatOpenAI(
        model=os.getenv("TOKENHUB_CHAT_MODEL", DEFAULT_TOKENHUB_CHAT_MODEL),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    answer = generate_answer(args.question, documents, llm)
    print("\n=== 回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
