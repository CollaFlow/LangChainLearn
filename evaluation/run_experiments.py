"""批量运行两种切分策略，并保存 Ragas 所需的实验结果。"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from basic_rag import (  # noqa: E402
    DEFAULT_DOCUMENT,
    build_vector_store,
    generate_answer,
    load_markdown,
    retrieve_documents,
    split_document,
)

EVALUATION_DIR = Path(__file__).resolve().parent
EVAL_PATH = EVALUATION_DIR / "cheque_book_eval.json"
RESULTS_PATH = EVALUATION_DIR / "cheque_book_results.json"


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("TOKENHUB_API_KEY")
    if not api_key:
        raise SystemExit("缺少 TOKENHUB_API_KEY，请先配置项目根目录的 .env。")
    base_url = os.getenv(
        "TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com/v1"
    )

    eval_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = (
        json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if RESULTS_PATH.exists()
        else []
    )
    existing = {
        (record["strategy"], record["user_input"]) for record in results
    }

    embeddings = OpenAIEmbeddings(
        model=os.getenv(
            "TOKENHUB_EMBEDDING_MODEL", "kinfra-text-embedding-0.6b"
        ),
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    llm = ChatOpenAI(
        model=os.getenv("TOKENHUB_CHAT_MODEL", "hy4"),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    document = load_markdown(DEFAULT_DOCUMENT)

    for strategy in ("recursive", "markdown"):
        pending = [
            item
            for item in eval_items
            if (strategy, item["user_input"]) not in existing
        ]
        if not pending:
            print(f"跳过 {strategy}：所有查询已有结果")
            continue

        chunks = split_document(document, strategy=strategy)
        vector_store = build_vector_store(chunks, embeddings)
        print(f"{strategy}：已建立索引，剩余 {len(pending)} 条查询")

        for item in pending:
            question = item["user_input"]
            documents = retrieve_documents(vector_store, question, k=3)
            response = generate_answer(question, documents, llm)
            results.append(
                {
                    "strategy": strategy,
                    "user_input": question,
                    "reference": item["reference"],
                    "retrieved_contexts": [
                        doc.page_content for doc in documents
                    ],
                    "response": response,
                }
            )
            RESULTS_PATH.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"已保存：{strategy} | {question}")


if __name__ == "__main__":
    main()
