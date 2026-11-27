"""使用 TokenHub 模型评估 RAG 实验结果。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.metrics import AnswerCorrectness, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

ROOT_DIR = Path(__file__).resolve().parent.parent


def create_tokenhub_client() -> OpenAI:
    """创建 TokenHub 的 OpenAI 兼容客户端。"""
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("TOKENHUB_API_KEY")
    if not api_key:
        raise SystemExit("缺少 TOKENHUB_API_KEY，请先配置项目根目录的 .env。")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com/v1"
        ),
    )


def create_judge_llm():
    """创建 Ragas 评分模型，但不在这里生成 RAG 回答。"""
    return llm_factory(
        os.getenv("TOKENHUB_CHAT_MODEL", "hy4"),
        client=create_tokenhub_client(),
        temperature=0,
        max_tokens=4096,
        reasoning_effort="no_think",
    )


def create_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """创建用于答案语义相似度计算的 TokenHub 向量模型。"""
    load_dotenv(ROOT_DIR / ".env")
    embeddings = OpenAIEmbeddings(
        model=os.getenv(
            "TOKENHUB_EMBEDDING_MODEL", "kinfra-text-embedding-0.6b"
        ),
        api_key=os.getenv("TOKENHUB_API_KEY"),
        base_url=os.getenv(
            "TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com/v1"
        ),
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    return LangchainEmbeddingsWrapper(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=("answer_correctness", "context_recall", "faithfulness"),
        default="answer_correctness",
    )
    args = parser.parse_args()

    results_path = Path(__file__).resolve().parent / "cheque_book_results.json"
    records = json.loads(results_path.read_text(encoding="utf-8"))
    samples = [
        SingleTurnSample(
            user_input=record["user_input"],
            reference=record["reference"],
            retrieved_contexts=record["retrieved_contexts"],
            response=record["response"],
        )
        for record in records
    ]

    judge_llm = create_judge_llm()

    output_path = Path(__file__).resolve().parent / f"ragas_{args.metric}.json"
    scores = []
    if output_path.exists():
        previous_scores = json.loads(output_path.read_text(encoding="utf-8"))
        for record, score in zip(records, previous_scores):
            if record["strategy"] == score["strategy"] and args.metric in score:
                scores.append(
                    {
                        "strategy": record["strategy"],
                        "user_input": record["user_input"],
                        args.metric: score[args.metric],
                    }
                )

    completed = {
        (score["strategy"], score["user_input"]) for score in scores
    }
    for record, sample in zip(records, samples, strict=True):
        key = (record["strategy"], record["user_input"])
        if key in completed:
            print(f"跳过已有评分：{key[0]} | {key[1]}")
            continue

        judge_embeddings = (
            create_judge_embeddings()
            if args.metric == "answer_correctness"
            else None
        )
        if args.metric == "answer_correctness":
            metric = AnswerCorrectness(
                llm=judge_llm, embeddings=judge_embeddings
            )
        elif args.metric == "context_recall":
            metric = ContextRecall(llm=judge_llm)
        else:
            metric = Faithfulness(llm=judge_llm)

        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[metric],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(timeout=600, max_workers=2),
            raise_exceptions=True,
        )
        scores.append(
            {
                "strategy": record["strategy"],
                "user_input": record["user_input"],
                **result.scores[0],
            }
        )
        output_path.write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已保存评分：{key[0]} | {key[1]}")

    print(json.dumps(scores, ensure_ascii=False, indent=2))
