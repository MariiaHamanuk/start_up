"""
RAG evaluation using RAGAS.
Measures: Faithfulness, Answer Relevancy, Context Precision.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions data/eval_questions.json

The script runs each benchmark question through the full RAG pipeline
(retrieval + optional reranking + generation) and scores the results.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.config import settings
from app.retrieval.chain import PROMPT, format_docs, get_llm, retrieve_and_rerank
from langchain_core.output_parsers import StrOutputParser


def run_rag(question: str) -> tuple[str, list[str]]:
    docs = retrieve_and_rerank(question, top_k=settings.top_k)
    answer = (PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": format_docs(docs),
        "question": question,
    })
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/eval_questions.json")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Benchmark file not found: {questions_path}")
        return

    benchmark = json.loads(questions_path.read_text(encoding="utf-8"))
    print(f"Running {len(benchmark)} benchmark questions...\n")

    data: dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in benchmark:
        q = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"  Q: {q[:70]}...")
        answer, contexts = run_rag(q)
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(data)

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.gemini_api_key,
        )
    )

    print("\nScoring with RAGAS (faithfulness, answer_relevancy, context_precision)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df: pd.DataFrame = result.to_pandas()

    print("\n" + "=" * 60)
    print("RAGAS BENCHMARK RESULTS")
    print("=" * 60)
    cols = ["question", "faithfulness", "answer_relevancy", "context_precision"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    print("\n--- Averages ---")
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric in df.columns:
            print(f"  {metric:<25}: {df[metric].mean():.3f}")

    output_path = Path("data/eval_results.json")
    df.to_json(output_path, orient="records", indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
