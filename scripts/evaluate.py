"""
RAG evaluation using Gemini-as-judge.
Implements faithfulness and answer relevancy without external eval frameworks.

Faithfulness:   does the answer only contain claims supported by the retrieved context?
Answer Relevancy: is the answer actually addressing the question?

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions data/eval_questions.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.retrieval.chain import PROMPT, format_docs, get_llm, retrieve_and_rerank


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are a strict evaluator. Given a context and an answer, score how faithful the answer is.
Faithful means: every claim in the answer is directly supported by the context.

Context:
{context}

Answer:
{answer}

Score from 0.0 to 1.0 where:
1.0 = every claim is supported by context
0.5 = some claims are supported, some are not
0.0 = answer contains claims not in the context (hallucination)

Reply with ONLY a number between 0.0 and 1.0.
""")

RELEVANCY_PROMPT = ChatPromptTemplate.from_template("""
You are a strict evaluator. Score how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer is partially relevant
0.0 = answer does not address the question

Reply with ONLY a number between 0.0 and 1.0.
""")


def parse_score(raw: str) -> float:
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        return 0.0


def evaluate_single(question: str, answer: str, context: str, llm) -> dict:
    faithfulness_chain = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    relevancy_chain = RELEVANCY_PROMPT | llm | StrOutputParser()

    faith_raw = faithfulness_chain.invoke({"context": context, "answer": answer})
    relev_raw = relevancy_chain.invoke({"question": question, "answer": answer})

    return {
        "faithfulness": parse_score(faith_raw),
        "answer_relevancy": parse_score(relev_raw),
    }


def run_rag(question: str) -> tuple[str, str]:
    docs = retrieve_and_rerank(question, top_k=settings.top_k)
    context = format_docs(docs)
    answer = (PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })
    return answer, context


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

    llm = get_llm()
    rows = []

    for item in benchmark:
        q = item["question"]
        print(f"  Q: {q[:70]}...")
        answer, context = run_rag(q)
        scores = evaluate_single(q, answer, context, llm)
        rows.append({
            "question": q,
            "answer": answer[:120] + "..." if len(answer) > 120 else answer,
            **scores,
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS (Gemini-as-judge)")
    print("=" * 65)
    print(df[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))

    print("\n--- Averages ---")
    print(f"  Faithfulness     : {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy : {df['answer_relevancy'].mean():.3f}")
    print(f"  Overall          : {df[['faithfulness','answer_relevancy']].mean().mean():.3f}")

    output_path = Path("data/eval_results.json")
    df.to_json(output_path, orient="records", indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
