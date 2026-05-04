from __future__ import annotations

from dataclasses import dataclass

from src.rag_pipeline import NovaBiteRAGStore


@dataclass
class EvalCase:
    query: str
    expected_snippet: str


def run_retrieval_eval() -> None:
    store = NovaBiteRAGStore()
    store.load_or_create()

    cases = [
        EvalCase("Do you have vegan pasta?", "Garden Vegan Pasta"),
        EvalCase("Is the chicken grilled or fried?", "flame-grilled"),
        EvalCase("What are your opening hours on weekends?", "Weekends"),
        EvalCase("Do you host birthday events?", "Birthday events are available"),
    ]

    hits = 0
    print("Retrieval evaluation (snippet match in top-k filtered docs)\n")
    for idx, case in enumerate(cases, start=1):
        docs = store.retrieve(case.query)
        text = "\n".join(d.page_content for d in docs)
        passed = case.expected_snippet.lower() in text.lower()
        hits += int(passed)
        print(f"{idx}. Query: {case.query}")
        print(f"   Expected snippet: {case.expected_snippet}")
        print(f"   Retrieved docs: {len(docs)}")
        print(f"   Pass: {passed}\n")

    accuracy = hits / len(cases) if cases else 0.0
    print(f"Hit accuracy: {hits}/{len(cases)} = {accuracy:.2%}")


if __name__ == "__main__":
    run_retrieval_eval()
