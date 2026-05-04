from __future__ import annotations

from src.agents import MainOrchestratorAgent


def run_cli() -> None:
    agent = MainOrchestratorAgent()
    print("NovaBite Assistant is ready. Type 'exit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye.")
            break
        if not user_input:
            continue

        result = agent.process(user_input)
        print(f"Assistant: {result.answer}")


if __name__ == "__main__":
    run_cli()
