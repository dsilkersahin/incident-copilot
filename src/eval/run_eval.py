from src.generation.answer import ask

QUESTIONS = [
    "How do I restart service X?",
    "Where is payment incident runbook?"
]

def run():
    for q in QUESTIONS:
        print("Q:", q)
        print("A:", ask(q))
        print("-" * 40)

if __name__ == "__main__":
    run()
