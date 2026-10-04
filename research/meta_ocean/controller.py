"""Common generate/evaluate/revise controller; GEPA edits this entire file."""


def search(*, task, starter, budget, seed, generate, evaluate, commit):
    incumbent = starter
    initial = evaluate(starter)
    best = initial.get("score", float("-inf"))
    if "id" in initial:
        commit(initial["id"])
    feedback = initial
    for _ in range(budget["evaluations"] - 1):
        response = generate(
            f"Improve this policy for the following task:\n{task}\n"
            "Return only complete Python defining class Solution(Policy), importing Policy "
            "from rsikit. No markdown or explanations. Keep it deterministic and row-independent; "
            "use per-row episode memory only when the task permits it.\n"
            f"Current policy:\n{incumbent}\nLatest feedback: {feedback}\n"
        )
        if "error" in response:
            break
        source = response["text"].strip()
        if source.startswith("```"):
            source = source.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        feedback = evaluate(source)
        if "id" in feedback and feedback["score"] > best:
            best, incumbent = feedback["score"], source
            commit(feedback["id"])
