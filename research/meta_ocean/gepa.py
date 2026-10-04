"""GEPA's adapter boundary: one example is the entire replicated benchmark."""

import json

from research.providers import BudgetExceeded

from .experiment import append_json, digest, progress, source_text

CONTRACT = """Evolve one Python search controller shared across the configured environment suite,
not a game policy. Each environment/replicate gets a fresh controller and full budget.
Preserve the entry point:
def search(*, task, starter, budget, seed, generate, evaluate, commit): ...
generate(prompt) -> {'text': str} or {'error': str}; all calls are metered.
evaluate(policy_source) -> {'id': str, 'score': float, 'scores': list} or {'error': str}.
Each evaluation costs one credit, including malformed, duplicate and failed proposals.
Each successful evaluation plays exactly ten games, each capped at 2,000 steps.
The task describes the actual step cap, action space, scoring and permitted episode memory.
commit(id) retains a previously evaluated policy as the final incumbent.
On timeout/failure the last committed policy survives; otherwise the common starter does.
Choose one final policy using search feedback only. Private audit results are unavailable
inside the controller. No network, credentials, simulator, host files or prior trial archive.
The task text describes observations/actions; policies import Policy from rsikit.
budget contains evaluations and output_tokens. Keep all work bounded and use only the oracles.
Return complete controller Python source only, without Markdown or explanation.
"""


class Adapter:
    def __init__(self, benchmark, edit, config, path):
        self.benchmark, self.edit, self.config, self.path = benchmark, edit, config, path
        self.calls = self.revisions = 0
        self.best = None
        self.latest = {}
        self.stopped = None

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        if len(batch) != 1:
            raise ValueError("A GEPA example must contain the whole benchmark")
        if self.calls >= self.config.benchmark_limit:
            raise BudgetExceeded("complete benchmark cap")
        self.calls += 1
        result = self.benchmark(candidate["controller"])
        self.latest[candidate["controller"]] = result
        return EvaluationBatch(
            outputs=[result],
            scores=[result["selection_score"]],
            trajectories=[result] if capture_traces else None,
        )

    def on_valset_evaluated(self, event):
        # GEPA's minibatch scores can be noisy. Only full GEPA validation promotes a winner.
        result = self.latest[event["candidate"]["controller"]]
        if (
            self.best is None
            or result["selection_score"] > self.best["scorecard"]["selection_score"]
        ):
            self.best = dict(controller=event["candidate"]["controller"], scorecard=result)
            progress(
                f"GEPA accepted controller: selection score {result['selection_score']:.4f}",
                kind="leaderboard",
                rows=[
                    dict(
                        id=digest(self.best["controller"])[:12],
                        name=self.config.objective,
                        description="Best controller (GEPA development validation)",
                        score=result["selection_score"],
                        generation=self.revisions,
                        extras={},
                    )
                ],
            )

    def log(self, message):
        append_json(self.path / "gepa_log.jsonl", dict(message=message))
        progress(f"GEPA: {str(message).splitlines()[0][:240]}" if message else "GEPA")

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {key: eval_batch.outputs for key in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        if self.stopped or self.revisions >= self.config.revisions:
            raise BudgetExceeded("controller revision cap")
        self.revisions += 1
        progress(f"GEPA editing controller: revision {self.revisions}/{self.config.revisions}")
        prompt = (
            CONTRACT + f"\nCampaign objective: {self.config.objective}. "
            "S is the equal-weight mean of per-environment normalized final scores. "
            "T and E are equal-weight per-environment means of actual output tokens and evaluated proposals. "
            "Efficiency campaigns qualify only at S >= 100. Optimize selection_score.\n"
            f"Current controller:\n{candidate['controller']}\n"
            f"Benchmark feedback:\n{json.dumps(reflective_dataset)}\n"
        )
        try:
            response = self.edit(prompt).strip()
        except Exception as exc:
            self.stopped = f"{type(exc).__name__}: {exc}"
            append_json(
                self.path / "edits.jsonl",
                dict(revision=self.revisions, prompt=prompt, error=self.stopped),
            )
            raise
        append_json(
            self.path / "edits.jsonl",
            dict(revision=self.revisions, prompt=prompt, response=response),
        )
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Syntax/runtime failures are evaluated as failed trials, not silently skipped revisions.
        return dict(controller=source_text(response))


def optimize(starter, benchmark, edit, config, path):
    import gepa

    adapter = Adapter(benchmark, edit, config, path)
    stop = None
    try:
        gepa.optimize(
            seed_candidate={"controller": starter},
            trainset=["development-suite"],
            adapter=adapter,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            candidate_selection_strategy="current_best",
            use_merge=False,
            max_metric_calls=config.benchmark_limit,
            seed=config.search_seed,
            run_dir=str(path / "gepa"),
            callbacks=[adapter],
            logger=adapter,
            cache_evaluation=False,
            raise_on_exception=True,
            stop_callbacks=lambda state: (
                bool(adapter.stopped)
                or adapter.revisions >= config.revisions
                or adapter.calls >= config.benchmark_limit
            ),
        )
    except BudgetExceeded as exc:
        stop = str(exc)
    if adapter.best is None:
        raise RuntimeError("GEPA did not complete an initial benchmark")
    return dict(
        adapter.best,
        benchmarks=adapter.calls,
        revisions=adapter.revisions,
        stop=stop or adapter.stopped,
    )
