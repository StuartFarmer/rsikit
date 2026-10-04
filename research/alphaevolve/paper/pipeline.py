"""Compatibility entry point for complete-round AlphaEvolve searches."""

from dataclasses import replace

from rsikit import search as run_search


async def search(
    generator,
    evaluate_batch,
    *,
    proposals: int,
    generation_concurrency: int = 4,
    evaluation_batch_size: int = 10,
    on_event=None,
) -> None:
    # Budgets supplied to this legacy entry point request additional original attempts.
    generator.config = replace(
        generator.config,
        proposals=generator._attempt_offset + len(generator.attempts) + proposals,
        batch_size=evaluation_batch_size,
        generation_concurrency=generation_concurrency,
    )

    def checkpoint(agent):
        agent.checkpoint()
        if on_event is not None:
            event = (
                "generated"
                if agent._round
                else "evaluation_failed"
                if agent._repairs
                else "evaluated"
            )
            on_event(event, list(agent._round.values()))

    await run_search(generator, evaluate_batch, on_checkpoint=checkpoint)
