"""Independent island founders and per-seed feedback over the local baseline."""

from slick import prompt

from ..edits import Mutation, Program
from ..original.agent import AlphaEvolve as OriginalAlphaEvolve
from ..original.agent import Guidance, _Candidate


class AlphaEvolve(OriginalAlphaEvolve):
    """Keep generation, repair, ranking, and evaluation mechanics shared."""

    def _founding_island(self, attempt_id: int) -> int | None:
        empty = [i for i, island in enumerate(self.islands) if island is None]
        return empty[(attempt_id - 1) % len(empty)] if empty else None

    def _register_founder(self, candidate: _Candidate, island_id: int) -> None:
        # Renaming an identical program must not found another island.
        if not any(
            island is not None and island.policy._implementation == candidate.policy._implementation
            for island in self.islands
        ):
            self._register(candidate, island_id)

    def reset_islands(self) -> None:
        if all(island is not None for island in self.islands):
            super().reset_islands()

    @prompt(template="improved/prompts/mutate.j2", output_type=Mutation)
    async def mutate(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Mutation,
    ) -> Mutation:
        return generated

    @prompt(template="improved/prompts/rewrite.j2", output_type=Program)
    async def rewrite(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Program,
    ) -> Program:
        return generated

    @prompt(template="improved/prompts/evolve_prompt.j2", output_type=Guidance)
    async def evolve_prompt(
        self, parent: _Candidate, ideas: list[dict], failures: list[dict], *, generated: Guidance
    ) -> str:
        return generated.instruction
