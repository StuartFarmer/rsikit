"""Independent island founders and per-seed feedback over the local baseline."""

from slick import parse, render

from rsikit.policy import PolicyDefinition

from ..generation import Mutation, _PolicyResponse, apply_edits
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
            island is not None and island.policy.source == candidate.policy.source
            for island in self.islands
        ):
            self._register(candidate, island_id)

    def reset_islands(self) -> None:
        if all(island is not None for island in self.islands):
            super().reset_islands()

    async def mutate(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        provider,
        record=None,
    ) -> PolicyDefinition:
        schema = Mutation.model_json_schema()
        context = render(
            "improved/prompts/mutate.j2",
            instance=self,
            schema=schema,
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        mutation = parse(raw, Mutation)
        return PolicyDefinition.from_text(
            apply_edits(parent.policy.source, mutation.edits),
            name=mutation.name,
            description=mutation.description,
        )

    async def rewrite(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        provider,
        record=None,
    ) -> PolicyDefinition:
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "improved/prompts/rewrite.j2",
            instance=self,
            schema=schema,
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    async def evolve_prompt(
        self, parent: _Candidate, ideas: list[dict], failures: list[dict], *, provider, record=None
    ) -> str:
        schema = Guidance.model_json_schema()
        context = render(
            "improved/prompts/evolve_prompt.j2",
            instance=self,
            schema=schema,
            parent=parent,
            ideas=ideas,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["meta_raw"] = raw
        generated = parse(raw, Guidance)
        return generated.instruction
