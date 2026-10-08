"""Uniform seed strategy adapted from SkyDiscover's initial_search_strategy.py."""


def select(population, state, rng):
    parent = rng.choice(population)
    count = state["inspirations"]
    examples = rng.sample(population, min(count + 1, len(population)))
    return {
        "parent_id": parent["id"],
        "operator": "free",
        "inspiration_ids": [p["id"] for p in examples if p["id"] != parent["id"]][:count],
    }
