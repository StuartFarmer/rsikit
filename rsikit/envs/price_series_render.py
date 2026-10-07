"""The Bitcoin performance chart adapted to any CSV asset."""

from .bitcoin_render import BitcoinRenderer


class PriceSeriesRenderer(BitcoinRenderer):
    """Keep the same charts, using asset labels and timestamps or row indices."""

    currency = "QUOTE CURRENCY"
    currency_symbol = ""
    step_label = "bar"
    value_key = "value"

    def __init__(self, env, *, policy_name="Baseline agent", split="training", policy_id=""):
        super().__init__(env, policy_name=policy_name, split=split, policy_id=policy_id)
        self.title = self.price_label = self.allocation_label = env.unwrapped.asset_name

    def _date_label(self, observation):
        game = self.unwrapped
        if game._timestamps:
            return (
                game._timestamps[game._index]
                .removesuffix("+00:00")
                .replace("T", " ")
                .removesuffix(" 00:00:00")
            )
        return str(int(observation[4]))

    def _value_label(self, value):
        return f"{value:,.4f}" if abs(value) < 10 else f"{value:,.0f}"
