"""Conservative shared model-call budgets and wire usage accounting."""

import contextvars
import math
import time
from decimal import Decimal

from slick.providers import OpenRouterAPI

CALL = contextvars.ContextVar("ocean_call", default={})
RESPONSE = contextvars.ContextVar("ocean_response", default=None)


class BudgetExceeded(RuntimeError):
    pass


class UsageOpenRouter(OpenRouterAPI):
    async def _asend(self, request):
        response = await super()._asend(request)
        event = RESPONSE.get()
        usage = getattr(response, "usage", None)
        if event is not None and usage is not None:
            event["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            event["actual_cost"] = event["usage"].get("cost")
        return response


class BudgetProvider:
    """Reserve full per-call limits before sending; failed/cancelled calls are charged.

    We never refund unknown usage or assume an API error means zero cost. The UTF-8
    byte count plus 1024 framing tokens bounds ordinary byte-tokenized text requests;
    no tools/images are allowed. Price ceilings must cover the chosen model/routing.
    Actual billing remains unknown when the API omits cost metadata.
    """

    def __init__(
        self,
        provider,
        *,
        max_calls,
        max_tokens,
        spend_cap,
        max_input_tokens,
        max_output_tokens,
        input_price,
        output_price,
        log=None,
    ):
        self.provider = provider
        self.limits = dict(
            max_calls=max_calls,
            max_tokens=max_tokens,
            spend_cap=spend_cap,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            input_price=input_price,
            output_price=output_price,
        )
        for name, value in self.limits.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.per_call_tokens = max_input_tokens + max_output_tokens
        self.per_call_cost = (
            Decimal(str(input_price)) * max_input_tokens
            + Decimal(str(output_price)) * max_output_tokens
        ) / 1000000
        self.calls = 0
        self.events = []
        self.stopped = None
        self.log = log

    @property
    def reserved_tokens(self):
        return self.calls * self.per_call_tokens

    @property
    def reserved_cost(self):
        return float(self.calls * self.per_call_cost)

    async def acall(self, context, **kwargs):
        if kwargs:
            raise ValueError("Ocean generation supports text-only calls without tools")
        reason = self.stopped
        if len(context.encode("utf-8")) + 1024 > self.limits["max_input_tokens"]:
            reason = "input token bound"
        elif self.calls >= self.limits["max_calls"]:
            reason = "call cap"
        elif self.reserved_tokens + self.per_call_tokens > self.limits["max_tokens"]:
            reason = "token cap"
        elif (self.calls + 1) * self.per_call_cost > Decimal(str(self.limits["spend_cap"])):
            reason = "spend reservation cap"
        if reason:
            self.stopped = reason
            raise BudgetExceeded(reason)
        # No await before reservation: concurrent requests cannot overspend the ledger.
        self.calls += 1
        event = dict(
            CALL.get(),
            call=self.calls,
            start=time.monotonic(),
            finish=None,
            status="running",
            usage=None,
            actual_cost=None,
            reserved_tokens=self.per_call_tokens,
            reserved_cost=float(self.per_call_cost),
        )
        self.events.append(event)
        token = RESPONSE.set(event)
        if self.log:
            self.log(dict(event, event="llm_start"))
        try:
            result = await self.provider.acall(context)
            event["status"] = "ok"
            return result
        except BaseException as exc:
            event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event["finish"] = time.monotonic()
            RESPONSE.reset(token)
            if self.log:
                self.log(dict(event, event="llm_finish"))
