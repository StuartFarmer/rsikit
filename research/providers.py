"""Conservative shared model-call budgets and wire usage accounting."""

import asyncio
import contextvars
import logging
import math
import random
import time
from decimal import Decimal

from slick.providers import OpenRouterAPI, ProviderError

try:
    from openai import APIConnectionError
except ImportError:
    CONNECTION_ERRORS = ()  # The SDK is an optional dependency for offline research.
else:
    CONNECTION_ERRORS = (APIConnectionError,)  # Includes APITimeoutError.

logger = logging.getLogger(__name__)

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
    Omitted cumulative limits are unlimited; prices are required only for a spend cap.
    Actual billing remains unknown when the API omits cost metadata.
    Connection failures retry three times within the caller's deadline; each attempt
    consumes a fresh reservation. Keep SDK retries disabled to avoid hidden requests.
    """

    def __init__(
        self,
        provider,
        *,
        max_calls=None,
        max_tokens=None,
        spend_cap=None,
        max_input_tokens,
        max_output_tokens,
        input_price=None,
        output_price=None,
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
            if value is None and name not in ("max_input_tokens", "max_output_tokens"):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if spend_cap is not None and (input_price is None or output_price is None):
            raise ValueError("spend_cap requires input_price and output_price")
        self.per_call_tokens = max_input_tokens + max_output_tokens
        self.per_call_cost = None
        if input_price is not None and output_price is not None:
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
        return float(self.calls * self.per_call_cost) if self.per_call_cost is not None else None

    def restore(self, events):
        """Replay the call ledger; even starts without a finish consume a reservation."""
        calls = {}
        for event in events:
            calls[event["call"]] = event
        self.calls = max(calls, default=0)
        self.events = list(calls.values())

    async def acall(self, context, **kwargs):
        if kwargs:
            raise ValueError("Ocean generation supports text-only calls without tools")
        for attempt in range(4):
            try:
                return await self._attempt(context)
            except ProviderError as exc:
                if attempt == 3 or not isinstance(exc.__cause__, CONNECTION_ERRORS):
                    raise
                delay = 2**attempt * random.uniform(0.75, 1.25)
                logger.warning(
                    "Model connection failed (%s); retry %s/3 in %.1fs",
                    type(exc.__cause__).__name__,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _attempt(self, context):
        reason = self.stopped
        if len(context.encode("utf-8")) + 1024 > self.limits["max_input_tokens"]:
            reason = "input token bound"
        elif self.limits["max_calls"] is not None and self.calls >= self.limits["max_calls"]:
            reason = "call cap"
        elif (
            self.limits["max_tokens"] is not None
            and self.reserved_tokens + self.per_call_tokens > self.limits["max_tokens"]
        ):
            reason = "token cap"
        elif self.limits["spend_cap"] is not None and (
            self.calls + 1
        ) * self.per_call_cost > Decimal(str(self.limits["spend_cap"])):
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
            reserved_cost=float(self.per_call_cost) if self.per_call_cost is not None else None,
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
