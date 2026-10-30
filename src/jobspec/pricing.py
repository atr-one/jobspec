"""Model prices for cost accounting.

THE PRICES BELOW ARE EXAMPLES. They were copied from the providers' public price lists when this
file was written and will go out of date. Check your provider's current price list before relying
on any cost this package reports, and override the table with ``--prices prices.json`` (a JSON
object ``{"model-name": [input_usd_per_1m, output_usd_per_1m]}``) or ``set_price``.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple

# USD per 1M tokens: (input, output). Example values - verify before use.
PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic, first-party API list prices
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    # OpenAI
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def set_price(model: str, input_per_1m: float, output_per_1m: float) -> None:
    PRICES[model] = (float(input_per_1m), float(output_per_1m))


def load_prices(path: str) -> None:
    """Merge a JSON price file into the table."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("a price file must be a JSON object of model -> [input, output]")
    for model, pair in data.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"price for {model!r} must be [input_usd_per_1m, output_usd_per_1m]")
        set_price(str(model), float(pair[0]), float(pair[1]))


def price_for(model: str) -> Optional[Tuple[float, float]]:
    """Exact name first, then the longest table key that prefixes the name (dated snapshots)."""
    if model in PRICES:
        return PRICES[model]
    keys = [k for k in PRICES if model.startswith(k)]
    if not keys:
        return None
    return PRICES[max(keys, key=len)]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    price = price_for(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def estimate_tokens(text: str) -> int:
    """A rough token count: about 4 characters per token for Latin script, 2.5 for others.

    Only for planning. Real counts come from the API usage block.
    """
    if not text:
        return 0
    latin = sum(1 for ch in text if ord(ch) < 0x250)
    other = len(text) - latin
    return math.ceil(latin / 4.0 + other / 2.5)
