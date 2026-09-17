"""Historical backtesting for the Fox agent.

The design goal is that this package contains NO strategy logic. It replays
market data through the same `Synthesizer` and `Discipline` objects the live
agent uses, so a backtest result reflects the real decision code rather than a
reimplementation that can drift from it.

Known and material limitation: the public Hyperliquid API does not expose
historical whale positions, so the smart money factor (40% of the live weight
and the only factor with evidence behind it) cannot be backtested. See
`backtest/market.py` for how that is handled explicitly rather than silently.
"""

__version__ = "0.1.0"
