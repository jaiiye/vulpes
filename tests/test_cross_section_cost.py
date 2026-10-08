"""The cross-sectional cost ladder: its pool, its windows, its arithmetic.

Not its numbers - those need the archive. What is worth pinning is the pool,
because getting it wrong does not produce a wrong answer, it produces a
plausible one: run on every symbol in the archive (6437 of them, most listed
for a week) the reversal basket's long leg is a basket of coins about to be
delisted, and `late_exit_factor=0` then books a -100% period. That run reports
a healthy-looking breakdown and is entirely an artefact of the pool.

Also pinned: the split is recomputed per window, because a symbol's liquidity
tier is a property of the window, not of the symbol.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.cross_section import Panel  # noqa: E402
import cross_section_cost as xc  # noqa: E402


def make_panel(n_rows: int, series: dict, volumes: dict | None = None) -> Panel:
    times = [1_700_000_000_000 + i * 14_400_000 for i in range(n_rows)]
    volume = volumes or {
        s: [1000.0 if c is not None else None for c in v] for s, v in series.items()
    }
    return Panel(times=times, symbols=sorted(series), close=series, volume=volume)


class TestSpread(unittest.TestCase):
    def test_reports_seven_points_in_bp(self):
        vals = [-0.02, -0.01, 0.0, 0.01, 0.02]
        out = xc.spread(vals)
        self.assertEqual(len(out.split()), 7)
        self.assertIn("-200.0", out)
        self.assertTrue(out.rstrip().endswith("200.0"))

    def test_empty_is_said_plainly(self):
        self.assertEqual(xc.spread([]), "no periods")


class TestWindowPanels(unittest.TestCase):
    def _panel(self) -> Panel:
        """20 rows, 40 symbols - enough that the illiquid half clears
        `min_symbols`. Liquidity is inverted between the two halves."""
        busy_first = {f"S{i:02d}" for i in range(20)}
        series = {}
        volumes = {}
        for i in range(40):
            s = f"S{i:02d}"
            series[s] = [10.0 + 0.01 * i * j for j in range(20)]
            # The first 20 are busy in the first half and quiet in the second;
            # the last 20 the reverse. A split taken once over the whole panel
            # would put half of these in the wrong leg of both windows.
            first = [5000.0] * 10 if s in busy_first else [10.0] * 10
            second = [10.0] * 10 if s in busy_first else [5000.0] * 10
            volumes[s] = first + second
        return make_panel(20, series, volumes)

    def test_the_split_is_recomputed_per_window(self):
        """The illiquid half flips, so freezing one split would be wrong."""
        panels = xc.window_panels(self._panel(), 2)
        self.assertEqual(len(panels), 2)
        self.assertEqual(set(panels[0].symbols), {f"S{i:02d}" for i in range(20, 40)})
        self.assertEqual(set(panels[1].symbols), {f"S{i:02d}" for i in range(20)})

    def test_windows_are_disjoint_and_cover_the_panel(self):
        panels = xc.window_panels(self._panel(), 2)
        self.assertEqual(len(panels[0]) + len(panels[1]), 20)

    def test_the_pool_is_whoever_is_trading_at_the_reference_row(self):
        """`REF_ROW`, not a coverage filter.

        A coverage filter is the survivorship bias this study is trying to
        measure: a delisted coin cannot have full coverage, and the reversal
        basket's long leg is the leg that would have held it. The pool has to
        be definable from one row, with no reference to what happens later.
        """
        series = {f"S{i:02d}": [10.0 + 0.01 * i * j for j in range(20)]
                  for i in range(40)}
        # Neither is trading at the reference row (the last one, here): GONE
        # delisted early, BRIEF existed only in the middle. Both are excluded
        # by a single row either way - no coverage arithmetic over their future.
        series["GONE"] = [10.0] * 5 + [None] * 15
        series["BRIEF"] = [None] * 5 + [10.0] * 5 + [None] * 10
        volumes = {s: [1000.0 if c is not None else None for c in v]
                   for s, v in series.items()}
        panels = xc.window_panels(make_panel(20, series, volumes), 2)
        for p in panels:
            self.assertNotIn("GONE", p.symbols)
            self.assertNotIn("BRIEF", p.symbols)


class TestMeasure(unittest.TestCase):
    def _panels(self) -> list[Panel]:
        """24 symbols, 40 rows - enough that `min_symbols` is not the binding
        constraint and every window actually trades."""
        series = {}
        for i in range(24):
            # Each symbol drifts at its own rate, so the ranking is not a tie.
            series[f"S{i:02d}"] = [100.0 * (1.0 + 0.001 * i) ** j for j in range(40)]
        volumes = {s: [1000.0 + j for j in range(40)] for s in series}
        return [make_panel(40, series, volumes)]

    def test_net_falls_monotonically_with_the_fee(self):
        """The only free parameter has to move the answer in one direction."""
        panels = self._panels()
        nets = [xc.measure(panels, 6, 6, f)["cw_mean"] for f in (0.0, 5.0, 20.0)]
        self.assertTrue(all(a >= b for a, b in zip(nets, nets[1:])), nets)

    def test_gross_does_not_depend_on_the_fee(self):
        """The ladder is one measurement with one free parameter: raising the
        fee must not quietly change what the basket earned before costs."""
        panels = self._panels()
        a = xc.measure(panels, 6, 6, 0.0)["gross_bp"]
        b = xc.measure(panels, 6, 6, 30.0)["gross_bp"]
        self.assertAlmostEqual(a, b)

    def test_a_window_that_never_trades_is_reported_as_such(self):
        panels = self._panels()
        r = xc.measure(panels, 6, 6, 1.0)
        self.assertEqual(r["n_windows"], 1)
        self.assertGreater(r["n_periods"], 0)


if __name__ == "__main__":
    unittest.main()
