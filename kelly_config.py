"""The one fractional-Kelly multiplier used to size bets.

Owner decision 2026-10-26: every bet this repo sizes uses 0.4 Kelly.
Import KELLY_FRACTION. Do not copy the number into a caller.

This is the post-solve haircut on full Kelly
(stake = bankroll * KELLY_FRACTION * f*). It is not a quote floor and it
is not a cap. golfer_cap, portfolio_cap, per-player caps, book caps, and
maker_quotes.KELLY_MIN_BY_TYPE stay in their own modules.
"""

KELLY_FRACTION = 0.4
