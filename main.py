#!/usr/bin/env python3
"""Polymarket "Next CEO of NVIDIA?" 4-hour-ahead quantile forecaster (v3b).

Entry for make.fun bounty polymarket-nvidia-ceo-forecast-01.

Reads every <id>.in case in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output). A case holds, per market,
the Yes price at every minute of the 72 hours up to a checkpoint; the output
is, per market, five quantiles (5%, 25%, 50%, 75%, 95%) of the price at each
of the next 240 minutes.

Method (per market, fitted only on the case's own prices, no network):
  * Volatility: sigma is the mean of the q90 of |1-minute moves| over three
    lookbacks (24h, 6h, 2h), so a fresh burst of moves widens the forecast
    fast while the longer windows keep it stable. Floored at 0.0003 and
    capped at 0.03 per sqrt-minute.
  * Shape: Student-t(5) quantiles, whose fatter tails price in the jumps
    these markets actually make. The inner quantiles (25%/75%) are tightened
    by 0.85: the realised 90% coverage of a plain t5 fan is only ~0.81-0.85
    while the centre is slightly too wide, so the fan is narrowed inside
    and kept full outside.
  * Horizon: spread grows as h**0.4, not sqrt(h): realised volatility of
    h-step changes grows slower than a random walk because the prices
    mean-revert (the 24 outcomes' Yes prices sum to about 1).
  * Median drift: the outcomes are mutually exclusive, so the sum of the
    24 Yes prices should be ~1. When it is not, each market's median is
    pulled toward its share of 1, linearly over the horizon:
    med(h) = x - (S-1) * (x/S) * (h/240), S = sum of last prices.
    The pull is asymmetric: 1.5x when pulling down (S > 1, prices
    collectively too high) and 0.5x when pulling up, because the
    symmetric drift under-corrects the downward case.
  * Momentum: the median also continues a quarter of the market's move
    over the last 30 minutes (capped at 0.05), linearly over the horizon;
    prices that just jumped tend to keep drifting the same way for a while.

Practice loss 0.260675 on the kit's six cases (house baseline 0.363589,
v2 entry 0.262644, v1 entry 0.269316). Standard library only, deterministic (no randomness,
no time, no I/O besides the case files), well inside 8 s / 1 CPU / 1024 MB
per case.
"""

import json
import math
import os
from statistics import NormalDist

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Student-t with 5 degrees of freedom, quantiles at the five levels.
Z_T5 = [-2.015048, -0.726687, 0.0, 0.726687, 2.015048]
INNER_TIGHTEN = 0.85  # 25%/75% quantiles are narrowed by this factor

SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.03
ALPHA = 0.4           # spread grows as h ** ALPHA
MOM_STRENGTH = 0.25   # fraction of the last-30-min move continued
MOM_LAG = 30          # minutes
MOM_CAP = 0.05


def clamp6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def q90_sigma(diffs):
    """90th percentile of |diffs|, rescaled to a normal sigma."""
    if not diffs:
        return 0.0
    a = sorted(abs(v) for v in diffs)
    return a[int(0.90 * (len(a) - 1))] / NormalDist().inv_cdf(0.95)


def market_sigma(prices):
    """Per-market 1-minute sigma: mean of the 24h, 6h and 2h q90 estimates."""
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR
    total = 0.0
    for window in (1440, 360, 120):
        w = prices[-window:] if n > window else prices
        total += q90_sigma([w[i + 1] - w[i] for i in range(len(w) - 1)])
    sig = total / 3.0
    return min(max(sig, SIGMA_FLOOR), SIGMA_CAP)


def zscores(quantiles):
    """Quantile z-scores: exact t5 for the competition's five levels,
    normal quantiles (with the same inner tightening) for any other set."""
    if list(quantiles) == QUANTILES:
        return [Z_T5[0], Z_T5[1] * INNER_TIGHTEN, 0.0,
                Z_T5[3] * INNER_TIGHTEN, Z_T5[4]]
    nd = NormalDist()
    out = []
    for t in quantiles:
        z = nd.inv_cdf(t)
        out.append(z * INNER_TIGHTEN if abs(z) < 1.0 else z)
    return out


def forecast_market(prices, horizon, quantiles, zs, sum_last):
    x = prices[-1] if prices else 0.5
    sigma = market_sigma(prices)
    # Share of 1 this market would hold if the 24 prices summed to 1.
    # Asymmetric bounce (v3b): when the prices sum above 1 the market is
    # collectively too high and the pull back down is under-applied by a
    # symmetric drift, so pull down 1.5x and pull up only 0.5x.
    drift = 0.0
    if sum_last > 0.0:
        drift = -(sum_last - 1.0) * (x / sum_last)
        drift *= 1.5 if drift < 0 else 0.5
    # Continuation of the last-30-minute move, capped.
    mom = 0.0
    if len(prices) > MOM_LAG:
        mom = max(-MOM_CAP, min(MOM_CAP, x - prices[-1 - MOM_LAG]))
    rows = []
    for h in range(1, horizon + 1):
        med = x + drift * (h / horizon) + MOM_STRENGTH * mom * (h / horizon)
        sd = sigma * (h ** ALPHA)
        row = [clamp6(med + z * sd) for z in zs]
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = max(1, int(case.get("horizon_minutes", 240)))
        quantiles = case.get("quantiles", QUANTILES)
        zs = zscores(quantiles)
        markets = case["markets"]
        sum_last = sum(m["prices"][-1] for m in markets if m.get("prices"))
        forecasts = [
            {
                "id": m["id"],
                "q": forecast_market(m.get("prices", []), horizon,
                                     quantiles, zs, sum_last),
            }
            for m in markets
        ]
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts},
                                separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
