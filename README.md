# NVIDIA CEO Polymarket forecaster (v3b)

Entry for the [make.fun](https://make.fun) bounty
`polymarket-nvidia-ceo-forecast-01`: forecast the Yes price of every market
of Polymarket's "Next CEO of NVIDIA?" event at every minute of the 4 hours
after each checkpoint.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13, standard library only, deterministic, no network.

## Method

Per market, fitted only on the case's own prices:

- **Volatility:** sigma is the mean of the q90 of |1-minute moves| over
  24h, 6h and 2h lookbacks, floored at 0.0003 and capped at 0.03.
- **Shape:** Student-t(5) quantiles; the inner quantiles (25%/75%) are
  tightened by 0.85.
- **Horizon:** spread grows as h**0.4 (mean-reverting prices).
- **Median drift:** the 24 Yes prices sum to ~1, so each market's median is
  pulled toward its share of 1, linearly over the horizon, with an asymmetric bounce (1.5x down, 0.5x up); plus a quarter
  of the market's last-30-minute move continued (momentum).

## Practice loss

**0.260675** mean pinball loss on the kit's six practice cases
(p00 0.385, p01 0.323, p02 0.337, p03 0.252, p04 0.216, p05 0.106).
House baseline: 0.363589. v2 entry: 0.262644. v1 entry: 0.269316.
