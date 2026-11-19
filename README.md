# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alkahfiarbi/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alkahfiarbi/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one: for every
PlayStation 5 trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its
Exophase earn rate at the case's target, from the house's reads of the
trophy page before the checkpoint.

## How it works

Exophase prints a *rate* — earners / players-tracked — and both sides move
through launch week, so a trophy's rate rises but the rise decelerates:
early players earn the first trophies in hours, later joiners dilute the
rate, and every rate bends toward the share of players who will ever earn
that trophy. A straight line in log-time therefore overshoots when pushed
four days ahead. The forecaster (`main.py`, Python 3.13 standard library
only, deterministic, no network) is a damped log-time trend on the logit
scale:

1. For each trophy, fit logit(rate) against ln(1 + hours since release)
   by least squares over the last 24 h of reads. Reads are de-duplicated
   first — Exophase holds stale counts between refreshes.
2. Shrink each trophy's slope toward the median slope of its tier
   (platinum / gold / silver / bronze).
3. Extrapolate from the fitted line at the last read, with the slope
   damped by 1 / (1 + 0.5 * log-time ahead) — the concave correction a
   saturating adoption curve needs.
4. Spread the five quantiles on the logit scale with a sigma combining a
   random-walk term, the fit's residual scatter, an extrapolation term,
   and a base term for Exophase's refresh jumps.

Constants were tuned on a backtest (`../gta6-backtest/backtest.py`, kept
out of this repo): a simulated launch week — players arriving on a power
law, each trophy earned after a lognormal delay, Exophase refreshing in
steps — scored with the contest's own pinball loss over 20 simulated
finals in 5 regimes. The tuned model beats the house trend baseline in
every simulated final (mean loss ~0.21 vs ~0.33) and the persistence
baseline in all but a post-cutoff player-count shock no forecaster can
see coming. The spread is kept at 0.75x the backtest optimum, because
the simulation's ground truth is smoother than reality.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes
`<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`)
to the output directory. The grader runs `python3 -P -s /bot/main.py`
with `/input` and `/output`; the program also honours
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=scdytkwz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
