"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, from the house's reads of the trophy page before the checkpoint.

Model (standard library only, deterministic, no network)
--------------------------------------------------------
Exophase prints a *rate*: earners / players-tracked.  Both sides move --
players join and earn -- so a trophy's rate rises through launch week,
but the rise decelerates: the first players earn the early trophies in
hours, later joiners dilute the rate, and every trophy's rate bends
toward the share of players who will ever earn it.  A straight line in
log-time therefore overshoots when it is pushed four days ahead.  The
model is a damped log-time trend on the logit scale:

1. For each trophy, fit logit(rate) against x = ln(1 + hours since
   release) by least squares over the last 24 h of reads (fewer when the
   case is shorter).  Reads are de-duplicated first: Exophase holds
   stale counts between refreshes, and a flat stretch must not outweigh
   the reads where something actually moved.
2. Shrink each trophy's slope toward the median slope of its own tier
   (platinum / gold / silver / bronze): one trophy's short history is
   noisy, and trophies of a tier progress alike.
3. Extrapolate from the fitted line at the last read, with the slope
   damped by 1 / (1 + KAPPA * (x_target - x_last)): the further ahead,
   the more the rise bends.  This is the concave correction a saturating
   adoption curve needs, and it is what keeps the forecast between the
   two house baselines -- persistence (which undershoots: it says
   nothing moves) and the house curve baseline (which overshoots: it
   extends the recent line undamped) -- instead of repeating either
   one's mistake.
4. The median is the damped line read at the target, clamped to
   [0.01, 99.99].  A trophy every tracked player holds stays held.

Spread (logit scale).  The quantiles are
q_tau = sigmoid(logit(median) + z_tau * sigma), z the normal quantiles.
sigma combines four honest sources of error, added in quadrature:
  - a random-walk term 0.375 * sqrt(horizon in days), the house
    baseline's own scale (0.5) narrowed by the backtest: rates drift
    while the page is unread;
  - the residual scatter of the trophy's own logit fit: a trophy whose
    reads jump around gets a wider band;
  - an extrapolation term 0.225 per unit of log-time between the last
    read and the target: the further ahead, the less the past binds;
  - a base 0.1875 for Exophase's own refresh behaviour (the page holds
    stale counts, then jumps: launch night's first refresh cut every
    rate, because the tracked-player count jumped 638 -> 3,868).

When the history is flat (the practice cases: every read identical),
every slope is ~0 and the forecast is persistence with a modest spread,
i.e. it degrades to the house baseline instead of inventing a trend.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
Reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR, else
/input) and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR,
else /output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (see module docstring) ---------------------------------
# Tuned on a backtest (see ../gta6-backtest/backtest.py): a simulated
# launch week -- players arriving on a power law, each trophy earned
# after a lognormal delay, Exophase refreshing in steps -- scored with
# the contest's own pinball loss over 20 simulated finals in 5 regimes.
# The tuned model beats the house trend baseline in every simulated
# final (mean loss ~0.21 vs ~0.33) and the persistence baseline in all
# but a post-cutoff player-count shock no forecaster can see coming.
# The spread is kept at 0.75x the backtest optimum: the simulation's
# ground truth is smoother than reality, so the shipped band stays
# honest about model error instead of chasing the simulated optimum.
FIT_WINDOW_HOURS = 24.0      # reads used for the fits: the recent regime
DAMP_KAPPA = 0.5             # slope damping per unit of log-time ahead
SIGMA_RW_PER_SQRT_DAY = 0.375  # random-walk spread, logit units / sqrt(day)
SIGMA_BASE = 0.1875          # irreducible refresh / model uncertainty
SIGMA_EXTRAP_PER_LOG_TIME = 0.225
SIGMA_FIT_MIN = 0.1125       # floor / cap for the fit-residual spread
SIGMA_FIT_MAX = 0.525
SIGMA_MAX = 1.60
SHRINK_PRIOR_N = 3.0        # pseudo-count for shrinkage to the tier slope


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (as the house baseline rounds)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 -> aware datetime (accepts a trailing Z and fractions)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std or None)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, None
    if n == 1:
        return ys[0], 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return a, b, resid


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    release = case.get("game", {}).get("release_date") or "2026-11-19"
    t0 = parse_time(release if "T" in release else release + "T00:00:00Z")
    target = parse_time(case["target"])

    def x_of(when):
        hours = (when - t0).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    # De-duplicate consecutive reads that change nothing: Exophase holds
    # stale counts between refreshes, and a flat stretch must not
    # outweigh the reads where something actually moved.
    reads = []
    prev_key = None
    for r in history:
        key = (r["players_tracked"], tuple(r["rates"]))
        if key != prev_key:
            reads.append(r)
            prev_key = key
    if not reads:
        reads = history

    last = reads[-1]
    last_at = parse_time(last["at"])
    x_last = x_of(last_at)
    x_target = x_of(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    dx = max(0.0, x_target - x_last)
    damp = 1.0 / (1.0 + DAMP_KAPPA * dx)

    # Fit window: the recent regime only.
    window_start = last_at.timestamp() - FIT_WINDOW_HOURS * 3600.0
    fit_reads = [r for r in reads if parse_time(r["at"]).timestamp() >= window_start]
    if len(fit_reads) < 3:
        fit_reads = reads
    xs = [x_of(parse_time(r["at"])) for r in fit_reads]

    # Last known rate per trophy (walk back over nulls).
    last_rate = []
    for i in range(n_tr):
        rate = None
        for r in reversed(reads):
            v = r["rates"][i]
            if v is not None:
                rate = float(v)
                break
        last_rate.append(0.0 if rate is None else rate)

    # --- per-trophy logit trend, shrunk to the tier slope -------------------
    fits = []
    fit_sigmas = []
    for i in range(n_tr):
        pts_x, pts_y = [], []
        for r, x in zip(fit_reads, xs):
            v = r["rates"][i]
            if v is None:
                continue
            pts_x.append(x)
            pts_y.append(logit(float(v)))
        a_i, b_i, resid = ols(pts_x, pts_y)
        fits.append((a_i, b_i))
        fit_sigmas.append(0.0 if resid is None else resid)

    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [fits[i][1] for i, t in enumerate(trophies) if t["tier"] == tier]
        if vals:
            tier_slopes[tier] = median(vals)
    global_slope = median([f[1] for f in fits]) if fits else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a_i, b_i = fits[i]
        tier_prior = tier_slopes.get(trophy["tier"], global_slope)
        # Shrinkage weight grows with how many reads inform the trophy.
        n_pts = sum(1 for r in fit_reads if r["rates"][i] is not None)
        w = n_pts / (n_pts + SHRINK_PRIOR_N)
        b_star = w * b_i + (1.0 - w) * tier_prior
        b_star = clamp(b_star, -0.2, 3.0)
        # Extrapolate from the fitted line at the last read, damped.
        anchor = a_i + b_i * x_last
        lg_med = anchor + b_star * dx * damp
        med = 100.0 * sigmoid(lg_med)
        med = clamp(med, 0.01, 99.99)
        # A trophy every tracked player holds stays held.
        if last_rate[i] >= 99.99:
            med = 99.99

        # --- spread ---------------------------------------------------------
        sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(max(0.0, horizon_days))
        sigma_fit = clamp(fit_sigmas[i], SIGMA_FIT_MIN, SIGMA_FIT_MAX)
        sigma_extrap = SIGMA_EXTRAP_PER_LOG_TIME * dx
        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_extrap ** 2 + SIGMA_BASE ** 2)
        sigma = min(sigma, SIGMA_MAX)

        lg = logit(med)
        qs = []
        for z in Z:
            q = 100.0 * sigmoid(lg + z * sigma)
            qs.append(round4(clamp(q, 0.0, 100.0)))
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
