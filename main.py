#!/usr/bin/env python3
"""Monte Carlo top-5 forecaster for pump.fun coins (make.fun program/v1).

Simulates log-market-cap trajectories for candidate coins plus a newcomer
field of not-yet-launched rockets, and derives per-place probabilities,
an ordered top 5, and per-place size distributions from the simulations.
Standard library only, deterministic (fixed seed, fixed path count).
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
N_PATHS = 4000  # ~6 s on the kit input; keeps Monte Carlo noise small
MAX_CANDIDATES = 90
PLACES = 5
N_BINS = 82
CHART_SIZE = 100
FLOOR_LOGCAP = 0.5  # log10 SOL a dead coin falls to (below any chart)
MIN_ROW_P = 0.006  # rows below this lose to the unnamed share; leave unnamed


def parse_time(s):
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_history(input_dir):
    """Return (marks, task): marks = [(at_ts, [row, ...]), ...] sorted by time."""
    marks = []
    path = os.path.join(input_dir, "history.ndjson")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict):
                    continue
                at = parse_time(d.get("at"))
                if at is None:
                    continue
                rows = d.get("rows")
                if not isinstance(rows, list):
                    rows = []
                clean = []
                seen = set()
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    mint = r.get("mint")
                    if not isinstance(mint, str) or not mint or mint in seen:
                        continue
                    try:
                        cap_lamports = int(str(r.get("market_cap_lamports")))
                    except (TypeError, ValueError):
                        continue
                    if cap_lamports <= 0:
                        continue
                    seen.add(mint)
                    clean.append({
                        "mint": mint,
                        "logcap": math.log10(cap_lamports) - 9.0,
                        "created": parse_time(r.get("created_at")),
                        "complete": bool(r.get("complete")),
                        "mayhem": bool(r.get("mayhem")),
                    })
                marks.append((at, clean))
    except OSError:
        pass
    marks.sort(key=lambda m: m[0])
    task = {}
    try:
        with open(os.path.join(input_dir, "task.json"),
                  "r", encoding="utf-8", errors="replace") as fh:
            task = json.load(fh)
    except (OSError, ValueError):
        task = {}
    if not isinstance(task, dict):
        task = {}
    return marks, task


def build_series(marks):
    """Per-mint time series: mint -> {"times": [...], "caps": [...],
    "created": ts|None, "complete": bool, "last_seen": ts}."""
    series = {}
    for at, rows in marks:
        for r in rows:
            s = series.get(r["mint"])
            if s is None:
                s = series[r["mint"]] = {
                    "times": [], "caps": [], "created": r["created"],
                    "complete": False, "last_seen": at,
                }
            s["times"].append(at)
            s["caps"].append(r["logcap"])
            if r["created"] is not None:
                if s["created"] is None or r["created"] < s["created"]:
                    s["created"] = r["created"]
            s["complete"] = s["complete"] or r["complete"]
            s["last_seen"] = at
    return series


def estimate_params(series, last_time, global_drift, global_vol):
    """Per-coin (drift/h, vol/h, hazard/h) with shrinkage to global priors."""
    params = {}
    for mint, s in series.items():
        caps = s["caps"]
        times = s["times"]
        n = len(caps)
        if n >= 2:
            wsum = 0.0
            dsum = 0.0
            vsum = 0.0
            steps = []
            for i in range(1, n):
                dt = (times[i] - times[i - 1]) / 3600.0
                if dt <= 0:
                    continue
                step = (caps[i] - caps[i - 1]) / dt
                w = 0.5 + 0.5 * (i / (n - 1))  # recency weight
                steps.append((step, w))
                wsum += w
                dsum += w * step
            drift_raw = dsum / wsum if wsum else global_drift
            for step, w in steps:
                vsum += w * (step - drift_raw) ** 2
            var_raw = vsum / wsum if wsum else global_vol ** 2
            vol_raw = math.sqrt(max(var_raw, 1e-4))
            k = n / (n + 3.0)  # shrinkage weight on own history
            drift = k * drift_raw + (1 - k) * global_drift
            vol = k * vol_raw + (1 - k) * global_vol
        else:
            drift, vol = global_drift, global_vol
        drift = max(-0.6, min(0.6, drift))
        vol = max(0.08, min(1.2, vol))
        age_h = None
        if s["created"] is not None:
            age_h = max(0.0, (last_time - s["created"]) / 3600.0)
        elif times:
            age_h = max(0.0, (last_time - times[0]) / 3600.0)
        # Age-dependent crash hazard: the young die fast.
        if age_h is None or age_h < 0.5:
            hazard = 0.30
        elif age_h < 1.5:
            hazard = 0.18
        elif age_h < 4.0:
            hazard = 0.10
        else:
            hazard = 0.06
        if s["complete"]:
            hazard *= 0.8
        params[mint] = {
            "last": caps[-1], "drift": drift, "vol": vol,
            "hazard": hazard, "age_h": age_h,
        }
    return params


def estimate_globals(series):
    """Global drift/vol priors from all per-step log-cap changes."""
    steps = []
    for s in series.values():
        caps, times = s["caps"], s["times"]
        for i in range(1, len(caps)):
            dt = (times[i] - times[i - 1]) / 3600.0
            if dt > 0:
                steps.append((caps[i] - caps[i - 1]) / dt)
    if len(steps) < 10:
        return -0.05, 0.35
    steps.sort()
    med = steps[len(steps) // 2]
    mean = sum(steps) / len(steps)
    var = sum((x - mean) ** 2 for x in steps) / len(steps)
    vol = math.sqrt(max(var, 0.01))
    return max(-0.3, min(0.1, med)), max(0.15, min(0.8, vol))


def estimate_newcomers(marks, series, last_time):
    """Rocket arrival rate (per hour) and cap distribution centre.

    A rocket is a coin that reaches the top 20 within 60 minutes of
    creation. Returns (lam_per_hour, mu_logcap, sd_logcap).
    """
    # One pass: first time each mint appears in a mark's top 20.
    first_top20 = {}
    for at, rows in marks:
        for r in rows[:20]:
            if r["mint"] not in first_top20:
                first_top20[r["mint"]] = at
    top20_early = 0
    for mint, s in series.items():
        if s["created"] is None:
            continue
        first = first_top20.get(mint)
        if first is not None and first - s["created"] <= 3600:
            top20_early += 1
    span_h = 0.0
    if marks:
        span_h = max(0.25, (marks[-1][0] - marks[0][0]) / 3600.0)
    lam = top20_early / span_h if span_h else 1.0
    lam = max(0.5, min(6.0, lam))
    # Rocket cap centre: near the top of the current top 10 (rockets that
    # reach the top 5 are drawn from the upper tail, not the median), with
    # a wide spread: rocket outcomes are extremely dispersed.
    last_rows = marks[-1][1] if marks else []
    top_caps = sorted((r["logcap"] for r in last_rows[:10]), reverse=True)
    mu = top_caps[min(2, len(top_caps) - 1)] if top_caps else 3.0
    return lam, mu, 0.65


def pick_candidates(marks, series, params, last_time):
    """Candidate coins for simulation, deterministic order (by mint)."""
    scored = {}
    if marks:
        last_rows = marks[-1][1]
        for pos, r in enumerate(last_rows[:60]):
            scored[r["mint"]] = scored.get(r["mint"], 0.0) + (60 - pos)
        for at, rows in marks[-3:]:
            for pos, r in enumerate(rows[:30]):
                scored[r["mint"]] = scored.get(r["mint"], 0.0) + (30 - pos) * 0.5
    # Momentum bonus: log-cap gain over the coin's observed life.
    for mint, s in series.items():
        if len(s["caps"]) >= 2:
            gain = s["caps"][-1] - s["caps"][0]
            scored[mint] = scored.get(mint, 0.0) + max(0.0, gain) * 20.0
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    cands = [m for m, _ in ranked[:MAX_CANDIDATES]]
    cands.sort()
    return cands


def bin_of(logcap, edges):
    """Size bin 0..81 for a log10-cap-in-SOL value."""
    if logcap < edges[0]:
        return 0
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= logcap:
            lo = mid + 1
        else:
            hi = mid
    # lo = number of edges <= logcap; bin = lo (bin i spans edges[i-1]..edges[i])
    b = lo
    if b > N_BINS - 1:
        b = N_BINS - 1
    return b


def simulate(cands, params, horizons, edges, lam, rocket_mu, rocket_sd, rng):
    """Monte Carlo. Returns (place_counts, size_counts, newcomer_counts).

    place_counts[place][mint] = paths where mint held that place (summed
    over reads); size_counts[place][bin] likewise; newcomer_counts[place]
    = paths where an unnamed newcomer held that place.
    """
    place_counts = [dict() for _ in range(PLACES)]
    size_counts = [[0] * N_BINS for _ in range(PLACES)]
    newcomer_counts = [0] * PLACES
    cand_params = [(m, params[m]) for m in cands]
    for _ in range(N_PATHS):
        for h in horizons:
            sq = math.sqrt(h)
            entries = []
            for mint, p in cand_params:
                # Crash hazard: dead coins fall off the chart.
                if rng.random() < 1.0 - (1.0 - p["hazard"]) ** h:
                    val = FLOOR_LOGCAP
                else:
                    # Mild mean reversion for very large caps.
                    rev = 0.10 * max(0.0, p["last"] - 3.0)
                    drift = p["drift"] - rev
                    val = p["last"] + drift * h + p["vol"] * sq * rng.gauss(0.0, 1.0)
                entries.append((val, mint))
            # Newcomer rockets: Poisson arrivals, lognormal caps.
            n_new = 0
            lam_h = lam * h
            if lam_h > 0:
                # Knuth's algorithm for small lambda*h.
                limit = math.exp(-lam_h)
                prod = 1.0
                while True:
                    prod *= rng.random()
                    if prod <= limit:
                        break
                    n_new += 1
                    if n_new > 12:
                        break
            for _ in range(n_new):
                val = rng.gauss(rocket_mu, rocket_sd)
                entries.append((val, None))
            entries.sort(key=lambda e: (-e[0], e[1] or ""))
            for k in range(PLACES):
                val, mint = entries[k]
                b = bin_of(val, edges)
                size_counts[k][b] += 1
                if mint is None:
                    newcomer_counts[k] += 1
                else:
                    d = place_counts[k]
                    d[mint] = d.get(mint, 0) + 1
    return place_counts, size_counts, newcomer_counts


def floor6(p):
    return math.floor(p * 1e6) / 1e6


def build_forecast(marks, task):
    places = int(task.get("places") or PLACES)
    places = max(1, min(PLACES, places))
    size_cfg = task.get("size") or {}
    edges = size_cfg.get("edges") or [2.0 + 0.05 * i for i in range(81)]
    edges = [float(e) for e in edges]
    while len(edges) < 81:
        edges.append(edges[-1] + 0.05 if edges else 2.0)
    reads = task.get("reads") or []
    read_times = [t for t in (parse_time(r) for r in reads) if t is not None]
    if not marks or not read_times:
        return fallback_forecast(edges)
    last_time = marks[-1][0]
    horizons = [max(0.05, (rt - last_time) / 3600.0) for rt in read_times]
    series = build_series(marks)
    if len(series) < 5:
        return fallback_forecast(edges, list(series.keys()))
    g_drift, g_vol = estimate_globals(series)
    params = estimate_params(series, last_time, g_drift, g_vol)
    lam, rocket_mu, rocket_sd = estimate_newcomers(marks, series, last_time)
    cands = pick_candidates(marks, series, params, last_time)
    rng = random.Random(SEED)
    place_counts, size_counts, newcomer_counts = simulate(
        cands, params, horizons, edges, lam, rocket_mu, rocket_sd, rng)
    total = float(N_PATHS * len(horizons))

    probabilities = []
    # Unnamed-mass floor: the simulation under-disperses newcomer wins,
    # and post-cut launches take a large share of top-5 places. The floor
    # grows with the observed rocket rate so calm histories stay sharp
    # while churning histories leave room for the unborn.
    unnamed_floor = max(0.15, min(0.45, 0.15 + 0.06 * lam))
    for k in range(places):
        counts = place_counts[k]
        rows = []
        for mint, c in counts.items():
            p = c / total
            if p >= MIN_ROW_P:
                rows.append((mint, p))
        rows.sort(key=lambda r: (-r[1], r[0]))
        rows = rows[:100]
        s = sum(p for _, p in rows)
        # Leave the newcomer + dropped mass unnamed; enforce the floor.
        cap = 1.0 - unnamed_floor
        if s > cap:
            scale = cap / s
            rows = [(m, p * scale) for m, p in rows]
        for mint, p in rows:
            pf = floor6(p)
            if pf > 0:
                probabilities.append(
                    {"mint": mint, "place": k + 1, "p": pf})

    # top5: lowest expected rank, counting non-placement as rank 6, so a
    # coin that rarely places cannot beat a coin that usually places.
    exp_rank = {}
    for mint in cands:
        placed_mass = 0.0
        er = 0.0
        for k in range(places):
            p = place_counts[k].get(mint, 0) / total
            placed_mass += p
            er += (k + 1) * p
        er += 6.0 * max(0.0, 1.0 - placed_mass)
        exp_rank[mint] = er
    top5 = sorted(
        cands,
        key=lambda m: (exp_rank[m],
                       -place_counts[0].get(m, 0), m))[:5]
    while len(top5) < 5:  # degenerate: pad with other seen mints
        for m in sorted(series.keys()):
            if m not in top5:
                top5.append(m)
                if len(top5) == 5:
                    break
        else:
            break
    if len(top5) < 5:
        return fallback_forecast(edges, top5)

    sizes = []
    for k in range(places):
        hist = size_counts[k]
        # Light uniform smoothing so no plausible bin is exactly zero.
        smooth = [c / total * 0.97 + 0.03 / N_BINS for c in hist]
        s = sum(smooth)
        rows = []
        for b in range(N_BINS):
            p = smooth[b] / s * 0.97
            if p >= 0.0005:
                rows.append((b, floor6(p)))
        for b, pf in rows:
            if pf > 0:
                sizes.append({"place": k + 1, "bin": b, "p": pf})

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def fallback_forecast(edges, known_mints=None):
    probs = []
    top5 = list(known_mints or [])[:5]
    pad = 0
    while len(top5) < 5:
        pad += 1
        top5.append("1" * (31 + pad))
    sizes = []
    for k in range(PLACES):
        for b in range(N_BINS):
            sizes.append({"place": k + 1, "bin": b, "p": 0.012})
    return {"probabilities": probs, "top5": top5, "sizes": sizes}


def main():
    input_dir = (os.environ.get("MAKEFUN_INPUT_DIR")
                 or os.environ.get("MAKEFUN_INPUT") or "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR")
    output_file = os.environ.get("MAKEFUN_OUTPUT")
    marks, task = load_history(input_dir)
    forecast = build_forecast(marks, task)
    data = json.dumps(forecast, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > 1048576:  # pragma: no cover - safety valve
        forecast = fallback_forecast([2.0 + 0.05 * i for i in range(81)])
        data = json.dumps(forecast, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    if output_file:
        out_path = output_file
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        out_dir = output_dir or "/output"
        os.makedirs(out_dir, exist_ok=True)
        name = task.get("output") or "forecast.json"
        if not isinstance(name, str) or "/" in name or "\\" in name:
            name = "forecast.json"
        out_path = os.path.join(out_dir, name)
    with open(out_path, "wb") as fh:
        fh.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
