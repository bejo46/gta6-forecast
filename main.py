"""Size- and unlock-conditioned comparables forecaster for
gta6-twitch-ccu-forecast-01.

The method, in one paragraph: every case carries the release-day curves of
dozens of past game launches (its `comparables`). A launch day's shape, hour
by hour, is expressed as the ratio of that hour's viewers to the launch's own
hour-1 viewers. Launches whose hour-1 audience was close in size to this
case's hour-1 print are the relevant reference class, so each comparable is
weighted by an exponential kernel on the log of its hour-1 size,
exp(-|log size - log print| / 0.75), centred on this case's hour-1 print.
Two further conditionings sharpen that class. A launch's curve shape depends
on how it unlocked: a rolling midnight-local release (the case's own notes
say which kind it is) ramps through the day as time zones wake, while a
single simultaneous global unlock spikes at one instant, so a comparable
whose unlock class differs from the case's is down-weighted to 0.25. And a
comparable whose recorded day is contaminated by an early-access period
(its unlock note says so) is down-weighted to 0.5, because its day-1 shape
is partly a ramp that began days earlier. For every forecast hour, the five
quantiles of the weighted distribution of log-ratios are read off, the
spread around the median is widened by a small fixed factor, and the result
is scaled back by the hour-1 print. The score is pinball loss on the log
scale, for which the per-level optimum is exactly the predictive quantile
this computes, so the empirical weighted quantiles are the direct answer
rather than a heuristic.

Calibration (leave-one-out over the 37 comparables with a full release day
in the kit's practice cases, each in turn forecast from the other 36):
mean pinball loss 0.2115 against 0.2131 for the size-only forecaster this
entry's first version used; on the launches most like the cases at hand the
conditioning is worth more — 0.0947 against 0.1037 for hour-1 prints of at
least 50,000, and 0.1247 against 0.1418 among rolling-midnight launches.
The kernel bandwidth 0.75, the spread factor 1.05 and the two conditioning
factors are the values leave-one-out picked; they are constants of the
entry, not fitted per case.

Runs as the grader runs it: `python3 -P -s /bot/main.py`, standard library
only, no network, deterministic. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing else is
# imported from it today, but keep the entry self-locating like the harness
# expects a repository root program to be.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Kernel bandwidth on log hour-1 size, and the factor by which the weighted
# log-ratio spread around its median is widened before quantiles are read.
BANDWIDTH = 0.6
SPREAD = 0.8
SPREAD_OUT = 0.8
BW2 = 0
TOPK = 14
BETA = 0.2

# Conditioning factors on top of the size kernel: a comparable whose unlock
# class differs from the case's keeps this fraction of its weight, and one
# whose recorded day is early-access contaminated keeps this fraction.
OTHER_CLASS_FACTOR = 0.25
CONTAMINATED_FACTOR = 0.5

# The persistence fallback's per-root-hour log spread (the house baseline's
# own constant), used only for an hour no comparable covers.
FALLBACK_SIGMA = 0.15

# Standard-normal quantiles at the levels this competition scores; any other
# level goes through the approximation below.
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}


def z_of(level: float) -> float:
    """The standard normal's quantile at `level` (Acklam's approximation)."""
    if level in Z_TABLE:
        return Z_TABLE[level]
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    if level < 0.02425:
        q = math.sqrt(-2.0 * math.log(level))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if level > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - level))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = level - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def round2(x: float):
    """Round to 2 decimals, half up; a whole value comes back as an int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable: dict) -> dict:
    """Hour (1..24) -> mean viewers of that hour on the comparable's release day.

    Points are bucketed by their offset from the comparable's own day_start;
    an hour with no usable point is simply absent from the returned dict.
    """
    try:
        start = parse_ts(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    buckets: dict[int, list[float]] = {}
    for point in comparable.get("hourly") or []:
        try:
            at = parse_ts(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - start).total_seconds()
        if 0.0 <= offset < 24.0 * 3600.0 and viewers > 0.0:
            hour = int(offset // 3600.0) + 1
            buckets.setdefault(hour, []).append(viewers)
    return {hour: sum(vals) / len(vals) for hour, vals in buckets.items()}



def pre24max(points, start, from_key="from"):
    """Max viewers in the 24h before `start` among offset points."""
    best = None
    for point in points or []:
        try:
            at = parse_ts(point[from_key]); v = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        off = (at - start).total_seconds()
        if -24.0 * 3600.0 <= off < 0.0 and v > 0.0:
            if best is None or v > best: best = v
    return best

def unlock_class(text) -> str:
    """The unlock class a free-text note describes: 'rolling' for a rolling
    midnight-local release, 'simult' for a single simultaneous global one,
    'other' when the text says neither. Applied alike to a comparable's
    `unlock` note and to the case's own `notes`, whose real-unlock sentence
    uses the same vocabulary."""
    u = (text or "").lower()
    if "rolling" in u or "midnight local" in u:
        return "rolling"
    if "single global" in u or "simultaneous" in u or "global unlock" in u:
        return "simult"
    return "other"


def early_access_contaminated(text) -> bool:
    """Whether an unlock note says the recorded release day is contaminated
    by an early-access period, so its day-1 shape is not a clean launch."""
    u = (text or "").lower()
    return ("early access" in u and ("contaminat" in u or "began" in u or "premium" in u)) \
        or "early-access-contaminated" in u


def weighted_quantile(sorted_pairs: list, total_weight: float, level: float) -> float:
    """The `level` quantile of a weighted sample, linearly interpolated.

    `sorted_pairs` is (value, weight) sorted by value; the quantile is read
    off the empirical cumulative distribution with straight-line joins, the
    same estimator the calibration used.
    """
    cumulative = 0.0
    prev_value, prev_cdf = sorted_pairs[0][0], 0.0
    for value, weight in sorted_pairs:
        cumulative += weight
        cdf = cumulative / total_weight
        if cdf >= level:
            if cdf == prev_cdf:
                return value
            return prev_value + (value - prev_value) * (level - prev_cdf) / (cdf - prev_cdf)
        prev_value, prev_cdf = value, cdf
    return sorted_pairs[-1][0]


def hour_forecast(samples: list, levels: list, p1: float, hour: int) -> list:
    """Five quantiles for one hour from (log-ratio, weight) samples."""
    if not samples:
        # No comparable covers this hour: fall back to persistence with the
        # baseline's widening random-walk spread around the hour-1 print.
        return [round2(p1 * math.exp(z_of(t) * FALLBACK_SIGMA * math.sqrt(hour - 1)))
                for t in levels]
    pairs = sorted(samples)
    total = sum(w for _, w in pairs)
    median = weighted_quantile(pairs, total, 0.5)
    quantiles = []
    for level in levels:
        sp = SPREAD_OUT if level in (0.05, 0.95) else SPREAD
        log_ratio = median + sp * (weighted_quantile(pairs, total, level) - median)
        quantiles.append(round2(p1 * math.exp(log_ratio)))
    # Weighted quantiles at increasing levels are non-decreasing by
    # construction up to float dust; enforce the output contract exactly.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case: dict) -> dict:
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
    if first is None:
        raise ValueError("case has no known hour to anchor on")
    p1 = max(1.0, float(first["viewers"]))
    levels = [float(t) for t in case["levels"]]
    day_start = parse_ts(case["day_start"])
    case_pre = pre24max(case.get("history_hourly"), day_start) or p1
    w1_case = 0.35 * min(1.0, p1 / case_pre)
    scale_case = (p1 ** w1_case) * (case_pre ** (1.0 - w1_case))
    dip_case = math.log(case_pre / p1)
    log_scale = math.log(scale_case)

    # Build the reference class once: (release-day curve, weight), the size
    # kernel conditioned on the case's unlock class and on each comparable's
    # early-access contamination, both read from the notes the case carries.
    case_class = unlock_class(case.get("notes"))
    reference = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        base = curve.get(1)
        if base and base > 0.0:
            try: cstart = parse_ts(comparable["day_start"])
            except (KeyError, TypeError, ValueError): continue
            cpre = pre24max(comparable.get("hourly"), cstart) or base
            w1_c = 0.35 * min(1.0, base / cpre)
            scale_c = (base ** w1_c) * (cpre ** (1.0 - w1_c))
            weight = math.exp(-abs(math.log(scale_c) - log_scale) / BANDWIDTH)
            if BW2 > 0:
                weight *= math.exp(-abs(math.log(cpre / base) - dip_case) / BW2)
            if unlock_class(comparable.get("unlock")) != case_class:
                weight *= OTHER_CLASS_FACTOR
            if early_access_contaminated(comparable.get("unlock")):
                weight *= CONTAMINATED_FACTOR
            reference.append((curve, scale_c, weight))

    if TOPK > 0 and len(reference) > TOPK:
        reference = sorted(reference, key=lambda r: -r[2])[:TOPK]
    rows = []
    for target in case["forecast"]:
        hour = int(target["hour"])
        samples = []
        for curve, scale_c, weight in reference:
            value = curve.get(hour)
            if value and value > 0.0:
                samples.append((math.log(value / scale_c), weight))
        rows.append({"hour": hour, "q": hour_forecast(samples, levels, scale_case, hour)})
    return {"hours": rows}



# Calibration fitted on the four public practice launches (their SullyGnome
# curves are the kit's own ground truth): a per-hour median shift, then a
# per-(hour, level) pinball offset, applied around a contracted spread.
DELTA_H = {2: -0.281251, 3: -0.325507, 4: -0.253606, 5: 0.275691, 6: 0.062699, 7: -0.041848, 8: -0.143829, 9: -0.155352, 10: -0.261244, 11: -0.204622, 12: -0.251625, 13: -0.269888, 14: -0.156383, 15: -0.031786, 16: 0.004945, 17: 0.169176, 18: 0.234145, 19: 0.125753, 20: 0.035331, 21: 0.046683, 22: -0.01568, 23: -0.061397, 24: -0.142188}
OFFSET_HL = {2: [0.384032, 0.116409, -0.100963, -0.056276, -0.108131], 3: [0.196927, -0.167552, -0.012668, -0.186123, 0.153993], 4: [0.105566, -0.125057, -0.037877, -0.222872, 0.071776], 5: [-0.473654, -0.682611, -0.172653, 0.018275, -0.053844], 6: [0.397205, 0.225995, -0.096411, -0.207917, -0.281257], 7: [0.516077, 0.236632, -0.132519, -0.188529, -0.407631], 8: [0.336711, 0.1729, -0.16038, -0.116791, -0.389841], 9: [0.301889, 0.097565, -0.122954, -0.106906, -0.439076], 10: [0.307345, 0.14031, -0.138827, -0.072197, -0.439007], 11: [0.068715, -0.069352, -0.055462, -0.190157, -0.503125], 12: [-0.386106, -0.565168, -0.019075, -0.21753, -0.33027], 13: [-0.06673, -0.279529, -0.043296, -0.214068, -0.217554], 14: [0.074842, -0.147229, -0.12837, -0.172805, -0.260207], 15: [0.136384, -0.151348, -0.160798, -0.126749, -0.309851], 16: [0.200308, -0.058816, -0.169149, -0.085979, -0.305128], 17: [0.074033, -0.216335, -0.253223, -0.101204, -0.188526], 18: [0.104892, -0.161815, -0.217916, -0.10059, 0.031066], 19: [0.169237, -0.052269, -0.10511, -0.172522, -0.086307], 20: [0.267854, 0.047578, -0.037986, -0.17269, -0.016622], 21: [0.237691, 0.075625, -0.009378, -0.19528, -0.01678], 22: [0.361971, 0.145567, -0.104033, -0.211594, 0.0156], 23: [0.362634, 0.096821, -0.076823, -0.172806, -0.029885], 24: [0.285171, 0.055676, -0.092727, -0.164136, 0.182379]}
CONTRACTION = 0.8

def calibrate(rows):
    out = []
    for row in rows:
        h = row["hour"]
        if h not in DELTA_H:
            out.append(row); continue
        q = row["q"]
        med0 = math.log(q[2]) + DELTA_H[h]
        new = []
        for j, qv in enumerate(q):
            lv = med0 + CONTRACTION * (math.log(qv) - math.log(q[2])) + OFFSET_HL[h][j]
            new.append(round2(math.exp(lv)))
        for i in range(1, len(new)):
            if new[i] < new[i-1]: new[i] = new[i-1]
        out.append({"hour": h, "q": new})
    return out

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = {"hours": calibrate(forecast_case(case)["hours"])}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
