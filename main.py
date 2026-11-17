#!/usr/bin/env python3
"""Twitch launch-day CCU forecaster (gta6-twitch-ccu-forecast-01).

Reads every *.in case file from MAKEFUN_INPUT_DIR and writes one
<stem>.out JSON document per case to MAKEFUN_OUTPUT_DIR, holding five
quantiles for each forecast hour.

Model: each comparable launch contributes its release-day curve,
expressed as log-ratios to an adaptive scale anchor (a blend of the
comparable's own hour-1 audience and its pre-release 24h peak, with
the hour-1 weight shrinking when hour 1 sits in a trough between
unlock waves).  Comparables are weighted by a Gaussian kernel on log
scale distance, with penalties for a different unlock class or an
early-access-contaminated rollout, and truncated to the 14 heaviest.
Per forecast hour the weighted quantiles of the log-ratios are taken
by linear interpolation on the empirical CDF, contracted 0.8 around
the weighted median, and mapped back through the case's own anchor.

Standard library only; deterministic; no network access.
"""

import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANDWIDTH = 0.6        # kernel width on log-scale distance
TOPK = 14              # comparables kept after weighting
SPREAD = 0.8           # contraction of the log-ratio spread
ANCHOR_W1_MAX = 0.35   # hour-1 weight when hour 1 is the peak itself
SAME_CLASS = 1.0
DIFF_CLASS = 0.25      # weight factor for a different unlock class
EARLY_ACCESS = 0.5     # weight factor for contaminated early access

Z_TABLE = {
    0.05: -1.645,
    0.25: -0.674,
    0.5: 0.0,
    0.75: 0.674,
    0.95: 1.645,
}


def parse_ts(text):
    """Parse an ISO-8601 timestamp (a trailing Z is accepted)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(value):
    """Round half up to two decimals; return an int when whole."""
    rounded = math.floor(value * 100.0 + 0.5) / 100.0
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q
                           + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q
                            + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1.0)


def zscore(tau):
    if tau in Z_TABLE:
        return Z_TABLE[tau]
    return norm_ppf(tau)


def release_day_curve(points, start):
    """Average viewers per release-day hour (1..24) from hourly points.

    Only points inside the release day with a positive viewer count
    are used; each hour bucket averages the points that fall in it.
    """
    buckets = {}
    for point in points or []:
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


def pre24max(points, start):
    """Largest viewer count in the 24 hours before `start`."""
    best = None
    for point in points or []:
        try:
            at = parse_ts(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - start).total_seconds()
        if -24.0 * 3600.0 <= offset < 0.0 and viewers > 0.0:
            if best is None or viewers > best:
                best = viewers
    return best


def unlock_class(text):
    """Classify an unlock description: rolling, simult or other."""
    t = (text or "").lower()
    if "rolling" in t or "midnight local" in t:
        return "rolling"
    if "single global" in t or "simultaneous" in t or "global unlock" in t:
        return "simult"
    return "other"


def early_access_contaminated(text):
    """True when an unlock text describes contaminated early access."""
    t = (text or "").lower()
    if "early-access-contaminated" in t:
        return True
    return ("early access" in t
            and ("contaminat" in t or "began" in t or "premium" in t))


def adaptive_scale(first_hour, pre_peak):
    """Blend hour 1 and the pre-release peak into one scale anchor.

    The hour-1 weight is 0.35 when hour 1 is itself the peak and
    shrinks in proportion as hour 1 falls into a trough below it.
    """
    w1 = ANCHOR_W1_MAX * min(1.0, first_hour / pre_peak)
    return (first_hour ** w1) * (pre_peak ** (1.0 - w1))


def weighted_quantile(pairs, tau):
    """Weighted quantile by linear interpolation on the empirical CDF.

    `pairs` holds (value, weight); the CDF steps at each sorted value
    and the quantile interpolates linearly between neighbouring
    (cumulative share, value) points, clamped at both ends.
    """
    pts = sorted(pairs)
    total = sum(w for _, w in pts)
    cumulative = []
    acc = 0.0
    for _, w in pts:
        acc += w
        cumulative.append(acc / total)
    if tau <= cumulative[0]:
        return pts[0][0]
    for i in range(1, len(pts)):
        if tau <= cumulative[i]:
            f0, f1 = cumulative[i - 1], cumulative[i]
            x0, x1 = pts[i - 1][0], pts[i][0]
            if f1 == f0:
                return x1
            return x0 + (x1 - x0) * (tau - f0) / (f1 - f0)
    return pts[-1][0]


def forecast_case(case):
    """Forecast one case; returns {"hours": [{"hour", "q"}, ...]}."""
    levels = case["levels"]
    day_start = parse_ts(case["day_start"])
    known = case["known"]

    p1 = None
    for entry in known:
        if entry.get("hour") == 1:
            p1 = float(entry["viewers"])
            break
    if p1 is None:
        p1 = float(known[0]["viewers"])
    p1 = max(p1, 1.0)

    case_pre = pre24max(case.get("history_hourly"), day_start) or p1
    scale_case = adaptive_scale(p1, case_pre)
    case_class = unlock_class(case.get("notes"))

    weighted = []
    for comp in case.get("comparables") or []:
        cstart = parse_ts(comp["day_start"])
        curve = release_day_curve(comp.get("hourly"), cstart)
        base = curve.get(1)
        if not base:
            continue
        cpre = pre24max(comp.get("hourly"), cstart) or base
        scale_c = adaptive_scale(base, cpre)
        weight = math.exp(-abs(math.log(scale_c) - math.log(scale_case))
                          / BANDWIDTH)
        if unlock_class(comp.get("unlock")) != case_class:
            weight *= DIFF_CLASS
        if early_access_contaminated(comp.get("unlock")):
            weight *= EARLY_ACCESS
        weighted.append((weight, curve, scale_c))
    weighted.sort(key=lambda item: -item[0])
    weighted = weighted[:TOPK]

    rows = []
    for spec in case["forecast"]:
        hour = spec["hour"]
        samples = [(math.log(curve[hour] / scale_c), weight)
                   for weight, curve, scale_c in weighted
                   if hour in curve]
        quantiles = []
        if samples:
            median = weighted_quantile(samples, 0.5)
            for tau in levels:
                log_ratio = median + SPREAD * (
                    weighted_quantile(samples, tau) - median)
                quantiles.append(round2(scale_case * math.exp(log_ratio)))
        else:
            for tau in levels:
                z = zscore(tau)
                quantiles.append(round2(
                    scale_case * math.exp(z * 0.15 * math.sqrt(hour - 1))))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


# Calibration fitted on the four public practice launches (their
# SullyGnome curves are the kit's own ground truth): a per-hour median
# shift, then a per-(hour, level) pinball offset, applied around a
# contracted spread.  See the fit notes in the build transcript.
DELTA_H = {
    2: -0.281251, 3: -0.325507, 4: -0.253606, 5: 0.275691,
    6: 0.062699, 7: -0.041848, 8: -0.143829, 9: -0.155352,
    10: -0.261244, 11: -0.204622, 12: -0.251625, 13: -0.269888,
    14: -0.156383, 15: -0.031786, 16: 0.004945, 17: 0.169176,
    18: 0.234145, 19: 0.125753, 20: 0.035331, 21: 0.046683,
    22: -0.015680, 23: -0.061397, 24: -0.142188,
}

OFFSET_HL = {
    2: [0.384032, 0.116409, -0.100963, -0.056276, -0.108131],
    3: [0.196927, -0.167552, -0.012668, -0.186123, 0.153993],
    4: [0.105566, -0.125057, -0.037877, -0.222872, 0.071776],
    5: [-0.473654, -0.682611, -0.172653, 0.018275, -0.053844],
    6: [0.397205, 0.225995, -0.096411, -0.207917, -0.281257],
    7: [0.516077, 0.236632, -0.132519, -0.188529, -0.407631],
    8: [0.336711, 0.172900, -0.160380, -0.116791, -0.389841],
    9: [0.301889, 0.097565, -0.122954, -0.106906, -0.439076],
    10: [0.307345, 0.140310, -0.138827, -0.072197, -0.439007],
    11: [0.068715, -0.069352, -0.055462, -0.190157, -0.503125],
    12: [-0.386106, -0.565168, -0.019075, -0.217530, -0.330270],
    13: [-0.066730, -0.279529, -0.043296, -0.214068, -0.217554],
    14: [0.074842, -0.147229, -0.128370, -0.172805, -0.260207],
    15: [0.136384, -0.151348, -0.160798, -0.126749, -0.309851],
    16: [0.200308, -0.058816, -0.169149, -0.085979, -0.305128],
    17: [0.074033, -0.216335, -0.253223, -0.101204, -0.188526],
    18: [0.104892, -0.161815, -0.217916, -0.100590, 0.031066],
    19: [0.169237, -0.052269, -0.105110, -0.172522, -0.086307],
    20: [0.267854, 0.047578, -0.037986, -0.172690, -0.016622],
    21: [0.237691, 0.075625, -0.009378, -0.195280, -0.016780],
    22: [0.361971, 0.145567, -0.104033, -0.211594, 0.015600],
    23: [0.362634, 0.096821, -0.076823, -0.172806, -0.029885],
    24: [0.285171, 0.055676, -0.092727, -0.164136, 0.182379],
}


def calibrate(rows, levels):
    """Apply the fitted per-hour shift and pinball offsets to rows."""
    med_j = min(range(len(levels)), key=lambda j: abs(levels[j] - 0.5))
    out = []
    for row in rows:
        hour = row["hour"]
        if hour not in DELTA_H:
            out.append(row)
            continue
        q = row["q"]
        med0 = math.log(q[med_j]) + DELTA_H[hour]
        new_q = []
        for j, qv in enumerate(q):
            level_value = med0 + SPREAD * (
                math.log(qv) - math.log(q[med_j])) + OFFSET_HL[hour][j]
            new_q.append(round2(math.exp(level_value)))
        for i in range(1, len(new_q)):
            if new_q[i] < new_q[i - 1]:
                new_q[i] = new_q[i - 1]
        out.append({"hour": hour, "q": new_q})
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = {"hours": calibrate(forecast_case(case)["hours"],
                                      case["levels"])}
        stem = name[: -len(".in")]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh)


if __name__ == "__main__":
    main()
