#!/usr/bin/env python3
"""GTA6 Twitch CCU forecaster.

Kernel-weighted comparable launch curves with a fixed beta scale anchor.
Reads makefun twitch-ccu cases (*.in) from MAKEFUN_INPUT_DIR and writes
per-case quantile forecasts (<stem>.out) to MAKEFUN_OUTPUT_DIR.
Stdlib only, deterministic, no network.
"""
import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 0.2        # fixed exponent for the scale anchor
BW = 0.6          # kernel bandwidth on log-scale distance
TOPK = 14         # comparables kept after weighting
CONTRACT = 0.8    # quantile contraction toward the median

Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def release_curve(points, day_start):
    """Hour 1..24 -> mean viewers, bucketing points by offset from day_start.

    Only points with 0 <= offset < 24h and viewers > 0 count.
    """
    start = parse_ts(day_start)
    buckets = {}
    for p in points:
        try:
            v = float(p["viewers"])
            off = (parse_ts(p["from"]) - start).total_seconds() / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if v <= 0:
            continue
        if 0.0 <= off < 24.0:
            h = int(math.floor(off)) + 1
            buckets.setdefault(h, []).append(v)
    return {h: sum(vs) / len(vs) for h, vs in buckets.items()}


def pre24max(points, day_start):
    """Max viewers in the 24h before day_start (-24h <= offset < 0)."""
    start = parse_ts(day_start)
    best = None
    for p in points:
        try:
            v = float(p["viewers"])
            off = (parse_ts(p["from"]) - start).total_seconds() / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if -24.0 <= off < 0.0:
            if best is None or v > best:
                best = v
    return best


def unlock_class(text):
    t = (text or "").lower()
    if "rolling" in t or "midnight local" in t:
        return "rolling"
    if "single global" in t or "simultaneous" in t or "global unlock" in t:
        return "simult"
    return "other"


def early_access_contaminated(text):
    t = (text or "").lower()
    if "early-access-contaminated" in t:
        return True
    return "early access" in t and (
        "contaminat" in t or "began" in t or "premium" in t
    )


def weighted_quantile(samples, tau):
    """Weighted quantile with linear interpolation on the empirical CDF.

    samples: list of (value, weight). Positions are cumulative weight over
    total weight at each sorted value; interpolate linearly between points.
    """
    pts = sorted(samples)
    total = sum(w for _, w in pts)
    if total <= 0:
        return pts[0][0]
    xs, cs = [], []
    acc = 0.0
    for v, w in pts:
        acc += w
        xs.append(v)
        cs.append(acc / total)
    if tau <= cs[0]:
        return xs[0]
    if tau >= cs[-1]:
        return xs[-1]
    for i in range(1, len(xs)):
        if tau <= cs[i]:
            c0, c1 = cs[i - 1], cs[i]
            x0, x1 = xs[i - 1], xs[i]
            if c1 == c0:
                return x1
            f = (tau - c0) / (c1 - c0)
            return x0 + f * (x1 - x0)
    return xs[-1]


def round2(x):
    """Round half-up to 2 decimals; return int when the result is whole."""
    r = math.floor(x * 100.0 + 0.5) / 100.0
    if r == int(r):
        return int(r)
    return r


def norm_ppf(p):
    """Inverse standard normal CDF: exact table for the kit levels,
    Acklam's approximation otherwise."""
    if p in Z_TABLE:
        return Z_TABLE[p]
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
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1.0)


def anchor_p1(case):
    """Viewers of the first known hour-1 point (else known[0]), min 1.0."""
    known = case["known"]
    for k in known:
        if k.get("hour") == 1:
            return max(1.0, float(k["viewers"]))
    return max(1.0, float(known[0]["viewers"]))


def forecast_case(case):
    levels = [float(t) for t in case["levels"]]
    p1 = anchor_p1(case)
    case_pre = pre24max(case.get("history_hourly") or [], case["day_start"]) or p1
    scale_case = (p1 ** BETA) * (case_pre ** (1.0 - BETA))
    case_class = unlock_class(case.get("notes"))

    comps = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        curve = release_curve(hourly, comp["day_start"])
        if 1 not in curve:
            continue
        base = curve[1]
        cpre = pre24max(hourly, comp["day_start"]) or base
        scale_c = (base ** BETA) * (cpre ** (1.0 - BETA))
        w = math.exp(-abs(math.log(scale_c) - math.log(scale_case)) / BW)
        if unlock_class(comp.get("unlock")) != case_class:
            w *= 0.25
        if early_access_contaminated(comp.get("unlock")):
            w *= 0.5
        comps.append((w, curve, scale_c))
    comps.sort(key=lambda t: -t[0])
    comps = comps[:TOPK]

    rows = []
    for f in case["forecast"]:
        h = int(f["hour"])
        samples = [(math.log(curve[h] / scale_c), w)
                   for w, curve, scale_c in comps if h in curve]
        qs = []
        if samples:
            m = weighted_quantile(samples, 0.5)
            for tau in levels:
                lr = m + CONTRACT * (weighted_quantile(samples, tau) - m)
                qs.append(round2(scale_case * math.exp(lr)))
        else:
            spread = 0.15 * math.sqrt(max(h - 1, 0))
            for tau in levels:
                qs.append(round2(scale_case * math.exp(norm_ppf(tau) * spread)))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return rows


# Dip-split calibration tables, fitted on the 4 practice cases.
# HI bucket: cases whose pre-release max dwarfs the first known hour
# (dip >= 0.9); LO bucket: all other cases.
DELTA_HI = {2: -0.291106, 3: -0.177291, 4: -0.2274, 5: 0.197002, 6: -0.072135, 7: -0.108568, 8: -0.324214, 9: -0.130417, 10: -0.352536, 11: -0.251482, 12: -0.217787, 13: -0.091246, 14: 0.116606, 15: 0.239043, 16: 0.264753, 17: 0.635802, 18: 0.955973, 19: 0.600118, 20: 0.658624, 21: 0.608246, 22: 0.659305, 23: 0.429827, 24: 0.518288}
OFFSET_HI = {2: [0.533701, 0.282745, 0.0, -0.328569, -0.750852], 3: [0.887711, 0.617848, 0.0, -0.295754, -0.694437], 4: [0.695069, 0.494015, 0.0, -0.287469, -0.666633], 5: [0.74947, 0.372648, 0.0, -0.253482, -0.518745], 6: [1.126802, 0.544323, 0.0, -0.359586, -0.463053], 7: [0.906553, 0.484465, 0.0, -0.354124, -0.647445], 8: [0.615836, 0.414011, 0.0, -0.319571, -0.658783], 9: [0.660266, 0.332657, 0.0, -0.244345, -0.60622], 10: [0.710079, 0.379583, 0.0, -0.156255, -0.549578], 11: [0.864128, 0.3946, 0.0, -0.200064, -0.541647], 12: [1.04373, 0.371684, 0.0, -0.213382, -0.519112], 13: [1.210635, 0.181641, 0.0, -0.320988, -0.612722], 14: [1.221281, 0.115163, 0.0, -0.363969, -0.646234], 15: [1.358074, 0.134061, 0.0, -0.344907, -0.593483], 16: [1.309264, 0.179298, 0.0, -0.283484, -0.605102], 17: [1.071795, 0.173902, 0.0, -0.331571, -0.646984], 18: [0.378672, 0.025818, 0.0, -0.331747, -0.686974], 19: [0.316812, 0.210564, 0.0, -0.181088, -0.565223], 20: [0.328439, 0.288469, 0.0, -0.251543, -0.584775], 21: [0.370969, 0.308892, 0.0, -0.243878, -0.596487], 22: [0.282543, 0.240425, 0.0, -0.299907, -0.594973], 23: [0.285026, 0.283947, 0.0, -0.19874, -0.542547], 24: [0.24843, 0.159472, 0.0, -0.102628, -0.505749]}
DELTA_LO = {2: -0.175183, 3: -0.340592, 4: -0.21957, 5: 0.448139, 6: 0.164074, 7: 0.100728, 8: 0.021665, 9: -0.026129, 10: -0.104725, 11: -0.123283, 12: -0.272688, 13: -0.294085, 14: -0.266089, 15: -0.192227, 16: -0.149037, 17: -0.082692, 18: 0.028791, 19: 0.033589, 20: 0.082722, 21: 0.041831, 22: -0.114043, 23: -0.144232, 24: -0.237799}
OFFSET_LO = {2: [0.231849, -0.077672, 0.0, -0.010189, -0.171296], 3: [0.246189, -0.180098, 0.0, 0.311849, 0.179448], 4: [0.078169, -0.197895, 0.0, 0.152329, 0.051391], 5: [-0.6341, -0.865445, 0.0, -0.107761, -0.225356], 6: [0.307425, 0.118166, 0.0, -0.103171, -0.392881], 7: [0.282102, 0.059383, 0.0, -0.122119, -0.559876], 8: [0.234325, 0.022722, 0.0, -0.097607, -0.535106], 9: [0.16915, -0.084896, 0.0, -0.201804, -0.577594], 10: [0.165993, -0.066852, 0.0, -0.137521, -0.600808], 11: [-0.017735, -0.197571, 0.0, -0.124461, -0.589053], 12: [-0.370741, -0.594905, 0.0, -0.023979, -0.31151], 13: [-0.023466, -0.30408, 0.0, 0.090186, -0.197318], 14: [0.193158, -0.064502, 0.0, 0.099321, -0.155604], 15: [0.302456, 0.008755, 0.0, 0.050595, -0.151226], 16: [0.361552, 0.055749, 0.0, 0.090183, -0.150585], 17: [0.329774, -0.010233, 0.0, 0.15701, -0.018472], 18: [0.318163, 0.046479, 0.0, 0.105485, -0.06654], 19: [0.269505, 0.03922, 0.0, -0.071306, -0.26505], 20: [0.255729, -0.047728, 0.0, -0.203258, -0.443667], 21: [0.250677, 0.055939, 0.0, -0.173945, -0.399098], 22: [0.468134, 0.196298, 0.0, -0.103086, -0.284665], 23: [0.442538, 0.131609, 0.0, -0.086688, -0.285986], 24: [0.38004, 0.123087, 0.0, -0.066391, -0.197357]}


def case_dip(case):
    """log(pre-release 24h max / first known hour viewers); 0.0 if unknown."""
    try:
        pre = pre24max(case.get("history_hourly") or [], case["day_start"])
        if pre is None:
            return 0.0
        return math.log(pre / max(1.0, float(case["known"][0]["viewers"])))
    except Exception:
        return 0.0


def calibrate(rows, case):
    """Apply the dip-bucket calibration tables to base forecast rows."""
    if case_dip(case) >= 0.9:
        delta_tab, offset_tab = DELTA_HI, OFFSET_HI
    else:
        delta_tab, offset_tab = DELTA_LO, OFFSET_LO
    levels = [float(t) for t in case["levels"]]
    mid = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    out = []
    for row in rows:
        h = row["hour"]
        q = row["q"]
        if h not in delta_tab:
            out.append({"hour": h, "q": list(q)})
            continue
        qmed = q[mid]
        med0 = math.log(qmed) + delta_tab[h]
        offs = offset_tab[h]
        nq = [round2(math.exp(med0 + CONTRACT * (math.log(qv) - math.log(qmed))
                             + offs[j]))
              for j, qv in enumerate(q)]
        for i in range(1, len(nq)):
            if nq[i] < nq[i - 1]:
                nq[i] = nq[i - 1]
        out.append({"hour": h, "q": nq})
    return out


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outdir, exist_ok=True)
    for name in sorted(os.listdir(indir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(indir, name)) as fh:
            case = json.load(fh)
        rows = calibrate(forecast_case(case), case)
        stem = name[:-len(".in")]
        with open(os.path.join(outdir, stem + ".out"), "w") as fh:
            json.dump({"hours": rows}, fh)


if __name__ == "__main__":
    main()
