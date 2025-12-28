from __future__ import annotations
import pandas as pd
from dateutil import parser

# Your calibrated zones
ZONES = {
    "Z1": (None, 105),
    "Z2": (106, 122),
    "Z3": (123, 135),
    "Z4": (136, 148),
    "Z5": (149, None),
}

def _classify_zone(hr: float) -> str:
    for z, (lo, hi) in ZONES.items():
        if lo is None and hr <= hi:
            return z
        if hi is None and hr >= lo:
            return z
        if lo is not None and hi is not None and lo <= hr <= hi:
            return z
    return "UNK"

def _to_day(ts: str) -> pd.Timestamp:
    # normalize to local-naive day
    dt = parser.parse(ts)
    # strip tz safely; keep date
    dt = dt.replace(tzinfo=None)
    return pd.Timestamp(dt).normalize()

def extract_metric_series(payloads, metric_name: str, value_key: str = "qty") -> pd.DataFrame:
    """
    Metrics expected shape:
      {"name": "...", "units": "...", "data": [{"date": "...", "qty": 12.3}, ...]}
    """
    rows = []
    for p in payloads:
        root = p.get("data", p)
        for m in root.get("metrics", []):
            if m.get("name") != metric_name:
                continue
            for d in m.get("data", []):
                if "date" not in d:
                    continue
                val = d.get(value_key)
                if val is None:
                    continue
                rows.append({"date": _to_day(d["date"]), metric_name: float(val)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sum-type metrics vs mean-type metrics
    sum_metrics = {"mindful_minutes", "alcohol_consumption"}
    if metric_name in sum_metrics:
        df = df.groupby("date", as_index=False)[metric_name].sum()
    else:
        df = df.groupby("date", as_index=False)[metric_name].mean()

    return df

def extract_sleep_analysis(payloads) -> pd.DataFrame:
    """
    sleep_analysis often includes: totalSleep, awake, rem, deep, core (hours)
    """
    rows = []
    for p in payloads:
        root = p.get("data", p)
        for m in root.get("metrics", []):
            if m.get("name") != "sleep_analysis":
                continue
            for d in m.get("data", []):
                if "date" not in d:
                    continue
                total = d.get("totalSleep")
                if total is None:
                    continue
                rows.append({
                    "date": _to_day(d["date"]),
                    "sleep_total_hr": float(total),
                    "sleep_awake_hr": float(d.get("awake", 0.0) or 0.0),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("date", as_index=False).mean(numeric_only=True)

def derive_sleep_score(df_sleep: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent derived score (0–100):
      - duration target 7.5h (70 points)
      - awake penalty up to 30 points (1h awake = max penalty)
    """
    if df_sleep.empty:
        return df_sleep

    out = df_sleep.copy()
    duration = out["sleep_total_hr"].clip(0, 9)
    duration_score = (duration / 7.5).clip(0, 1) * 70

    awake = out["sleep_awake_hr"].clip(0, 3)
    awake_penalty = (awake / 1.0).clip(0, 1) * 30

    out["sleep_score_derived"] = (duration_score + (30 - awake_penalty)).clip(0, 100)
    return out[["date", "sleep_score_derived"]]

def compute_zone_minutes_and_hrr1(payloads) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses workouts[].heartRateData[] entries with Avg + date.
    Assumes each HR bucket ~ 1 minute (good enough for trends).
    HRR1: peak_hr - hr_at_~60s_post_peak (approx using next bucket >= peak+60s).
    """
    daily_rows = []
    hrr_rows = []

    for p in payloads:
        root = p.get("data", p)
        for w in root.get("workouts", []):
            name = w.get("name", "Unknown")
            hr_points = w.get("heartRateData") or []
            if not hr_points:
                continue

            pts = []
            for hp in hr_points:
                if hp.get("Avg") is None or hp.get("date") is None:
                    continue
                t = parser.parse(hp["date"]).replace(tzinfo=None)
                pts.append((t, float(hp["Avg"])))

            if len(pts) < 5:
                continue
            pts.sort(key=lambda x: x[0])

            # day bucketing
            day = _to_day(w["end"]) if w.get("end") else pd.Timestamp(pts[-1][0]).normalize()

            # zone minutes
            zmins = {z: 0 for z in ZONES}
            for _, hr in pts:
                z = _classify_zone(hr)
                if z in zmins:
                    zmins[z] += 1

            total = sum(zmins.values())
            if total == 0:
                continue

            daily_rows.append({
                "date": day,
                "z1_min": zmins["Z1"],
                "z2_min": zmins["Z2"],
                "z3_min": zmins["Z3"],
                "z4_min": zmins["Z4"],
                "z5_min": zmins["Z5"],
                "cardio_min": total,
                "zone2_pct": 100.0 * zmins["Z2"] / total,
            })

            # HRR1
            peak_idx = max(range(len(pts)), key=lambda i: pts[i][1])
            peak_t, peak_hr = pts[peak_idx]
            target_t = peak_t + pd.Timedelta(seconds=60)

            post_hr = None
            for t, hr in pts[peak_idx:]:
                if t >= target_t:
                    post_hr = hr
                    break
            if post_hr is None:
                post_hr = pts[-1][1]

            hrr_rows.append({
                "date": day,
                "hrr1": float(peak_hr - post_hr),
            })

    daily = pd.DataFrame(daily_rows)
    hrr = pd.DataFrame(hrr_rows)

    if not daily.empty:
        daily = daily.groupby("date", as_index=False).sum(numeric_only=True)
        daily["zone2_pct"] = 100.0 * daily["z2_min"] / daily["cardio_min"].where(daily["cardio_min"] > 0, 1)

    if not hrr.empty:
        hrr = hrr.groupby("date", as_index=False).mean(numeric_only=True)

    return daily, hrr

def build_phase1_daily(payloads) -> pd.DataFrame:
    # Metrics (names are as commonly found in exports; we’ll adjust if yours differs)
    hrv = extract_metric_series(payloads, "heart_rate_variability")
    rhr = extract_metric_series(payloads, "resting_heart_rate")
    rr  = extract_metric_series(payloads, "respiratory_rate")
    bd  = extract_metric_series(payloads, "breathing_disturbances")
    mm  = extract_metric_series(payloads, "mindful_minutes")
    alc = extract_metric_series(payloads, "alcohol_consumption")

    sleep = extract_sleep_analysis(payloads)
    sleep_score = derive_sleep_score(sleep)

    workouts_daily, hrr = compute_zone_minutes_and_hrr1(payloads)

    # merge outer on date
    dfs = [workouts_daily, hrr, hrv, rhr, rr, bd, mm, alc, sleep_score]
    base = None
    for d in dfs:
        if d is None or d.empty:
            continue
        base = d if base is None else base.merge(d, on="date", how="outer")

    if base is None:
        return pd.DataFrame(columns=["date"])

    base = base.sort_values("date")
    return base
