"""SNR-анализ для эксперимента по block-параметрам с latency.

Поддерживает CSV формата snr_block_*_latency.csv:
    design_idx, run_id, MaxMessageCount, AbsoluteMaxBytes, PreferredMaxBytes,
    BatchTimeout, tps,
    tps_create, tps_change, tps_query_all, tps_query_one,
    lat_avg_create, lat_avg_change, lat_avg_query_all, lat_avg_query_one,
    lat_max_create, lat_max_change, lat_max_query_all, lat_max_query_one.

Использование:
    cd ml
    python snr_block_latency_analysis.py --input snr_block_10x3_latency.csv
    python snr_block_latency_analysis.py --input snr_block_10x3_latency.csv \
        --plot tps_vs_batchtimeout.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Группы метрик. "direction": LB = larger-is-better (TPS), SB = smaller-is-better (latency).
METRIC_GROUPS = [
    ("TPS (write rounds — should depend on block params)", [
        ("tps_create", "LB"),
        ("tps_change", "LB"),
    ]),
    ("TPS (read-only rounds — should NOT depend on block params)", [
        ("tps_query_all", "LB"),
        ("tps_query_one", "LB"),
    ]),
    ("Avg latency (write rounds — should depend on block params)", [
        ("lat_avg_create", "SB"),
        ("lat_avg_change", "SB"),
    ]),
    ("Avg latency (read-only rounds — should NOT depend)", [
        ("lat_avg_query_all", "SB"),
        ("lat_avg_query_one", "SB"),
    ]),
    ("Max latency (write rounds)", [
        ("lat_max_create", "SB"),
        ("lat_max_change", "SB"),
    ]),
    ("Mean TPS (legacy avg over 4 rounds — diluted by queries)", [
        ("tps", "LB"),
    ]),
]


def _is_pos(x: np.ndarray) -> bool:
    return bool(np.all(x > 0))


def variance_snr(values: pd.Series, groups: pd.Series) -> tuple[float, float, float, float]:
    """Variance-based SNR.

    SNR = Var(point means) / mean(Var within point).
    Возвращает: SNR, 10*log10(SNR) в дБ, std(point means), sqrt(mean within-var).
    """
    df = pd.DataFrame({"y": values.astype(float), "g": groups})
    df = df.dropna(subset=["y"])
    if df.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    means = df.groupby("g")["y"].mean()
    within_var = df.groupby("g")["y"].var(ddof=1).fillna(0.0)
    between_var = means.var(ddof=1)
    mean_within = within_var.mean()
    if mean_within <= 0 or not np.isfinite(between_var) or not np.isfinite(mean_within):
        snr = float("nan")
        snr_db = float("nan")
    else:
        snr = between_var / mean_within
        snr_db = 10.0 * math.log10(snr) if snr > 0 else float("-inf")
    return snr, snr_db, math.sqrt(between_var) if between_var > 0 else 0.0, math.sqrt(mean_within)


def taguchi_snr_db(values: np.ndarray, direction: str) -> float:
    """Taguchi S/N в дБ. LB = larger-is-better, SB = smaller-is-better.

    LB: -10*log10(mean(1/y^2))
    SB: -10*log10(mean(y^2))
    Если все значения нули — для SB это +inf (нет шума), вернём NaN
    (бессмысленный кейс, всё равно метрика близка к "детерминистическому 0").
    """
    y = values.astype(float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    if direction == "LB":
        y = y[y > 0]
        if y.size == 0:
            return float("nan")
        return float(-10.0 * math.log10(np.mean(1.0 / (y ** 2))))
    elif direction == "SB":
        mse = float(np.mean(y ** 2))
        if mse <= 0:
            return float("nan")
        return float(-10.0 * math.log10(mse))
    else:
        raise ValueError(direction)


def per_point_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("design_idx")[col].agg(["mean", "std", "min", "max"]).round(4)
    g["CV%"] = (g["std"] / g["mean"] * 100).round(2)
    g["range"] = (g["max"] - g["min"]).round(4)
    return g


def analyze_metric(df: pd.DataFrame, col: str, direction: str) -> dict:
    s = df[col].astype(float)
    snr_var, snr_db, sig, noise = variance_snr(s, df["design_idx"])
    means = df.groupby("design_idx")[col].mean()
    rng_abs = float(means.max() - means.min())
    rng_rel = rng_abs / float(means.mean()) * 100 if means.mean() != 0 else float("nan")
    # Taguchi считаем "по точкам" — каждой точке свой score, потом среднее по точкам.
    # Это даёт Taguchi-like SNR distribution и общий показатель.
    per_pt_taguchi = []
    for _, sub in df.groupby("design_idx")[col]:
        per_pt_taguchi.append(taguchi_snr_db(sub.to_numpy(), direction))
    taguchi_mean = float(np.nanmean(per_pt_taguchi))
    taguchi_range = float(np.nanmax(per_pt_taguchi) - np.nanmin(per_pt_taguchi))
    return {
        "metric": col,
        "dir": direction,
        "mean": float(means.mean()),
        "between_std": sig,
        "within_std": noise,
        "range_abs": rng_abs,
        "range_rel%": rng_rel,
        "SNR_var": snr_var,
        "SNR_var_dB": snr_db,
        "Taguchi_mean_dB": taguchi_mean,
        "Taguchi_range_dB": taguchi_range,
    }


def print_summary(df: pd.DataFrame) -> None:
    pts = sorted(df["design_idx"].unique())
    n = len(pts)
    m_per_pt = df.groupby("design_idx").size()
    print(f"Файл: {df.shape[0]} строк; N={n} точек; M={m_per_pt.min()}…{m_per_pt.max()} повторов.")
    print("Колонки параметров видны в полном CSV; ниже — варианты диапазонов:")
    for p in ["MaxMessageCount", "AbsoluteMaxBytes", "PreferredMaxBytes", "BatchTimeout"]:
        if p in df.columns:
            u = sorted(df[p].unique())
            print(f"  {p:<20}: {len(u)} уникальных значений, range=[{u[0]}, {u[-1]}]")
    print()


def format_table(rows: list[dict]) -> str:
    cols = [
        "metric", "dir", "mean", "between_std", "within_std",
        "range_abs", "range_rel%", "SNR_var", "SNR_var_dB",
        "Taguchi_mean_dB", "Taguchi_range_dB",
    ]
    headers = {
        "metric": "metric", "dir": "dir",
        "mean": "mean", "between_std": "σ_between",
        "within_std": "σ_within", "range_abs": "range",
        "range_rel%": "range%", "SNR_var": "SNR",
        "SNR_var_dB": "SNR(dB)", "Taguchi_mean_dB": "Taguchi(dB)",
        "Taguchi_range_dB": "Taguchi_Δ(dB)",
    }
    widths = {c: max(len(headers[c]), max((len(_fmt_cell(r, c)) for r in rows), default=0)) for c in cols}
    line = " | ".join(headers[c].ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    out = [line, sep]
    for r in rows:
        out.append(" | ".join(_fmt_cell(r, c).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def _fmt_cell(r: dict, c: str) -> str:
    v = r.get(c, "")
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        if c in ("SNR_var",):
            return f"{v:.3f}"
        if c.endswith("dB"):
            return f"{v:+.2f}"
        if c.endswith("%"):
            return f"{v:.2f}"
        if abs(v) < 0.01 or abs(v) >= 1000:
            return f"{v:.4g}"
        return f"{v:.4f}"
    return str(v)


def maybe_plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("tps_create",     "TPS — Create a car (write)",     axes[0, 0]),
        ("tps_change",     "TPS — Change car owner (rw)",    axes[0, 1]),
        ("lat_avg_create", "Avg latency — Create (s)",       axes[1, 0]),
        ("lat_avg_change", "Avg latency — Change (s)",       axes[1, 1]),
    ]
    grp = df.groupby("design_idx")
    bt = grp["BatchTimeout"].first().to_numpy()
    order = np.argsort(bt)
    bt_sorted = bt[order]
    for col, title, ax in panels:
        means = grp[col].mean().to_numpy()[order]
        stds = grp[col].std(ddof=1).to_numpy()[order]
        ax.errorbar(bt_sorted, means, yerr=stds, fmt="o-", capsize=3)
        ax.set_title(title)
        ax.set_xlabel("BatchTimeout (s)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Block params SNR experiment — sorted by BatchTimeout (mean ± std over M repeats)",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"График сохранён: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SNR-анализ для block-эксперимента с latency")
    parser.add_argument("--input", required=True, help="Путь к CSV с колонками tps_*, lat_avg_*, lat_max_*")
    parser.add_argument("--plot", default=None, help="Если указан — сохранить мульти-панельный график")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"файл не найден: {in_path}")
    df = pd.read_csv(in_path)
    print(f"\n=== Файл: {in_path} ===")
    print_summary(df)

    for group_title, metrics in METRIC_GROUPS:
        rows = []
        for col, direction in metrics:
            if col not in df.columns:
                continue
            rows.append(analyze_metric(df, col, direction))
        if not rows:
            continue
        print(f"\n### {group_title}")
        print(format_table(rows))

    print("\n=== Per-point stats для ключевых метрик (write rounds) ===")
    for col in ["tps_create", "lat_avg_create", "tps_change", "lat_avg_change"]:
        if col in df.columns:
            print(f"\n-- {col} --")
            print(per_point_stats(df, col).to_string())

    if args.plot:
        maybe_plot(df, Path(args.plot))


if __name__ == "__main__":
    main()
