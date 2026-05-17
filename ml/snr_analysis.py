"""Анализ результатов SNR-эксперимента: считает SNR (delta-method) и строит график tps(delay).

Используется формула из работы:

    SNR = var_theta E_xi F / E_theta var_xi F

Estimator:
    F_bar_m(n)   = (1/M) * sum_m F(x0, theta_n, xi_{n,m})           # среднее по M в точке n
    F_bar_nm     = (1/N) * sum_n F_bar_m(n)                          # глобальное среднее
    numerator    = (1/N) * sum_n (F_bar_m(n) - F_bar_nm)^2           # разброс сигнала
    denominator  = (1/N) * sum_n ((1/M) * sum_m (F - F_bar_m(n))^2)  # средний разброс шума
    SNR_hat      = numerator / denominator
    SNR_db       = 10 * log10(SNR_hat)

Поддерживаются три формата CSV (распознаются автоматически):
  - delay_multi:  delay_s, run_id, tps_create, tps_change, tps_query_all,
                  tps_query_one, tps_avg     (новый snr_delay_experiment.py)
  - delay_sweep:  delay_s, run_id, tps       (старый 1D-формат)
  - block_design: design_idx, run_id, BatchTimeout, MaxMessageCount,
                  AbsoluteMaxBytes, PreferredMaxBytes, tps   (4D SLHC)

Для delay_multi считается отдельный SNR для каждого из 5 TPS-метрик
(4 раунда + среднее) и рисуется одна фигура с 5 субплотами.

Использование:
    python snr_analysis.py --csv snr_delay_results.csv
    python snr_analysis.py --csv snr_block_results.csv --plot tps_vs_delay.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_snr(df: pd.DataFrame, theta_col: str, value_col: str) -> dict:
    """Считает SNR по delta-методу.

    df ожидается длинного формата: одна строка = (theta, xi-realization).
    """
    df = df.dropna(subset=[value_col]).copy()

    # F_bar_m(n) — среднее по M в каждой точке n
    per_theta_mean = df.groupby(theta_col)[value_col].mean()
    per_theta_var_ddof0 = df.groupby(theta_col)[value_col].var(ddof=0)
    per_theta_count = df.groupby(theta_col)[value_col].count()

    N = len(per_theta_mean)
    F_bar_nm = per_theta_mean.mean()  # глобальное среднее средних

    # числитель: разброс F_bar_m по theta
    numerator = float(((per_theta_mean - F_bar_nm) ** 2).mean())
    # знаменатель: средний по theta внутренний разброс (M-усреднение)
    denominator = float(per_theta_var_ddof0.mean())

    snr = numerator / denominator if denominator > 0 else math.inf
    snr_db = 10 * math.log10(snr) if snr > 0 and math.isfinite(snr) else float("nan")

    return {
        "N": N,
        "M_per_theta": per_theta_count.to_dict(),
        "F_bar_nm": float(F_bar_nm),
        "numerator (var_theta E_xi F)": numerator,
        "denominator (E_theta var_xi F)": denominator,
        "SNR": snr,
        "SNR_db": snr_db,
        "per_theta_mean": per_theta_mean,
        "per_theta_std": df.groupby(theta_col)[value_col].std(ddof=0),
    }


def plot_tps_vs_delay(stats: dict, theta_label: str, value_label: str, out_path: Path | None) -> None:
    mean_series = stats["per_theta_mean"].sort_index()
    std_series = stats["per_theta_std"].reindex(mean_series.index)

    x = mean_series.index.to_numpy()
    y = mean_series.to_numpy()
    s = std_series.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, y - s, y + s, alpha=0.25, color="tab:purple", label="mean ± std")
    ax.plot(x, y, marker="o", color="tab:purple", label="mean")
    ax.set_xlabel(theta_label)
    ax.set_ylabel(value_label)
    ax.set_title(f"{value_label} as a function of {theta_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    snr_db_text = f"SNR = {stats['SNR']:.2f} ({stats['SNR_db']:.2f} dB)"
    ax.text(
        0.02, 0.98, snr_db_text,
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"),
    )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"график сохранён в {out_path}")
    else:
        plt.show()


def plot_tps_vs_delay_block(df: pd.DataFrame, stats: dict, delay_col: str, out_path: Path | None) -> None:
    """Скаттер TPS vs delay для многомерного дизайна (delay — один из ≥2 параметров).

    Каждая design-точка получает свою (mean ± std) над M повторами.
    Остальные параметры показываются в подписи рядом с маркером.
    """
    per_idx_mean = df.groupby("design_idx")["tps"].mean()
    per_idx_std = df.groupby("design_idx")["tps"].std(ddof=0)
    per_idx_delay = df.groupby("design_idx")[delay_col].first()

    # упорядочим по delay для красоты
    order = per_idx_delay.sort_values().index
    x = per_idx_delay.loc[order].to_numpy()
    y = per_idx_mean.loc[order].to_numpy()
    s = per_idx_std.loc[order].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, y, yerr=s, fmt="o", color="tab:purple",
                ecolor="tab:purple", elinewidth=1.2, capsize=4, label="design points (mean ± std)")
    # тонкая линия между соседними по delay точками (просто чтобы глазу легче)
    ax.plot(x, y, color="tab:purple", alpha=0.35, linewidth=1)

    ax.set_xlabel(f"{delay_col} (delay, seconds)")
    ax.set_ylabel("tps")
    ax.set_title("TPS vs delay  (4D SLHC, остальные block-параметры варьируются)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    snr_text = f"SNR = {stats['SNR']:.2f} ({stats['SNR_db']:.2f} dB)\nN={stats['N']}, M(avg)={np.mean(list(stats['M_per_theta'].values())):.1f}"
    ax.text(0.02, 0.98, snr_text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"график сохранён в {out_path}")
    else:
        plt.show()


TPS_COLUMNS_MULTI = ["tps_create", "tps_change", "tps_query_all", "tps_query_one", "tps_avg"]
TPS_PRETTY = {
    "tps_create":    "Create a car (write)",
    "tps_change":    "Change car owner (write)",
    "tps_query_all": "Query all cars (read)",
    "tps_query_one": "Query a car (read)",
    "tps_avg":       "Average over all rounds",
}


def load_results(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Возвращает (df, mode), где mode ∈ {'delay_multi', 'delay_sweep', 'block_design'}."""
    df = pd.read_csv(csv_path)
    if {"design_idx", "run_id", "tps", "BatchTimeout"}.issubset(df.columns):
        return df, "block_design"
    if {"delay_s", "run_id", *TPS_COLUMNS_MULTI}.issubset(df.columns):
        return df, "delay_multi"
    if {"delay_s", "run_id", "tps"}.issubset(df.columns):
        return df, "delay_sweep"
    raise ValueError(f"непонятный формат CSV, колонки: {list(df.columns)}")


def plot_tps_vs_delay_multi(df: pd.DataFrame, stats_per_col: dict, out_path: Path | None) -> None:
    """Одна фигура с 5 субплотами: 4 раунда + tps_avg. Каждый — TPS vs delay со своим SNR."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    axes = axes.flatten()
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for ax, col, color in zip(axes, TPS_COLUMNS_MULTI, palette):
        stats = stats_per_col[col]
        mean_series = stats["per_theta_mean"].sort_index()
        std_series = stats["per_theta_std"].reindex(mean_series.index)

        x = mean_series.index.to_numpy()
        y = mean_series.to_numpy()
        s = std_series.to_numpy()

        ax.fill_between(x, y - s, y + s, alpha=0.25, color=color, label="mean ± std")
        ax.plot(x, y, marker="o", color=color, label="mean")
        ax.set_xlabel("delay, seconds")
        ax.set_ylabel("TPS")
        ax.set_title(TPS_PRETTY[col])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        snr_text = (
            f"SNR = {stats['SNR']:.2f}\n"
            f"({stats['SNR_db']:.2f} dB)\n"
            f"N={stats['N']}, M(avg)={np.mean(list(stats['M_per_theta'].values())):.1f}"
        )
        ax.text(0.02, 0.98, snr_text, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"), fontsize=9)

    # последний пустой subplot (3x2 = 6, у нас 5 графиков) — скроем
    for ax in axes[len(TPS_COLUMNS_MULTI):]:
        ax.set_visible(False)

    fig.suptitle("TPS vs delay  (4 раунда + среднее, SNR по каждому)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"график сохранён в {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="snr_delay_results.csv", help="Файл с результатами эксперимента")
    parser.add_argument("--plot", default="tps_vs_delay.png", help="Куда сохранить график (или 'show')")
    args = parser.parse_args()

    df, mode = load_results(Path(args.csv))
    print(f"Режим CSV                     : {mode}")

    if mode == "delay_multi":
        stats_per_col = {}
        print("\nSNR по каждой TPS-метрике (delta-method):")
        for col in TPS_COLUMNS_MULTI:
            stats = compute_snr(df, theta_col="delay_s", value_col=col)
            stats_per_col[col] = stats
            print(
                f"  {TPS_PRETTY[col]:<28} N={stats['N']:<2} "
                f"F̄={stats['F_bar_nm']:>8.2f}  "
                f"var_θ E_ξ F={stats['numerator (var_theta E_xi F)']:>10.2f}  "
                f"E_θ var_ξ F={stats['denominator (E_theta var_xi F)']:>10.2f}  "
                f"SNR={stats['SNR']:>7.3f} ({stats['SNR_db']:>6.2f} dB)"
            )
        out_path = None if args.plot == "show" else Path(args.plot)
        plot_tps_vs_delay_multi(df, stats_per_col, out_path)
        return

    theta_col = "design_idx" if mode == "block_design" else "delay_s"
    stats = compute_snr(df, theta_col=theta_col, value_col="tps")

    print(f"N (точек по theta)            : {stats['N']}")
    print(f"M (повторов на точку)         : {stats['M_per_theta']}")
    print(f"Среднее TPS (F̄)              : {stats['F_bar_nm']:.4f}")
    print(f"Числитель var_θ E_ξ F         : {stats['numerator (var_theta E_xi F)']:.6f}")
    print(f"Знаменатель E_θ var_ξ F       : {stats['denominator (E_theta var_xi F)']:.6f}")
    print(f"SNR                           : {stats['SNR']:.4f}")
    print(f"SNR_db                        : {stats['SNR_db']:.3f} dB")

    out_path = None if args.plot == "show" else Path(args.plot)
    if mode == "block_design":
        plot_tps_vs_delay_block(df, stats, delay_col="BatchTimeout", out_path=out_path)
    else:
        plot_tps_vs_delay(stats, theta_label="delay, seconds", value_label="tps", out_path=out_path)


if __name__ == "__main__":
    main()
