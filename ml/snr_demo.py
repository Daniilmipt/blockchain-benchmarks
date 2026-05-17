"""Демо: проверка пайплайна анализа SNR на данных, которые уже лежат в репо.

На checkpoint1 нет scores_snr.txt, но есть `design/Y_initial` — TPS на 317
точках начального плана эксперимента (Latin Hypercube). Для каждой точки
там M=1, поэтому полноценный SNR с разделением сигнал/шум по этой выборке
оценить нельзя, но можно показать общий разброс TPS и положение нашего
синтетического примера tps(delay) относительно него.

После того как прогоните snr_delay_experiment.py, получите свой CSV
и считайте по нему snr_analysis.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from snr_analysis import compute_snr, plot_tps_vs_delay


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def overview_design_data():
    print("=" * 64)
    print("1) Что лежит в design/Y_initial (BO-инициализация, M=1 на точку)")
    print("=" * 64)
    y_path = REPO_ROOT / "design" / "Y_initial"
    if not y_path.exists():
        print(f"(не найден {y_path}, пропускаем)")
        return
    y = pd.read_csv(y_path, header=None).iloc[:, 0]
    print(f"Точек:                        {len(y)}")
    print(f"TPS min/max:                  {y.min():.2f} / {y.max():.2f}")
    print(f"Mean ± std:                   {y.mean():.2f} ± {y.std(ddof=0):.2f}")
    print(f"Размах (max-min):             {y.max() - y.min():.2f}")
    print("Каждая точка измерена один раз, поэтому шум и сигнал в этих")
    print("данных неразличимы. Полноценный SNR нужно мерить отдельным")
    print("свипом с M>>1 в каждой точке (см. snr_delay_experiment.py).\n")


def synthetic_delay_demo():
    """Имитация эксперимента tps(delay) (форма из Figure 2 + шум).

    Только чтобы показать ожидаемый вид графика и SNR-расчёта.
    Реальные значения должны прийти из snr_delay_experiment.py.
    """
    print("=" * 64)
    print("2) Синтетика tps(delay) (форма из Figure 2 статьи, N=10, M=10)")
    print("=" * 64)
    rng = np.random.default_rng(42)
    delays = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    mean_tps = np.array([23.6, 23.1, 23.4, 26.1, 35.5, 28.1, 28.3, 29.6, 29.6, 29.4])
    noise_std = np.array([0.6, 0.7, 0.8, 1.4, 3.0, 1.4, 1.1, 1.0, 0.9, 0.9])

    rows = []
    for theta_n, mu, sd in zip(delays, mean_tps, noise_std):
        for m in range(1, 11):
            rows.append((int(theta_n), m, float(rng.normal(mu, sd))))
    df = pd.DataFrame(rows, columns=["delay_s", "run_id", "tps"])
    df.to_csv(HERE / "snr_delay_results_demo.csv", index=False)
    print(f"синтетические данные → {HERE / 'snr_delay_results_demo.csv'}")

    stats = compute_snr(df, theta_col="delay_s", value_col="tps")
    print(f"N                   : {stats['N']}")
    print(f"Среднее TPS         : {stats['F_bar_nm']:.4f}")
    print(f"var_θ E_ξ F         : {stats['numerator (var_theta E_xi F)']:.6f}")
    print(f"E_θ var_ξ F         : {stats['denominator (E_theta var_xi F)']:.6f}")
    print(f"SNR                 : {stats['SNR']:.4f}")
    print(f"SNR_db              : {stats['SNR_db']:.3f} dB")

    plot_tps_vs_delay(
        stats, theta_label="delay, seconds", value_label="tps",
        out_path=HERE / "tps_vs_delay_demo.png",
    )


if __name__ == "__main__":
    overview_design_data()
    synthetic_delay_demo()
