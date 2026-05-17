"""SNR-эксперимент: TPS как функция delay-паузы перед benchmark.

Что такое delay:
    Это просто time.sleep между deployCC и caliper launch — пауза, в течение
    которой сеть «остывает» / достигает стационарного состояния перед замером.
    Никакие параметры Fabric (включая Orderer.BatchTimeout) при этом не меняются —
    BatchTimeout, MaxMessageCount и пр. оставлены под отдельный эксперимент с BO.

Что меряем:
    Caliper прогоняет 4 раунда: Create a car, Change car owner, Query all cars,
    Query a car. На каждый раунд получаем свой Throughput (TPS). Сохраняем все 4
    значения + среднее по ним. Итого 5 TPS-метрик на каждый прогон.

Эксперимент:
    theta_n = delay_n,  n = 1..N
    Для каждого theta_n  M раз: start_network -> deployCC -> sleep(delay) -> caliper.
    На каждый прогон в CSV пишется одна строка: delay_s,run_id,tps_create,
    tps_change,tps_query_all,tps_query_one,tps_avg.

CSV пишется построчно, --resume пропускает уже собранные (delay, run_id),
а строки с любыми NaN в TPS перевыполняются заново.

Использование (из ml/):
    python snr_delay_experiment.py
    python snr_delay_experiment.py --delays 10 20 30 40 50 60 70 80 90 -m 10
"""

import argparse
import math
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from network_manager import (  # noqa: E402
    start_network,
    start_benchmark,
    stop_network,
    observe_round_tps,
    ROUND_KEYS,
)


CSV_COLUMNS = ["delay_s", "run_id"] + ROUND_KEYS


def measure_tps_once(delay_seconds: int) -> dict:
    """Один прогон: network up + deployCC -> sleep(delay) -> caliper -> parse -> down."""
    try:
        start_network()
        if delay_seconds > 0:
            print(f"[sleep] {delay_seconds}s стационаризация перед benchmark")
            time.sleep(delay_seconds)
        start_benchmark()
        return observe_round_tps()
    finally:
        stop_network()


def _row_is_complete(parts: list) -> bool:
    """В CSV-строке должны быть все 5 TPS и ни одного NaN/пусто."""
    if len(parts) != len(CSV_COLUMNS):
        return False
    for cell in parts[2:]:
        if cell == "" or cell.lower() == "nan":
            return False
        try:
            v = float(cell)
        except ValueError:
            return False
        if math.isnan(v):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="SNR experiment: TPS vs delay (sleep before benchmark)")
    parser.add_argument(
        "--delays", type=int, nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        help="Список значений delay (секунды паузы перед запуском caliper).",
    )
    parser.add_argument(
        "-m", "--repeats", type=int, default=10,
        help="Сколько раз повторить замер при фиксированном delay (M).",
    )
    parser.add_argument(
        "--output", type=str, default="snr_delay_results.csv",
        help="CSV-файл с результатами.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Не перезатирать output: успешные строки сохраняются, NaN-и переразвернутся.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    existing: set[tuple[int, int]] = set()

    if args.resume and output.exists():
        # читаем все строки, разделяем на «удачные» (полные) и «неудачные»
        with open(output, "r") as f:
            first = next(f, None)
            kept_rows: list[str] = []
            for line in f:
                parts = line.strip().split(",")
                if _row_is_complete(parts):
                    existing.add((int(parts[0]), int(parts[1])))
                    kept_rows.append(line if line.endswith("\n") else line + "\n")
        # перезаписываем CSV, оставляя только полные строки + новую шапку (вдруг колонки сменились)
        with open(output, "w") as f:
            f.write(",".join(CSV_COLUMNS) + "\n")
            f.writelines(kept_rows)
        print(f"[resume] пропустим {len(existing)} уже собранных (delay, run_id)")
    else:
        with open(output, "w") as f:
            f.write(",".join(CSV_COLUMNS) + "\n")

    for delay in args.delays:
        print(f"\n##### delay = {delay}s #####")

        for m in range(1, args.repeats + 1):
            if (delay, m) in existing:
                print(f"[skip] delay={delay}s run={m} (уже в CSV)")
                continue
            print(f"\n=== delay={delay}s, run {m}/{args.repeats} ===")
            t0 = time.time()
            tps = {k: float("nan") for k in ROUND_KEYS}
            try:
                tps = measure_tps_once(delay)
            except Exception as e:
                print(f"[error] delay={delay}s run={m}: {e}")
            dt = time.time() - t0
            print(f"[result] delay={delay}s run={m}: {tps} ({dt:.1f}s)")
            with open(output, "a") as f:
                cells = [str(delay), str(m)] + [
                    f"{tps[k]:.6f}" if tps[k] == tps[k] else "nan"
                    for k in ROUND_KEYS
                ]
                f.write(",".join(cells) + "\n")

    print(f"\nГотово. Результаты в {output}")


if __name__ == "__main__":
    main()
