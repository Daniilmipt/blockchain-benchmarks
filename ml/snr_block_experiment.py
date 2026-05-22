# -*- coding: utf-8 -*-
"""SNR-эксперимент по 4 channel-параметрам Orderer-а (широкие границы).

Параметры (только из configtx.yaml; core.yaml org{1..4} не трогаем):
    - Orderer.BatchSize.MaxMessageCount   : 10 … 5000
    - Orderer.BatchSize.AbsoluteMaxBytes  : 10 … 200    MB
    - Orderer.BatchSize.PreferredMaxBytes : 512 … 10000 KB
    - Orderer.BatchTimeout                : 500 … 5000  ms  (= 0.5 … 5 s)

BatchTimeout внутри хранится/сэмплируется в **миллисекундах** (целое), а в CSV
пишется в **секундах** (float). В YAML configtx пишем "Xs", либо "Xms" если не
кратно секунде.

Использование:
    cd ml
    # 1) Сгенерировать SLHC (N=10) и сохранить дизайн рядом с design/X_initial:
    python snr_block_experiment.py --design-only -n 10 --seed 42 \
        --save-design ../design/X_initial_block.csv

    # 2) Прогнать эксперимент по сохранённому файлу с M=10 повторов:
    python snr_block_experiment.py \
        --design-file ../design/X_initial_block.csv -m 10 \
        --output snr_block_results.csv

    # Можно и в одну команду — сгенерировать, сохранить и сразу прогнать:
    python snr_block_experiment.py -n 10 -m 10 --seed 42 \
        --save-design ../design/X_initial_block.csv \
        --output snr_block_results.csv

    # Опционально добор BO поверх SLHC:
    python snr_block_experiment.py --design-file ../design/X_initial_block.csv \
        -m 10 --bo-iters 30 --af EI
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pySOT.experimental_design import SymmetricLatinHypercube

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from network_manager import (  # noqa: E402
    start_network,
    start_benchmark,
    stop_network,
    observe_round_metrics,
    ROUND_SHORTS,
)

REPO_ROOT = HERE.parent
CONFIGTX_PATH = REPO_ROOT / "networks" / "configtx" / "configtx.yaml"
DEFAULT_BLOCK_DESIGN_CSV = REPO_ROOT / "design" / "X_initial_block.csv"


# Широкие границы для SNR-разведки. BatchTimeout — целочисленные миллисекунды.
BLOCK_PARAMS = [
    {"name": "MaxMessageCount",   "bounds": (10, 5000),    "unit": ""},
    {"name": "AbsoluteMaxBytes",  "bounds": (10, 200),     "unit": " MB"},
    {"name": "PreferredMaxBytes", "bounds": (512, 10000),  "unit": " KB"},
    {"name": "BatchTimeout",      "bounds": (500, 5000),   "unit": "ms (в CSV — секунды)"},
]


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def batch_timeout_to_yaml(ms: int) -> str:
    ms = max(1, int(ms))
    return f"{ms // 1000}s" if ms % 1000 == 0 else f"{ms}ms"


def apply_block_params(configtx: dict, values: dict) -> None:
    """Пишем BatchSize/BatchTimeout в корневой Orderer и во все Profiles.<*>.Orderer."""
    mmc = int(values["MaxMessageCount"])
    abs_mb = int(values["AbsoluteMaxBytes"])
    pref_kb = int(values["PreferredMaxBytes"])
    t_ms = int(values["BatchTimeout_ms"])

    def _write(orderer):
        if not orderer:
            return
        bs = orderer.get("BatchSize")
        if bs is not None:
            if "MaxMessageCount" in bs:
                bs["MaxMessageCount"] = mmc
            if "AbsoluteMaxBytes" in bs:
                bs["AbsoluteMaxBytes"] = f"{abs_mb} MB"
            if "PreferredMaxBytes" in bs:
                bs["PreferredMaxBytes"] = f"{pref_kb} KB"
        if "BatchTimeout" in orderer:
            orderer["BatchTimeout"] = batch_timeout_to_yaml(t_ms)

    _write(configtx.get("Orderer"))
    profiles = configtx.get("Profiles", {}) or {}
    for profile in profiles.values():
        _write((profile or {}).get("Orderer"))


def measure_once(delay_seconds: int = 0) -> dict:
    """Один прогон: network up + deployCC -> sleep(delay) -> caliper -> parse -> down.

    Возвращает плоский dict со всеми метриками по раундам (см.
    network_manager.observe_round_metrics) + ключ 'tps' (= tps_avg) для
    обратной совместимости с прежним кодом/анализом.
    """
    try:
        start_network()
        if delay_seconds and delay_seconds > 0:
            print(f"[sleep] {delay_seconds}s стационаризация перед benchmark")
            time.sleep(delay_seconds)
        start_benchmark()
        m = observe_round_metrics()
        m["tps"] = m.get("tps_avg", float("nan"))
        return m
    finally:
        stop_network()


def empty_metrics() -> dict:
    """Возвращает dict со всеми ключами метрик, заполненный NaN."""
    keys = (
        [f"tps_{s}" for s in ROUND_SHORTS]
        + [f"lat_avg_{s}" for s in ROUND_SHORTS]
        + [f"lat_max_{s}" for s in ROUND_SHORTS]
        + [f"lat_min_{s}" for s in ROUND_SHORTS]
        + ["tps_avg", "tps"]
    )
    return {k: float("nan") for k in keys}


# Порядок колонок метрик в CSV (фиксирован для совместимости анализа)
METRIC_COLUMNS = (
    ["tps"]
    + [f"tps_{s}" for s in ROUND_SHORTS]
    + [f"lat_avg_{s}" for s in ROUND_SHORTS]
    + [f"lat_max_{s}" for s in ROUND_SHORTS]
)


def design_column_names():
    return [p["name"] for p in BLOCK_PARAMS]


def generate_design(n_pts: int, seed: int) -> np.ndarray:
    """SLHC в 4D, все оси целочисленные (BatchTimeout — в ms)."""
    np.random.seed(seed)
    dim = len(BLOCK_PARAMS)
    if n_pts < 2 * dim + 1:
        print(
            f"[warn] для SLHC рекомендуется num_pts >= 2*dim+1 = {2 * dim + 1}; "
            f"запрошено {n_pts}",
            file=sys.stderr,
        )
    lb = np.array([float(p["bounds"][0]) for p in BLOCK_PARAMS])
    ub = np.array([float(p["bounds"][1]) for p in BLOCK_PARAMS])
    int_var = np.arange(dim, dtype=int)
    slhd = SymmetricLatinHypercube(dim=dim, num_pts=n_pts)
    pts = slhd.generate_points(lb=lb, ub=ub, int_var=int_var)
    return np.round(pts).astype(int)


def save_design_csv(path: Path, design: np.ndarray, seed: int) -> None:
    """Сохранить N×4 дизайн в CSV (BatchTimeout — миллисекунды)."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(design, columns=design_column_names())
    with open(path, "w") as f:
        f.write(f"# seed={seed}; SLHC по 4 блок-параметрам; BatchTimeout в ms (500..5000)\n")
    df.to_csv(path, mode="a", index=False)


def load_design_csv(path: Path) -> np.ndarray:
    """Загрузить N×4 из CSV (комментарии с # игнорируются)."""
    path = path.resolve()
    if not path.is_file():
        sys.exit(f"файл дизайна не найден: {path}")
    df = pd.read_csv(path, comment="#")
    names = design_column_names()
    missing = [n for n in names if n not in df.columns]
    if missing:
        sys.exit(
            f"{path}: нет колонок {missing}. Ожидаются: {', '.join(names)} "
            "(BatchTimeout — целые миллисекунды)"
        )
    if len(df) < 1:
        sys.exit(f"{path}: пустой файл дизайна")
    raw = df[names].to_numpy(dtype=float)
    out = np.empty_like(raw, dtype=int)
    for j, p in enumerate(BLOCK_PARAMS):
        lo, hi = p["bounds"]
        out[:, j] = np.clip(np.round(raw[:, j]), lo, hi).astype(int)
    return out


def point_to_values(point) -> dict:
    """Точка дизайна → словарь для apply_block_params и для CSV."""
    b = BLOCK_PARAMS
    mmc = int(np.clip(int(round(point[0])), b[0]["bounds"][0], b[0]["bounds"][1]))
    abs_b = int(np.clip(int(round(point[1])), b[1]["bounds"][0], b[1]["bounds"][1]))
    pref_kb = int(np.clip(int(round(point[2])), b[2]["bounds"][0], b[2]["bounds"][1]))
    t_ms = int(np.clip(int(round(point[3])), b[3]["bounds"][0], b[3]["bounds"][1]))
    return {
        "MaxMessageCount": mmc,
        "AbsoluteMaxBytes": abs_b,
        "PreferredMaxBytes": pref_kb,
        "BatchTimeout_ms": t_ms,
        "BatchTimeout_sec": t_ms / 1000.0,
    }


def _fmt(v) -> str:
    return f"{v:.6f}" if isinstance(v, float) and v == v else "nan"


def run_slhc_phase(design: np.ndarray, m_repeats: int, output: Path, resume: bool, delay_seconds: int = 0) -> pd.DataFrame:
    cols = ["design_idx", "run_id"] + design_column_names() + METRIC_COLUMNS
    existing = set()
    if resume and output.exists():
        df0 = pd.read_csv(output)
        for _, row in df0.iterrows():
            existing.add((int(row["design_idx"]), int(row["run_id"])))
    else:
        pd.DataFrame(columns=cols).to_csv(output, index=False)

    for n, point in enumerate(design, start=1):
        values = point_to_values(point)
        print("\n" + "=" * 60)
        print(
            f"design point {n}/{len(design)}: "
            f"MaxMessageCount={values['MaxMessageCount']}, "
            f"AbsoluteMaxBytes={values['AbsoluteMaxBytes']} MB, "
            f"PreferredMaxBytes={values['PreferredMaxBytes']} KB, "
            f"BatchTimeout={values['BatchTimeout_sec']:.3f}s ({values['BatchTimeout_ms']} ms)"
        )
        print("=" * 60)

        configtx = load_yaml(CONFIGTX_PATH)
        apply_block_params(configtx, values)
        dump_yaml(CONFIGTX_PATH, configtx)

        for m in range(1, m_repeats + 1):
            if (n, m) in existing:
                print(f"[skip] design_idx={n} run_id={m} (уже в CSV)")
                continue
            print(f"\n--- design_idx={n}, run {m}/{m_repeats} ---")
            t0 = time.time()
            try:
                metrics = measure_once(delay_seconds=delay_seconds)
            except Exception as e:
                print(f"[error] design_idx={n} run={m}: {e}")
                metrics = empty_metrics()
            dt = time.time() - t0
            print(
                f"[result] design_idx={n} run={m}: "
                f"tps={metrics.get('tps')} "
                f"lat_avg_create={metrics.get('lat_avg_create')}s "
                f"lat_avg_change={metrics.get('lat_avg_change')}s "
                f"({dt:.1f}s)"
            )
            row_vals = [
                str(values["MaxMessageCount"]),
                str(values["AbsoluteMaxBytes"]),
                str(values["PreferredMaxBytes"]),
                f"{values['BatchTimeout_sec']:.6f}",
            ] + [_fmt(metrics.get(c)) for c in METRIC_COLUMNS]
            with open(output, "a") as f:
                f.write(f"{n},{m},{','.join(row_vals)}\n")

    return pd.read_csv(output)


def run_bo_phase(slhc_df: pd.DataFrame, n_iters: int, af: str) -> None:
    """Опциональный GPyOpt поверх SLHC-средних. Домены big: integer-диапазоны полны."""
    import GPyOpt

    print("\n" + "#" * 60)
    print(f"BO phase: af={af}, iters={n_iters}")
    print("#" * 60)

    means = slhc_df.groupby("design_idx").agg(
        {**{p["name"]: "first" for p in BLOCK_PARAMS}, "tps": "mean"}
    )
    X = means[design_column_names()].to_numpy(dtype=float)
    # В CSV BatchTimeout — секунды; для домена/GPyOpt нужны ms.
    tb_lo, tb_hi = BLOCK_PARAMS[3]["bounds"]
    X[:, 3] = np.clip(np.round(X[:, 3] * 1000.0), tb_lo, tb_hi)
    Y = means[["tps"]].to_numpy(dtype=float)

    domain = []
    for p in BLOCK_PARAMS:
        lo, hi = int(p["bounds"][0]), int(p["bounds"][1])
        domain.append({"name": p["name"], "type": "discrete", "domain": tuple(range(lo, hi + 1))})

    def bo_objective(x):
        x = np.atleast_2d(x)[0]
        values = point_to_values([int(round(x[0])), int(round(x[1])), int(round(x[2])), int(round(x[3]))])
        configtx = load_yaml(CONFIGTX_PATH)
        apply_block_params(configtx, values)
        dump_yaml(CONFIGTX_PATH, configtx)
        try:
            m = measure_once()
            tps = m.get("tps", 0.0)
            if not (tps == tps):
                tps = 0.0
        except Exception as e:
            print(f"[bo error] {values}: {e}")
            tps = 0.0
        print(f"[bo eval] {values} -> tps={tps}")
        return tps

    opt = GPyOpt.methods.BayesianOptimization(
        f=bo_objective,
        domain=domain,
        acquisition_type=af,
        initial_design_type="random",
        initial_design_numdata=1,
        maximize=True,
        normalize_Y=True,
        num_cores=1,
        X=X,
        Y=Y,
    )
    opt.run_optimization(max_iter=n_iters)
    print(f"\nBO best x: {opt.x_opt}\nBO best y: {opt.fx_opt}")


def main():
    parser = argparse.ArgumentParser(description="SNR-эксперимент по 4 channel-параметрам Orderer-а (широкие границы)")
    parser.add_argument("-n", "--num-pts", type=int, default=10, help="N точек в SLHC (если без --design-file)")
    parser.add_argument("-m", "--repeats", type=int, default=3, help="M повторов каждой точки (default 3 — быстрая итерация; для финального прогона ставим больше)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для SLHC")
    parser.add_argument(
        "--design-file",
        type=str,
        default=None,
        help="CSV с дизайном (колонки = именам BLOCK_PARAMS, BatchTimeout в ms). "
             "Если указан — SLHC не генерируется, читаются точки из файла.",
    )
    parser.add_argument(
        "--save-design",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Сохранить сгенерированный дизайн в CSV (можно вместе с --design-only). "
             f"Рекомендация: {DEFAULT_BLOCK_DESIGN_CSV.relative_to(REPO_ROOT)}",
    )
    parser.add_argument("--output", default="snr_block_results.csv", help="CSV с результатами")
    parser.add_argument("--resume", action="store_true", help="Не перезатирать output, продолжить эксперимент")
    parser.add_argument("--design-only", action="store_true", help="Только построить/напечатать дизайн и выйти")
    parser.add_argument("--bo-iters", type=int, default=0, help="Итерации BO поверх SLHC (default 0)")
    parser.add_argument("--af", default="EI", choices=["EI", "LCB", "MPI"], help="Acquisition function (default EI)")
    parser.add_argument(
        "--delay", type=int, default=0,
        help="Секунды паузы (time.sleep) между network up/deployCC и caliper. По умолчанию 0.",
    )
    args = parser.parse_args()

    if not CONFIGTX_PATH.exists() and not args.design_only:
        sys.exit(f"не найден {CONFIGTX_PATH}")

    print("Параметры эксперимента (широкий диапазон):")
    for p in BLOCK_PARAMS:
        if p["name"] == "BatchTimeout":
            lo_s = p["bounds"][0] / 1000.0
            hi_s = p["bounds"][1] / 1000.0
            print(f"  {p['name']:<20} bounds={p['bounds']} ms  (≈ {lo_s}…{hi_s} s)  {p['unit']}")
        else:
            print(f"  {p['name']:<20} bounds={p['bounds']}  unit='{p['unit']}'")

    if args.design_file:
        design_path = Path(args.design_file)
        if not design_path.is_absolute():
            design_path = (Path.cwd() / design_path).resolve()
        design = load_design_csv(design_path)
        print(f"\nДизайн загружен из файла: {design_path} (N={len(design)})")
    else:
        design = generate_design(args.num_pts, args.seed)

    if args.save_design:
        save_path = Path(args.save_design)
        if not save_path.is_absolute():
            save_path = (Path.cwd() / save_path).resolve()
        save_design_csv(save_path, design, args.seed)
        print(f"\nДизайн сохранён: {save_path}")

    disp = pd.DataFrame(design, columns=design_column_names()).copy()
    disp["BatchTimeout_s"] = disp["BatchTimeout"] / 1000.0
    disp = disp.drop(columns=["BatchTimeout"])
    src = "из файла" if args.design_file else "SLHC"
    print(f"\nДизайн {src} (N={len(design)}); BatchTimeout_s — секунды:")
    print(disp.to_string(index=True))

    if args.design_only:
        return

    out_path = Path(args.output)
    if args.delay:
        print(f"\nИспользуется пауза перед benchmark: delay={args.delay}s")
    slhc_df = run_slhc_phase(design, args.repeats, out_path, args.resume, delay_seconds=args.delay)
    print(f"\nSLHC-фаза завершена. Записано строк: {len(slhc_df)}. Файл: {out_path}")

    if args.bo_iters > 0:
        run_bo_phase(slhc_df, args.bo_iters, args.af)


if __name__ == "__main__":
    main()
