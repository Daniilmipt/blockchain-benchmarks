"""SNR-эксперимент по 4 channel-параметрам Orderer-а.

Параметры (только из configtx.yaml, ничего не трогаем в core.yaml org{1..4}):
    - Orderer.BatchSize.MaxMessageCount
    - Orderer.BatchSize.AbsoluteMaxBytes (MB)
    - Orderer.BatchSize.PreferredMaxBytes (KB)
    - Orderer.BatchTimeout (s)

Все остальные 313 параметров (per-org core.yaml + raft + всё остальное)
остаются на дефолтных значениях из YAML — 4 организации поднимаются как есть.

Диапазоны границ — та же логика, что и в domain.read_and_filter_config:
    [max(0, default // 10 * 7),  default // 10 * 13 + 5]   (~70..130% от дефолта)

План:
    1. Сгенерировать SLHC из pySOT, N точек в 4D-пространстве (default N=10).
       Все 4 параметра дискретные → int_var.
    2. Для каждой точки M раз (default M=10) перезапустить сеть и benchmark,
       записать (design_idx, m, BatchTimeout, MaxMessageCount, AbsoluteMaxBytes,
       PreferredMaxBytes, tps) в CSV.
    3. Опционально (--bo-iters K, default 0) после SLHC-сбора запустить
       GPyOpt-BO с заданным AF (default EI) на тех же 4 параметрах,
       стартуя с уже собранных средних по точкам как warm start.

После эксперимента отдельно прогоняется snr_analysis.py --csv ... для расчёта
SNR и графика TPS vs delay.

Использование:
    cd ml
    python snr_block_experiment.py                           # N=10, M=10, без BO
    python snr_block_experiment.py -n 10 -m 10 --bo-iters 50 --af EI
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
    observe_data,
)

REPO_ROOT = HERE.parent
CONFIGTX_PATH = REPO_ROOT / "networks" / "configtx" / "configtx.yaml"


# тот же расчёт границ, что в ml/domain.read_and_filter_config (ветка discrete)
def discrete_bounds(default: int) -> tuple[int, int]:
    return (max(0, (default // 10) * 7), (default // 10) * 13 + 5)


# 4 параметра орда. unit_suffix добавляется в YAML после числа.
BLOCK_PARAMS = [
    {"name": "MaxMessageCount",   "default": 10,  "unit": ""},
    {"name": "AbsoluteMaxBytes",  "default": 99,  "unit": " MB"},
    {"name": "PreferredMaxBytes", "default": 512, "unit": " KB"},
    {"name": "BatchTimeout",      "default": 2,   "unit": "s"},
]
for p in BLOCK_PARAMS:
    p["bounds"] = discrete_bounds(p["default"])


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def apply_block_params(configtx: dict, values: dict[str, int]) -> None:
    """Пишем BatchSize.{Max,Abs,Pref} и BatchTimeout и в корневой Orderer,
    и во все Profiles.<*>.Orderer (так как configtxgen берёт значения из профиля).
    """
    def _write(orderer: dict | None) -> None:
        if not orderer:
            return
        bs = orderer.get("BatchSize")
        if bs is not None:
            if "MaxMessageCount" in bs:
                bs["MaxMessageCount"] = int(values["MaxMessageCount"])
            if "AbsoluteMaxBytes" in bs:
                bs["AbsoluteMaxBytes"] = f"{int(values['AbsoluteMaxBytes'])} MB"
            if "PreferredMaxBytes" in bs:
                bs["PreferredMaxBytes"] = f"{int(values['PreferredMaxBytes'])} KB"
        if "BatchTimeout" in orderer:
            orderer["BatchTimeout"] = f"{int(values['BatchTimeout'])}s"

    _write(configtx.get("Orderer"))
    profiles = configtx.get("Profiles", {}) or {}
    for profile in profiles.values():
        _write((profile or {}).get("Orderer"))


def measure_tps_once() -> float:
    """Один прогон: network up + deploy CC -> caliper -> parse -> down."""
    try:
        start_network()
        start_benchmark()
        return observe_data()
    finally:
        stop_network()


def generate_design(n_pts: int, seed: int) -> np.ndarray:
    """SLHC через pySOT в 4-мерном пространстве BLOCK_PARAMS, все дискретные."""
    np.random.seed(seed)
    dim = len(BLOCK_PARAMS)
    if n_pts < 2 * dim + 1:
        print(
            f"[warn] для SLHC рекомендуется num_pts >= 2*dim+1 = {2*dim+1}; "
            f"запрошено {n_pts}",
            file=sys.stderr,
        )
    lb = np.array([float(p["bounds"][0]) for p in BLOCK_PARAMS])
    ub = np.array([float(p["bounds"][1]) for p in BLOCK_PARAMS])
    int_var = np.arange(dim, dtype=int)

    slhd = SymmetricLatinHypercube(dim=dim, num_pts=n_pts)
    pts = slhd.generate_points(lb=lb, ub=ub, int_var=int_var)
    return pts.astype(int)


def run_slhc_phase(design: np.ndarray, m_repeats: int, output: Path, resume: bool) -> pd.DataFrame:
    cols = ["design_idx", "run_id"] + [p["name"] for p in BLOCK_PARAMS] + ["tps"]
    existing = set()
    if resume and output.exists():
        df0 = pd.read_csv(output)
        for _, row in df0.iterrows():
            existing.add((int(row["design_idx"]), int(row["run_id"])))
    else:
        pd.DataFrame(columns=cols).to_csv(output, index=False)

    for n, point in enumerate(design, start=1):
        values = {p["name"]: int(v) for p, v in zip(BLOCK_PARAMS, point)}
        print("\n" + "=" * 60)
        print(f"design point {n}/{len(design)}: {values}")
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
                tps = measure_tps_once()
            except Exception as e:
                print(f"[error] design_idx={n} run={m}: {e}")
                tps = float("nan")
            dt = time.time() - t0
            print(f"[result] design_idx={n} run={m}: tps={tps} ({dt:.1f}s)")
            with open(output, "a") as f:
                tps_str = f"{tps:.6f}" if tps == tps else "nan"
                vals_str = ",".join(str(values[p["name"]]) for p in BLOCK_PARAMS)
                f.write(f"{n},{m},{vals_str},{tps_str}\n")

    return pd.read_csv(output)


def run_bo_phase(slhc_df: pd.DataFrame, n_iters: int, af: str) -> None:
    """Опциональный BO-«добор» поверх SLHC-данных.

    Берём средние tps по каждой design-точке как warm-start (X, y) и крутим
    GPyOpt в 4D-пространстве BLOCK_PARAMS. Целевая функция — measure_tps_once
    (с подстановкой block-параметров).
    """
    import GPyOpt

    print("\n" + "#" * 60)
    print(f"BO phase: af={af}, iters={n_iters}")
    print("#" * 60)

    means = slhc_df.groupby("design_idx").agg(
        {**{p["name"]: "first" for p in BLOCK_PARAMS}, "tps": "mean"}
    )
    X = means[[p["name"] for p in BLOCK_PARAMS]].to_numpy(dtype=float)
    Y = means[["tps"]].to_numpy(dtype=float)

    domain = [
        {
            "name": p["name"],
            "type": "discrete",
            "domain": tuple(range(int(p["bounds"][0]), int(p["bounds"][1]) + 1)),
        }
        for p in BLOCK_PARAMS
    ]

    def bo_objective(x):
        x = np.atleast_2d(x)[0]
        values = {p["name"]: int(round(v)) for p, v in zip(BLOCK_PARAMS, x)}
        configtx = load_yaml(CONFIGTX_PATH)
        apply_block_params(configtx, values)
        dump_yaml(CONFIGTX_PATH, configtx)
        try:
            tps = measure_tps_once()
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
    parser = argparse.ArgumentParser(description="SNR эксперимент по 4 channel-параметрам Orderer-а")
    parser.add_argument("-n", "--num-pts", type=int, default=10, help="N точек в SLHC (default 10)")
    parser.add_argument("-m", "--repeats", type=int, default=10, help="M повторов каждой точки (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для SLHC")
    parser.add_argument("--output", default="snr_block_results.csv", help="CSV с результатами")
    parser.add_argument("--resume", action="store_true", help="Не перезатирать output, продолжить эксперимент")
    parser.add_argument("--design-only", action="store_true", help="Только напечатать дизайн и выйти")
    parser.add_argument("--bo-iters", type=int, default=0, help="Дополнительные итерации BO после SLHC (default 0 = только SLHC)")
    parser.add_argument("--af", default="EI", choices=["EI", "LCB", "MPI"], help="Acquisition function для BO (default EI)")
    args = parser.parse_args()

    if not CONFIGTX_PATH.exists():
        sys.exit(f"не найден {CONFIGTX_PATH}")

    print("Параметры эксперимента:")
    for p in BLOCK_PARAMS:
        print(f"  {p['name']:<20} default={p['default']:<5} bounds={p['bounds']}  unit='{p['unit']}'")

    design = generate_design(args.num_pts, args.seed)
    print(f"\nSLHC дизайн (N={len(design)}):")
    print(pd.DataFrame(design, columns=[p["name"] for p in BLOCK_PARAMS]).to_string(index=True))

    if args.design_only:
        return

    out_path = Path(args.output)
    slhc_df = run_slhc_phase(design, args.repeats, out_path, args.resume)
    print(f"\nSLHC-фаза завершена. Записано строк: {len(slhc_df)}. Файл: {out_path}")

    if args.bo_iters > 0:
        run_bo_phase(slhc_df, args.bo_iters, args.af)


if __name__ == "__main__":
    main()
