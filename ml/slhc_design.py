"""Генерация симметричного латинского гиперкуба (SLHC) для swept-параметров.

Использует pySOT.experimental_design.SymmetricLatinHypercube — стабильный
вариант 0.3.3 (последний релиз, июнь 2020). Уже фигурирует в ml/requirements.txt.

Полученная сетка сохраняется в CSV (одна строка = одна точка), в порядке
параметров из текущего domain.controllable_params. Дальше её можно скармливать
в snr_delay_experiment.py / отдельный benchmark-раннер.

Пример:
    # 50 точек по всему 317-мерному домену:
    python slhc_design.py --num-pts 50 --out slhc_full.csv

    # SLHC только по подмножеству параметров (например, BatchTimeout +
    # MaxMessageCount, остальные на дефолтах):
    python slhc_design.py --num-pts 20 --grep "BatchTimeout|MaxMessageCount" \\
        --out slhc_subset.csv

CSV-формат: первая строка — заголовок с именами выбранных параметров.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pySOT.experimental_design import SymmetricLatinHypercube

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from globals import set_model_type  # noqa: E402

CONFIGS = [
    "../networks/configtx/configtx.yaml",
    "../networks/compose/docker/peercfg-org1/core.yaml",
    "../networks/compose/docker/peercfg-org2/core.yaml",
    "../networks/compose/docker/peercfg-org3/core.yaml",
    "../networks/compose/docker/peercfg-org4/core.yaml",
]


def select_params(all_params: list, grep: str | None, config_idx: int | None) -> list[int]:
    """Возвращает индексы выбранных параметров в `all_params`."""
    idxs = list(range(len(all_params)))
    if grep is not None:
        import re
        pat = re.compile(grep)
        idxs = [i for i in idxs if pat.search(all_params[i]["name"])]
    if config_idx is not None:
        idxs = [i for i in idxs if all_params[i]["config idx"] == config_idx]
    return idxs


def main():
    parser = argparse.ArgumentParser(description="SLHC design generator")
    parser.add_argument("--num-pts", type=int, required=True, help="N — число точек гиперкуба")
    parser.add_argument("--grep", default=None, help="Регулярка для фильтра имён параметров")
    parser.add_argument("--config-idx", type=int, default=None, help="Фильтр по config idx")
    parser.add_argument("--out", default="slhc_design.csv", help="Куда сохранить CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    set_model_type("BO")
    from domain import read_configs, controllable_params
    read_configs(CONFIGS)

    sel = select_params(controllable_params, args.grep, args.config_idx)
    if not sel:
        sys.exit("выборка параметров пустая (проверьте --grep/--config-idx)")
    params = [controllable_params[i] for i in sel]

    dim = len(params)
    if args.num_pts < 2 * dim + 1:
        print(
            f"[warn] pySOT SymmetricLatinHypercube требует num_pts >= 2*dim+1 "
            f"(=2*{dim}+1={2 * dim + 1}); запрошено {args.num_pts}. "
            f"С большой вероятностью генерация упадёт с 'No valid design found'.",
            file=sys.stderr,
        )

    lb = np.array([float(p["bounds"][0]) for p in params])
    ub = np.array([float(p["bounds"][1]) for p in params])
    int_var = np.array(
        [j for j, p in enumerate(params) if p["type"] == "discrete"],
        dtype=int,
    )

    slhd = SymmetricLatinHypercube(dim=dim, num_pts=args.num_pts)
    points = slhd.generate_points(lb=lb, ub=ub, int_var=int_var)

    df = pd.DataFrame(points, columns=[p["name"] for p in params])
    df.to_csv(args.out, index=False)
    print(f"Сгенерировано {args.num_pts} точек в {dim}-мерном SLHC.")
    print(f"int-параметры: {len(int_var)} / continuous: {dim - len(int_var)}.")
    print(f"Сохранено в {args.out}")


if __name__ == "__main__":
    main()
