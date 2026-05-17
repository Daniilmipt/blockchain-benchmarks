"""Печатает текущий BO-домен из ml/domain.py.

Полезно, чтобы понять, какие параметры BO реально перебирает и в каких границах,
без запуска самого BO.

Запуск (из ml/):
    python explain_domain.py
    python explain_domain.py --grep BatchTimeout
    python explain_domain.py --config-idx 0    # только параметры из configtx
"""

import argparse
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grep", default=None, help="Подстрока в имени параметра (без учёта регистра)")
    parser.add_argument("--config-idx", type=int, default=None, help="Показать только из этого config idx")
    args = parser.parse_args()

    set_model_type("BO")
    from domain import read_configs, controllable_params

    read_configs(CONFIGS)

    print("Список конфигов:")
    for i, cfg in enumerate(CONFIGS):
        print(f"  [{i}] {cfg}")

    items = controllable_params
    if args.grep is not None:
        s = args.grep.lower()
        items = [p for p in items if s in p["name"].lower()]
    if args.config_idx is not None:
        items = [p for p in items if p["config idx"] == args.config_idx]

    print(f"\nВсего параметров в домене: {len(controllable_params)}  показываем: {len(items)}\n")
    print(f"{'idx':>4} {'cfg':>3} {'type':<10} {'default':>10}  {'bounds':<22}  name")
    print("-" * 120)
    for i, p in enumerate(items, start=1):
        b = p["bounds"]
        bounds_str = f"({b[0]:g}, {b[1]:g})"
        print(f"{i:>4} {p['config idx']:>3} {p['type']:<10} {p['default value']!r:>10}  {bounds_str:<22}  {p['name']}")


if __name__ == "__main__":
    main()
