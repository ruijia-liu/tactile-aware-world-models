"""Generate fixed assignments before simulating or training."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import random

import numpy as np

MATERIALS = list(product(("s03", "s55"), ("p06", "p80")))


def development(stage):
    branches = stage == "branches"
    first, per_cell = (100, 6) if branches else (400, 2)
    rng = np.random.default_rng(90638 + first)
    conditions = list(product(("visible", "mask"), ("stable", "perturbed")))
    order_balance = None if branches else rng.integers(0, 2, size=4)
    rows = []
    for i, slot in enumerate(rng.permutation(len(conditions) * per_cell)):
        cell, repeat = divmod(int(slot), per_cell)
        visibility, contact = conditions[cell]
        route = first + i
        material = (repeat + cell) % 4
        order = ["tight", "hold", "relieved"] if branches else ["blind", "touch"]
        if not branches:
            material = 2 * repeat + ((cell // 2 + cell % 2) % 2)
            if repeat != order_balance[cell]:
                order.reverse()
        offset = float((-1)**repeat * rng.uniform(.002, .004)) if contact == "perturbed" else 0.
        assigned_order = rng.permutation(order).tolist() if branches else order
        row = dict(route_index=route, geometry_seed=88113+route, visibility=visibility,
                   contact=contact, contact_offset_y_m=offset, plug_material=MATERIALS[material][0],
                   pad_material=MATERIALS[material][1],
                   split=("train" if repeat < 4 else "select") if branches else "closed_loop_validation",
                   pair_action_order=assigned_order)
        if not branches:
            row["hold_seconds"] = .35
        rows.append(row)
    return rows


def confirmation():
    assignment_seed = 20261123
    rng = random.Random(assignment_seed)
    routes = list(range(1000, 1100))
    rng.shuffle(routes)
    rows = []
    for cell, (visibility, contact) in enumerate(product(("visible", "mask"), ("nominal", "perturbed"))):
        # Four materials appear six or seven times in each cell, 25 times overall.
        materials = MATERIALS * 6 + [MATERIALS[cell]]
        blind_first = [True] * (12 + cell % 2) + [False] * (13 - cell % 2)
        positive = 12 + int(visibility == "mask")
        signs = [1] * positive + [-1] * (25 - positive)
        for seq in (materials, blind_first, signs):
            rng.shuffle(seq)
        for i, (plug, pads) in enumerate(materials):
            index = routes.pop()
            family = f"route{index:03d}"
            offset = signs[i] * rng.randint(2000, 4000) / 1_000_000 if contact == "perturbed" else 0.
            first = ["blind", "touch"] if blind_first[i] else ["touch", "blind"]
            rows.append(dict(route_index=index, family_id=family, geometry_seed=88113 + index,
                             reset_seed=331106 + index, scenario_id=f"{family}_{plug}_{pads}",
                             plug_material=plug, pad_material=pads, contact_offset_y_m=offset,
                             contact=contact, visibility=visibility, pair_action_order=first))
    rows.sort(key=lambda row: row["route_index"])
    if (len({r["geometry_seed"] for r in rows}) != 100
            or set(Counter((r["visibility"], r["contact"]) for r in rows).values()) != {25}):
        raise ValueError("Invalid scenario allocation")
    return dict(protocol="flexitac-paired-confirmation-v1", assignment_seed=assignment_seed,
                scenario_count=100, episodes_per_scenario=2, trials=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for filename, rows in (("branches.json", development("branches")),
                           ("validation.json", development("validation")), ("manifest.json", confirmation())):
        path = args.output / filename
        data = (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Existing assignment differs: {path}")
        else:
            path.write_bytes(data)
        print(path)


if __name__ == "__main__":
    main()
